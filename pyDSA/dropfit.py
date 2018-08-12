# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA
# Version: 0.1

# This file is part of pyDSA

# pyDSA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy.interpolate as spint
import scipy.optimize as spopt
import scipy.misc as spmisc
from IMTreatment import Points, Profile
import IMTreatment.plotlib as pplt
from .helpers import fit_circle, fit_ellipse, get_ellipse_points


__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"

# TODO: remove tests fot self.fits existence: they always exist !
# TODO: recheck all the docstrings


class DropFit(object):
    def __init__(self, baseline, x_bounds, y_bounds):
        """
        A droplet edges fitting.

        Parameters
        ----------
        im: Image object
            Image at the origin of the fit
        """
        self.baseline = baseline
        self.thetas = None
        self.triple_pts = None
        self.thetas_triple = None
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.colors = pplt.get_color_cycles()

    def display(self):
        """
        """
        plt.axis('equal')
        plt.gca().set_adjustable('box')
        # Display baseline
        if self.baseline is not None:
            x0 = self.x_bounds[0]
            xf = self.x_bounds[-1]
            x0 -= np.abs(xf - x0)*.1
            xf += np.abs(xf - x0)*.1
            self.baseline.display(x0, xf, color=self.colors[0])

    def _get_angle_display_lines(self):
        bs_angle = self.baseline.tilt_angle*180/np.pi
        lines = []
        length = (self.y_bounds[-1] - self.y_bounds[0])/3
        if self.thetas is not None:
            theta1 = (self.thetas[0] + bs_angle)/180*np.pi
            theta2 = (self.thetas[1] + bs_angle)/180*np.pi
            xy_inter = self._get_inters_base_fit()
            y1 = xy_inter[0][1]
            y2 = xy_inter[1][1]
            x1 = xy_inter[0][0]
            x2 = xy_inter[1][0]
            lines.append([[x1, x1 + length*np.cos(theta1)],
                          [y1, y1 + length*np.sin(theta1)]])
            lines.append([[x2, x2 + length*np.cos(theta2)],
                          [y2, y2 + length*np.sin(theta2)]])
        else:
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
        if self.thetas_triple is not None:
            # contact angle with triple points
            theta1 = (self.thetas_triple[0] + bs_angle)/180*np.pi
            theta2 = (self.thetas_triple[1] + bs_angle)/180*np.pi
            xy_inter = self.triple_pts
            y1 = xy_inter[0][1]
            y2 = xy_inter[1][1]
            x1 = xy_inter[0][0]
            x2 = xy_inter[1][0]
            lines.append([[x1, x1 + length*np.cos(theta1)],
                          [y1, y1 + length*np.sin(theta1)]])
            lines.append([[x2, x2 + length*np.cos(theta2)],
                          [y2, y2 + length*np.sin(theta2)]])
        else:
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
        return lines

    def get_ridge_height(self):

        """
        Return the ridge height.
        """
        pts = self.triple_pts
        if pts is None:
            return np.nan, np.nan
        heights = [self.baseline.get_distance_to_baseline(pt)
                   for pt in pts]
        return np.array(heights)

    def get_base_diameter(self):
        """
        Return the base diameter.
        """
        pt1, pt2 = self._get_inters_base_fit()
        diam = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        return diam


class DropSplineFit(DropFit):
    def __init__(self, fits, x_bounds, y_bounds, baseline):
        """
        A droplet edges spline fitting.

        Parameters
        ----------
        im: Image object
            Image at the origin of the fit
        fits: List of 2 functions
            Spline fit for the two drop edges.
        """
        super().__init__(baseline=baseline,
                         x_bounds=x_bounds,
                         y_bounds=y_bounds)
        self.fits = fits

    def _detect_triple_points_as_x_minima(self, edge_number, verbose):
        fit = self.fits[edge_number]
        y = np.linspace(self.x_bounds[0], self.x_bounds[1], 100)
        dy = (y[-1] - y[0])/100
        def zerofun(y):
            return spmisc.derivative(fit, y, dx=dy, n=1, order=3)
        def dzerofun(y):
            return spmisc.derivative(zerofun, y, dx=dy, n=1, order=3)
        y0 = y[0]
        deriv = zerofun(y)
        deriv_sign = deriv[1:]*deriv[:-1]
        inds = np.where(deriv_sign < 0)[0] + 1
        # If no x minima
        if len(inds) == 0:
            if verbose:
                plt.figure()
                plt.plot(y, zerofun(y), 'o-')
                plt.xlabel('y')
                plt.xlabel('dy/dx')
                plt.title('x-minima method failed: no minima')
                plt.grid()
            return None
        # Choose first point with right curvature sign
        for indi in inds:
            curv = dzerofun(y[indi])
            if ((edge_number == 0 and curv < 0) or
                (edge_number == 1 and curv > 0)):
                ind = indi
                break
        # Find the accurate position
        y0 = spopt.brentq(zerofun, y[ind-1], y[ind])
        # verbose
        if verbose:
            plt.figure()
            plt.plot(y, zerofun(y), 'o-')
            plt.xlabel('y')
            plt.xlabel('dy/dx')
            plt.axvline(y[ind - 1], ls='--', color='k')
            plt.axvline(y[ind], ls='--', color='k')
            plt.title('x-minima method')
            plt.grid()
        return [fit(y0), y0]

    def _detect_triple_points_as_curvature_change(self, edge_number, verbose):
        fit = self.fits[edge_number]
        y = np.linspace(self.x_bounds[0], self.x_bounds[1], 100)
        dy = (y[-1] - y[0])/100
        def zerofun(y):
            dxdy = spmisc.derivative(fit, y, dx=dy, n=1, order=3)
            dxdy2 = spmisc.derivative(fit, y, dx=dy, n=2, order=3)
            return dxdy2/(1 + dxdy**2)**(3/2)
        def dzerofun(y):
            return spmisc.derivative(zerofun, y, dx=dy, n=1, order=5)
        if verbose:
            plt.figure()
            plt.plot(y, zerofun(y), 'o-')
            plt.xlabel('y')
            plt.ylabel('d^2y/dx^2')
            plt.title('curvature method')
            plt.grid()
        # Get the triple point iteratively
        while True:
            y0 = self._get_curvature_root(y=y, zerofun=zerofun,
                                          verbose=verbose)
            if y0 is None:
                return None
            # check if the triple point is curvature coherent,
            # else, find the next one
            deriv = dzerofun(y0)
            if (edge_number == 0 and deriv < 0) or (edge_number == 1
                                                    and deriv > 0):
                y = y[y > y0]
            else:
                break
            # sanity check
            if len(y) == 0:
                if verbose:
                    warnings.warn('Cannot find a decent triple point')
                return None
        # Ok, good to go
        if verbose:
            plt.figure()
            plt.plot(y, zerofun(y), 'o-')
            plt.plot(y0, 0, "ok")
            plt.xlabel('y')
            plt.xlabel('d^2y/dx^2')
            plt.grid()
        return [fit(y0), y0]

    def _get_curvature_root(self, y, zerofun, verbose):
        try:
            # get last point with opposite sign
            y0 = y[0]
            yf = y[-1]
            while True:
                if zerofun(y0)*zerofun(yf) > 0 and \
                    abs(y0 - yf)/((y0 + yf)/2) > 0.01:
                    yf -= (yf - y0)*1/10
                else:
                    break
            # get exact position
            y0 = spopt.brentq(zerofun, y0, yf)
            return y0
        except (RuntimeError, ValueError) as m:
            if verbose:
                warnings.warn('Cannot find a triple point here.'
                              '\nYou should try a different fitting.'
                              '\nError message:{}'.format(m))
            return None

    def detect_triple_points(self, verbose=False, use_x_minima=False):
        """
        Compute the triple points (water, oil and air interfaces) position.

        Parameters
        ==========
        use_x_minima: boolean
            If True, try to define the triple point as the minimal x values and
            fall back to the curvature method if necessary.
            (default to False).

        Returns
        =======
        tripl_pts: 2x2 array of numbers
           Position of the triple points for each edge ([pt1, pt2])
        """
        # Checks
        if self.fits is None:
            raise Exception("You should computing fitting first with 'fit()'")
        tripl_pts = [None, None]
        for i in [0, 1]:
            # Try the x minima method
            if use_x_minima:
                tp = self._detect_triple_points_as_x_minima(edge_number=i,
                                                            verbose=verbose)
                if tp is None:
                    tripl_pts[i] = [np.nan, np.nan]
                    if verbose:
                        print("Use of the x minima failed")
                else:
                    tripl_pts[i] = tp
                    continue
            # Try the curvature change method
            tp = self._detect_triple_points_as_curvature_change(edge_number=i,
                                                                verbose=verbose)
            if tp is None:
                tripl_pts[i] = [np.nan, np.nan]
                if verbose:
                    print("Use of the curvature change failed")
            else:
                tripl_pts[i] = tp
        # Store and return
        if verbose:
            plt.show()
        self.triple_pts = tripl_pts
        return tripl_pts

    def _get_inters_base_fit(self, verbose=False):
        flat = False
        try:
            bfun = self.baseline.get_baseline_fun(along_y=True)
        except Exception:
            flat = True
        xys = []
        for i in range(2):
            sfun = self.fits[i]
            if np.isnan(sfun(0)):
                x_inter = np.nan
                y_inter = np.nan
            elif flat:
                y_inter = self.baseline.pt1[1]
                x_inter = sfun(y_inter)
            else:
                y_inter = spopt.fsolve(lambda y: bfun(y) - sfun(y),
                                       (self.baseline.pt1[1] +
                                        self.baseline.pt2[1])/2)[0]
                x_inter = bfun(y_inter)
            xys.append([x_inter, y_inter])
        if verbose:
            y = np.linspace(self.y_bounds[0], self.y_bounds[-1], 100)
            x = np.linspace(self.baseline.pt1[0], self.baseline.pt2[0], 100)
            bfun = self.baseline.get_baseline_fun()
            plt.figure()
            plt.plot([xys[0][0], xys[1][0]], [xys[0][1], xys[1][1]], "ok",
                     label="intersection")
            plt.plot(x, bfun(x), label="base")
            plt.plot(sfun(y), y, label="fit")
            plt.legend()
            plt.show()
        return xys

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        bs_angle = self.baseline.tilt_angle*180/np.pi
        # Compute base contact angle
        xy_inter = self._get_inters_base_fit()
        self.thetas = self._compute_fitting_angle_at_pts(xy_inter)
        # correct regardin baseline angle
        self.thetas -= bs_angle
        # Compute triple point contact angle
        if self.triple_pts is not None:
            xy_tri = self.triple_pts
            self.thetas_triple = self._compute_fitting_angle_at_pts(xy_tri)
            # correct regardin baseline angle
            self.thetas_triple -= bs_angle
        # display if asked
        if verbose:
            self.display()

    def _compute_fitting_angle_at_pts(self, pts):
        thetas = []
        for i in range(2):
            x_inter, y_inter = pts[i]
            sfun = self.fits[i]
            # Get gradient
            dy = (self.x_bounds[1] - self.x_bounds[0])/100
            deriv = spmisc.derivative(sfun, y_inter, dx=dy)
            theta = np.pi*1/2 - np.arctan(deriv)
            thetas.append(theta/np.pi*180)
        return np.array(thetas)

    def display(self, displ_fits=True, displ_ca=True,
                displ_tp=True, *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if self.fits is not None and displ_fits:
            xy_inter = self._get_inters_base_fit()
            y1 = np.linspace(xy_inter[0][1],
                             self.y_bounds[-1],
                             1000)
            y2 = np.linspace(xy_inter[1][1],
                             self.y_bounds[-1],
                             1000)
            x1 = self.fits[0](y1)
            x2 = self.fits[1](y2)
            plt.plot(x1, y1, color=self.colors[1])
            plt.plot(x2, y2, color=self.colors[1])
        # Display contact angles
        if displ_ca:
            lines = self._get_angle_display_lines()
            for line in lines:
                plt.plot(line[0], line[1],
                         color=self.colors[0])
                plt.plot(line[0][0], line[0][1],
                         color=self.colors[0])
        # Display triple points
        if self.triple_pts is not None and displ_tp:
            for i in [0, 1]:
                plt.plot(self.triple_pts[i][0],
                         self.triple_pts[i][1],
                         marker="o",
                         color=self.colors[4])


class DropCircleFit(DropFit):
    def __init__(self, xyc, R, baseline, x_bounds, y_bounds):
        super().__init__(baseline=baseline,
                         x_bounds=x_bounds,
                         y_bounds=y_bounds)
        self.fits = [xyc, R]

    def _get_inters_base_fit(self):
        """
        """
        # getting intersection points
        # from: http://mathworld.wolfram.com/Circle-LineIntersection.html
        (xc, yc), r = self.fits
        xbas1 = self.baseline.pt1[0] - xc
        xbas2 = self.baseline.pt2[0] - xc
        ybas1 = self.baseline.pt1[1] - yc
        ybas2 = self.baseline.pt2[1] - yc
        dx = xbas2 - xbas1
        dy = ybas2 - ybas1
        dr = (dx**2 + dy**2)**.5
        D = xbas1*ybas2 - xbas2*ybas1
        #
        x1 = (D*dy + np.sign(dy)*dx*(r**2*dr**2 - D**2)**.5)/dr**2
        x2 = (D*dy - np.sign(dy)*dx*(r**2*dr**2 - D**2)**.5)/dr**2
        y1 = (-D*dx + abs(dy)*(r**2*dr**2 - D**2)**.5)/dr**2
        y2 = (-D*dx - abs(dy)*(r**2*dr**2 - D**2)**.5)/dr**2
        #
        x1 += xc
        x2 += xc
        y1 += yc
        y2 += yc
        return [[x1, y1], [x2, y2]]

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        bs_angle = self.baseline.tilt_angle*180/np.pi
        # Compute circle fits contact angles
        (xc, yc), R = self.fits
        pts = self._get_inters_base_fit()
        thetas = []
        for pt in pts:
            theta = np.pi/2 + np.arctan((yc - pt[1])/(xc - pt[0]))
            thetas.append(theta)
        self.thetas = np.array(thetas)*180/np.pi
        # correct regarding the baseline angle
        self.thetas -= bs_angle
        # display if asked
        if verbose:
            self.display()

    def display(self, displ_fits=True, displ_ca=True,
                *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if self.fits is not None and displ_fits:
            xyc, R = self.fits
            plt.plot(xyc[0], xyc[1], marker='o',
                     color=self.colors[5])
            circ = plt.Circle((xyc[0], xyc[1]), radius=R,
                              color=self.colors[5],
                              fill=False)
            plt.gca().add_artist(circ)
        # Display contact angles
        if displ_ca:
            lines = self._get_angle_display_lines()
            for line in lines:
                plt.plot(line[0], line[1],
                         color=self.colors[0])
                plt.plot(line[0][0], line[0][1],
                         color=self.colors[0])


class DropCirclesFit(DropFit):
    def __init__(self, xyc, Rs, baseline, triple_pts, x_bounds, y_bounds):
        super().__init__(baseline=baseline,
                         x_bounds=x_bounds,
                         y_bounds=y_bounds)
        self.fits = [[xyc[i], Rs[i]] for i in range(len(xyc))]
        self.triple_pts = triple_pts

    def _get_inters_base_fit(self):
        """
        """
        # getting intersection points
        # from: http://mathworld.wolfram.com/Circle-LineIntersection.html
        (xc, yc), r = self.fits[0]
        xbas1 = self.baseline.pt1[0] - xc
        xbas2 = self.baseline.pt2[0] - xc
        ybas1 = self.baseline.pt1[1] - yc
        ybas2 = self.baseline.pt2[1] - yc
        dx = xbas2 - xbas1
        dy = ybas2 - ybas1
        dr = (dx**2 + dy**2)**.5
        D = xbas1*ybas2 - xbas2*ybas1
        #
        x1 = (D*dy + np.sign(dy)*dx*(r**2*dr**2 - D**2)**.5)/dr**2
        x2 = (D*dy - np.sign(dy)*dx*(r**2*dr**2 - D**2)**.5)/dr**2
        y1 = (-D*dx + abs(dy)*(r**2*dr**2 - D**2)**.5)/dr**2
        y2 = (-D*dx - abs(dy)*(r**2*dr**2 - D**2)**.5)/dr**2
        #
        x1 += xc
        x2 += xc
        y1 += yc
        y2 += yc
        return [[x1, y1], [x2, y2]]

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        bs_angle = self.baseline.tilt_angle*180/np.pi
        # Compute circle fits contact angles
        (xc, yc), R = self.fits[0]
        pts = self._get_inters_base_fit()
        thetas = []
        for pt in pts:
            theta = np.pi/2 + np.arctan((yc - pt[1])/(xc - pt[0]))
            thetas.append(theta)
        self.thetas = np.array(thetas)*180/np.pi
        # correct regarding baseline angle
        self.thetas -= bs_angle
        # display if asked
        if verbose:
            self.display()

    def display(self, displ_fits=True, displ_ca=True,
                displ_tp=True, *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if self.fits is not None and displ_fits:
            for cf in self.fits:
                xyc, R = cf
                plt.plot(xyc[0], xyc[1], marker='o',
                         color=self.colors[5])
                circ = plt.Circle((xyc[0], xyc[1]), radius=R,
                                  color=self.colors[5],
                                  fill=False)
                plt.gca().add_artist(circ)
        # Display contact angles
        if displ_ca:
            lines = self._get_angle_display_lines()
            for line in lines:
                plt.plot(line[0], line[1],
                         color=self.colors[0])
                plt.plot(line[0][0], line[0][1],
                         color=self.colors[0])
        # Display triple points from circle fits
        if self.triple_pts is not None and displ_tp:
            for tp in self.triple_pts:
                plt.plot(tp[0], tp[1], marker='o',
                         color=self.colors[5])


class DropEllipseFit(DropFit):
    def __init__(self, xyc, R1, R2, theta, baseline, x_bounds, y_bounds):
        super().__init__(baseline, x_bounds, y_bounds)
        self.fits = [xyc, R1, R2, theta]

    def _get_inters_base_fit(self):
        """
        """
        (h, k), a, b, theta = self.fits
        # Rotate the baseline in the ellipse referential
        tmpbs = self.baseline.copy()
        tmpbs.set_origin(h, k)
        tmpbs.rotate(-theta)
        tmpbs.set_origin(-h, -k)
        m, c = tmpbs.coefs
        # from: http://ambrsoft.com/TrigoCalc/Circles2/Ellipse/EllipseLine.htm
        eps = c - k
        delta = c + m*h
        # x
        A = h*b**2 - m*a**2*eps
        B = a*b*(a**2*m**2 + b**2 - delta**2 - k**2 + 2*delta*k)**.5
        C = a**2*m**2 + b**2
        x1 = (A + B)/C
        x2 = (A - B)/C
        # y
        D = b**2*delta + k*a**2*m**2
        E = a*b*m*(a**2*m**2 + b**2 - delta**2 - k**2 + 2*delta*k)**.5
        F = C
        y1 = (D + E)/F
        y2 = (D - E)/F
        # Rotate back the point in the base referential
        xs = [x1, x2]
        ys = [y1, y2]
        x1, x2 = h + np.cos(theta)*(xs - h) - np.sin(theta)*(ys - k)
        y1, y2 = k + np.sin(theta)*(xs - h) + np.cos(theta)*(ys - k)
        if x1 < x2:
            return [[x1, y1], [x2, y2]]
        else:
            return [[x2, y2], [x1, y1]]

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        bs_angle = self.baseline.tilt_angle*180/np.pi
        # Compute ellipse fit contact angle
        xy_inter = self._get_inters_base_fit()
        bs_angle = self.baseline.tilt_angle
        (xc, yc), R1, R2, theta = self.fits
        thetas = []
        for i, xy in enumerate(xy_inter):
            x_ref = (xy[0] - xc)*np.cos(theta) + (xy[1] - yc)*np.sin(theta)
            y_ref = (xy[1] - yc)*np.cos(theta) - (xy[0] - xc)*np.sin(theta)
            slope = -(R2**2*x_ref)/(R1**2*y_ref)
            thet = np.arctan(slope)
            if np.sin(thet) < 0:
                thet += np.pi
            thet += (theta - bs_angle)
            thetas.append(thet)
        self.thetas = np.array(thetas)*180/np.pi
        # display if asked
        if verbose:
            self.display()

    def display(self, displ_fits=True, displ_ca=True,
                *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if displ_fits:
            (xc, yc), R1, R2, theta = self.fits
            elxs, elys = get_ellipse_points(xc, yc, R1, R2, theta, res=100)
            rxs = [xc + R1*np.cos(theta),
                   xc,
                   xc + R2*np.cos(theta + np.pi/2)]
            rys = [yc + R1*np.sin(theta),
                   yc,
                   yc + R2*np.sin(theta + np.pi/2)]
            plt.plot(rxs, rys, color=self.colors[3], ls=":")
            plt.plot(elxs, elys, color=self.colors[3])
            plt.plot(xc, yc, color=self.colors[3], marker='o', ls='none')
        # Display contact angles
        if displ_ca:
            lines = self._get_angle_display_lines()
            for line in lines:
                plt.plot(line[0], line[1],
                         color=self.colors[0])
                plt.plot(line[0][0], line[0][1],
                         color=self.colors[0])