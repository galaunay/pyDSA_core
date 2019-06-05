# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA_core

# This file is part of pyDSA_core

# pyDSA_core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.misc as spmisc
import IMTreatment.plotlib as pplt
from . import helpers


__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
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
        self.fits = None
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

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        return np.array([[np.nan], [np.nan]])

    def get_drop_center(self):
        raise NotImplementedError("Not implemented yet")

    def get_drop_position(self):
        """ Return the position of the droplet edges. """
        try:
            return self._get_inters_base_fit()
        except:
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])

    def get_base_diameter(self):
        """
        Return the base diameter.
        """
        pt1, pt2 = self._get_inters_base_fit()
        diam = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        return diam

    def get_drop_height(self):
        """ Return the height of the droplet center. """
        xyc = self.get_drop_center()
        hb = self.baseline.get_projection_to_baseline(xyc)[1]
        hmax = np.max(self.y_bounds)
        return hmax - hb

    def get_drop_area(self):
        raise NotImplementedError("Not implemented yet")

    def get_drop_volume(self):
        raise NotImplementedError("Not implemented yet")

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
        t = np.linspace(0, 1, 100)
        dt = 1/100
        def zerofun(t):
            dxdt = spmisc.derivative(fit[0], t, dx=dt, n=1, order=3)
            dydt = spmisc.derivative(fit[1], t, dx=dt, n=1, order=3)
            dxdy = dxdt*(1/dydt)
            return dxdy
        def dzerofun(t):
            return spmisc.derivative(zerofun, t, dx=dt, n=1, order=5)
        t0 = t[0]
        deriv = zerofun(t)
        deriv_sign = deriv[1:]*deriv[:-1]
        inds = np.where(deriv_sign < 0)[0] + 1
        # If no x minima
        if len(inds) == 0:
            if verbose:
                plt.figure()
                plt.plot(t, zerofun(t), 'o-')
                plt.xlabel('y')
                plt.xlabel('dy/dx')
                plt.title('x-minima method failed: no minima')
                plt.grid()
            return None
        # Choose first point with right curvature sign
        for indi in inds:
            curv = dzerofun(t[indi])
            if ((edge_number == 0 and curv < 0) or
                (edge_number == 1 and curv > 0)):
                ind = indi
                break
        # Find the accurate position
        t0 = spopt.brentq(zerofun, t[ind-1], t[ind])
        # verbose
        if verbose:
            plt.figure()
            plt.plot(t, zerofun(t), 'o-')
            plt.xlabel('y')
            plt.xlabel('dy/dx')
            plt.axvline(t[ind - 1], ls='--', color='k')
            plt.axvline(t[ind], ls='--', color='k')
            plt.title('x-minima method')
            plt.grid()
        return [fit[0](t0), fit[1](t0)]

    def _detect_triple_points_as_curvature_change(self, edge_number, verbose):
        fit = self.fits[edge_number]
        t = np.linspace(0, 1, 100)
        dt = 1/100
        def zerofun(t):
            dxdt = spmisc.derivative(fit[0], t, dx=dt, n=1, order=3)
            dydt = spmisc.derivative(fit[1], t, dx=dt, n=1, order=3)
            dxdy = dxdt*(1/dydt)
            dxdt2 = spmisc.derivative(fit[0], t, dx=dt, n=2, order=3)
            dydt2 = spmisc.derivative(fit[1], t, dx=dt, n=2, order=3)
            dxdy2 = dxdt2*(1/dydt2)
            return dxdy2/(1 + dxdy**2)**(3/2)
        def dzerofun(t):
            return spmisc.derivative(zerofun, t, dx=dt, n=1, order=5)
        if verbose:
            plt.figure()
            plt.plot(t, zerofun(t), 'o-')
            plt.xlabel('y')
            plt.ylabel('d^2y/dx^2')
            plt.title('curvature method')
            plt.grid()
        # Get the triple point iteratively
        while True:
            t0 = self._get_curvature_root(t=t, zerofun=zerofun,
                                          verbose=verbose)
            if t0 is None:
                return None
            # check if the triple point is curvature coherent,
            # else, find the next one
            deriv = dzerofun(t0)
            if (edge_number == 0 and deriv < 0) or (edge_number == 1
                                                    and deriv > 0):
                t = t[t > t0]
            else:
                break
            # sanity check
            if len(t) == 0:
                if verbose:
                    warnings.warn('Cannot find a decent triple point')
                return None
        # Ok, good to go
        if verbose:
            plt.figure()
            plt.plot(t, zerofun(t), 'o-')
            plt.plot(t0, 0, "ok")
            plt.xlabel('y')
            plt.xlabel('d^2y/dx^2')
            plt.grid()
        return [fit[0](t0), fit[1](t0)]

    def _get_curvature_root(self, t, zerofun, verbose):
        try:
            # get last point with opposite sign
            t0 = t[0]
            tf = t[-1]
            while True:
                if zerofun(t0)*zerofun(tf) > 0 and \
                    abs(t0 - tf)/((t0 + tf)/2) > 0.01:
                    tf -= (tf - t0)*1/10
                else:
                    break
            # get exact position
            t0 = spopt.brentq(zerofun, t0, tf)
            return t0
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
            tp = self._detect_triple_points_as_curvature_change(
                edge_number=i,
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
            sfunx, sfuny = self.fits[i]
            if np.isnan(sfunx(0)):
                x_inter = np.nan
                y_inter = np.nan
            elif flat:
                y_inter = self.baseline.pt1[1]
                t_inter = spopt.fsolve(lambda t: y_inter - sfuny(t), 0.5)[0]
                x_inter = sfunx(t_inter)
            else:
                t_inter = spopt.fsolve(lambda t: bfun(sfuny(t)) - sfunx(t),
                                       0.5)[0]
                x_inter = sfunx(t_inter)
                y_inter = sfuny(t_inter)
            xys.append([x_inter, y_inter])
        if verbose:
            x = np.linspace(self.baseline.pt1[0], self.baseline.pt2[0], 100)
            bfun = self.baseline.get_baseline_fun()
            plt.figure()
            plt.plot([xys[0][0], xys[1][0]], [xys[0][1], xys[1][1]], "ok",
                     label="intersection")
            plt.plot(x, bfun(x), label="base")
            t = np.linspace(0, 1, 100)
            plt.plot(sfunx(t), sfuny(t), label="fit")
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
        # correct regarding baseline angle
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
            sfunx, sfuny = self.fits[i]
            t_inter = spopt.fsolve(lambda t: y_inter - sfuny(t), 0.5)
            # Get gradient
            dt = 0.01
            derivx = spmisc.derivative(sfunx, t_inter, dx=dt)
            derivy = spmisc.derivative(sfuny, t_inter, dx=dt)
            theta = np.arctan2(derivy, derivx)[0]
            theta = theta % (2*np.pi)
            thetas.append(theta/np.pi*180)
        return np.array(thetas)

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        (fit1x, fit1y), (fit2x, fit2y) = self.fits
        t = np.linspace(0, 1, resolution)
        x1 = fit1x(t)
        y1 = fit1y(t)
        x2 = fit2x(t)
        y2 = fit2y(t)
        xs = np.concatenate((x1, [np.nan], x2))
        ys = np.concatenate((y1, [np.nan], y2))
        pts = [xs, ys]
        return pts

    def display(self, displ_fits=True, displ_ca=True,
                displ_tp=True, *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if self.fits is not None and displ_fits:
            t = np.linspace(0, 1, 1000)
            x1 = self.fits[0][0](t)
            y1 = self.fits[0][1](t)
            x2 = self.fits[1][0](t)
            y2 = self.fits[1][1](t)
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

    def get_drop_center(self):
        """ Return the center of the droplet 2D projection. """
        t = np.linspace(0, 1, 1000)
        xs = np.concatenate((self.fits[0][0](t), self.fits[1][0](t)[::-1]))
        ys = np.concatenate((self.fits[0][1](t), self.fits[1][1](t)[::-1]))
        # shoelace
        # Credit to Elfego Ruiz-Gutierrez
        A = self.get_drop_area()
        z = xs * np.roll(ys, 1) - ys * np.roll(xs, 1)
        xx = xs + np.roll(xs, 1)
        yy = ys + np.roll(ys, 1)
        center = np.r_[np.dot(xx, z), np.dot(yy, z)]/A/6.0
        return center

    def get_drop_area(self):
        """ Return the droplet 2D projection area. """
        t = np.linspace(0, 1, 1000)
        xs = np.concatenate((self.fits[0][0](t), self.fits[1][0](t)[::-1]))
        ys = np.concatenate((self.fits[0][1](t), self.fits[1][1](t)[::-1]))
        # Shoelace method
        area = 0.5*np.abs(np.dot(xs, np.roll(ys, 1))
                          - np.dot(ys, np.roll(xs, 1)))
        return area

    def get_drop_volume(self):
        raise Exception('Cannot get the drop volume from a spline fit')


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
        if dy == 0:
            sign_dy = 1
        else:
            sign_dy = np.sign(dy)
        x1 = (D*dy + sign_dy*dx*(r**2*dr**2 - D**2)**.5)/dr**2
        x2 = (D*dy - sign_dy*dx*(r**2*dr**2 - D**2)**.5)/dr**2
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
        pt1, pt2 = self._get_inters_base_fit()
        thetas = [- np.pi/2 + np.arctan2((yc - pt1[1]), (xc - pt1[0])),
                  np.pi/2 + np.arctan2((yc - pt2[1]), (xc - pt2[0]))]
        # Be sure to be in the right side of the baseline
        thetas[0] = thetas[0]
        thetas[1] = thetas[1]
        beta = self.baseline.tilt_angle
        if (beta - thetas[0]) % (2*np.pi) < np.pi:
            thetas[0] = (thetas[0] + np.pi) % (2*np.pi)
        if (beta - thetas[1]) % (2*np.pi) < np.pi:
            thetas[1] = (thetas[1] + np.pi) % (2*np.pi)
        # Convert to degree
        self.thetas = np.array(thetas)*180/np.pi
        # correct regarding the baseline angle
        self.thetas -= bs_angle
        # display if asked
        if verbose:
            self.display()

    def get_drop_center(self):
        """
        Return the center of the drop.
        """
        xc, yc = self.fits[0]
        # r = self.fits[1]
        # h = yc - self.baseline.get_projection_to_baseline([xc, yc])[1]
        # Need to be in the baseline referential...
        print("Warning: not in the baseline referential")
        return np.array(xc, yc, dtype=float)

    def get_drop_area(self):
        """
        Return the area of the 2D drop projection.
        """
        r = self.fits[1]
        xc, yc = self.fits[0]
        h = yc - self.baseline.get_projection_to_baseline([xc, yc])[1]
        theta = 2*(np.pi - np.arccos(h/r))
        # from wikipedia: circular segments
        area = r**2/2*(theta - np.sin(theta))
        return area

    def get_drop_volume(self):
        """
        Return the drop volume.
        """
        # From worlfram alpha on spehrical caps
        xyc, R = self.fits
        h = self.get_drop_height()
        V = 1/3*np.pi*h**2*(3*R - h)
        return V

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        (xc, yc) = self.fits[0]
        radius = self.fits[1]
        theta = np.linspace(0, np.pi*2, resolution)
        x = xc + radius*np.cos(theta)
        y = yc + radius*np.sin(theta)
        pts = [x, y]
        return pts

    def display(self, displ_fits=True, displ_ca=True, displ_center=True,
                *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if self.fits is not None and displ_center:
            xyc, R = self.fits
            plt.plot(xyc[0], xyc[1], marker='o',
                     color=self.colors[5])
        if self.fits is not None and displ_fits:
            xyc, R = self.fits
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

    def _compute_fitting_angle(self):
        thetas = []
        xyc = self.fits[0][0]
        sides = [1, -1]
        for i in range(2):
            xyc2 = self.fits[i+1][0]
            theta = sides[i]*np.pi/2 + np.arctan2(xyc[1] - xyc2[1],
                                                  xyc[0] - xyc2[0])
            theta = theta % (2*np.pi)
            thetas.append(theta/np.pi*180)
        return np.array(thetas)

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
        # Compute contact angles at triple point
        if self.triple_pts is not None:
            self.thetas_triple = self._compute_fitting_angle()
            # correct regardin baseline angle
            self.thetas_triple -= bs_angle
        # display if asked
        if verbose:
            self.display()

    def get_drop_center(self):
        """
        Return the center of the drop.
        """
        return np.array(self.fits[0][0],
                        dtype=float)

    def get_drop_height(self):
        """
        Return the drop height.
        """
        hb = self.baseline.get_projection_to_baseline(self.fits[0][0])[1]
        return (self.fits[0][0][1] - hb) + self.fits[0][1]

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        (xc, yc), radius = self.fits[0]
        (xc1, yc1), radius1 = self.fits[1]
        (xc2, yc2), radius2 = self.fits[2]
        theta = np.linspace(0, np.pi*2, resolution)
        # main circle
        x = xc + radius*np.cos(theta)
        y = yc + radius*np.sin(theta)
        # small circle 1
        x1 = xc1 + radius1*np.cos(theta)
        y1 = yc1 + radius1*np.sin(theta)
        # small circle 1
        x2 = xc2 + radius2*np.cos(theta)
        y2 = yc2 + radius2*np.sin(theta)
        # returning
        pts = [np.concatenate([x, [np.nan], x1, [np.nan], x2]),
               np.concatenate([y, [np.nan], y1, [np.nan], y2])]
        return np.array(pts)

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
        if np.any(np.isnan([h, k, a, b, theta])):
            return [[np.nan, np.nan], [np.nan, np.nan]]
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

    def get_drop_center(self):
        """
        Return the center of the drop.
        """
        return np.array(self.fits[0], dtype=float)

    def get_drop_height(self):
        """
        Return the drop height.
        """
        hb = self.baseline.get_projection_to_baseline(self.fits[0])[1]
        xyc, R1, R2, theta = self.fits
        R = ((R1*np.sin(theta))**2 + (R2*np.cos(theta))**2)**.5
        return (xyc[1] - hb) + R

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        (xc, yc), R1, R2, theta = self.fits
        xs, ys = helpers.get_ellipse_points(xc, yc, R1, R2, theta, res=resolution)
        pts = [xs, ys]
        return pts

    def display(self, displ_fits=True, displ_ca=True,
                *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if displ_fits:
            (xc, yc), R1, R2, theta = self.fits
            elxs, elys = helpers.get_ellipse_points(xc, yc, R1, R2, theta, res=100)
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


class DropEllipsesFit(DropFit):
    def __init__(self, xyc1, R1a, R1b, theta1, xyc2, R2a, R2b, theta2,
                 baseline, x_bounds, y_bounds):
        super().__init__(baseline, x_bounds, y_bounds)
        self.fits = [[xyc1, R1a, R1b, theta1],
                     [xyc2, R2a, R2b, theta2]]

    def _get_inters_base_ellipse(self, fit):
        """
        """
        (h, k), a, b, theta = fit
        if np.any(np.isnan([h, k, a, b, theta])):
            return [[np.nan, np.nan], [np.nan, np.nan]]
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

    def _get_inters_base_fit(self):
        inter1 = self._get_inters_base_ellipse(self.fits[0])
        inter2 = self._get_inters_base_ellipse(self.fits[1])
        return [inter1[0], inter2[1]]

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
        thetas = []
        for i, xy in enumerate(xy_inter):
            (xc, yc), R1, R2, theta = self.fits[i]
            x_ref = (xy[0] - xc)*np.cos(theta) + (xy[1] - yc)*np.sin(theta)
            y_ref = (xy[1] - yc)*np.cos(theta) - (xy[0] - xc)*np.sin(theta)
            thet = np.arctan2(-(R2**2*x_ref), (R1**2*y_ref))
            if i == 1:
                thet += np.pi
            thet += (theta - bs_angle)
            thetas.append(thet % (2*np.pi))
        self.thetas = np.array(thetas)*180/np.pi
        # display if asked
        if verbose:
            self.display()

    def get_drop_center(self):
        """
        Return the center of the drop.
        """
        return np.array([(self.fits[0][0] + self.fits[1][0])/2,
                         (self.fits[0][1] + self.fits[1][1])/2],
                        dtype=float)

    def get_fit_as_points(self, resolution=100):
        """
        Return a representation of the fit as point coordinates.
        """
        inters = self._get_inters_base_fit()
        xs = []
        ys = []
        for i in range(2):
            (xc, yc), R1, R2, theta = self.fits[i]
            txs, tys = helpers.get_ellipse_points(xc, yc, R1, R2, theta,
                                                  res=resolution)
            # remove useless part
            if i == 0:
                filt = np.logical_and(txs < xc,
                                      tys > inters[i][1])
            else:
                filt = np.logical_and(txs > xc,
                                      tys > inters[i][1])
            txs = txs[filt]
            tys = tys[filt]
            # Nothing remaining
            if len(txs) == 0:
                txs = [np.nan]
                tys = [np.nan]
                xs.append(txs)
                ys.append(tys)
                continue
            # Make sure it starts at the intersection point
            ind = np.argmax(abs(tys - np.roll(tys, 1)))
            txs = np.roll(txs, -ind)
            tys = np.roll(tys, -ind)
            #
            xs.append(txs)
            ys.append(tys)
        # concatenate
        if i == 0:
            xs = np.concatenate((xs[0], [np.nan], xs[1], [np.nan]))
            ys = np.concatenate((ys[0], [np.nan], ys[1], [np.nan]))
        else:
            xs = np.concatenate((xs[0], [np.nan], xs[1][::-1], [np.nan]))
            ys = np.concatenate((ys[0], [np.nan], ys[1][::-1], [np.nan]))
        pts = [xs, ys]
        return pts

    def display(self, displ_fits=True, displ_ca=True,
                *args, **kwargs):
        """
        """
        super().display()
        # Display fits
        if displ_fits:
            for i in range(2):
                (xc, yc), R1, R2, theta = self.fits[i]
                elxs, elys = helpers.get_ellipse_points(xc, yc, R1, R2, theta,
                                                        res=100)
                # Filter out wrong sides
                if i == 0:
                    filt = elxs < xc
                else:
                    filt = elxs > xc
                elxs = elxs[filt]
                elys = elys[filt]
                # be sure to display ellipse radius on the right side
                coef = 1
                if np.cos(theta) > 0 and i == 0:
                    coef = -1
                if np.cos(theta) < 0 and i == 1:
                    coef = -1
                # plot
                rxs = [xc + coef*abs(R1*np.cos(theta)),
                       xc,
                       xc + coef*R2*np.cos(theta + np.pi/2)]
                rys = [yc + coef*R1*np.sin(theta),
                       yc,
                       yc + coef*R2*np.sin(theta + np.pi/2)]
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
