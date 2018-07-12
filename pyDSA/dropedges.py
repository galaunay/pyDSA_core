# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/DSA_analyzer
# Version: 0.1

# This file is part of DSA_analyzer

# DSA_analyzer is distributed in the hope that it will be useful,
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


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


def dummy_function(x):
    try:
        return np.array([np.nan]*len(x))
    except TypeError:
        return np.nan


class DropEdges(Points):
    def __init__(self, xy, im, type):
        """
        """
        super().__init__(xy=xy,
                         unit_x=im.unit_x,
                         unit_y=im.unit_y)
        self._type = type
        self.drop_edges = self._separate_drop_edges()
        self.edges_fits = None
        self.triple_pts = None
        self.circle_fits = None
        self.ellipse_fit = None
        self.circle_triple_pts = None
        self.baseline = im.baseline
        self.thetas = None
        self.thetas_triple = None
        self.thetas_circ = None
        self.thetas_ellipse = None
        self.im_axe_x = im.axe_x
        self.im_axe_y = im.axe_y
        self.im_dx = im.dx
        self.im_dy = im.dy
        self.colors = pplt.get_color_cycles()

    def _separate_drop_edges(self):
        """
        Separate the two sides of the drop.
        """
        # check if edges are present...
        if len(self.xy) == 0:
            return None, None
        ind_sort = np.argsort(self.xy[:, 0])
        xs = self.xy[:, 0][ind_sort]
        ys = self.xy[:, 1][ind_sort]
        dxs = xs[1::] - xs[0:-1]
        dxs_sorted = np.sort(dxs)
        if dxs_sorted[-1] > 10*dxs_sorted[-2]:
            # ind of the needle center (needle)
            ind_cut = np.argmax(dxs) + 1
        else:
            # ind of the higher point (no needle)
            ind_cut = np.argmax(ys)
        xs1 = xs[0:ind_cut]
        ys1 = ys[0:ind_cut]
        xs2 = xs[ind_cut::]
        ys2 = ys[ind_cut::]
        # ensure only one y per x
        new_y1 = np.sort(list(set(ys1)))
        new_x1 = [np.mean(xs1[y == ys1]) for y in new_y1]
        new_y2 = np.sort(list(set(ys2)))
        new_x2 = [np.mean(xs2[y == ys2]) for y in new_y2]
        # smooth to avoid stepping
        de1 = Profile(new_y1, new_x1,
                      unit_x=self.unit_y,
                      unit_y=self.unit_x)
        de2 = Profile(new_y2, new_x2,
                      unit_x=self.unit_y,
                      unit_y=self.unit_x)
        de1.smooth(tos='gaussian', size=1, inplace=True)
        de2.smooth(tos='gaussian', size=1, inplace=True)
        return de1, de2

    def rotate(self, angle, inplace=False):
        """
        Rotate the edge.

        Parameters
        ----------
        angle: number
            Rotation angle in radian.
        """
        # Checks
        if inplace:
            tmpp = self
        else:
            tmpp = self.copy()
        super(DropEdges, tmpp).rotate(angle, inplace=True)
        for de in tmpp.drop_edges:
            de.rotate(angle=angle, inplace=True)
        if tmpp.baseline is not None:
            tmpp.baseline.rotate(angle=angle)
        tmpp.edges_fits = None
        tmpp.triple_pts = None
        tmpp.circle_fits = None
        tmpp.circle_triple_pts = None
        return tmpp

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
        if self.edges_fits is None:
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

    def _detect_triple_points_as_x_minima(self, edge_number, verbose):
        fit = self.edges_fits[edge_number]
        y = self.drop_edges[edge_number].x
        y = np.linspace(y[0], y[-1], 100)
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
        fit = self.edges_fits[edge_number]
        y = self.drop_edges[edge_number].x
        y = np.linspace(y[0], y[-1], 100)
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

    def fit(self, k=5, s=.75, verbose=False):
        """
        Compute a spline fitting for the droplet shape.

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline.  Must be <= 5.
            Default is k=5.
        s : float, optional
            Smoothing factor between 0 (not smoothed) and 1 (very smoothed)
            Default to 0.75
        """
        # Chec if edges are present
        if self.drop_edges[0] is None and self.drop_edges[1] is None:
            return None, None
        # Prepare drop edge for interpolation
        # TODO: Find a more efficient fitting
        de1, de2 = self.drop_edges
        x1 = de1.y
        y1 = de1.x
        x2 = de2.y
        y2 = de2.x
        spline1 = None
        spline2 = None
        # Don't fit if no edge
        if len(y1) == 0:
            spline1 = dummy_function
        if len(y2) == 0:
            spline2 = dummy_function
        # spline interpolation
        max_smooth_carac_len = np.max([self.im_dx*len(self.im_axe_x),
                                       self.im_dy*len(self.im_axe_y)])
        min_smooth_carac_len = np.max([self.im_dx, self.im_dy])/6
        max_smooth_carac_len = min_smooth_carac_len*2
        min_smooth_fact = np.max([len(x1), len(x2)])*min_smooth_carac_len**2
        max_smooth_fact = np.max([len(x1), len(x2)])*max_smooth_carac_len**2
        s = max_smooth_fact - (max_smooth_fact - min_smooth_fact)*s
        if verbose:
            print("Used 's={}'".format(s))
        if spline1 is None:
            try:
                spline1 = spint.UnivariateSpline(y1, x1, k=k, s=s)
            except:
                spline1 = dummy_function
                if verbose:
                    print("Fitting failed for edge number one")
        if spline2 is None:
            try:
                spline2 = spint.UnivariateSpline(y2, x2, k=k, s=s)
            except:
                spline2 = dummy_function
                if verbose:
                    print("Fitting failed for edge number one")
        # Verbose if necessary
        if verbose:
            tmp_y1 = np.linspace(y1.min(), y1.max(), 1000)
            tmp_y2 = np.linspace(y2.min(), y2.max(), 1000)
            plt.figure()
            plt.plot(de1.y, de1.x, "o")
            plt.plot(x1, y1, 'xb')
            plt.plot(spline1(tmp_y1), tmp_y1, 'r')
            plt.plot(de2.y, de2.x, "o")
            plt.plot(x2, y2, 'xb')
            plt.plot(spline2(tmp_y2), tmp_y2, 'r')
            plt.axis('equal')
            plt.show()
        # store
        self.edges_fits = [spline1, spline2]
        # delete variables derived from the previous fitting
        self.triple_pts = None
        self.thetas = None
        self.thetas_triple = None
        # return
        return spline1, spline2

    def fit_ellipse(self, verbose=False):
        """
        Fit the drop edges with an ellipse.
        """
        xs = self.xy[:, 0]
        ys = self.xy[:, 1]
        (xc, yc), R1, R2, theta = fit_ellipse(xs, ys)
        self.ellipse_fit = (xc, yc), R1, R2, theta
        if verbose:
            plt.figure()
            self.display()
            thetas = np.linspace(0, np.pi*2, 100)
            xs = xc + R1*np.cos(thetas)
            ys = yc + R2*np.sin(thetas)
            new_xs = xc + np.cos(theta)*(xs - xc) - np.sin(theta)*(ys - yc)
            new_ys = yc + np.cos(theta)*(ys - yc) + np.sin(theta)*(xs - xc)
            plt.figure()
            self.display()
            plt.plot(xc, yc, 'ok')
            plt.plot(new_xs, new_ys)
            plt.show()

    def fit_LY(self, interp_res=100, method=None, verbose=False):
        # get data
        z1_min = self.triple_pts[0][1]
        z1 = np.linspace(z1_min,
                         np.max(self.xy[:, 1]),
                         interp_res)
        z2_min = self.triple_pts[1][1]
        z2 = np.linspace(z2_min,
                         np.max(self.xy[:, 1]),
                         interp_res)
        dz = z1[1] - z1[0]
        r1 = abs(self.edges_fits[0](z1) - self.get_drop_position())
        r2 = abs(self.edges_fits[1](z2) - self.get_drop_position())
        if verbose:
            plt.figure()
            self.display()
            plt.figure()
            plt.plot(r1, z1, label="edge1")
            plt.plot(r2, z2, label="edge2")
            plt.xlabel('r')
            plt.ylabel('z')
            plt.legend()
            plt.show()
        # Get gradients
        r1 = np.asarray(r1)
        r1p = np.gradient(r1, dz)
        r1pp = np.gradient(r1p, dz)
        z1p = np.gradient(z1, r1)
        z1pp = np.gradient(z1p, r1)
        if verbose:
            plt.figure()
            plt.plot(r1, z1/np.mean(z1), label="z1")
            plt.plot(r1, z1p/np.mean(z1p), label="z1p")
            plt.plot(r1, z1pp/np.mean(z1pp), label="z1pp")
            plt.xlabel('r')
            plt.ylabel('z')
            plt.legend()
            plt.show()
        # Get curvature
        # R1_inv = r1pp/(1 + r1p**2)**(3/2)
        # R2_inv = 1/(r1*(1 + r1p**2)**(1/2))
        R1_inv = (z1p/(r1*(1 + z1p**2)**.5))
        R2_inv = (z1pp/(1 + z1p**2)**(3/2))
        if verbose:
            R1 = 1/(z1p/(r1*(1 + z1p**2)**.5))
            R2 = 1/(z1pp/(1 + z1p**2)**(3/2))
            plt.figure()
            plt.plot(r1, z1)
            plt.xlabel('r')
            plt.ylabel('z')
            plt.figure()
            plt.plot(R1, z1, label="R1")
            plt.plot(R2, z1, label="R2")
            plt.xlabel('r')
            plt.ylabel('z')
            plt.legend()
            plt.show()
        # fit for right side
        def minfun(args):
            A, B = args
            res = R1_inv + R2_inv - (A + B*z1)
            return np.sum(res**2)
        res1 = spopt.minimize(minfun, [1, 1], method=method)
        if not res1.success:
            raise Exception('Fitting sigma failed:\n{}'.format(res1.message))
        A, B = res1.x
        if verbose:
            plt.figure()
            ls = R1_inv + R2_inv
            plt.plot(ls, z1, label="left equation side")
            rs = A + B*z1
            plt.plot(rs, z1, label="left equation side")
            plt.xlabel('r')
            plt.ylabel('z')
            plt.legend()
            plt.show()
        # Fit for left side (Curvature radius)
        def minfun(args, dz, z):
            r1 = args
            z1p = np.gradient(z1, r1)
            z1pp = np.gradient(z1p, r1)
            R1_inv = (z1p/(r1*(1 + z1p**2)**.5))
            R2_inv = (z1pp/(1 + z1p**2)**(3/2))
            res = (R1_inv + R2_inv) - (A + B*z1)
            return np.sum(res**2)
        res2 = spopt.minimize(minfun, r1, (dz, z1), method=method)
        if not res2.success:
            raise Exception('Fitting drop edge failed:\n{}'
                            .format(res2.message))
        new_r1 = res2.x
        if verbose:
            plt.figure()
            plt.plot(r1, z1, label="Spline fitting")
            plt.plot(new_r1, z1, label="LY fitting")
            plt.legend()
            plt.show()
        return z1, - new_r1 + self.get_drop_position()

    def fit_circle(self, verbose=False):
        """
        Fit a circle to the edges.

        Ignore the lower part of the drop if triple points are presents.
        """
        # get data
        ys1 = self.drop_edges[0].y
        xs1 = self.drop_edges[0].x
        ys2 = self.drop_edges[1].y
        xs2 = self.drop_edges[1].x
        if self.triple_pts is not None:
            if not np.any(np.isnan(self.triple_pts[0])):
                ytp = self.triple_pts[0][1]
                filt = ys1 > ytp
                xs1 = xs1[filt]
                ys1 = ys1[filt]
            if not np.any(np.isnan(self.triple_pts[1])):
                ytp = self.triple_pts[1][1]
                filt = ys2 > ytp
                xs2 = xs2[filt]
                ys2 = ys2[filt]
        xs = np.concatenate((ys1, ys2))
        ys = np.concatenate((xs1, xs2))
        # Fit circles
        c, R = fit_circle(xs, ys)
        self.circle_fits = [[c, R]]
        # verbose
        if verbose:
            plt.plot(xs, ys, '.k')
            plt.plot(c[0], c[1], "ok")
            theta = np.linspace(0, 2*np.pi, 100)
            tmpxs = c[0] + R*np.cos(theta)
            tmpys = c[1] + R*np.sin(theta)
            plt.plot(tmpxs, tmpys, "-k")
            plt.axis('image')
            plt.show()
        # return
        return self.circle_fits

    def fit_circles(self, sigma_max=None, soft_constr=False,
                    verbose=False):
        """
        Fit circles to the edges, cutting them if a triple point is
        present.

        If a triple point is present, the ridge fits are made to be tangent
        to the drop and to the baseline.

        Parameters
        ==========
        sigma_max: number
            If specified, points too far from the fit are iteratively removed
            until:
            std(R) < mean(R)*sigma_max
            With R the radii.

        """
        # Check if multiple or simple circles need to be detected
        simple_circ = False
        if self.triple_pts is None:
            simple_circ = True
        else:
            if np.any(np.isnan(np.array(self.triple_pts))):
                simple_circ = True
        # Three circles detection
        if not simple_circ:
            # get data
            tp1 = self.triple_pts[0]
            tp2 = self.triple_pts[1]
            xs1 = self.drop_edges[0].y
            ys1 = self.drop_edges[0].x
            xs2 = self.drop_edges[1].y
            ys2 = self.drop_edges[1].x
            # separate at the triple point
            ind_sep1 = np.where(ys1 > tp1[1])[0][0]
            ind_sep2 = np.where(ys2 > tp2[1])[0][0]
            xs_o1 = xs1[0:ind_sep1]
            ys_o1 = ys1[0:ind_sep1]
            xs_o2 = xs2[0:ind_sep2]
            ys_o2 = ys2[0:ind_sep2]
            xs_d = np.concatenate((xs1[ind_sep1::], xs2[ind_sep2::][::-1]))
            ys_d = np.concatenate((ys1[ind_sep1::], ys2[ind_sep2::][::-1]))
            # Fit circles
            # (use the middle circle fit to enhance the ridge fits)
            c_d, Rd = fit_circle(xs_d, ys_d,
                                 sigma_max=sigma_max)
            c_o1, R1 = fit_circle(xs_o1, ys_o1, self.baseline,
                                  sigma_max=sigma_max,
                                  tangent_circ=(c_d, Rd),
                                  soft_constr=soft_constr)
            c_o2, R2 = fit_circle(xs_o2, ys_o2, self.baseline,
                                  sigma_max=sigma_max,
                                  tangent_circ=(c_d, Rd),
                                  soft_constr=soft_constr)
            # check for abnormal values
            if R1 > Rd or R2 > Rd or c_o1[0] > c_d[0] or c_o2[0] < c_d[0]:
                return None
            else:
                self.circle_fits = [[c_o1, R1], [c_o2, R2], [c_d, Rd]]
            # Get triple points based from circles
            xtp1 = (c_o1[0]*Rd + c_d[0]*R1)/(Rd + R1)
            ytp1 = (c_o1[1]*Rd + c_d[1]*R1)/(Rd + R1)
            xtp2 = (c_o2[0]*Rd + c_d[0]*R2)/(Rd + R2)
            ytp2 = (c_o2[1]*Rd + c_d[1]*R2)/(Rd + R2)
            self.circle_triple_pts = [[xtp1, ytp1], [xtp2, ytp2]]
        # Simple one circle detection
        else:
            self.fit_circle(verbose=verbose)
        # Return
        return self.circle_fits

    def display(self, displ_edges=True, displ_fits=True, displ_ca=True,
                displ_tp=True, displ_circ_tp=True, displ_circ=True,
                displ_ellipse=True, *args, **kwargs):
        """
        """
        # super().display(*args, **kwargs)
        if displ_edges:
            for edg in self.drop_edges:
                plt.plot(edg.y, edg.x, color='k', marker='o')
        plt.axis('equal')
        # Display baseline
        if self.baseline is not None:
            x0 = np.min(self.xy[:, 0])
            xf = np.max(self.xy[:, 0])
            x0 -= np.abs(xf - x0)*.1
            xf += np.abs(xf - x0)*.1
            self.baseline.display(x0, xf, color=self.colors[0])
        # Display fits
        if self.edges_fits is not None and displ_fits:
            xy_inter = self._get_inters_base_fit()
            y1 = np.linspace(xy_inter[0][1],
                             np.max(self.xy[:, 1]),
                             1000)
            y2 = np.linspace(xy_inter[1][1],
                             np.max(self.xy[:, 1]),
                             1000)
            x1 = self.edges_fits[0](y1)
            x2 = self.edges_fits[1](y2)
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
        # Display circles
        if self.circle_fits is not None and displ_circ:
            for cf in self.circle_fits:
                xyc, R = cf
                plt.plot(xyc[0], xyc[1], marker='o',
                         color=self.colors[5])
                circ = plt.Circle((xyc[0], xyc[1]), radius=R,
                                  color=self.colors[5],
                                  fill=False)
                plt.gca().add_artist(circ)
        # Display ellipse
        if self.ellipse_fit is not None and displ_ellipse:
            (xc, yc), R1, R2, theta = self.ellipse_fit
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
        # Display triple points from circle fits
        if self.circle_triple_pts is not None and displ_circ_tp:
            for tp in self.circle_triple_pts:
                plt.plot(tp[0], tp[1], marker='o',
                         color=self.colors[5])

    def _get_angle_display_lines(self):
        bs_angle = self.baseline.tilt_angle*180/np.pi
        lines = []
        # contact angle with solid
        if len(self.xy) == 0:
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
            lines.append([[np.nan, np.nan],
                          [np.nan, np.nan]])
            return lines
        length = (np.max(self.xy[:, 1]) - np.min(self.xy[:, 1]))/3
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
        if self.thetas_circ is not None:
            # circle fit projected contact angle
            theta1 = (self.thetas_circ[0] + bs_angle)*np.pi/180
            theta2 = (self.thetas_circ[1] + bs_angle)*np.pi/180
            xy_inter = self._get_inters_base_circle_fit()
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
        if self.thetas_ellipse is not None:
            # Ellipse fit projected contact angle
            theta1 = (self.thetas_ellipse[0] + bs_angle)*np.pi/180
            theta2 = (self.thetas_ellipse[1] + bs_angle)*np.pi/180
            xy_inter = self._get_inters_base_ellipse_fit()
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

    def _get_inters_base_fit(self, verbose=False):
        flat = False
        try:
            bfun = self.baseline.get_baseline_fun(along_y=True)
        except Exception:
            flat = True
        xys = []
        for i in range(2):
            sfun = self.edges_fits[i]
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
            y = np.linspace(np.min(self.xy[:, 1]),
                            np.max(self.xy[:, 1]), 100)
            x = np.linspace(self.baseline.pt1[0],
                            self.baseline.pt2[0],
                            100)
            bfun = self.baseline.get_baseline_fun()
            plt.figure()
            plt.plot([xys[0][0], xys[1][0]], [xys[0][1], xys[1][1]], "ok",
                     label="intersection")
            plt.plot(x, bfun(x), label="base")
            plt.plot(sfun(y), y, label="fit")
            plt.legend()
            plt.show()
        return xys

    def _get_inters_base_circle_fit(self):
        """
        """
        if self.circle_fits is None:
            raise Exception()
        # getting intersection points
        # from: http://mathworld.wolfram.com/Circle-LineIntersection.html
        (xc, yc), r = self.circle_fits[0]
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

    def _get_inters_base_ellipse_fit(self):
        """
        """
        (h, k), a, b, theta = self.ellipse_fit
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

    def get_ridge_height(self, from_circ_fit=False):

        """
        Return the ridge height.

        Parameters
        ==========
        from_circ_fit: boolean
            If true, use the triple points found by the
            circle fits.
        """
        if from_circ_fit:
            pts = self.circle_triple_pts
        else:
            pts = self.triple_pts
        if pts is None:
            return np.nan, np.nan
        heights = []
        for pt in pts:
            x = pt[0]
            y = pt[1]
            y0 = self.baseline.get_baseline_fun()(x)
            y -= y0
            heights.append(y)
        return heights

    def get_base_diameter(self, from_circ_fit=False):
        """
        Return the base diameter.

        Parameters
        ==========
        from_circ_fit: boolean
            If True, us te circle fits (more accurate),
            else, use the spline fits.
        """
        if from_circ_fit:
            pt1, pt2 = self._get_inters_base_circle_fit()
        else:
            pt1, pt2 = self._get_inters_base_fit()
        diam = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        return diam

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
        if self.edges_fits is not None:
            xy_inter = self._get_inters_base_fit()
            self.thetas = self._compute_fitting_angle_at_pts(xy_inter)
            # correct regardin baseline angle
            self.thetas -= bs_angle
        # Compute circle fits contact angles
        if self.circle_fits is not None:
            (xc, yc), R = self.circle_fits[0]
            pts = self._get_inters_base_circle_fit()
            thetas = []
            for pt in pts:
                theta = np.pi/2 + np.arctan((yc - pt[1])/(xc - pt[0]))
                thetas.append(theta)
            self.thetas_circ = np.array(thetas)*180/np.pi
            # correct regardin baseline angle
            self.thetas_circ -= bs_angle
        # Compute triple point contact angle
        if self.triple_pts is not None:
            xy_tri = self.triple_pts
            self.thetas_triple = self._compute_fitting_angle_at_pts(xy_tri)
            # correct regardin baseline angle
            self.theta_triple -= bs_angle
        # Compute ellipse fit contact angle
        if self.ellipse_fit is not None:
            xy_inter = self._get_inters_base_ellipse_fit()
            bs_angle = self.baseline.tilt_angle
            (xc, yc), R1, R2, theta = self.ellipse_fit
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
            self.thetas_ellipse = np.array(thetas)*180/np.pi
        # display if asked
        if verbose:
            self.display()

    def _compute_fitting_angle_at_pts(self, pts):
        thetas = []
        for i in range(2):
            x_inter, y_inter = pts[i]
            sfun = self.edges_fits[i]
            # Get gradient
            dy = self.drop_edges[i].x
            dy = (dy[-1] - dy[0])/100
            deriv = spmisc.derivative(sfun, y_inter, dx=dy)
            theta = np.pi*1/2 - np.arctan(deriv)
            thetas.append(theta/np.pi*180)
        return np.array(thetas)



        # Should use laplace-young fitting
