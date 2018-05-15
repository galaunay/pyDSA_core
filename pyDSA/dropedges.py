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
import scipy.linalg as splin
import scipy.optimize as spopt
import scipy.misc as spmisc
from IMTreatment import Points, Profile
import IMTreatment.plotlib as pplt


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
        self.baseline = im.baseline
        self.thetas = None
        self.thetas_triple = None
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
            y0 = self._get_curvature_root(y=y, zerofun=zerofun, verbose=verbose)
            if y0 is None:
                return None
            # check if the triple point is curvature coherent,
            # else, find the next one
            deriv = dzerofun(y0)
            if (edge_number == 0 and deriv < 0) or (edge_number == 1 and deriv > 0):
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
                    yf -= (yf - y0)*1/5
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
        # return
        return spline1, spline2

    def fit_ellipse(self):
        """
        Fit the drop edges with an ellipse.
        """
        raise Exception('Not implemented yet')
        def fit_ellipse(x, y):
            x = x[:, np.newaxis]
            y = y[:, np.newaxis]
            D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
            S = np.dot(D. T, D)
            C = np.zeros([6, 6])
            C[0, 2] = C[2, 0] = 2
            C[1, 1] = -1
            E, V = splin.eig(np.dot(splin.inv(S), C))
            n = np.argmax(np.abs(E))
            a = V[:, n]
            b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
            num = b*b - a*c
            x0 = (c*d - b*f)/num
            y0 = (a*f - b*d)/num
            return np.array([x0, y0])
        # get data
        x1 = self.drop_edges[0].y
        y1 = self.drop_edges[0].x
        plt.figure()
        plt.plot(x1, y1, 'ok')
        a = fit_ellipse(x1, y1)
        plt.plot(a[0], a[1], 'or')

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

    def display(self, *args, **kwargs):
        """
        """
        # super().display(*args, **kwargs)
        for edg in self.drop_edges:
            plt.plot(edg.y, edg.x, color='k', marker='o')
        plt.axis('equal')
        # Display baseline
        x0 = np.min(self.xy[:, 0])
        xf = np.max(self.xy[:, 0])
        x0 -= np.abs(xf - x0)*.1
        xf += np.abs(xf - x0)*.1
        self.baseline.display(x0, xf, color=self.colors[0])
        # Display fits
        if self.edges_fits is not None:
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
        if self.thetas is not None:
            lines = self._get_angle_display_lines()
            for line in lines:
                plt.plot(line[0], line[1],
                         color=self.colors[0])
                plt.plot(line[0][0],
                         line[0][1],
                         color=self.colors[0])
        # Display triple points
        if self.triple_pts is not None:
            for i in [0, 1]:
                plt.plot(self.triple_pts[i][0],
                         self.triple_pts[i][1],
                         marker="o",
                         color=self.colors[4])

    def _get_angle_display_lines(self):
        lines = []
        # contact angle with solid
        if self.thetas is not None:
            length = (np.max(self.xy[:, 1]) - np.min(self.xy[:, 1]))/3
            theta1 = self.thetas[0]/180*np.pi
            theta2 = self.thetas[1]/180*np.pi
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
            length = (np.max(self.xy[:, 1]) - np.min(self.xy[:, 1]))/3
            theta1 = self.thetas_triple[0]/180*np.pi
            theta2 = self.thetas_triple[1]/180*np.pi
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

    def get_drop_base(self):
        """
        Return the drop base.

        Returns
        =======
        x1, x2: numbers
            Coordinates of the drop base.
        """
        if self.edges_fits is None:
            return [np.nan, np.nan]
        pt1, pt2 = self._get_inters_base_fit()
        return np.sort([pt1[0], pt2[0]])

    def get_drop_position(self):
        """
        Return the drop position along x (drop center)

        Returns
        =======
        x : number
            Position of the drop
        """
        return np.mean(self.get_drop_base())

    def get_drop_base_radius(self):
        """
        Return the drop base radius.
        """
        db = self.get_drop_base()
        return db[1] - db[0]

    def get_drop_radius(self):
        """
        Return the drop base radius. based on the triple points.
        """
        # Use triple points if present
        if self.triple_pts is not None and not np.any(np.isnan(self.triple_pts)):
            return abs(self.triple_pts[0][0] - self.triple_pts[1][0])
        else:
            return np.nan

    def get_drop_height(self):
        """
        Return the evolution of the drop height.
        """
        h = []
        for edge in self.point_sets:
            h.append(np.max(edge.xy[:, 1]))
        return h

    def get_ridge_height(self):
        """
        Return the ridge height.
        """
        if self.triple_pts is None:
            return np.nan, np.nan
        heights = []
        for pt in self.triple_pts:
            x = pt[0]
            y = pt[1]
            y0 = self.baseline.get_baseline_fun()(x)
            y -= y0
            heights.append(y)
        return heights

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        if self.edges_fits is None:
            raise Exception("You should compute fitting first with 'fit()'")
        # Compute base contact angle
        xy_inter = self._get_inters_base_fit()
        self.thetas = self._compute_fitting_angle_at_pts(xy_inter)
        # Compute triple point contact angle
        if self.triple_pts is not None:
            xy_tri = self.triple_pts
            self.thetas_triple = self._compute_fitting_angle_at_pts(xy_tri)
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
            theta =  np.pi*1/2 - np.arctan(deriv)
            thetas.append(theta/np.pi*180)
        return thetas



        # Should use laplace-young fitting
