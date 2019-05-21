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
import matplotlib.pyplot as plt
import scipy.interpolate as spint
import scipy.optimize as spopt
from IMTreatment import Points, Profile
import IMTreatment.plotlib as pplt
from .helpers import fit_circle, fit_ellipse, get_ellipse_points
from .dropfit import DropCircleFit, DropEllipseFit, DropSplineFit, \
    DropCirclesFit, DropEllipsesFit


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


def dummy_function(x):
    try:
        return np.array([np.nan]*len(x))
    except TypeError:
        return np.nan


def poly_function_from_coefs(coefs):
    def fun(x):
        nmb_coefs = len(coefs)
        x = np.asarray(x)
        return np.sum([coefs[i]*x**(nmb_coefs - 1 - i)
                       for i in range(nmb_coefs)], axis=0)
    return fun


class DropEdges(Points):
    def __init__(self, xy, im, type):
        """
        """
        super().__init__(xy=xy,
                         unit_x=im.unit_x,
                         unit_y=im.unit_y)
        self._type = type
        self.drop_edges = None
        self.baseline = im.baseline
        self.im_axe_x = im.axe_x
        self.im_axe_y = im.axe_y
        self.x_bounds = [im.axe_x[0], im.axe_x[-1]]
        self.y_bounds = [im.axe_y[0], im.axe_y[-1]]
        self.im_dx = im.dx
        self.im_dy = im.dy
        self.colors = pplt.get_color_cycles()

    def _separate_drop_edges(self):
        """
        Separate the two sides of the drop.
        """
        # check if edges are present...
        if len(self.xy) == 0:
            self.drop_edges = [None]*4
            return [None]*4
        # Get cut position
        ind_sort = np.argsort(self.xy[:, 0])
        xs = self.xy[:, 0][ind_sort]
        ys = self.xy[:, 1][ind_sort]
        dxs = xs[1::] - xs[0:-1]
        dxs_sorted = np.sort(dxs)
        if dxs_sorted[-1] > 10*dxs_sorted[-2]:
            # ind of the needle center (needle)
            ind_cut = np.argmax(dxs) + 1
            x_cut = xs[ind_cut]
        else:
            # ind of the higher point (no needle)
            ind_cut = np.argmax(ys)
            x_cut = xs[ind_cut]
        # Separate according to cut position
        xs1 = xs[xs < x_cut]
        xs2 = xs[xs > x_cut]
        ys1 = ys[xs < x_cut]
        ys2 = ys[xs > x_cut]
        # # Ensure only one y per x (Use hash table for optimization)
        new_y1 = np.sort(list(set(ys1)))
        dic = {y1: 0 for y1 in new_y1}
        nmb = {y1: 0 for y1 in new_y1}
        for i in range(len(ys1)):
            dic[ys1[i]] += xs1[i]
            nmb[ys1[i]] += 1
        new_x1 = [dic[y1]/nmb[y1] for y1 in new_y1]
        new_y2 = np.sort(list(set(ys2)))
        dic = {y2: 0 for y2 in new_y2}
        nmb = {y2: 0 for y2 in new_y2}
        for i in range(len(ys2)):
            dic[ys2[i]] += xs2[i]
            nmb[ys2[i]] += 1
        new_x2 = [dic[y2]/nmb[y2] for y2 in new_y2]
        # smooth to avoid stepping
        # Profile also automatically sort along y
        de1 = Profile(new_y1, new_x1)
        de2 = Profile(new_y2, new_x2)
        de1.smooth(tos='gaussian', size=1, inplace=True)
        de2.smooth(tos='gaussian', size=1, inplace=True)
        # parametrize
        if len(de1) != 0:
            t1s = np.cumsum(((de1.x[1:] - de1.x[0:-1])**2
                             + (de1.y[1:] - de1.y[0:-1])**2)**.5)
            t1s = np.concatenate(([0], t1s))
            t1s /= t1s[-1]
        else:
            t1s = []
        if len(de2) != 0:
            t2s = np.cumsum(((de2.x[1:] - de2.x[0:-1])**2
                             + (de2.y[1:] - de2.y[0:-1])**2)**.5)
            t2s = np.concatenate(([0], t2s))
            t2s /= t2s[-1]
        else:
            t2s = []
        dex1 = Profile(t1s, de1.y, unit_x="", unit_y=self.unit_x)
        dex2 = Profile(t2s, de2.y, unit_x="", unit_y=self.unit_x)
        dey1 = Profile(t1s, de1.x, unit_x="", unit_y=self.unit_y)
        dey2 = Profile(t2s, de2.x, unit_x="", unit_y=self.unit_y)
        # evenlify
        if len(dex1) != 0:
            dtx1 = self.im_dx/abs(dex1.y[-1] - dex1.y[0])
            dty1 = self.im_dy/abs(dey1.y[-1] - dey1.y[0])
            dt1 = np.min([dtx1, dty1])
            dex1 = dex1.evenly_space(dx=dt1)
            dey1 = dey1.evenly_space(dx=dt1)
        if len(dex2) != 0:
            dtx2 = self.im_dx/abs(dex2.y[-1] - dex2.y[0])
            dty2 = self.im_dy/abs(dey2.y[-1] - dey2.y[0])
            dt2 = np.min([dtx2, dty2])
            dex2 = dex2.evenly_space(dx=dt2)
            dey2 = dey2.evenly_space(dx=dt2)
        # store
        self.drop_edges = (dex1, dey1, dex2, dey2)
        return (dex1, dey1, dex2, dey2)

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
        if tmpp.drop_edges is not None:
            for de in tmpp.drop_edges:
                de.rotate(angle=angle, inplace=True)
        if tmpp.baseline is not None:
            tmpp.baseline.rotate(angle=angle)
        tmpp.edges_fits = None
        tmpp.triple_pts = None
        tmpp.circle_fits = None
        tmpp.circle_triple_pts = None
        return tmpp

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

        Returns:
        --------
        fit: DropFit object
            .
        """
        return self.fit_spline(k=k, s=s, verbose=verbose)

    def fit_spline(self, k=5, s=.75, verbose=False):
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

        Returns:
        --------
        fit: DropFit object
            .
        """
        # Chec if edges are present
        if self.drop_edges is None:
            self._separate_drop_edges()
        if self.drop_edges[0] is None and self.drop_edges[2] is None:
            dsf = DropSplineFit(x_bounds=[np.nan, np.nan],
                                y_bounds=[np.nan, np.nan],
                                baseline=self.baseline,
                                fits=[dummy_function, dummy_function])
            return dsf
        # Prepare drop edge for interpolation
        # TODO: Find a more efficient fitting
        dex1, dey1, dex2, dey2 = self.drop_edges
        spline1 = None
        spline2 = None
        # Don't fit if no edge
        if len(dex1) == 0:
            spline1 = dummy_function
        if len(dex2) == 0:
            spline2 = dummy_function
        # spline interpolation
        min_smooth_carac_len = np.max([self.im_dx, self.im_dy])/6
        max_smooth_carac_len = min_smooth_carac_len*2
        max_len = np.max([len(dex1), len(dex2)])
        min_smooth_fact = max_len*min_smooth_carac_len**2
        max_smooth_fact = max_len*max_smooth_carac_len**2
        s = max_smooth_fact - (max_smooth_fact - min_smooth_fact)*s
        max_smooth_carac_len = np.max([self.im_axe_x[-1] - self.im_axe_x[0],
                                       self.im_axe_y[-1] - self.im_axe_y[0]])
        if verbose:
            print("Used 's={}'".format(s))
        if spline1 is None:
            try:
                spx1 = spint.UnivariateSpline(dex1.x, dex1.y, k=k, s=s)
                spy1 = spint.UnivariateSpline(dey1.x, dey1.y, k=k, s=s)
                spline1 = [spx1, spy1]
            except:
                spline1 = (dummy_function, dummy_function)
                if verbose:
                    print("Fitting failed for edge number one")
        if spline2 is None:
            try:
                spx2 = spint.UnivariateSpline(dex2.x, dex2.y, k=k, s=s)
                spy2 = spint.UnivariateSpline(dey2.x, dey2.y, k=k, s=s)
                spline2 = [spx2, spy2]
            except:
                spline2 = (dummy_function, dummy_function)
                if verbose:
                    print("Fitting failed for edge number one")
        # Verbose if necessary
        if verbose:
            t = np.linspace(0, 1, 1000)
            plt.figure()
            plt.plot(dex1.y, dey1.y, "o")
            plt.plot(spline1[0](t), spline1[1](t), 'r')
            plt.plot(dex2.y, dey2.y, "o")
            plt.plot(spline2[0](t), spline2[1](t), 'r')
            plt.axis('equal')
            plt.show()
        # return
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        dsf = DropSplineFit(x_bounds=x_bounds,
                            y_bounds=y_bounds,
                            baseline=self.baseline,
                            fits=[spline1, spline2])
        return dsf

    def fit_polyline(self, deg=5, verbose=False):
        """
        Fit the drop edge with a poly line.

        Parameters
        ----------
        deg: integer
            Polynomial degree (default to 5).

        Returns
        -------
        fit: DropFit object
            .
        """
        if self.drop_edges is None:
            self._separate_drop_edges()
        if self.drop_edges[0] is None and self.drop_edges[2] is None:
            dsf = DropSplineFit(x_bounds=[np.nan, np.nan],
                                y_bounds=[np.nan, np.nan],
                                baseline=self.baseline,
                                fits=[dummy_function, dummy_function])
            return dsf
        # Prepare drop edge for interpolation
        # TODO: Find a more efficient fitting
        dex1, dey1, dex2, dey2 = self.drop_edges
        polyline1 = None
        polyline2 = None
        # Don't fit if no edge
        if len(dey1) == 0:
            polyline1 = (dummy_function, dummy_function)
        if len(dey2) == 0:
            polyline2 = (dummy_function, dummy_function)
        # fit
        if polyline1 is None:
            try:
                psx1 = np.polyfit(dex1.x, dex1.y, deg)
                psy1 = np.polyfit(dey1.x, dey1.y, deg)
                polyline1 = [poly_function_from_coefs(psx1),
                             poly_function_from_coefs(psy1)]
            except:
                polyline1 = (dummy_function, dummy_function)
                if verbose:
                    print("Fitting failed for edge number one")
        if polyline2 is None:
            try:
                psx2 = np.polyfit(dex2.x, dex2.y, deg)
                psy2 = np.polyfit(dey2.x, dey2.y, deg)
                polyline2 = [poly_function_from_coefs(psx2),
                             poly_function_from_coefs(psy2)]
            except:
                polyline2 = (dummy_function, dummy_function)
                if verbose:
                    print("Fitting failed for edge number two")
        # Return
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        dsf = DropSplineFit(x_bounds=x_bounds,
                            y_bounds=y_bounds,
                            baseline=self.baseline,
                            fits=[polyline1, polyline2])
        return dsf

    def fit_ellipse(self, triple_pts=None, verbose=False):
        """
        Fit the drop edges with an ellipse.

        Remove points under the triple points if possible.

        Returns:
        --------
        fit: DropFit object
            .
        """
        if self.drop_edges is None:
            self._separate_drop_edges()
        dex1, dey1, dex2, dey2 = self.drop_edges
        if dex1 is None or dex2 is None:
            des = DropEllipseFit(xyc=[np.nan, np.nan], R1=np.nan,
                                 R2=np.nan, theta=np.nan,
                                 baseline=self.baseline,
                                 x_bounds=[np.nan, np.nan],
                                 y_bounds=[np.nan, np.nan])
            return des
        xs1 = dex1.y
        ys1 = dey1.y
        xs2 = dex2.y
        ys2 = dey2.y
        if triple_pts is not None:
            tp1 = triple_pts[0]
            tp2 = triple_pts[1]
            if not np.isnan(tp1[1]):
                filt1 = ys1 > tp1[1]
                xs1 = xs1[filt1]
                ys1 = ys1[filt1]
            if not np.isnan(tp2[1]):
                filt2 = ys2 > tp2[1]
                xs2 = xs2[filt2]
                ys2 = ys2[filt2]
            if verbose:
                plt.figure()
                self.display()
                plt.figure()
                plt.plot(xs1, ys1)
                plt.plot(xs2, ys2)
                plt.axis('equal')
                plt.show()
        xs = np.concatenate((xs1, xs2))
        ys = np.concatenate((ys1, ys2))
        (xc, yc), R1, R2, theta = fit_ellipse(xs, ys)
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
        # Return
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        des = DropEllipseFit(xyc=[xc, yc], R1=R1, R2=R2, theta=theta,
                             baseline=self.baseline,
                             x_bounds=x_bounds,
                             y_bounds=y_bounds)
        return des

    def fit_ellipses(self, triple_pts=None, verbose=False):
        """
        Fit the drop edges with two ellipses (one on each sides).

        Remove points under the triple points if possible.

        Returns:
        --------
        fit: DropFit object
            .
        """
        if self.drop_edges is None:
            self._separate_drop_edges()
        dex1, dey1, dex2, dey2 = self.drop_edges
        if dex1 is None or dex2 is None:
            des = DropEllipseFit(xyc=[np.nan, np.nan], R1=np.nan,
                                 R2=np.nan, theta=np.nan,
                                 baseline=self.baseline,
                                 x_bounds=[np.nan, np.nan],
                                 y_bounds=[np.nan, np.nan])
            return des
        xs1 = dex1.y
        ys1 = dey1.y
        xs2 = dex2.y
        ys2 = dey2.y
        if triple_pts is not None:
            tp1 = triple_pts[0]
            tp2 = triple_pts[1]
            if not np.isnan(tp1[1]):
                filt1 = ys1 > tp1[1]
                xs1 = xs1[filt1]
                ys1 = ys1[filt1]
            if not np.isnan(tp2[1]):
                filt2 = ys2 > tp2[1]
                xs2 = xs2[filt2]
                ys2 = ys2[filt2]
            if verbose:
                plt.figure()
                self.display()
                plt.figure()
                plt.plot(xs1, ys1)
                plt.plot(xs2, ys2)
                plt.axis('equal')
                plt.show()
        fit1 = fit_ellipse(xs1, ys1)
        fit2 = fit_ellipse(xs2, ys2)
        if verbose:
            plt.figure()
            self.display()
            thetas = np.linspace(0, np.pi*2, 100)
            for fit in [fit1, fit2]:
                (xc, yc), R1, R2, theta = fit
                xs = xc + R1*np.cos(thetas)
                ys = yc + R2*np.sin(thetas)
                new_xs = xc + np.cos(theta)*(xs - xc) - np.sin(theta)*(ys - yc)
                new_ys = yc + np.cos(theta)*(ys - yc) + np.sin(theta)*(xs - xc)
                plt.figure()
                self.display()
                plt.plot(xc, yc, 'ok')
                plt.plot(new_xs, new_ys)
            plt.show()
        # Return
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        des = DropEllipsesFit(xyc1=fit1[0], R1a=fit1[1], R1b=fit1[2],
                              theta1=fit1[3],
                              xyc2=fit2[0], R2a=fit2[1], R2b=fit2[2],
                              theta2=fit2[3],
                              baseline=self.baseline,
                              x_bounds=x_bounds,
                              y_bounds=y_bounds)
        return des

    def fit_LY(self, interp_res=100, method=None, verbose=False):
        """
        Fit the drop edges with the Laplace-young equations.

        Returns:
        --------
        fit: DropFit object
            .
        """
        raise Exception('Not implemented (properly) yes')
        # get data
        z1_min = triple_pts[0][1]
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
        # TODO: Need to return a DropFit object !
        return z1, - new_r1 + self.get_drop_position()

    def fit_circle(self, triple_pts=None, verbose=False):
        """
        Fit a circle to the edges.

        Ignore the lower part of the drop if triple points are presents.
        """
        # get data
        if self.drop_edges is None:
            self._separate_drop_edges()
        dex1, dey1, dex2, dey2 = self.drop_edges
        if dex1 is None or dex2 is None:
            dcf = DropCircleFit(xyc=[np.nan, np.nan], R=np.nan,
                                baseline=self.baseline,
                                x_bounds=[np.nan, np.nan],
                                y_bounds=[np.nan, np.nan])
            return dcf
        xs1 = dex1.y
        ys1 = dey1.y
        xs2 = dex2.y
        ys2 = dey2.y
        if triple_pts is not None:
            if not np.any(np.isnan(triple_pts[0])):
                ytp = triple_pts[0][1]
                filt = ys1 > ytp
                xs1 = xs1[filt]
                ys1 = ys1[filt]
            if not np.any(np.isnan(triple_pts[1])):
                ytp = triple_pts[1][1]
                filt = ys2 > ytp
                xs2 = xs2[filt]
                ys2 = ys2[filt]
        xs = np.concatenate((xs1, xs2))
        ys = np.concatenate((ys1, ys2))
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
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        dcf = DropCircleFit(xyc=c, R=R, baseline=self.baseline,
                            x_bounds=x_bounds, y_bounds=y_bounds)
        return dcf

    def fit_circles(self, triple_pts, sigma_max=None, soft_constr=False,
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
        # get data
        tp1 = triple_pts[0]
        tp2 = triple_pts[1]
        if self.drop_edges is None:
            self._separate_drop_edges()
        dex1, dey1, dex2, dey2 = self.drop_edges
        # check
        if dex1 is None or dex2 is None:
            dcf = DropCirclesFit(xyc=[[np.nan, np.nan],
                                      [np.nan, np.nan],
                                      [np.nan, np.nan]],
                                 Rs=[np.nan, np.nan, np.nan],
                                 baseline=self.baseline,
                                 triple_pts=[[np.nan, np.nan],
                                             [np.nan, np.nan]],
                                 x_bounds=[np.nan, np.nan],
                                 y_bounds=[np.nan, np.nan])
            return dcf
        #
        xs1 = dex1.y
        ys1 = dey1.y
        xs2 = dex2.y
        ys2 = dey2.y
        # reorder
        if ys1[0] > ys1[-1]:
            ys1 = ys1[::-1]
            xs1 = xs1[::-1]
        if ys2[0] > ys2[-1]:
            ys2 = ys2[::-1]
            xs2 = xs2[::-1]
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
        # Get triple points based from circles
        xtp1 = (c_o1[0]*Rd + c_d[0]*R1)/(Rd + R1)
        ytp1 = (c_o1[1]*Rd + c_d[1]*R1)/(Rd + R1)
        xtp2 = (c_o2[0]*Rd + c_d[0]*R2)/(Rd + R2)
        ytp2 = (c_o2[1]*Rd + c_d[1]*R2)/(Rd + R2)
        circle_triple_pts = [[xtp1, ytp1], [xtp2, ytp2]]
        # Return
        x_bounds = [np.min(self.xy[:, 0]), np.max(self.xy[:, 0])]
        y_bounds = [np.min(self.xy[:, 1]), np.max(self.xy[:, 1])]
        dcf = DropCirclesFit(xyc=[c_d, c_o1, c_o2],
                             Rs=[Rd, R1, R2],
                             baseline=self.baseline,
                             triple_pts=circle_triple_pts,
                             x_bounds=x_bounds,
                             y_bounds=y_bounds)
        return dcf

    def display(self, *args, **kwargs):
        """
        """
        if self.drop_edges is not None:
            dex1, dey1, dex2, dey2 = self.drop_edges
            xs1 = dex1.y
            ys1 = dey1.y
            xs2 = dex2.y
            ys2 = dey2.y
            plt.plot(xs1, ys1, color='k', marker='o')
            plt.plot(xs2, ys2, color='k', marker='o')
        else:
            plt.plot(self.xy[:, 0], self.xy[:, 1], color='k', marker='o')
        plt.axis('equal')
        plt.gca().set_adjustable('box')
        # Display baseline
        if self.baseline is not None:
            x0 = np.min(self.xy[:, 0])
            xf = np.max(self.xy[:, 0])
            x0 -= np.abs(xf - x0)*.1
            xf += np.abs(xf - x0)*.1
            self.baseline.display(x0, xf, color=self.colors[0])
