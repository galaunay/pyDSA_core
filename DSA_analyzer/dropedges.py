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
import matplotlib.pyplot as plt
import scipy.interpolate as spint
import scipy.optimize as spopt
import scipy.misc as spmisc
from IMTreatment import Points
import IMTreatment.plotlib as pplt


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class DropEdges(Points):
    def __init__(self, baseline, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.drop_edges = self._separate_drop_edges()
        self.edges_fits = None
        self.triple_pts = None
        self.baseline = baseline
        self.thetas = None

    def _separate_drop_edges(self):
        """
        Separate the two sides of the drop.
        """
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
        de1 = Points(list(zip(xs1, ys1)), unit_x=self.unit_x,
                     unit_y=self.unit_y)
        de2 = Points(list(zip(xs2, ys2)), unit_x=self.unit_x,
                     unit_y=self.unit_y)
        return de1, de2

    def detect_triple_points(self, verbose=False):
        """
        Compute the triple points (water, oil and air interfaces) position.

        Returns
        =======
        tripl_pts: 2x2 array of numbers
           Position of the triple points for each edge ([pt1, pt2])
        """
        if self.edges_fits is None:
            raise Exception("You should computing fitting first with 'fit()'")
        tripl_pts = []
        if verbose:
            plt.figure()
        for i in [0, 1]:
            fit = self.edges_fits[i]
            y = self.drop_edges[i].xy[:, 1]
            dy = (y[-1] - y[0])/100
            def zerofun(y):
                return spmisc.derivative(fit, y, dx=dy, n=2, order=3)
            def dzerofun(y):
                return spmisc.derivative(fit, y, dx=dy, n=3, order=5)
            try:
                y0 = spopt.newton(zerofun, (y[0] + y[-1])/2, dzerofun)
            except RuntimeError:
                raise Exception('Cannot find the triple point here.'
                                '\nYou should try a different fitting.')
            tripl_pts.append([fit(y0), y0])
            if verbose:
                plt.plot(y, zerofun(y))
        if verbose:
            plt.xlabel('y')
            plt.xlabel('d^2y/dx^2')
            plt.show()
        self.triple_pts = tripl_pts
        return tripl_pts

    def fit(self, k=5, s=None, verbose=False):
        """
        Get a fitting for the droplet shape.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        """
        # Prepare drop edge for interpolation
        # TODO: Find a more efficient fitting
        de1, de2 = self._separate_drop_edges()
        x1 = de1.xy[:, 0]
        y1 = de1.xy[:, 1]
        new_y1 = np.sort(list(set(y1)))
        new_x1 = [np.mean(x1[y == y1]) for y in new_y1]
        x2 = de2.xy[:, 0]
        y2 = de2.xy[:, 1]
        new_y2 = np.sort(list(set(y2)))
        new_x2 = [np.mean(x2[y == y2]) for y in new_y2]
        # spline interpolation
        s = s or len(y1)/10
        try:
            spline1 = spint.UnivariateSpline(new_y1, new_x1, k=k, s=s)
            spline2 = spint.UnivariateSpline(new_y2, new_x2, k=k, s=s)
        except:
            spline1 = lambda x: np.nan
            spline2 = lambda x: np.nan
        # store
        self.edges_fits = [spline1, spline2]
        # Verbose if necessary
        if verbose:
            plt.figure()
            de1.display()
            plt.plot(new_x1, new_y1, '--')
            plt.plot(spline1(new_y1), new_y1, 'r')
            de2.displanew_y()
            plt.plot(new_x2, new_y2, '--')
            plt.plot(spline2(new_y2), new_y2, 'r')
            plt.axis('equal')
            plt.show()
        return spline1, spline2

    def fit_LY(self, interp_res=100, verbose=False):
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
        print('=== Fitting sigma and pressure')
        res1 = spopt.minimize(minfun, [1, 1])
        if not res1.success:
            raise Exception('Fitting sigma failed...')
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
        print('=== Fitting edges')
        res2 = spopt.minimize(minfun, r1, (dz, z1))
        if not res2.success:
            raise Exception('Fitting drop edge failed...')
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
        super().display(*args, **kwargs)
        colors = pplt.get_color_cycles()
        # Display baseline
        self.baseline.display(color=colors[0])
        # Display fits
        if self.fit is not None:
            xy_inter = self._get_inters_base_fit()
            y1 = np.linspace(xy_inter[0][1],
                             np.max(self.xy[:, 1]),
                             1000)
            y2 = np.linspace(xy_inter[1][1],
                             np.max(self.xy[:, 1]),
                             1000)
            x1 = self.edges_fits[0](y1)
            x2 = self.edges_fits[1](y2)
            plt.plot(x1, y1, color=colors[1])
            plt.plot(x2, y2, color=colors[1])
        # Display contact angle
        if self.thetas is not None:
            length = (y1[-1] - y1[0])/3
            theta1 = self.thetas[0]/180*np.pi
            theta2 = self.thetas[1]/180*np.pi
            plt.plot([x1[0], x1[0] + length*np.sin(theta1)],
                     [y1[0], y1[0] + length*np.cos(theta1)],
                     color=colors[3])
            plt.plot([x2[0], x2[0] + length*np.sin(theta2)],
                     [y2[0], y2[0] + length*np.cos(theta2)],
                     color=colors[3])
            plt.plot(x2[0], y2[0], color=colors[3], marker='o',
                     ls='none')
        # Display triple points
        if self.triple_pts is not None:
            for i in [0, 1]:
                plt.plot(self.triple_pts[i][0],
                         self.triple_pts[i][1],
                         marker="o",
                         color=colors[4])

    def _get_inters_base_fit(self):
        bfun = self.baseline.get_baseline_fun(along_y=True)
        xys = []
        for i in range(2):
            sfun = self.edges_fits[i]
            y_inter = spopt.fsolve(lambda y: bfun(y) - sfun(y),
                                   (self.baseline.pt1[1] +
                                    self.baseline.pt2[1])/2)[0]
            x_inter = bfun(y_inter)
            xys.append([x_inter, y_inter])
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
            raise Exception("You should computing fitting first with 'fit()'")
        x1 = self.edges_fits[0](self.xy[0, 1])
        x2 = self.edges_fits[1](self.xy[0, 1])
        return np.sort([x1, x2])

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

    def compute_contact_angle(self, verbose=False):
        """
        Compute the contact angles.

        Returns
        =======
        thetas : 2x1 array of numbers
           Contact angles in °
        """
        if self.edges_fits is None:
            raise Exception("You should computing fitting first with 'fit()'")
        # Find interesection between fitting and baseline
        xy_inter = self._get_inters_base_fit()
        thetas = []
        for i in range(2):
            x_inter, y_inter = xy_inter[i]
            sfun = self.edges_fits[i]
            # Get gradient
            dy = self.drop_edges[i].xy[:, 1]
            dy = (dy[-1] - dy[0])/100
            deriv = spmisc.derivative(sfun, y_inter, dx=dy)
            thetas.append(np.arctan(deriv)*180/np.pi)
            # display if asked
            if verbose:
                self.display()
                plt.plot(x_inter, y_inter, 'bo')
                dy = dy*10
                plt.plot([x_inter, x_inter + dy*deriv],
                         [y_inter, y_inter + dy], 'b')
                plt.axis('equal')
        self.thetas = thetas
        return [thetas[0], thetas[1]]



        # Should use laplace-young fitting
