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
import unum
import copy


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class Baseline(object):
    def __init__(self, pts=None, xmin=None, xmax=None):
        """

        """
        self.pt1 = None
        self.pt2 = None
        self.xy = []
        self.coefs = []
        self.tilt_angle = None
        if pts is not None:
            self.from_points(pts=pts, xmin=xmin, xmax=xmax)

    def __eq__(self, other):
        if not isinstance(other, Baseline):
            return False
        for arg in ['pt1', 'pt2', 'xy', 'coefs', 'tilt_angle']:
            try:
                getattr(self, arg)[0]
                if not np.allclose(getattr(self, arg), getattr(other, arg)):
                    return False
            except (TypeError, IndexError):
                if not np.isclose(getattr(self, arg), getattr(other, arg)):
                    return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def from_points(self, pts, xmin=None, xmax=None):
        if len(pts) == 2 and ((xmin is None and xmax is None)
                              or (xmin == pts[0][0] and xmax == pts[1][0])):
            self.pt1 = np.array(pts[0], dtype=float)
            self.pt2 = np.array(pts[1], dtype=float)
        else:
            self.pt1, self.pt2 = self.get_baseline_from_points(pts,
                                                               xmin=xmin,
                                                               xmax=xmax)
        # get coefs
        self.xy = [[self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]]]
        slope = (self.pt2[1] - self.pt1[1])/(self.pt2[0] - self.pt1[0])
        intercept = self.pt2[1] - slope*self.pt2[0]
        self.coefs = [slope, intercept]
        self.tilt_angle = np.arctan(self.coefs[0])

    @staticmethod
    def get_baseline_from_points(pts, xmin=None, xmax=None):
        pos = np.array(pts)
        a, b = np.polyfit(pos[:, 0], pos[:, 1], 1)
        if xmin is None:
            xmin = np.min(pos[:, 0])
        if xmax is None:
            xmax = np.max(pos[:, 0])
        res = np.array([[xmin, a*xmin + b], [xmax, a*xmax + b]],
                       dtype=float)
        return res

    def get_baseline_fun(self, along_y=False):
        a, b = self.coefs
        # check if flat:
        if abs(a) < abs(b)*1e-10 and along_y:
            raise Exception("baseline is flat")
        if along_y:
            def fun(y):
                return (y - b)/a
        else:
            def fun(x):
                return a*x + b
        return fun

    def get_projection_to_baseline(self, xy):
        """
        Return the projection of the given point on the baseline.
        """
        vec1 = np.array([self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1]])
        vec2 = np.array([xy[0] - self.pt1[0], xy[1] - self.pt1[1]])
        vec3 = np.dot(vec1, vec2)/np.dot(vec1, vec1)*vec1
        return np.array([vec3[0] + self.pt1[0], vec3[1] + self.pt1[1]])

    def get_distance_to_baseline(self, xy):
        """
        Return the distance between the given point and the baseline/
        """
        bspos = self.get_projection_to_baseline(xy)
        return np.linalg.norm(xy - bspos)

    def scale(self, scalex=None, scaley=None):
        """
        Scale the baseline.
        """
        if isinstance(scalex, unum.Unum):
            scalex = float(scalex.asNumber())
        if isinstance(scaley, unum.Unum):
            scaley = float(scaley.asNumber())
        # print(scalex)
        # print(scaley)
        pt1 = self.pt1
        pt2 = self.pt2
        if scalex is not None:
            pt1[0] *= scalex
            pt2[0] *= scalex
        if scaley is not None:
            pt1[1] *= scaley
            pt2[1] *= scaley
        if (scalex is not None) or (scaley is not None):
            self.from_points([pt1, pt2])

    def set_origin(self, xo, yo):
        """
        Change the origin of the baseline to the new point.
        """
        x = np.array([self.pt1[0], self.pt2[0]])
        y = np.array([self.pt1[1], self.pt2[1]])
        new_x = x - xo
        new_y = y - yo
        self.from_points([[new_x[0], new_y[0]],
                          [new_x[1], new_y[1]]])

    def rotate(self, angle):
        """
        Rotate the baseline.

        Parameters
        ----------
        angle: number
            Rotation angle in radian.
        """
        # Checks
        x = np.array([self.pt1[0], self.pt2[0]])
        y = np.array([self.pt1[1], self.pt2[1]])
        new_x = x*np.cos(angle) - y*np.sin(angle)
        new_y = y*np.cos(angle) + x*np.sin(angle)
        self.from_points([[new_x[0], new_y[0]],
                          [new_x[1], new_y[1]]])

    def display(self, x0=None, xf=None, *args, **kwargs):
        if x0 is None:
            x0 = self.xy[0][0]
            y0 = self.xy[1][0]
        else:
            fun = self.get_baseline_fun()
            y0 = fun(x0)
        if xf is None:
            xf = self.xy[0][1]
            yf = self.xy[1][1]
        else:
            fun = self.get_baseline_fun()
            yf = fun(xf)
        plt.plot([x0, xf], [y0, yf], *args, **kwargs)
