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


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
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
        if pts is not None:
            self.from_points(pts=pts, xmin=xmin, xmax=xmax)

    def from_points(self, pts, xmin=None, xmax=None):
        if len(pts) == 2 and xmin is None and xmax is None:
            self.pt1 = np.array(pts[0], dtype=float)
            self.pt2 = np.array(pts[1], dtype=float)
        else:
            self.pt1, self.pt2 = self.get_baseline_from_points(pts,
                                                               xmin=xmin,
                                                               xmax=xmax)
        # get coefs
        self.xy = [[self.pt1[0], self.pt2[0]], [self.pt1[1], self.pt2[1]]]
        self.coefs = np.polyfit(self.xy[0], self.xy[1], 1)

    def get_baseline_from_points(self, pts, xmin=None, xmax=None):
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

    def scale(self, scalex=None, scaley=None):
        """
        Scale the baseline.
        """
        pt1 = self.pt1
        pt2 = self.pt2
        if scalex is not None:
            pt1 *= scalex
        if scaley is not None:
            pt2 *= scaley
        if (scalex is not None) or (scaley is not None):
            self.from_points([pt1, pt2])

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
