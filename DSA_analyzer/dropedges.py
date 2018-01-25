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
from IMTreatment import Points


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class DropEdges(Points):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.drop_edges = self._separate_drop_edges()

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

    def get_fitting(self, k=5, s=None, verbose=False):
        """
        Get a fitting for the droplet shape.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        """
        print('TODO: to clean')
        # Prepare drop edge for interpolation
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
        s = s or len(new_y1)/10
        spline1 = spint.UnivariateSpline(new_y1, new_x1, k=k, s=s)
        spline2 = spint.UnivariateSpline(new_y2, new_x2, k=k, s=s)
        if verbose:
            plt.figure()
            de1.display()
            plt.plot(spline1(new_y1), new_y1, 'r')
            de2.display()
            plt.plot(spline2(new_y2), new_y2, 'r')
            plt.axis('equal')
            plt.show()
        return spline1, spline2

    def get_contact_angle(self, k=5, s=None, verbose=False):
        print('TODO: to clean')
        # get spline
        de1, de2 = self._separate_drop_edges()
        y1 = de1.xy[:, 1]
        y2 = de2.xy[:, 1]
        spline1, spline2 = self.get_fitting(k=k, s=s, verbose=verbose)
        y1 = np.linspace(np.min(y1), np.max(y1), len(y1)*10)
        x1 = spline1(y1)
        y2 = np.linspace(np.min(y2), np.max(y2), len(y2)*10)
        x2 = spline2(y2)
        # Get gradients
        print('TODO: better way to estimate gradients...')
        grad1 = np.gradient(x1, y1[1] - y1[0])
        grad2 = np.gradient(x2, y2[1] - y2[0])
        if verbose:
            plt.figure()
            plt.plot(y1, grad1)
            plt.plot(y2, grad2)
            plt.title('gradients')
            plt.show()
        return grad1[0], grad2[0]
