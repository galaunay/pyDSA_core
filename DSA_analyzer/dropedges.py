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
        self.edges_fits = None

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
            de2.display()
            plt.plot(new_x2, new_y2, '--')
            plt.plot(spline2(new_y2), new_y2, 'r')
            plt.axis('equal')
            plt.show()
        return spline1, spline2

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

    def get_drop_base_radius(self):
        """
        Return the drop base radius.
        """
        db = self.get_drop_base()
        return db[1] - db[0]

    def get_contact_angle(self, k=5, s=None, verbose=False):
        raise Exception('Not yet implemented')
        # Should use laplace-young fitting
