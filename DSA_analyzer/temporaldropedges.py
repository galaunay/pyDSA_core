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
from IMTreatment.utils import ProgressCounter
from IMTreatment import TemporalPoints, Profile
from .dropedges import DropEdges


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class TemporalDropEdges(TemporalPoints):
    def __init__(self, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for i in range(len(self.point_sets)):
            yield self.point_sets[i]

    def __getitem__(self, i):
        return self.point_sets[i]

    def fit(self, k=5, s=None, verbose=False):
        """
        Get a fitting for the droplets shape.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        """
        if verbose:
            pg = ProgressCounter("Fitting droplet interfaces", "Done",
                                 len(self.point_sets), 'edges', 5)
        for edge in self.point_sets:
            edge.fit(k=k, s=s, verbose=False)
            if verbose:
                pg.print_progress()

    def detect_triple_points(self, verbose=False):
        """
        Compute the triple points (water, oil and air interfaces) positions.

        Returns
        =======
        tripl_pts: 2x2 array of numbers
           Position of the triple points for each edge ([pt1, pt2])
        """
        if verbose:
            pg = ProgressCounter("Detecting triple point positions", "Done",
                                 len(self.point_sets), 'triple points', 5)
        for edge in self.point_sets:
            edge.detect_triple_points()
            if verbose:
                pg.print_progress()

    def get_drop_base(self):
        """
        Return the drops base.
        """
        dbs = []
        for edge in self.point_sets:
            dbs.append(edge.get_drop_base())
        return np.array(dbs)

    def get_drop_base_radius(self):
        """
        Return the drops base radius.
        """
        radii = []
        for edge in self.point_sets:
            radii.append(edge.get_drop_base_radius())
        return radii

    def get_contact_angle(self, smooth=None):
        """
        Return the drop contact angles."
        """
        thetas = []
        for edge in self.point_sets:
            thetas.append(edge.get_contact_angle())
        if smooth:
            thetas = np.array(thetas)
            tmp_prof1 = Profile(np.arange(len(thetas[:, 0])),
                                thetas[:, 0])
            tmp_prof1.smooth(size=smooth, inplace=True)
            tmp_prof2 = Profile(np.arange(len(thetas[:, 1])),
                                thetas[:, 1])
            tmp_prof2.smooth(size=smooth, inplace=True)
            thetas = np.array([[tmp_prof1.y[i], tmp_prof2.y[i]]
                               for i in range(len(thetas))])
        return np.array(thetas)
