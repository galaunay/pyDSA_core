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
from IMTreatment import TemporalPoints, Profile, plotlib as pplt
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
        self.baseline = None
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

    def compute_contact_angle(self, smooth=None):
        """
        Compute the drop contact angles."
        """
        thetas = []
        thetas_triple = []
        for edge in self.point_sets:
            edge.compute_contact_angle()
            thetas.append(edge.thetas)
            if edge.thetas_triple is not None:
                thetas_triple.append(edge.thetas_triple)
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
            if len(thetas_triple) > 2:
                thetas_triple = np.array(thetas_triple)
                tmp_prof1 = Profile(np.arange(len(thetas_triple[:, 0])),
                                    thetas_triple[:, 0])
                tmp_prof1.smooth(size=smooth, inplace=True)
                tmp_prof2 = Profile(np.arange(len(thetas_triple[:, 1])),
                                    thetas_triple[:, 1])
                tmp_prof2.smooth(size=smooth, inplace=True)
                thetas_triple = np.array([[tmp_prof1.y[i], tmp_prof2.y[i]]
                                        for i in range(len(thetas_triple))])
        return np.array(thetas), np.array(thetas_triple)

    def display(self, *args, **kwargs):
        # Display points
        if self[0].fit is None:
            kwargs['cpkw'] = {}
            kwargs['cpkw']['aspect'] = 'equal'
            super().display(*args, **kwargs)
        # Display baseline
        self.baseline.display(color=self[0].colors[0])
        # Display fitting
        if self[0].fit is not None:
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for edge in self.point_sets:
                xy_inter = edge._get_inters_base_fit()
                y1 = np.linspace(xy_inter[0][1],
                                 np.max(edge.xy[:, 1]),
                                 1000)
                y2 = np.linspace(xy_inter[1][1],
                                 np.max(edge.xy[:, 1]),
                                 1000)
                x1 = edge.edges_fits[0](y1)
                x2 = edge.edges_fits[1](y2)
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
            db1 = pplt.Displayer(x1s, y1s, color=self[0].colors[1])
            db2 = pplt.Displayer(x2s, y2s, color=self[0].colors[1])
            pplt.ButtonManager(db1)
            pplt.ButtonManager(db2)
        # Display triple points
        if self[0].triple_pts is not None:
            xs = [[edge.triple_pts[i][0]
                  for i in [0, 1]]
                  for edge in self.point_sets]
            ys = [[edge.triple_pts[i][1]
                  for i in [0, 1]]
                  for edge in self.point_sets]
            db = pplt.Displayer(xs, ys, ls='none', marker='o',
                                kind='plot',
                                color=self[0].colors[2])
            pplt.ButtonManager(db)
        # Display contact angles
        lines = [edge._get_angle_display_lines()
                 for edge in self.point_sets]
        lines = [[line[0] for line in lines],
                 [line[1] for line in lines],
                 [line[2] for line in lines],
                 [line[3] for line in lines]]
        for line in lines:
            line = np.array(line)
            db = pplt.Displayer(line[:, 0], line[:, 1],
                                kind='plot', color=self[0].colors[0])
            pplt.ButtonManager(db)
