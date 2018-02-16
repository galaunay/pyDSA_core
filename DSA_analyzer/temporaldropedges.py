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
        Compute a spline fitting for the droplets shape.

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline.  Must be <= 5.
            Default is k=5.
        s : float or None, optional
            Positive smoothing factor used to choose the number of knots.
            Smaller number means better fitted curves.
            If None (default), a default value will be inferred from the data.
            If 0, spline will interpolate through all data points.
        """
        if verbose:
            pg = ProgressCounter("Fitting droplet interfaces", "Done",
                                 len(self.point_sets), 'edges', 5)
        for edge in self.point_sets:
            edge.fit(k=k, s=s, verbose=False)
            if verbose:
                pg.print_progress()

    def detect_triple_points(self, smooth=None, verbose=False):
        """
        Compute the triple points (water, oil and air interfaces) positions.

        Parameters
        ==========
        smooth: number
           Smoothing factor for the triple point position.

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
        if smooth is not None:
            self.smooth_triple_points(tos='gaussian', size=smooth)

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

    def get_drop_radius(self):
        """
        Return the drops base radius based on the triple points position.
        """
        radii = []
        for edge in self.point_sets:
            radii.append(edge.get_drop_radius())
        return radii

    def compute_contact_angle(self, smooth=None, verbose=False):
        """
        Compute the drop contact angles."
        """
        if verbose:
            pg = ProgressCounter("Getting contact angles", "Done",
                                 len(self.point_sets), 'images', 5)
        thetas = []
        thetas_triple = []
        for edge in self.point_sets:
            edge.compute_contact_angle()
            thetas.append(edge.thetas)
            if edge.thetas_triple is not None:
                thetas_triple.append(edge.thetas_triple)
            if verbose:
                pg.print_progress()
        if smooth:
            self.smooth_contact_angle(tos='gaussian', size=smooth)

    def smooth_triple_points(self, tos='gaussian', size=None):
        """
        Smooth the position of the triple point.

        Parameters
        ==========
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        """
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        t = []
        mask = []
        for i, edge in enumerate(self.point_sets):
            t.append(self.times[i])
            if edge.triple_pts is not None:
                x1.append(edge.triple_pts[0][0])
                x2.append(edge.triple_pts[1][0])
                y1.append(edge.triple_pts[0][1])
                y2.append(edge.triple_pts[1][1])
                mask.append(False)
            else:
                x1.append(np.nan)
                x2.append(np.nan)
                y1.append(np.nan)
                y2.append(np.nan)
                mask.append(True)
        if not np.all(np.isnan(x1)):
            x1 = Profile(t, x1).smooth(tos='gaussian', size=size).y
            x2 = Profile(t, x2).smooth(tos='gaussian', size=size).y
            y1 = Profile(t, y1).smooth(tos='gaussian', size=size).y
            y2 = Profile(t, y2).smooth(tos='gaussian', size=size).y
            for i, edge in enumerate(self.point_sets):
                if not mask[i]:
                    edge.triple_pts = [[x1[i], y1[i]], [x2[i], y2[i]]]

    def smooth_contact_angle(self, tos='gaussian', size=None):
        """
        Smooth the contact angles.

        Parameters
        ==========
        tos : string, optional
            Type of smoothing, can be 'uniform' (default) or 'gaussian'
            (See ndimage module documentation for more details)
        size : number, optional
            Size of the smoothing (is radius for 'uniform' and
            sigma for 'gaussian').
            Default is 3 for 'uniform' and 1 for 'gaussian'.
        """
        theta1 = []
        theta2 = []
        theta3 = []
        theta4 = []
        t = []
        mask = []
        mask2 = []
        for i, edge in enumerate(self.point_sets):
            t.append(self.times[i])
            if edge.thetas is not None:
                theta1.append(edge.thetas[0])
                theta2.append(edge.thetas[1])
                mask.append(False)
            else:
                theta1.append(np.nan)
                theta2.append(np.nan)
                mask.append(True)
            if edge.thetas_triple is not None:
                theta3.append(edge.thetas_triple[0])
                theta4.append(edge.thetas_triple[1])
                mask2.append(False)
            else:
                theta3.append(np.nan)
                theta4.append(np.nan)
                mask2.append(True)
        # smooth
        if not np.all(mask):
            theta1 = Profile(t, theta1).smooth(tos='gaussian', size=size).y
            theta2 = Profile(t, theta2).smooth(tos='gaussian', size=size).y
            for i, edge in enumerate(self.point_sets):
                if not mask[i]:
                    edge.thetas = [theta1[i], theta2[i]]
        if not np.all(mask2):
            theta3 = Profile(t, theta3).smooth(tos='gaussian', size=size).y
            theta4 = Profile(t, theta4).smooth(tos='gaussian', size=size).y
            for i, edge in enumerate(self.point_sets):
                if not mask2[i]:
                    edge.thetas_triple = [theta3[i], theta4[i]]

    def display(self, *args, **kwargs):
        #
        length = len(self.point_sets)
        # Display points
        if self[0].edges_fits is None:
            kwargs['cpkw'] = {}
            kwargs['cpkw']['aspect'] = 'equal'
            super().display(*args, **kwargs)
        # Display fitting
        if self[0].edges_fits is not None:
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
            if len(x1s) != length or len(x2s) != length:
                raise Exception()
            db1 = pplt.Displayer(x1s, y1s, color=self[0].colors[1])
            db2 = pplt.Displayer(x2s, y2s, color=self[0].colors[1])
            pplt.ButtonManager(db1)
            pplt.ButtonManager(db2)
        # Display triple points
        xs = []
        ys = []
        for edge in self.point_sets:
            if edge.triple_pts is None:
                xs.append([np.nan, np.nan])
                ys.append([np.nan, np.nan])
            else:
                xs.append([edge.triple_pts[0][0],
                            edge.triple_pts[1][0]])
                ys.append([edge.triple_pts[0][1],
                            edge.triple_pts[1][1]])
        if len(xs) != length or len(ys) != length:
            raise Exception()
        if not np.all(np.isnan(xs)):
            db = pplt.Displayer(xs, ys, ls='none', marker='o',
                                kind='plot',
                                color=self[0].colors[2])
            pplt.ButtonManager(db)
        # Display contact angles
        if np.any([edge.thetas for edge in self.point_sets] is not None):
            lines = [edge._get_angle_display_lines()
                     for edge in self.point_sets]
            lines1 = []
            lines2 = []
            lines3 = []
            lines4 = []
            for line in lines:
                lines1.append(line[0])
                lines2.append(line[1])
                if len(line) > 2:
                    lines3.append(line[2])
                    lines4.append(line[3])
                else:
                    lines3.append([[np.nan, np.nan], [np.nan, np.nan]])
                    lines4.append([[np.nan, np.nan], [np.nan, np.nan]])
            for line in [lines1, lines2, lines3, lines4]:
                line = np.array(line)
                if not np.all(np.isnan(line)):
                    if len(line[:, 0]) != length or len(line[:, 1]) != length:
                        raise Exception()
                    db = pplt.Displayer(line[:, 0], line[:, 1],
                                        kind='plot', color=self[0].colors[0])
                    pplt.ButtonManager(db)

    def display_summary(self):
        """
        Display a summary of the drop parameters evolution.
        """
        bdp = self.get_drop_base()
        radii = self.get_drop_base_radius()
        radiit = self.get_drop_radius()
        thetas = []
        thetas_triple = []
        triple_pts1 = []
        triple_pts2 = []
        for edge in self.point_sets:
            if edge.thetas is not None:
                thetas.append(edge.thetas)
            else:
                thetas.append([np.nan, np.nan])
            if edge.thetas_triple is not None:
                thetas_triple.append(edge.thetas_triple)
            else:
                thetas_triple.append([np.nan, np.nan])
            if edge.triple_pts is not None:
                triple_pts1.append(edge.triple_pts[0])
                triple_pts2.append(edge.triple_pts[1])
            else:
                triple_pts1.append([np.nan, np.nan])
                triple_pts2.append([np.nan, np.nan])
        thetas = np.asarray(thetas)
        thetas_triple = np.asarray(thetas_triple)
        triple_pts1 = np.asarray(triple_pts1)
        triple_pts2 = np.asarray(triple_pts2)
        ts = np.arange(0, len(self.point_sets)*self.dt, self.dt)[0: len(bdp)]
        fig, axs = plt.subplots(2, 1)
        # Drop dimensions
        plt.sca(axs[0])
        plt.plot(bdp[:, 0], ts, label="Contact (left)")
        plt.plot(bdp[:, 1], ts, label="Contact (right)")
        plt.plot(radii, ts, label="Base radius")
        if (not np.all(np.isnan(triple_pts1)) and
            not np.all(np.isnan(triple_pts2))):
            plt.plot(radiit, ts, label="Drop radius")
            plt.plot(triple_pts1[:, 0], ts, label="Triple point (left)")
            plt.plot(triple_pts2[:, 0], ts, label="Triple point (right)")
        plt.xlabel('[um]')
        plt.ylabel('Time [s]')
        plt.legend(loc=0)
        # Contact angles
        plt.sca(axs[1])
        plt.plot(-thetas[:, 0], ts, label="Angle (left)")
        plt.plot(180 - thetas[:, 1], ts, label="Angle (right)")
        if not np.all(np.isnan(thetas_triple)):
            plt.plot(-thetas_triple[:, 0], ts,
                     label="Angle at triple point (left)",
                     marker=",", ls="-")
            plt.plot(180 - thetas_triple[:, 1], ts,
                     label="Angle at triple point (right)",
                     marker=",", ls="-")
        plt.ylabel('Time [s]')
        plt.xlabel('[Deg]')
        plt.legend(loc=0)
