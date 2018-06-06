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

    def fit(self, k=5, s=0.75, verbose=False):
        """
        Compute a spline fitting for the droplets shape.

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline.  Must be <= 5.
            Default is k=5.
        s : float, optional
            Smoothing factor between 0 (not smoothed) and 1 (very smoothed)
            Default to 0.75
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting droplet interfaces",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges', perc_interv=5)
        for edge in self.point_sets:
            edge.fit(k=k, s=s, verbose=False)
            if verbose:
                pg.print_progress()

    def detect_triple_points(self, smooth=None, use_x_minima=False,
                             verbose=False):
        """
        Compute the triple points (water, oil and air interfaces) positions.

        Parameters
        ==========
        smooth: number
           Smoothing factor for the triple point position.
        use_x_minima: boolean
            If True, try to define the triple point as the minimal x values and
            fall back to the curvature method if necessary.
            (default to False).

        Returns
        =======
        tripl_pts: 2x2 array of numbers
           Position of the triple points for each edge ([pt1, pt2])
        """
        if verbose:
            pg = ProgressCounter(init_mess="Detecting triple point positions",
                                 nmb_max=len(self.point_sets),
                                 name_things='triple points',
                                 perc_interv=5)
        for edge in self.point_sets:
            try:
                edge.detect_triple_points(use_x_minima=use_x_minima)
            except Exception:
                pass
            if verbose:
                pg.print_progress()
        if smooth is not None:
            self.smooth_triple_points(tos='gaussian', size=smooth)

    def fit_circles(self, sigma_max=None, verbose=False, nmb_pass=1):
        """
        Fit circles to the edges, cutting them if a triple point is
        present.

        Parameters
        ==========
        sigma_max: number
            If specified, points too far from the fit are iteratively removed
            until:
            std(R) < mean(R)*sigma_max
            With R the radii.
        nmb_pass: positive integer
            If superior to 1, specify the number of pass to make.
            Addintional passes use the previously triple points detected by
            circle fits to more accurately detect the next ones.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting circles to edges",
                                 nmb_max=len(self.point_sets)*nmb_pass,
                                 name_things='edges',
                                 perc_interv=5)
        # backup triple points
        tps_back = [edge.triple_pts for edge in self.point_sets]
        #  passes
        for i in range(nmb_pass):
            self.smooth_triple_points('gaussian', size=10)
            for edge in self.point_sets:
                try:
                    edge.fit_circles(sigma_max=sigma_max)
                except Exception:
                    pass
                if verbose:
                    pg.print_progress()
            for edge in self.point_sets:
                edge.triple_pts = edge.circle_triple_pts
        # restore triple point
        for i, tps in enumerate(tps_back):
            self.point_sets[i].triple_pts = tps

    def get_contact_angles(self):
        """
        Return the drop contact angles.
        """
        thetas = []
        thetas_triple = []
        for edge in self.point_sets:
            if edge.thetas is not None:
                thetas.append(edge.thetas)
            else:
                thetas.append([np.nan, np.nan])
            if edge.thetas_triple is not None:
                thetas_triple.append(edge.thetas_triple)
            else:
                thetas_triple.append([np.nan, np.nan])
        thetas = np.asarray(thetas)
        thetas_triple = np.asarray(thetas_triple)
        return thetas, thetas_triple

    def get_triple_points(self):
        """
        Return the drop triple points.
        """
        triple_pts1 = []
        triple_pts2 = []
        for edge in self.point_sets:
            if edge.triple_pts is not None:
                triple_pts1.append(edge.triple_pts[0])
                triple_pts2.append(edge.triple_pts[1])
            else:
                triple_pts1.append([np.nan, np.nan])
                triple_pts2.append([np.nan, np.nan])
        triple_pts1 = np.asarray(triple_pts1)
        triple_pts2 = np.asarray(triple_pts2)
        return triple_pts1, triple_pts2

    def get_ridge_heights(self, from_circ_fit=False):
        """
        Return the ridge heights.

        Parameters
        ==========
        from_circ_fit: boolean
            If true, use the triple points found by the
            circle fits.
        """
        height1 = []
        height2 = []
        for edge in self.point_sets:
            h1, h2 = edge.get_ridge_height(from_circ_fit=from_circ_fit)
            height1.append(h1)
            height2.append(h2)
        return (np.array(height1, dtype=float),
                np.array(height2, dtype=float))

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
            pg = ProgressCounter(init_mess="Getting contact angles",
                                 nmb_max=len(self.point_sets),
                                 name_things='images',
                                 perc_interv=5)
        for edge in self.point_sets:
            try:
                edge.compute_contact_angle()
            except Exception:
                pass
            if verbose:
                pg.print_progress()
        if smooth is not None:
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

    def display(self, displ_pts=True, displ_fit=True,
                displ_tp=True, displ_ca=True,
                displ_circ=True, displ_circ_tp=True,
                *args, **kwargs):
        #
        length = len(self.point_sets)
        kwargs['cpkw'] = {}
        kwargs['cpkw']['aspect'] = 'equal'
        kwargs['cpkw']['color'] = 'k'
        kwargs['cpkw']['marker'] = 'x'
        displs = []
        # Display points
        if self[0].drop_edges is not None and displ_pts:
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for edge in self.point_sets:
                if edge.drop_edges[0] is None:
                    x1s.append([])
                    x2s.append([])
                    y1s.append([])
                    y2s.append([])
                else:
                    x1s.append(edge.drop_edges[0].y)
                    x2s.append(edge.drop_edges[1].y)
                    y1s.append(edge.drop_edges[0].x)
                    y2s.append(edge.drop_edges[1].x)
            if len(x1s) != length or len(x2s) != length:
                raise Exception()
            db1 = pplt.Displayer(x1s, y1s, color='k', marker="o")
            db2 = pplt.Displayer(x2s, y2s, color='k', marker="o")
            displs.append(db1)
            displs.append(db2)
        # Display fitting
        if self[0].edges_fits is not None and displ_fit:
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for edge in self.point_sets:
                if edge.drop_edges[0] is None:
                    x1s.append([np.nan]*1000)
                    x2s.append([np.nan]*1000)
                    y1s.append([np.nan]*1000)
                    y2s.append([np.nan]*1000)
                else:
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
            displs.append(db1)
            displs.append(db2)
        # Display triple points
        if displ_tp:
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
                displs.append(db)
        # Display contact angles
        if np.any([edge.thetas for edge in self.point_sets] is not None) \
           and displ_ca:
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
                    displs.append(db)
        # Display circles
        if displ_circ:
            thetas = np.linspace(0, np.pi*2, 100)
            # loop on different circles
            for i in range(3):
                xs = []
                ys = []
                xps = []
                yps = []
                # Loop on times
                for j in range(len(self.point_sets)):
                    if self[j].circle_fits is None:
                        for l in [xs, ys]:
                            l.append(np.nan*thetas)
                        for l in [xps, yps]:
                            l.append([np.nan])
                        continue
                    if i >= len(self[j].circle_fits):
                        for l in [xs, ys]:
                            l.append(np.nan*thetas)
                        for l in [xps, yps]:
                            l.append([np.nan])
                        continue
                    if self[j].circle_fits[i] is None:
                        for l in [xs, ys]:
                            l.append(np.nan*thetas)
                        for l in [xps, yps]:
                            l.append([np.nan])
                        continue
                    # plot the circle !
                    (xc, yc), R = self[j].circle_fits[i]
                    xs.append(xc + R*np.cos(thetas))
                    ys.append(yc + R*np.sin(thetas))
                    xps.append([xc])
                    yps.append([yc])
                if not np.all(np.isnan(ys)):
                    db = pplt.Displayer(xs, ys, kind='plot',
                                        color=self[0].colors[5])
                    displs.append(db)
                if not np.all(np.isnan(yps)):
                    dbp = pplt.Displayer(xps, yps, kind='plot',
                                         marker='o',
                                         color=self[0].colors[5])
                    displs.append(dbp)
        # Display triple points from circle fits
        if displ_circ_tp:
            xs = []
            ys = []
            for i in range(len(self.point_sets)):
                # check
                if self[i].circle_triple_pts is None:
                    for l in [xs, ys]:
                        l.append([np.nan, np.nan])
                    continue
                xs.append([self[i].circle_triple_pts[0][0],
                           self[i].circle_triple_pts[1][0]])
                ys.append([self[i].circle_triple_pts[0][1],
                           self[i].circle_triple_pts[1][1]])
            if not np.all(np.isnan(ys)):
                db = pplt.Displayer(xs, ys, kind='plot',
                                    marker='o', ls='none',
                                    color=self[0].colors[5])
                displs.append(db)

        # Add button manager
        bm = pplt.ButtonManager(displs)
        return bm

    def display_ridge_evolution(self, tis=None, from_circ_fit=False):
        """
        Display an interactive plot with the ridge height evolution.
        """
        # Get dat
        rh1, rh2 = self.get_ridge_heights(from_circ_fit=from_circ_fit)
        rh = rh1.copy()
        rh[np.isnan(rh1)] = rh2[np.isnan(rh1)]
        filt = np.logical_and(~np.isnan(rh1), ~np.isnan(rh2))
        rh[filt] = (rh1[filt] + rh2[filt])/2
        # Check if there is ridge to display
        if not np.any(filt):
            raise Exception('No ridge detected, cannot display them...')

        # Display images + edges
        plt.figure()
        bmgr = self.display()
        if tis is not None:
            tis.display()
        # Display ridge evolution
        fig2 = plt.figure()
        #     function to navigate through images
        def on_click(event):
            if fig2.canvas.manager.toolbar._active is not None:
                return None
            x = event.xdata
            ind = np.argmin(abs(x - self.times))
            bmgr.ind = ind
            bmgr.update()
        fig2.canvas.mpl_connect('button_press_event', on_click)
        displ2 = pplt.Displayer([[xi] for xi in self.times],
                                [[yi] for yi in rh],
                                data_type="points",
                                color=self[0].colors[2],
                                edgecolors='k')
        bmgr2 = pplt.ButtonManager([displ2])
        plt.plot(self.times, rh1, zorder=0)
        plt.plot(self.times, rh2, zorder=0)
        plt.ylim(ymin=0)
        bmgr.link_to_other_graph(bmgr2)
        return bmgr, bmgr2

    def display_summary(self, figsize=None):
        """
        Display a summary of the drop parameters evolution.
        """
        bdp = self.get_drop_base()
        radii = self.get_drop_base_radius()
        radiit = self.get_drop_radius()
        thetas, thetas_triple = self.get_contact_angles()
        triple_pts1, triple_pts2 = self.get_triple_points()
        ts = np.arange(0, len(self.point_sets)*self.dt, self.dt)[0: len(bdp)]
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        # Drop dimensions
        plt.sca(axs[0])
        plt.plot(bdp[:, 0], ts, label="Contact (left)")
        plt.plot(bdp[:, 1], ts, label="Contact (right)")
        plt.plot(radii, ts, label="Base radius")
        if (not np.all(np.isnan(triple_pts1)) and
            not np.all(np.isnan(triple_pts2))):
            plt.plot(radiit, ts, label="Drop base length")
            plt.plot(triple_pts1[:, 0], ts, label="Triple point (left)")
            plt.plot(triple_pts2[:, 0], ts, label="Triple point (right)")
        plt.xlabel('x {}'.format(self.unit_x.strUnit()))
        plt.ylabel('Time {}'.format(self.unit_times.strUnit()))
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
        plt.ylabel('Time {}'.format(self.unit_times.strUnit()))
        plt.xlabel('[Deg]')
        plt.legend(loc=0)
