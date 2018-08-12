# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA
# Version: 0.1

# This file is part of pyDSA

# pyDSA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import matplotlib.pyplot as plt
from IMTreatment.utils import ProgressCounter
from IMTreatment import TemporalPoints, Profile, plotlib as pplt
from . import helpers as hlp


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class TemporalFits(object):
    def __init__(self, fits, temporaledges):
        """

        """
        self.baseline = temporaledges.baseline
        self.fits = fits
        self.times = temporaledges.times
        self.unit_times = temporaledges.unit_times
        self.dt = temporaledges.dt
        self.unit_x = temporaledges.unit_x
        self.unit_y = temporaledges.unit_y

    def __iter__(self):
        for i in range(len(self.fits)):
            yield self.fits[i]

    def __getitem__(self, i):
        return self.fits[i]

    def compute_contact_angle(self, verbose=False):
        """
        Compute the drop contact angles.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Getting contact angles",
                                 nmb_max=len(self.fits),
                                 name_things='images',
                                 perc_interv=5)
        for fit in self.fits:
            try:
                fit.compute_contact_angle()
            except Exception:
                pass
            if verbose:
                pg.print_progress()

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
        # Get triple points as profiles
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        t = []
        mask = []
        for i, fit in enumerate(self.fits):
            t.append(self.times[i])
            if fit.triple_pts is not None:
                x1.append(fit.triple_pts[0][0])
                x2.append(fit.triple_pts[1][0])
                y1.append(fit.triple_pts[0][1])
                y2.append(fit.triple_pts[1][1])
                mask.append(False)
            else:
                x1.append(np.nan)
                x2.append(np.nan)
                y1.append(np.nan)
                y2.append(np.nan)
                mask.append(True)
        # Smooth the profiles
        if not np.all(np.isnan(x1)):
            x1 = Profile(t, x1).smooth(tos='gaussian', size=size).y
            x2 = Profile(t, x2).smooth(tos='gaussian', size=size).y
            y1 = Profile(t, y1).smooth(tos='gaussian', size=size).y
            y2 = Profile(t, y2).smooth(tos='gaussian', size=size).y
            # Replace by the smoothed version
            for i, fit in enumerate(self.fits):
                if not mask[i]:
                    fit.triple_pts = [[x1[i], y1[i]], [x2[i], y2[i]]]

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
        # Get the contact angles as profiles
        theta1 = []
        theta2 = []
        theta3 = []
        theta4 = []
        t = []
        mask = []
        mask2 = []
        for i, fit in enumerate(self.fits):
            t.append(self.times[i])
            if fit.thetas is not None:
                theta1.append(fit.thetas[0])
                theta2.append(fit.thetas[1])
                mask.append(False)
            else:
                theta1.append(np.nan)
                theta2.append(np.nan)
                mask.append(True)
            if fit.thetas_triple is not None:
                theta3.append(fit.thetas_triple[0])
                theta4.append(fit.thetas_triple[1])
                mask2.append(False)
            else:
                theta3.append(np.nan)
                theta4.append(np.nan)
                mask2.append(True)
        # smooth
        if not np.all(mask):
            theta1 = Profile(t, theta1).smooth(tos='gaussian', size=size).y
            theta2 = Profile(t, theta2).smooth(tos='gaussian', size=size).y
            for i, fit in enumerate(self.fits):
                if not mask[i]:
                    fit.thetas = [theta1[i], theta2[i]]
        if not np.all(mask2):
            theta3 = Profile(t, theta3).smooth(tos='gaussian', size=size).y
            theta4 = Profile(t, theta4).smooth(tos='gaussian', size=size).y
            for i, fit in enumerate(self.fits):
                if not mask2[i]:
                    fit.thetas_triple = [theta3[i], theta4[i]]

    def get_contact_angles(self):
        """
        Return the drop contact angles.
        """
        thetas = []
        for fit in self.fits:
            if fit.thetas is not None:
                thetas.append(fit.thetas)
            else:
                thetas.append([np.nan, np.nan])
        thetas = np.asarray(thetas)
        return thetas

    def get_triple_points(self):
        """
        Return the drop triple points.
        """
        triple_pts1 = []
        triple_pts2 = []
        for fit in self.fits:
            if fit.triple_pts is not None:
                triple_pts1.append(fit.triple_pts[0])
                triple_pts2.append(fit.triple_pts[1])
            else:
                triple_pts1.append([np.nan, np.nan])
                triple_pts2.append([np.nan, np.nan])
        triple_pts1 = np.asarray(triple_pts1)
        triple_pts2 = np.asarray(triple_pts2)
        return triple_pts1, triple_pts2

    def get_ridge_height(self):
        """
        Return the ridge heights.
        """
        height1 = []
        height2 = []
        for fit in self.fits:
            h1, h2 = fit.get_ridge_height()
            height1.append(h1)
            height2.append(h2)
        return (np.array(height1, dtype=float),
                np.array(height2, dtype=float))

    def get_base_diameter(self):
        """
        Return the base diameters.
        """
        diams = []
        for fit in self.fits:
            diams.append(fit.get_base_diameter())
        return np.asarray(diams)

    def _get_inters_base_fit(self):
        inter = []
        for fit in self.fits:
            inter.append(fit._get_inters_base_fit())
        return np.asarray(inter)

    def display(self, displ_tp=True, displ_ca=True):
        displs = []
        length = len(self.fits)
        # Display triple points
        if displ_tp:
            xs = []
            ys = []
            for fit in self.fits:
                if fit.triple_pts is None:
                    xs.append([np.nan, np.nan])
                    ys.append([np.nan, np.nan])
                else:
                    xs.append([fit.triple_pts[0][0],
                               fit.triple_pts[1][0]])
                    ys.append([fit.triple_pts[0][1],
                               fit.triple_pts[1][1]])
            if len(xs) != length or len(ys) != length:
                raise Exception()
            if not np.all(np.isnan(xs)):
                db = pplt.Displayer(xs, ys, ls='none', marker='o',
                                    kind='plot',
                                    color=self[0].colors[2])
                displs.append(db)
        # Display contact angles
        if np.any([fit.thetas for fit in self.fits] is not None) \
           and displ_ca:
            lines = [fit._get_angle_display_lines()
                     for fit in self.fits]
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
        # Add button manager
        if len(displs) != 0:
            bm = pplt.ButtonManager(displs)
            return bm

    def display_summary(self, figsize=None):
        """
        Display a summary of the drop parameters evolution.
        """
        bdp = self._get_inters_base_fit()
        radii = self.get_base_diameter()
        thetas = self.get_contact_angles()
        # triple_pts1, triple_pts2 = self.get_triple_points()
        ts = np.arange(0, len(self.fits)*self.dt, self.dt)[0: len(bdp)]
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        # Drop dimensions
        plt.sca(axs[0])
        plt.plot(bdp[:, 0, 0], ts, label="Contact (left)")
        plt.plot(bdp[:, 1, 0], ts, label="Contact (right)")
        plt.plot(radii, ts, label="Base radius")
        # if (not np.all(np.isnan(triple_pts1)) and
        #     not np.all(np.isnan(triple_pts2))):
        #     plt.plot(triple_pts1[:, 0], ts, label="Triple point (left)")
            # plt.plot(triple_pts2[:, 0], ts, label="Triple point (right)")
        plt.xlabel('x {}'.format(self.unit_x.strUnit()))
        plt.ylabel('Time {}'.format(self.unit_times.strUnit()))
        plt.legend(loc=0)
        # Contact angles
        plt.sca(axs[1])
        plt.plot(-thetas[:, 0], ts, label="Angle (left)")
        plt.plot(180 - thetas[:, 1], ts, label="Angle (right)")
        # if not np.all(np.isnan(thetas_triple)):
        #     plt.plot(-thetas_triple[:, 0], ts,
        #              label="Angle at triple point (left)",
        #              marker=",", ls="-")
        #     plt.plot(180 - thetas_triple[:, 1], ts,
        #              label="Angle at triple point (right)",
        #              marker=",", ls="-")
        plt.ylabel('Time {}'.format(self.unit_times.strUnit()))
        plt.xlabel('[Deg]')
        plt.legend(loc=0)

class TemporalSplineFits(TemporalFits):
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
                                 nmb_max=len(self.fits),
                                 name_things='triple points',
                                 perc_interv=5)
        for fit in self.fits:
            try:
                fit.detect_triple_points(use_x_minima=use_x_minima)
            except Exception:
                pass
            if verbose:
                pg.print_progress()
        if smooth is not None:
            self.smooth_triple_points(tos='gaussian', size=smooth)

    def display(self, displ_tp=True, displ_ca=True):
        super().display(displ_tp=displ_tp, displ_ca=displ_ca)
        displs = []
        # Display spline fitting
        x1s = []
        y1s = []
        x2s = []
        y2s = []
        for fit in self.fits:
            xy_inter = fit._get_inters_base_fit()
            y1 = np.linspace(xy_inter[0][1],
                             fit.y_bounds[1],
                             1000)
            y2 = np.linspace(xy_inter[1][1],
                             fit.y_bounds[1],
                             1000)
            x1 = fit.fits[0](y1)
            x2 = fit.fits[1](y2)
            if np.any(np.isnan(x2)):
                filt = np.logical_not(np.isnan(x2))
                x2 = x2[filt]
                y2 = y2[filt]
            if np.any(np.isnan(x1)):
                filt = np.logical_not(np.isnan(x1))
                x1 = x1[filt]
                y1 = y1[filt]
            x1s.append(x1)
            x2s.append(x2)
            y1s.append(y1)
            y2s.append(y2)
        x1s = np.asarray(x1s)
        x2s = np.asarray(x2s)
        y1s = np.asarray(y1s)
        y2s = np.asarray(y2s)
        if x1s.shape[1] != 0:
            db1 = pplt.Displayer(x1s, y1s, color=self[0].colors[1])
            displs.append(db1)
        if x2s.shape[1] != 0:
            db2 = pplt.Displayer(x2s, y2s, color=self[0].colors[1])
            displs.append(db2)
        # Add button manager
        if len(displs) != 0:
            bm = pplt.ButtonManager(displs)
            return bm


class TemporalCircleFits(TemporalFits):
    def display(self, displ_tp=True, displ_ca=True):
        super().display(displ_tp=displ_tp, displ_ca=displ_ca)
        displs = []
        # Display circle
        thetas = np.linspace(0, np.pi*2, 100)
        xs = []
        ys = []
        xps = []
        yps = []
        # Loop on times
        for fit in self.fits:
            if fit.fits is None:
                for l in [xs, ys]:
                    l.append(np.nan*thetas)
                for l in [xps, yps]:
                    l.append([np.nan])
                continue
            # plot the circle !
            (xc, yc), R = fit.fits
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
        # Add button manager
        if len(displs) != 0:
            bm = pplt.ButtonManager(displs)
            return bm


class TemporalEllipseFits(TemporalFits):
    def display(self, displ_tp=True, displ_ca=True):
        super().display(displ_tp=displ_tp, displ_ca=displ_ca)
        displs = []
        # Display ellipse fits
        res = 100
        xs = []
        ys = []
        xps = []
        yps = []
        # Loop on times
        for fit in self.fits:
            if fit.fits is None:
                for l in [xs, ys]:
                    l.append([np.nan]*res)
                for l in [xps, yps]:
                    l.append([np.nan])
                continue
            # plot the circle !
            (xc, yc), R1, R2, theta = fit.fits
            elxs, elys = hlp.get_ellipse_points(xc, yc, R1, R2, theta)
            xs.append(elxs)
            ys.append(elys)
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
        # Add button manager
        if len(displs) != 0:
            bm = pplt.ButtonManager(displs)
            return bm


class TemporalCirclesFits(TemporalFits):
    def display(self, displ_tp=True, displ_ca=True):
        super().display(displ_tp=displ_tp, displ_ca=displ_ca)
        displs = []
        # Display circles
        thetas = np.linspace(0, np.pi*2, 100)
        # loop on different circles
        for i in range(3):
            xs = []
            ys = []
            xps = []
            yps = []
            # Loop on times
            for fit in self.fits:
                if fit.fits is None:
                    for l in [xs, ys]:
                        l.append(np.nan*thetas)
                    for l in [xps, yps]:
                        l.append([np.nan])
                    continue
                if i >= len(fit.fits):
                    for l in [xs, ys]:
                        l.append(np.nan*thetas)
                    for l in [xps, yps]:
                        l.append([np.nan])
                    continue
                if fit.fits[i] is None:
                    for l in [xs, ys]:
                        l.append(np.nan*thetas)
                    for l in [xps, yps]:
                        l.append([np.nan])
                    continue
                # plot the circle !
                (xc, yc), R = fit.fits[i]
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
        # Add button manager
        if len(displs) != 0:
            bm = pplt.ButtonManager(displs)
            return bm
