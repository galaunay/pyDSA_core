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
from IMTreatment.utils import ProgressCounter
from IMTreatment import TemporalPoints, plotlib as pplt
from .temporalfits import TemporalCircleFits, TemporalSplineFits, \
    TemporalEllipseFits, TemporalCirclesFits, TemporalEllipsesFits
from . import dropedges, dropfit


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
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
        return self.fit_spline(k=k, s=s, verbose=verbose)

    def fit_spline(self, k=5, s=0.75, iteration_hook=None, verbose=False):
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
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting droplet interfaces",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges', perc_interv=5)
        fits = []
        for i, edge in enumerate(self.point_sets):
            try:
                fits.append(edge.fit_spline(k=k, s=s, verbose=False))
            except Exception:
                fits.append(dropfit.DropFit(baseline=edge.baseline,
                                            x_bounds=edge.x_bounds,
                                            y_bounds=edge.y_bounds))
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.point_sets))
        # return
        tf = TemporalSplineFits(fits=fits, temporaledges=self)
        return tf

    def fit_polyline(self, deg=5, iteration_hook=None, verbose=False):
        """
        Compute a polyline fitting for the droplets shape.

        Parameters
        ----------
        deg : integer
            Degree of the polynomial fitting.
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting droplet interfaces",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges', perc_interv=5)
        fits = []
        for i, edge in enumerate(self.point_sets):
            try:
                fits.append(edge.fit_polyline(deg=deg, verbose=False))
            except Exception:
                fits.append(dropfit.DropFit(baseline=edge.baseline,
                                            x_bounds=edge.x_bounds,
                                            y_bounds=edge.y_bounds))
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.point_sets))
        # return
        tf = TemporalSplineFits(fits=fits, temporaledges=self)
        return tf

    def fit_circle(self, triple_pts=None, iteration_hook=None, verbose=False):
        """
        Fit a circle to the edges.

        Ignore the lower part of the drop if triple points are presents.

        Parameters
        ----------
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting circles to edges",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges',
                                 perc_interv=5)
        fits = []
        for i, edge in enumerate(self.point_sets):
            try:
                fits.append(edge.fit_circle(triple_pts=triple_pts))
            except Exception:
                fits.append(dropfit.DropFit(baseline=edge.baseline,
                                            x_bounds=edge.x_bounds,
                                            y_bounds=edge.y_bounds))
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.point_sets))
        # return
        tf = TemporalCircleFits(fits=fits, temporaledges=self)
        return tf

    def fit_ellipse(self, triple_pts=None, iteration_hook=None, verbose=False):
        """
        Fit an ellipse to the edges.

        Ignore the lower part of the drop if triple points are presents.

        Parameters
        ----------
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting ellipses to edges",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges',
                                 perc_interv=5)
        fits = []
        for i, edge in enumerate(self.point_sets):
            try:
                fits.append(edge.fit_ellipse(triple_pts=triple_pts))
            except Exception:
                fits.append(dropfit.DropFit(baseline=edge.baseline,
                                            x_bounds=edge.x_bounds,
                                            y_bounds=edge.y_bounds))
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.point_sets))
        # return
        tf = TemporalEllipseFits(fits=fits, temporaledges=self)
        return tf

    def fit_ellipses(self, triple_pts=None, iteration_hook=None, verbose=False):
        """
        Fit two ellipses (one on each side) to the edges.

        Ignore the lower part of the drop if triple points are presents.

        Parameters
        ----------
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting ellipses to edges",
                                 nmb_max=len(self.point_sets),
                                 name_things='edges',
                                 perc_interv=5)
        fits = []
        for i, edge in enumerate(self.point_sets):
            try:
                fits.append(edge.fit_ellipses(triple_pts=triple_pts))
            except Exception:
                fits.append(dropfit.DropFit(baseline=edge.baseline,
                                            x_bounds=edge.x_bounds,
                                            y_bounds=edge.y_bounds))
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.point_sets))
        # return
        tf = TemporalEllipsesFits(fits=fits, temporaledges=self)
        return tf

    def fit_circles(self, triple_pts, sigma_max=None, verbose=False,
                    nmb_pass=1, soft_constr=False,
                    iteration_hook=None):
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
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        if verbose:
            pg = ProgressCounter(init_mess="Fitting circles to edges",
                                 nmb_max=len(self.point_sets)*nmb_pass,
                                 name_things='edges',
                                 perc_interv=5)
        # check
        if len(triple_pts) != len(self.point_sets):
            raise Exception('Not the right ner of triple points')
        #  passes
        tf = None
        for i in range(nmb_pass):
            if tf is not None:
                tf.smooth_triple_points('gaussian', size=10)
                triple_pts = tf.get_triple_points()
            fits = []
            for tp, edge in zip(triple_pts, self.point_sets):
                try:
                    fits.append(edge.fit_circles(triple_pts=tp,
                                                 sigma_max=sigma_max,
                                                 soft_constr=soft_constr))
                except Exception:
                    fits.append(dropfit.DropFit(baseline=edge.baseline,
                                                x_bounds=edge.x_bounds,
                                                y_bounds=edge.y_bounds))
                if verbose:
                    pg.print_progress()
                if iteration_hook is not None:
                    iteration_hook(i, len(self.point_sets)*nmb_pass)
            tf = TemporalCirclesFits(fits=fits, temporaledges=self)
        # return
        return tf

    def display(self, *args, **kwargs):
        #
        length = len(self.point_sets)
        displs = []

        # Display points
        for edge in self.point_sets:
            if edge.drop_edges is None:
                edge._separate_drop_edges()
        if self[0].drop_edges is not None:
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
                    y1s.append(edge.drop_edges[1].y)
                    x2s.append(edge.drop_edges[2].y)
                    y2s.append(edge.drop_edges[3].y)
            if len(x1s) != length or len(x2s) != length:
                raise Exception()
            db1 = pplt.Displayer(x1s, y1s, color='k', marker="o")
            db2 = pplt.Displayer(x2s, y2s, color='k', marker="o")
            displs.append(db1)
            displs.append(db2)
        # Add button manager
        if len(displs) != 0:
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
