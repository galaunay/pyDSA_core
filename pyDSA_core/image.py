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

import cv2
import os
import json
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import scipy.ndimage as spim
import warnings
from IMTreatment import ScalarField, Profile
from IMTreatment.utils import make_unit
import IMTreatment.plotlib as pplt
from .dropedges import DropEdges
from .baseline import Baseline


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class Image(ScalarField):
    def __init__(self, filepath=None, cache_infos=True):
        """
        Class representing a greyscale image.
        """
        super().__init__()
        self.baseline = None
        self.colors = pplt.get_color_cycles()
        if filepath is not None:
            self.filepath = os.path.abspath(filepath)
            self.infofile_path = os.path.splitext(self.filepath)[0] + ".info"
            self.cache_infos = cache_infos
        else:
            self.filepath = None
            self.infofile_path = None
            self.cache_infos = False

    def __eq__(self, other):
        if not isinstance(other, Image):
            return False
        if not super().__eq__(other):
            return False
        if not self.baseline == other.baseline:
            return False
        return True

    def display(self, *args, **kwargs):
        displ_baseline = True
        if 'displ_baseline' in kwargs.keys():
            displ_baseline = kwargs.pop('displ_baseline')
        super().display(*args, **kwargs)
        if self.baseline is not None and displ_baseline:
            bx, by = self.baseline.xy
            plt.plot(bx, by, color=self.colors[0], ls='none', marker='o')
            plt.plot(bx, by, color=self.colors[0], ls='-')

    def set_baseline(self, pt1, pt2):
        """
        Set the baseline.

        Parameters
        ==========
        pt1, pt2 : 2x1 arrays of numbers
            Points defining the baseline.
        """
        self.baseline = Baseline([pt1, pt2],
                                 xmax=self.axe_x[-1],
                                 xmin=self.axe_x[0])
        if self.cache_infos:
            self._dump_infos()

    def choose_baseline(self):
        """
        Choose baseline position interactively.

        Select as many points as you want (click on points to delete them),
        and close the figure when done.

        You can adjust the position of the point closer to your mouse with
        arrows, ctrl+arrows and shift+arrows.
        """
        pos = []
        eps = .01*(self.axe_x[-1] - self.axe_x[0])
        dx, dy = self.dx, self.dy
        # get cursor position on click
        fig = plt.figure()
        pts = plt.plot([], marker="o", ls="none", mfc=self.colors[0],
                       mec='k')[0]
        hl_pt = plt.plot([], marker="o", ls="none", mec=self.colors[0],
                         mfc='none',
                         ms=10)[0]
        baseline = plt.plot([], ls="-", color=self.colors[0])[0]
        bs = Baseline()

        def onclick(event):
            # toolbar want the focus !
            if fig.canvas.manager.toolbar._active is not None:
                return None
            # get the position
            xy = [event.xdata, event.ydata]
            diffs = [(xy[0] - xyi[0])**2 + (xy[1] - xyi[1])**2
                     for xyi in pos]
            # check if close to an existing point
            closes = diffs < eps**2
            if np.any(closes):
                for i in np.arange(len(pos) - 1, -1, -1):
                    if closes[i]:
                        del pos[i]
            else:
                pos.append(xy)
            if len(pos) == 0:
                hl_pt.set_data([np.nan, np.nan])
            else:
                hl_pt.set_data(np.array(pos[-1:]).transpose())
            redraw_if_necessary()

        def on_mouse_move(event):
            # toolbar want the focus !
            if fig.canvas.manager.toolbar._active is not None:
                return None
            ind = get_closest_point(event.xdata, event.ydata)
            if ind is not None:
                hl_pt.set_data(np.array(pos[ind:ind+1]).transpose())
                fig.canvas.draw()

        def redraw_if_necessary():
            if len(pos) != 0:
                pts.set_data(np.array(pos).transpose())
                if len(pos) > 1:
                    bs.from_points(pos, xmin=self.axe_x[0],
                                   xmax=self.axe_x[-1])
                    baseline.set_data(bs.xy)
                fig.canvas.draw()

        def get_closest_point(x, y):
            # get the position
            if x is None or y is None:
                return None
            if len(pos) == 0:
                return None
            xy = [x, y]
            diffs = [(xy[0] - xyi[0])**2 + (xy[1] - xyi[1])**2
                     for xyi in pos]
            # check if close to an existing point
            ind = np.argmin(diffs)
            return ind

        def onpress(event):
            x, y = event.xdata, event.ydata
            ind = get_closest_point(x, y)
            if ind is None:
                return None
            if event.key == 'up':
                pos[ind][1] += dy
            elif event.key == 'down':
                pos[ind][1] -= dy
            elif event.key == 'left':
                pos[ind][0] -= dx
            elif event.key == 'right':
                pos[ind][0] += dx
            elif event.key == 'shift+up':
                pos[ind][1] += 10*dy
            elif event.key == 'shift+down':
                pos[ind][1] -= 10*dy
            elif event.key == 'shift+left':
                pos[ind][0] -= 10*dx
            elif event.key == 'shift+right':
                pos[ind][0] += 10*dx
            elif event.key == 'ctrl+up':
                pos[ind][1] += 0.1*dy
            elif event.key == 'ctrl+down':
                pos[ind][1] -= 0.1*dy
            elif event.key == 'ctrl+left':
                pos[ind][0] -= 0.1*dx
            elif event.key == 'ctrl+right':
                pos[ind][0] += 0.1*dx
            # Update indicator
            hl_pt.set_data(np.array(pos[ind:ind+1]).transpose())
            redraw_if_necessary()

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        fig.canvas.mpl_connect('key_press_event', onpress)
        self.display(cmap=plt.cm.binary_r)
        plt.title("Put some points on the baseline."
                  "\nYou can remove points by clicking on it."
                  "\nClose the window when you are happy with the baseline")
        plt.show(block=True)
        if len(pos) < 2:
            if self.baseline is not None:
                return self.baseline
            else:
                return None
        # use linear interpolation to get baseline
        self.baseline = Baseline(pos, xmin=self.axe_x[0],
                                 xmax=self.axe_x[-1])
        if self.cache_infos:
            self._dump_infos()
        return self.baseline

    def scale_interactive(self):
        """
        Scale the Image interactively.
        """
        sc = ScaleChooser(self)
        if self.cache_infos:
            self._dump_infos()
        return sc.ret_values

    def scale(self, scalex=None, scaley=None, scalev=None, inplace=True):
        """
        Scale the Image.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values
        """
        if inplace:
            super().scale(scalex=scalex, scaley=scaley, scalev=scalev,
                          inplace=True)
            tmpim = self
        else:
            tmpim = super().scale(scalex=scalex, scaley=scaley, scalev=scalev,
                                  inplace=False)
        if tmpim.baseline is not None:
            tmpim.baseline.scale(scalex=scalex, scaley=scaley)
        if tmpim.cache_infos:
            tmpim._dump_infos()
        return tmpim

    def set_origin(self, x, y):
        """
        Modify the axis in order to place the origin at the given point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        super().set_origin(x, y)
        if self.baseline is not None:
            bpt1 = self.baseline.pt1
            bpt2 = self.baseline.pt2
            bpt1[0] -= x
            bpt2[0] -= x
            bpt1[1] -= y
            bpt2[1] -= y
            self.baseline.from_points([bpt1, bpt2])
        if self.cache_infos:
            self._dump_infos()

    def _dump_infos(self):
        # Gather old information if necessary
        if os.path.isfile(self.infofile_path):
            with open(self.infofile_path, 'r') as f:
                dic = json.load(f)
        else:
            dic = {}
        # Update with new values
        unit_x = self.unit_x.strUnit()[1:-1]
        unit_y = self.unit_y.strUnit()[1:-1]
        if self.baseline is not None:
            pt1 = list(self.baseline.pt1)
            pt2 = list(self.baseline.pt2)
        else:
            pt1 = None
            pt2 = None
        new_dic = {"dx": self.dx,
                   "dy": self.dy,
                   "x0": self.axe_x[0],
                   "y0": self.axe_y[0],
                   "baseline_pt1": pt1,
                   "baseline_pt2": pt2,
                   "unit_x": unit_x,
                   "unit_y": unit_y}
        dic.update(new_dic)
        # Write back infos
        with open(self.infofile_path, 'w+') as f:
            json.dump(dic, f)

    def _import_infos(self):
        if not os.path.isfile(self.infofile_path):
            return None
        try:
            with open(self.infofile_path, 'r') as f:
                dic = json.load(f)
        except:
            warnings.warn('Corrupted infofile, reinitializing...')
            os.remove(self.infofile_path)
            return None
        # Update with infos
        dx = dic['dx']
        dy = dic['dy']
        self.scale(scalex=dx/(self.axe_x[1] - self.axe_x[0]),
                   scaley=dy/(self.axe_y[1] - self.axe_y[0]),
                   inplace=True)
        try:  # Compatibility
            x0 = dic['x0']
            y0 = dic['y0']
            self.axe_x += (x0 - self.axe_x[0])
            self.axe_y += (y0 - self.axe_y[0])
        except KeyError:
            pass
        self.unit_x = dic['unit_x']
        self.unit_y = dic['unit_y']
        base1 = dic['baseline_pt1']
        base2 = dic['baseline_pt2']
        if base1 is not None and base2 is not None:
            self.set_baseline(base1, base2)

    def choose_tp_interactive(self):
        """
        Choose triple points interactively.
        """
        sc = TriplePointChooser(self)
        return sc.rh

    def take_measure(self, kind='all'):
        """
        Allow to interactively take a measure on the image.

        Parameters
        ----------
        kind: string
           Kind of measurement to make.
           Can be 'dx', 'dy', 'dl' ot 'all'.
        """
        mt = MeasuringTool(self, kind=kind)
        return mt.measures

    def edge_detection_canny(self, threshold1=None, threshold2=None,
                             base_max_dist=15, size_ratio=.5,
                             nmb_edges=2, ignored_pixels=2,
                             keep_exterior=True,
                             remove_included=True,
                             smooth_size=None,
                             dilatation_steps=1,
                             verbose=False, debug=False):
        """
        Perform edge detection using canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        base_max_dist: integers
            Maximal distance (in pixel) between the baseline and
            the beginning of the drop edge (default to 15).
        size_ratio: number
            Minimum size of edges, regarding the bigger detected one
            (default to 0.5).
        nmb_edges: integer
            Number of maximum expected edges (default to 2).
        ignored_pixels: integer
            Number of pixels ignored around the baseline
            (default to 2).
            Putting a small value to this allow to avoid
            small surface defects to be taken into account.
        keep_exterior: boolean
            If True (default), only keep the exterior edges.
        remove_included: boolean
            If True (default), remove edges included in other edges
        smooth_size: number
            If specified, the image is smoothed before
            performing the edge detection.
            (can be useful to put this to 1 to get rid of compression
             artefacts on images).
        dilatation_steps: positive integer
            Number of dilatation/erosion steps.
            Increase this if the drop edges are discontinuous.
        """
        # check for baseline
        if self.baseline is None:
            raise Exception('You should set the baseline first.')
        if not np.isclose(self.dx, self.dy):
            warnings.warn('dx is different than dy, results can be weird...')
        # Get adapted thresholds
        if threshold2 is None or threshold1 is None:
            # hist = self.get_histogram(cum=True,
            #                           bins=int((self.max - self.min)/10))
            mini = 0
            maxi = 255
            hist = cv2.calcHist([self.values], [0], None,
                                [maxi - mini], [mini, maxi])
            hist = np.cumsum(hist[:, 0])
            hist = Profile(np.arange(mini, maxi), hist)
            threshold1 = hist.get_value_position(
                hist.min + (hist.max - hist.min)/2)[0]
            threshold2 = np.max(hist.x)*.5
        # Remove useless part of the image
        tmp_im = self.crop(intervy=[np.min([self.baseline.pt2[1],
                                            self.baseline.pt1[1]]), np.inf],
                           inplace=False)
        # Smooth if asked
        if smooth_size is not None:
            if smooth_size != 0:
                tmp_im.smooth(tos='gaussian', size=smooth_size, inplace=True)
        if verbose:
            plt.figure()
            tmp_im.display()
            plt.title('Initial image')
        #======================================================================
        # Perform Canny detection
        #======================================================================
        im = np.array(tmp_im.values, dtype=np.uint8)
        im_edges = cv2.Canny(image=im, threshold1=threshold1,
                             threshold2=threshold2)
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  im_edges, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Canny edge detection \nwith th1={} and th2={}'
                      .format(threshold1, threshold2))
        #======================================================================
        # Remove points behind the baseline (and too close to)
        #======================================================================
        fun = self.baseline.get_baseline_fun()
        ign_dy = ignored_pixels*self.dy
        max_y = np.max([self.baseline.pt2[1] + ign_dy,
                        self.baseline.pt1[1] + ign_dy])
        for j in range(im_edges.shape[1]):
            y = tmp_im.axe_y[j]
            if y > max_y:
                continue
            for i in range(im_edges.shape[0]):
                x = tmp_im.axe_x[i]
                if y < fun(x) + ign_dy:
                    im_edges[i, j] = 0
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  im_edges, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Removed points under baseline')
        #======================================================================
        # Dilatation / erosion to ensure line continuity
        #======================================================================
        if dilatation_steps > 0:
            im_edges = cv2.dilate(im_edges,
                                  iterations=dilatation_steps,
                                  kernel=np.array([[1, 1, 1],
                                                   [1, 1, 1],
                                                   [1, 1, 1]]))
        # im_edges = spim.binary_erosion(im_edges, iterations=dilatation_steps,
        #                                structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  im_edges, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Dilatation / erosion step')
        #======================================================================
        # Keep only the bigger edges
        #======================================================================
        nmb, labels = cv2.connectedComponents(im_edges)
        # safeguard
        if nmb > 1000:
            raise Exception("Too many edges detected, you may want to use the "
                            "'smooth' parameter")
        # labels, nmb = spim.label(im_edges, np.ones((3, 3)))
        nmb_edge = nmb
        if debug:
            print(f"    Initial number of edges: {nmb_edge}")
        dy = self.axe_y[1] - self.axe_y[0]
        if nmb_edge > 1:
            #==================================================================
            # Remove small patches
            #==================================================================
            sizes = [np.sum(labels == label)
                     for label in np.arange(1, nmb + 1)]
            crit_size = np.sort(sizes)[-1]*size_ratio
            for i, size in enumerate(sizes):
                if size < crit_size:
                    im_edges[labels == i+1] = 0
                    labels[labels == i+1] = 0
                    nmb_edge -= 1
                    if debug:
                        print(f"Label {i+1} removed because too small")
                        print(f"    Remaining edges: {nmb_edge}")
            if verbose:
                plt.figure()
                im = Image()
                im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                      im_edges, mask=tmp_im.mask,
                                      unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
                im.display()
                plt.title('Removed small edges')
            #==================================================================
            # Remove lines not touching the baseline
            #==================================================================
            if nmb_edge > nmb_edges:
                for i in range(np.max(labels)):
                    # Untested ! (next line)
                    ys = tmp_im.axe_y[np.sum(labels == i+1, axis=0) > 0]
                    if len(ys) == 0:
                        continue
                    min_y = np.min(ys)
                    mdist = base_max_dist*dy
                    if min_y > mdist + np.max([self.baseline.pt1[1],
                                               self.baseline.pt2[1]]):
                        if debug:
                            print(f"Label {i+1} removed because not touching "
                                  "baseline")
                            print(f"    Remaining edges: {nmb_edge}")
                        im_edges[labels == i+1] = 0
                        labels[labels == i+1] = 0
                        nmb_edge -= 1
                if verbose:
                    plt.figure()
                    im = Image()
                    im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                          im_edges, mask=tmp_im.mask,
                                          unit_x=tmp_im.unit_x,
                                          unit_y=tmp_im.unit_y)
                    im.display()
                    plt.title('Removed edge not touching the baseline')
            #==============================================================================
            # Remove edges that are included in another edge
            #==============================================================================
            if nmb_edge > 1 and remove_included:
                maxs = []
                mins = []
                # Get upper and lower bounds for edges
                for i in np.arange(1, nmb + 1):
                    xs = tmp_im.axe_x[np.sum(labels == i, axis=1) > 0]
                    if len(xs) == 0:
                        mins.append(None)
                        maxs.append(None)
                    else:
                        mins.append(np.min(xs))
                        maxs.append(np.max(xs))
                # Remove edge if included in another one
                for i in range(nmb):
                    if mins[i] is None:
                        continue
                    for j in range(nmb):
                        if mins[j] is None:
                            continue
                        if mins[i] < mins[j] and maxs[j] < maxs[i]:
                            if debug:
                                print(f"Label {j+1} removed because"
                                      " included in another one")
                                print(f"    Remaining edges: {nmb_edge}")
                            im_edges[labels == j+1] = 0
                            labels[labels == j+1] = 0
                            nmb_edge -= 1
                if verbose:
                    plt.figure()
                    im = Image()
                    im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                          im_edges, mask=tmp_im.mask,
                                          unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
                    im.display()
                    plt.title('Removed edge included in another edge')
            #==================================================================
            # Keep only the exterior edges
            #==================================================================
            if nmb_edge > 2 and keep_exterior:
                mean_xs = []
                for i in np.arange(1, nmb + 1):
                    xs = tmp_im.axe_x[np.sum(labels == i, axis=1) > 0]
                    if len(xs) == 0:
                        mean_xs.append(np.nan)
                    else:
                        mean_xs.append(np.mean(xs))
                mean_xs = np.asarray(mean_xs)
                # mean_xs[np.isnan(mean_xs)] = np.mean(mean_xs[~np.isnan(mean_xs)])
                mean_xs_indsort = np.argsort(mean_xs)
                for i in mean_xs_indsort[1:-1]:
                    if np.isnan(mean_xs[i+1]):
                        break
                    if debug:
                        print(f"Label {i+1} removed because not exterior")
                        print(f"    Remaining edges: {nmb_edge}")
                    im_edges[labels == i+1] = 0
                    labels[labels == i+1] = 0
                    nmb_edge -= 1
                if verbose:
                    plt.figure()
                    im = Image()
                    im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                          im_edges, mask=tmp_im.mask,
                                          unit_x=tmp_im.unit_x,
                                          unit_y=tmp_im.unit_y)
                    im.display()
                    plt.title('Removed the interior edges')
            #==============================================================================
            # Let only the maximum allowed number of edges
            #==============================================================================
            if nmb_edge > nmb_edges and keep_exterior:
                sizes = [np.sum(labels == label)
                         for label in np.arange(1, nmb + 1)]
                indsort = np.argsort(sizes)
                for ind in indsort[:-nmb_edges]:
                    im_edges[ind + 1 == labels] = 0
                nmb_edge = nmb_edges
                if verbose:
                    plt.figure()
                    im = Image()
                    im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                          im_edges, mask=tmp_im.mask,
                                          unit_x=tmp_im.unit_x,
                                          unit_y=tmp_im.unit_y)
                    im.display()
                    plt.title('Removed small edges because too numerous')


        #======================================================================
        # Check and Return
        #======================================================================
        # Get points coordinates
        xs, ys = np.where(im_edges)
        axx = tmp_im.axe_x
        axy = tmp_im.axe_y
        xs = [axx[x] for x in xs]
        ys = [axy[y] for y in ys]
        xys = list(zip(xs, ys))
        # Check if there is something remaining
        if len(xys) < 10:
            if verbose:
                plt.show(block=False)
            raise Exception("Didn't found any drop here...")
        if verbose:
            plt.figure()
            self.display()
            xys = np.array(xys)
            plt.plot(xys[:, 0], xys[:, 1], ".k")
            plt.title('Initial image + edge points')
        edge = DropEdges(xy=xys, im=self, type="canny")
        return edge

    def edge_detection_contour(self, nmb_edges=2, ignored_pixels=2,
                               level=0.5, size_ratio=.5,
                               verbose=False):
        """
        Perform edge detection using contour.

        Parameters
        ==========
        nmb_edges: integer
            Number of maximum expected edges (default to 2).
        size_ratio: number
            Minimum size of edges, regarding the bigger detected one
            (default to 0.5).
        level: number
            Normalized level of the drop contour.
            Should be between 0 (black) and 1 (white).
            Default to 0.5.
        ignored_pixels: integer
            Number of pixels ignored around the baseline
            (default to 2).
            Putting a small value to this allow to avoid
            small surface defects to be taken into account.
        """
        if self.baseline is None:
            raise Exception('You should set the baseline first.')
        if self.dx != self.dy:
            warnings.warn('dx is different than dy, results can be weird...')
        # Get level
        level = self.min + level*(self.max - self.min)
        # Get contour
        import skimage.measure as skim
        contour = skim.find_contours(self.values, level)
        # put in the right units
        for cont in contour:
            cont[:, 0] *= self.dx
            cont[:, 0] += self.axe_x[0]
            cont[:, 1] *= self.dy
            cont[:, 1] += self.axe_y[0]
        # Remove contours below the baseline (or too close to)
        fun = self.baseline.get_baseline_fun()
        ign_dy = ignored_pixels*self.dy
        max_y = np.max([self.baseline.pt2[1] + ign_dy,
                        self.baseline.pt1[1] + ign_dy])
        new_contour = []
        for cont in contour:
            new_cont = []
            for xy in cont:
                if xy[1] > max_y + ign_dy:
                    new_cont.append(xy)
                elif xy[1] > fun(xy[0]) + ign_dy:
                    new_cont.append(xy)
            if len(new_cont) != 0:
                new_contour.append(new_cont)
        contour = new_contour
        # Keep only the bigger edges
        contour = sorted(contour, key=lambda x: -len(x))
        contour = contour[0:nmb_edges]
        threshold = len(contour[0])*size_ratio
        contour = [cont for cont in contour if len(cont) > threshold]
        # concatenate
        xys = np.concatenate(contour)
        # Check if there is something remaining
        if len(xys) < 10:
            raise Exception("Didn't found any drop here...")
        # return
        edge = DropEdges(xy=xys, im=self, type="contour")
        return edge

    edge_detection = edge_detection_canny


class ScaleChooser(object):
    def __init__(self, im):
        """

        """
        self.fig = plt.figure()
        self.pos = []
        self.im = im
        self.eps = .01*(self.im.axe_x[-1] - self.im.axe_x[0])
        self.pts = plt.plot([], marker="o", ls="none", mec='w', mfc='k')[0]
        self.axplot = plt.gca()
        self.ret_values = [None, None]
        self.real_dist = None
        self.real_unit = None
        # Display
        self.im.display(cmap=plt.cm.binary_r)
        plt.title("Scaling step:\n"
                  "Choose two points separated by a known distance.")
        # Connect click event on graph
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # Add ok button
        button_kwargs = {"color": "w", "hovercolor": [.5]*3}
        self.axok = mpl.axes.Axes(self.fig, [0.88, 0.02, 0.1, 0.05])
        self.bnok = Button(self.axok, 'Done', **button_kwargs)
        self.bnok.on_clicked(self.done)        # set up the apsect
        self.fig.add_axes(self.axok)
        # Add text
        self.axtxt = mpl.axes.Axes(self.fig, [0.68, 0.02, 0.1, 0.05])
        self.btntxt = TextBox(self.axtxt,
                              "Distance ('0.7mm' for example):", "")
        self.btntxt.on_submit(self.on_submit)
        self.fig.add_axes(self.axtxt)
        # show the plot
        plt.show(block=True)

    def on_submit(self, txt):
        match = re.match(r'\s*([0-9.]+)\s*(.*)\s*', txt)
        self.real_dist = float(match.groups()[0])
        self.real_unit = match.groups()[1]

    def onclick(self, event):
        # toolbar want the focus !
        if self.fig.canvas.manager.toolbar._active is not None:
            return None
        # Do nothing if not above an axe
        if event.inaxes != self.axplot:
            return None
        # get the position
        xy = [event.xdata, event.ydata]
        diffs = [(xy[0] - xyi[0])**2 + (xy[1] - xyi[1])**2
                 for xyi in self.pos]
        # check if close to an existing point
        closes = diffs < self.eps**2
        if np.any(closes):
            for i in np.arange(len(self.pos) - 1, -1, -1):
                if closes[i]:
                    del self.pos[i]
        elif len(self.pos) >= 2:
            pass
        else:
            self.pos.append(xy)
        # redraw
        if len(self.pos) != 0:
            self.pts.set_data(np.array(self.pos).transpose())
        else:
            self.pts.set_data(np.empty((2, 2)))
        self.fig.canvas.draw()

    def done(self, event):
        if len(self.pos) != 2:
            return None, None
        actual_width = ((self.pos[0][0] - self.pos[1][0])**2 +
                        (self.pos[0][1] - self.pos[1][1])**2)**.5
        # Getting wanted length and unity
        wanted_width = self.real_dist
        wanted_unity = make_unit(self.real_unit)
        # Compute and set the scale
        scale = wanted_width/actual_width
        self.im.scale(scalex=scale, scaley=scale, inplace=True)
        self.im.unit_x = wanted_unity
        self.im.unit_y = wanted_unity
        self.ret_values = scale, wanted_unity
        plt.close(self.fig)


class TriplePointChooser(object):
    def __init__(self, im):
        """
        """
        self.fig = plt.figure()
        if im.baseline is None:
            raise Exception()
        self.basefun = im.baseline.get_baseline_fun()
        self.pos = []
        self.basepos = []
        self.rh = []
        self.im = im
        self.eps = .01*(self.im.axe_x[-1] - self.im.axe_x[0])
        self.pts = plt.plot([], marker="o", ls=":", color='k',
                            mec='w', mfc='k')[0]
        self.texts = []
        self.axplot = plt.gca()
        # Display
        self.im.display(cmap=plt.cm.binary_r)
        plt.title("Click on the triple point positions.")
        # Connect click event on graph
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # show the plot
        plt.show(block=True)

    def onclick(self, event):
        # toolbar want the focus !
        if self.fig.canvas.manager.toolbar._active is not None:
            return None
        # Do nothing if not above an axe
        if event.inaxes != self.axplot:
            return None
        # get the position
        xy = [event.xdata, event.ydata]
        diffs = [(xy[0] - xyi[0])**2 + (xy[1] - xyi[1])**2
                 for xyi in self.pos]
        # check if close to an existing point
        closes = diffs < self.eps**2
        if np.any(closes):
            for i in np.arange(len(self.pos) - 1, -1, -1):
                if closes[i]:
                    del self.pos[i]
                    del self.rh[i]
                    del self.basepos[i]
                    self.texts[i].remove()
                    del self.texts[i]
        else:
            self.pos.append(xy)
            self.basepos.append(self.im.baseline.get_projection_to_baseline(xy))
            self.rh.append(self.im.baseline.get_distance_to_baseline(xy))
            if self.im.unit_y.strUnit() != "[]":
                unit = self.im.unit_y.strUnit()[1:-1]
            else:
                unit = ""
            self.texts.append(plt.text(self.pos[-1][0],
                                       self.pos[-1][1] + self.eps,
                                       f"h={self.rh[-1]:.2f}{unit}",
                                       horizontalalignment='center',
                                       verticalalignment='baseline'))
        # redraw
        if len(self.pos) != 0:
            pos = np.concatenate([[self.pos[i],
                                   self.basepos[i],
                                   [np.nan, np.nan]]
                                  for i in range(len(self.pos))])
            self.pts.set_data(np.array(pos).transpose())
        else:
            self.pts.set_data(np.empty((2, 2)))
        self.fig.canvas.draw()


class MeasuringTool(object):
    def __init__(self, im, kind='all'):
        """
        """
        self.im = im
        self.kind = kind
        #
        self.fig = plt.figure()
        self.im.display(cmap=plt.cm.binary_r)
        self.axplot = plt.gca()
        #
        self.pts_couples = [[]]
        self.measures = {'dx': [], 'dy': [], 'dl': []}
        self.plots = []
        self.texts = []
        self.plot_args = {'marker': 'o', 'ls': '-'}
        #
        self.eps = .01*(self.im.axe_x[-1] - self.im.axe_x[0])
        # Connect click event on graph
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # show the plot
        plt.show(block=True)

    def onclick(self, event):
        # toolbar want the focus !
        if self.fig.canvas.manager.toolbar._active is not None:
            return None
        # Do nothing if not above an axe
        if event.inaxes != self.axplot:
            return None
        # get the position
        xy = [event.xdata, event.ydata]
        # if fist point of a couple
        if len(self.pts_couples[-1]) == 0:
            self.pts_couples[-1].append(xy)
            plot = plt.plot([xy[0]], [xy[1]], **self.plot_args)[0]
            self.plots.append(plot)
        # if second point of a couple
        else:
            self.pts_couples[-1].append(xy)
            xs = [self.pts_couples[-1][0][0], xy[0]]
            ys = [self.pts_couples[-1][0][1], xy[1]]
            self.plots[-1].set_data([xs, ys])
            self.pts_couples.append([])
            # add text
            if self.im.unit_x != self.im.unit_y:
                unit = ""
            else:
                unit = self.im.unit_x.strUnit()
            text = ""
            print(f"Points {len(self.pts_couples)-1}:")
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            dl = (dx**2 + dy**2)**.5
            if self.kind in ['dx', 'all']:
                print(f"dx={dx} {unit}")
                text += f'dx={dx:.2f} {unit}\n'
            if self.kind in ['dy', 'all']:
                print(f"dy={dy} {unit}")
                text += f'dy={dy:.2f} {unit}\n'
            if self.kind in ['dl', 'all']:
                print(f"dl={dl} {unit}")
                text += f'dl={dl:.2f} {unit}\n'
            text = text.rstrip('\n')
            self.texts.append(plt.text(np.mean(xs),
                                       np.mean(ys),
                                       text,
                                       horizontalalignment='center',
                                       verticalalignment='baseline',
                                       bbox=dict(facecolor='white', alpha=0.5)))
            # record
            self.measures['dx'].append(dx)
            self.measures['dy'].append(dy)
            self.measures['dl'].append(dl)
        self.fig.canvas.draw()
