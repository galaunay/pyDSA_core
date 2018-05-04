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

import cv2
import re
import unum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button, TextBox
import scipy.ndimage as spim
import skimage.measure as skim
import warnings
from IMTreatment import ScalarField, Points
from IMTreatment.utils import make_unit
import IMTreatment.plotlib as pplt
from .dropedges import DropEdges
from .baseline import Baseline


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class Image(ScalarField):
    def __init__(self, filepath=None):
        """
        Class representing a greyscale image.
        """
        super().__init__()
        self.baseline = None
        self.colors = pplt.get_color_cycles()
        self.filepath = filepath

    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        if self.baseline is not None:
            bx, by = self.baseline.xy
            plt.plot(bx, by, color='g', ls='none', marker='o')
            plt.plot(bx, by, color='g', ls='-')

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

    def choose_baseline(self):
        """
        Choose baseline position interactively.

        Select as many points as you want (click on points to delete them),
        and close the figure when done.
        """
        pos = []
        eps = .01*(self.axe_x[-1] - self.axe_x[0])
        # get cursor position on click
        fig = plt.figure()
        pts = plt.plot([], marker="o", ls="none", mec='w', mfc='k')[0]
        baseline = plt.plot([], ls="-", color="k")[0]
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
            # redraw if necessary
            if len(pos) != 0:
                pts.set_data(np.array(pos).transpose())
                if len(pos) > 1:
                    bs.from_points(pos)
                    baseline.set_data(bs.xy)
                fig.canvas.draw()
        fig.canvas.mpl_connect('button_press_event', onclick)
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
        return self.baseline

    def scale_interactive(self):
        """
        Scale the Image interactively.
        """
        sc = ScaleChooser(self)
        return sc.ret_values

    def scale(self, scalex=None, scaley=None, scalev=None, inplace=True):
        """
        Scale the Image.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values
        """
        super().scale(scalex=scalex, scaley=scaley, scalev=scalev,
                      inplace=inplace)
        if self.baseline is not None:
            self.baseline.scale(scalex=scalex, scaley=scaley)

    def binarize(self, method='adaptative', threshold=None, inplace=False):
        """
        Binarize the image.

        Parameters
        ==========
        method: string in ['adaptative', 'simple', 'otsu']
            Method used to binarize (see opencv documention)
        threshold: number
            Threshold for the 'simple' method
        inplace: boolean
            To binarize in place or not.

        Returns:
        ========
        new_img: Image object
            Binarized image
        """
        if method == 'simple':
            pass
        elif method == 'adaptative':
            pass
        elif method == 'otsu':
            pass
        else:
            raise ValueError

    def _binarize_simple(self, threshold, inplace=False):
        """
        Binarize the image using a threshold.

        Parameters
        ==========
        threshold: number
            Threshold.
        inplace: boolean
            To binarize in place or not.

        Returns:
        ========
        new_img: Image object
            Binarized image
        """
        if inplace:
            tmp_im = self
        else:
            tmp_im = self.copy()
        ret_val, new_img = cv2.threshold(tmp_im.data,
                                         threshold,
                                         255,
                                         cv2.THRESH_BINARY)
        tmp_im.data = new_img
        tmp_im.used_threshold = int(ret_val)
        return tmp_im

    def _binarize_adaptative(self, kind='gaussian', size=101, inplace=False):
        """
        Binarize the image using a threshold.

        Parameters
        ==========
        kind: string in ['gaussian', 'mean']
            Type of threshold computation.
        size: number
            Size of the area used to binarize. Should be odd.
        threshold: number
            Threshold.
        inplace: boolean
            To binarize in place or not.

        Returns:
        ========
        new_img: Image object
            Binarized image
        """
        # checks
        if kind == 'gaussian':
            kind = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        elif kind == 'mean':
            kind = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            raise ValueError
        if size <= 0:
            raise ValueError
        if size % 2 != 1:
            size += 1
            warnings.warn(f"size should be odd and has been set to {size}"
                          f" instead of {size - 1}")
        #
        if inplace:
            tmp_im = self
        else:
            tmp_im = self.copy()
        new_img = cv2.adaptiveThreshold(tmp_im.data,
                                        255,
                                        kind,
                                        cv2.THRESH_BINARY,
                                        size,
                                        1)
        tmp_im.data = new_img
        tmp_im.used_threshold = 'adapt'
        return tmp_im

    def _binarize_otsu(self, inplace=True):
        """
        Binarize the image using a threshold choosen using the image histogram.

        Parameters
        ==========
        inplace: boolean
            To binarize in place or not.

        Returns:
        ========
        new_img: Image object
            Binarized image
        """
        if inplace:
            tmp_im = self
        else:
            tmp_im = self.copy()
        ret_val, new_img = cv2.threshold(self.data,
                                         0,
                                         255,
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tmp_im.data = new_img
        tmp_im.used_threshold = int(ret_val)
        return tmp_im

    def fill_holes(self, size=2, iterations=1, inplace=False):
        """
        Fill the holes in the image.

        Parameters
        ==========
        size: integer
            Size of the hole to fill.
        iterations: integet
            Number of iterations

        Returns
        =======
        tmp_im : Image object
            Transformation of the initial image where holes have been filled.
        """
        raise Exception('Obsolete')
        if inplace:
            tmp_im = self
        else:
            tmp_im = self.copy()
        # first, remove holes
        tmp_im.data = spim.binary_closing(tmp_im.data,
                                          structure=np.ones((size, size)),
                                          iterations=iterations)
        # remove black border
        border = size + iterations
        tmp_im.data = tmp_im.data[border:-border, border:-border]
        # second, only keep biggest white area
        labels, nmb = spim.label(tmp_im.data)
        sizes = [np.sum(labels == label) for label in np.arange(1, nmb + 1)]
        bigger_label = np.argmax(sizes) + 1
        labels[labels != bigger_label] = 0
        tmp_im.data = labels
        # third, only keep biggest black area
        tmp_data = tmp_im.data.copy()
        tmp_data[tmp_im.data == 1] = 0
        tmp_data[tmp_im.data == 0] = 1
        labels, nmb = spim.label(tmp_data)
        sizes = [np.sum(labels == label) for label in np.arange(1, nmb + 1)]
        bigger_label = np.argmax(sizes) + 1
        labels[labels != bigger_label] = 1
        # store
        tmp_data = labels.copy()
        tmp_data[labels == 1] = 0
        tmp_data[labels == 0] = 1
        tmp_im.data = tmp_data
        return tmp_im

    def edge_detection_canny(self, threshold1=None, threshold2=None,
                             base_max_dist=15, size_ratio=.5,
                             nmb_edges=2, ignored_pixels=2,
                             keep_exterior=True,
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
        """
        # check for baseline
        if self.baseline is None:
            raise Exception('You should set the baseline first.')
        if self.dx != self.dy:
            warnings.warn('dx is different than dy, results can be weird...')
        # Get thresholds from histograms (Otsu method)
        if threshold2 is None or threshold1 is None:
            # # Otsu method
            # threshold2, _ = cv2.threshold(np.array(self.values,
            #                                        dtype=np.uint8),
            #                               0, 255,
            #                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # threshold1 = threshold2/2
            # DSA adapted method
            hist = self.get_histogram(cum=True)
            threshold1 = hist.get_value_position(hist.max/2)[0]
            threshold2 = np.max(hist.x)
        # remove useless part of the image
        tmp_im = self.crop(intervy=[np.min([self.baseline.pt2[1],
                                            self.baseline.pt1[1]]), np.inf],
                           inplace=False)
        if verbose:
            plt.figure()
            tmp_im.display()
            plt.title('Initial image')
        # Perform Canny detection
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
        # remove points behind the baseline (and too close to)
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
        # dilatation / erosion to ensure line continuity
        im_edges = spim.binary_dilation(im_edges, iterations=1)
        im_edges = spim.binary_erosion(im_edges, iterations=1)
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  im_edges, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Dilatation / erosion step')
        # Keep only the bigger edges
        labels, nmb = spim.label(im_edges, np.ones((3, 3)))
        nmb_edge = nmb
        X, Y = np.meshgrid(tmp_im.axe_x, tmp_im.axe_y, indexing="ij")
        dy = self.axe_y[1] - self.axe_y[0]
        # Let only the maximum allowed number of edges
        if np.max(labels) > nmb_edges:
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
        if nmb_edge > 1:
            # Remove small patches
            sizes = [np.sum(labels == label)
                     for label in np.arange(1, nmb + 1)]
            crit_size = np.sort(sizes)[-1]*size_ratio
            for i, size in enumerate(sizes):
                if size < crit_size:
                    im_edges[labels == i+1] = 0
                    labels[labels == i+1] = 0
                    nmb_edge -= 1

            if verbose:
                plt.figure()
                im = Image()
                im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                      im_edges, mask=tmp_im.mask,
                                      unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
                im.display()
                plt.title('Removed small edges')
            # Remove if not touching the baseline
            for i in range(nmb):
                ys = Y[labels == i+1]
                if len(ys) == 0:
                    continue
                min_y = np.min(ys)
                mdist = base_max_dist*dy
                if min_y > mdist + np.max([self.baseline.pt1[1],
                                           self.baseline.pt2[1]]):
                    if debug:
                        print(f"Label {i+1} removed because not touching "
                              "baseline")
                    im_edges[labels == i+1] = 0
                    labels[labels == i+1] = 0
                    nmb_edge -= 1
            if verbose:
                plt.figure()
                im = Image()
                im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                      im_edges, mask=tmp_im.mask,
                                      unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
                im.display()
                plt.title('Removed edge not touching the baseline')
            # keep only the two exterior edges
            if nmb_edge > 2 and keep_exterior:
                mean_xs = [np.mean(X[labels == label])
                           for label in np.arange(1, nmb + 1)]
                mean_xs = np.asarray(mean_xs)
                mean_xs[np.isnan(mean_xs)] = np.mean(mean_xs[~np.isnan(mean_xs)])
                mean_xs_indsort = np.argsort(mean_xs)
                for i in mean_xs_indsort[1:-1]:
                    if debug:
                        print(f"Label {i+1} removed because not exterior")
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
                plt.title('Removed the interior edged')
        # Get points coordinates
        xs, ys = np.where(im_edges)
        xs = [tmp_im.axe_x[x] for x in xs]
        ys = [tmp_im.axe_y[y] for y in ys]
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

    def circle_detection(self, dp=1., minDist=10, verbose=False):
        """
        Detect the circle present in the image.

        Parameters
        ==========
        dp: number
            Inverse ratio of the accumulator resolution.
            (Put a bigger number to get more circles !)
        minDist: number
            Minimum distance between two circle centers.

        Notes
        =====
        Use the opencv HoughCircle function.
        """
        values = np.array(self.values, dtype=np.uint8)
        circles = cv2.HoughCircles(values,
                                   cv2.HOUGH_GRADIENT,
                                   dp=dp,
                                   minDist=minDist)[0]
        # Adapt to current axes
        print(circles)
        circles = np.array(circles)
        tmp = circles[:, 0].copy()
        circles[:, 0] = circles[:, 1]*self.dx
        circles[:, 1] = tmp*self.dy
        circles[:, 2] *= (self.dy + self.dx)/2
        print(self.dx, self.dy)
        print(circles)
        # display if asked
        if verbose:
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Wedge
            plt.figure()
            self.display()
            patches = []
            for (x, y, r) in circles:
                patches.append(Wedge((x, y), r, 0, 360, width=self.axe_x[-1]*.001))
                plt.plot(x, y, "ok")
            p = PatchCollection(patches)
            plt.gca().add_collection(p)
            plt.show()


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
