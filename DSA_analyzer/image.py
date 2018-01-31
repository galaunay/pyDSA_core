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
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import warnings
from IMTreatment import ScalarField, Points
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
    def __init__(self):
        """
        Class representing a greyscale image.
        """
        super().__init__()
        self.baseline = None

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
        plt.show()
        # use linear interpolation to get baseline
        self.baseline = Baseline(pos, xmin=self.axe_x[0],
                                 xmax=self.axe_x[-1])
        return self.baseline

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

    def edge_detection(self, threshold1=None, threshold2=None, verbose=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        if self.dx != self.dy:
            warnings.warn('dx is different than dy, results can be weird...')
        if verbose:
            plt.figure()
            self.display()
            plt.title('Initial image')
        # Get thresholds from histograms
        if threshold2 is None or threshold1 is None:
            hist = cv2.calcHist(np.array(self.values, dtype=np.uint8),
                                [0], None, [255], [0, 255])
            cumhist = np.cumsum(hist)
            cumhist -= np.min(cumhist)
            cumhist /= np.max(cumhist)
            if threshold1 is None:
                threshold1 = np.argwhere(cumhist > 0.2)[0][0]
            if threshold2 is None:
                threshold2 = np.argwhere(cumhist > 0.8)[0][0]
        # remove useless part of the image
        tmp_im = self.crop(intervy=[np.min([self.baseline.pt2[1],
                                            self.baseline.pt1[1]]), np.inf],
                           inplace=False)
        # Perform Canny detection
        im_edges = cv2.Canny(np.array(tmp_im.values, dtype=np.uint8),
                             threshold1, threshold2)
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  im_edges, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Canny edge detection')
        # Keep only the bigger edges
        labels, nmb = spim.label(im_edges, np.ones((3, 3)))
        if np.max(labels) > 1:
            sizes = [np.sum(labels == label) for label in np.arange(1, nmb + 1)]
            crit_size = np.sort(sizes)[-2]
            for i, size in enumerate(sizes):
                if size < crit_size:
                    im_edges[labels == i+1] = 0
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(tmp_im.axe_x, tmp_im.axe_y,
                                  labels, mask=tmp_im.mask,
                                  unit_x=tmp_im.unit_x, unit_y=tmp_im.unit_y)
            im.display()
            plt.title('Canny edge detection + blob selection')
        # Get points coordinates
        xs, ys = np.where(im_edges)
        xs = [tmp_im.axe_x[x] for x in xs]
        ys = [tmp_im.axe_y[y] for y in ys]
        # remove points behind the baseline
        xys = []
        a, b = np.polyfit(self.baseline.xy[0],
                          self.baseline.xy[1],
                          1)
        for x, y in zip(xs, ys):
            if y > self.baseline.pt1[1] and y > self.baseline.pt2[1]:
                xys.append([x, y])
                continue
            if y > a*x + b:
                xys.append([x, y])
        if verbose:
            plt.figure()
            self.display()
            xys = np.array(xys)
            plt.plot(xys[:, 0], xys[:, 1], ".k")
            plt.title('Initial image + edge points')
        return DropEdges(xy=xys, unit_x=self.unit_x, unit_y=self.unit_y,
                         baseline=self.baseline)

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
