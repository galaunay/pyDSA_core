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
import scipy.interpolate as spint
import warnings
import copy
from IMTreatment import ScalarField, Profile


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
            pt1, pt2 = self.baseline
            plt.plot([pt1[0], pt2[0]],
                     [pt1[1], pt2[1]],
                     color='g',
                     ls='none',
                     marker='o')
            plt.plot([pt1[0], pt2[0]],
                     [pt1[1], pt2[1]],
                     color='g',
                     ls='-')

    def set_baseline(self, pt1, pt2):
        """
        Set the baseline.

        Parameters
        ==========
        pt1, pt2 : 2x1 arrays of numbers
            Points defining the baseline.
        """
        self.baseline = [pt1, pt2]

    def get_histogram(self, cum=False, normalized=False):
        """
        Return the image histogram.

        Parameters
        ==========
        cum: boolean
            If True, get a cumulative histogram.

        Returns
        =======
        hist: array of numbers
            Histogram.
        """
        hist, xs = np.histogram(self.values.flatten(),
                                bins=255,
                                density=normalized)
        xs = xs[0:-1] + np.mean(xs[0:2])
        if cum:
            hist = np.cumsum(hist)

        return Profile(xs, hist, mask=False, unit_x=self.unit_values,
                       unit_y="counts")

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
        raise Exception('Need proper axis implementation to handle baseline modifs')
        self.baseline = [[self.baseline[0][0] + border*self.dx,
                          self.baseline[0][1] + border*self.dx],
                         [self.baseline[1][0] + border*self.dx,
                          self.baseline[1][1] + border*self.dx]]
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

    def edge_detection(self, threshold1=None, threshold2=None, inplace=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        if inplace:
            tmp_im = self
        else:
            tmp_im = self.copy()
        # Get thresholds from histograms
        if threshold2 is None or threshold1 is None:
            cumhist = self.get_histogram(cum=True).y
            cumhist -= np.min(cumhist)
            cumhist /= np.max(cumhist)
            if threshold1 is None:
                threshold1 = np.argwhere(cumhist > 0.2)[0][0]
            if threshold2 is None:
                threshold2 = np.argwhere(cumhist > 0.8)[0][0]
        # Perform Canny detection
        im_edges = cv2.Canny(np.array(tmp_im.values, dtype=np.uint8),
                             threshold1, threshold2)
        # Keep only the bigger edge
        labels, nmb = spim.label(im_edges, np.ones((3, 3)))
        sizes = [np.sum(labels == label) for label in np.arange(1, nmb + 1)]
        bigger_label = np.argmax(sizes) + 1
        im_edges[labels != bigger_label] = 0
        # Get points coordinates
        xs, ys = np.where(im_edges)
        return xs, ys


class ImageSequence(object):
    pass


class FeatureExtractor(object):
    pass


class ContactAngleExtractor(object):
    pass


class ContactAngleHysteresisExtractor(object):
    pass


if __name__ == '__main__':
    # Create image
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/data/Test"\
           " Sample 2.bmp"
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).transpose()
    data = data[:, ::-1]
    axe_x = np.arange(0, data.shape[0])
    axe_y = np.arange(0, data.shape[1])
    im = Image()
    im.import_from_arrays(axe_x, axe_y, data, unit_x="px", unit_y="px")
    # set baseline
    pt1 = [604.8, 68.6]
    pt2 = [157.6, 72.3]
    im.set_baseline(pt1, pt2)
    # Simple display
    im.display()
    plt.title('Raw image')
    # make Canny edge detection
    xs, ys = im.edge_detection()
    plt.plot(xs, ys, ".r")
    plt.title('Canny edge detection + area selection')
    # Fit
    raise Exception('Todo')
    spl = spint.UnivariateSpline(xs, ys, k=3)
    order = 5
    zx = np.polyfit(range(len(xs)), xs, order)
    fx = np.poly1d(zx)
    zy = np.polyfit(range(len(ys)), ys, order)
    fy = np.poly1d(zy)
    plt.plot(fx(range(len(xs))), fy(range(len(ys))), "--k")
    plt.show()
