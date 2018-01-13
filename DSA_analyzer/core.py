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
import warnings
import copy


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class Image(object):
    def __init__(self):
        """
        Class representing a greyscale image.
        """
        self.path = None
        self.data = None
        self.used_threshold = None
        self.baseline = None
        self.dx = None

    def import_from_file(self, filepath, dx=1):
        """
        Import greyscale image from an image file.
        """
        self.path = filepath
        data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if data.shape == [0, 0]:
            raise IOError()
        self.data = np.array(data, dtype=np.uint8)
        self.dx = dx

    def import_from_array(self, data, dx=1):
        """
        Import greyscale image from an array.
        """
        self.path = None
        data = np.array(data, dtype=np.uint8)
        if data.ndim != 2:
            raise ValueError
        if np.any(data > 255) or np.any(data < 0):
            raise ValueError()
        self.data = data
        self.dx = dx

    def copy(self):
        return copy.deepcopy(self)

    def display(self):
        plt.imshow(self.data, cmap='gray', interpolation='nearest',
                   extent=[0, self.dx*self.data.shape[1],
                           0, self.dx*self.data.shape[0]])
        plt.xticks([]), plt.yticks([])
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

    def edge_detection(self):
        pass


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
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/data/Test Sample 2.bmp"
    im = Image()
    im.import_from_file(path)
    # set baseline
    pt1 = [604.8, 68.6]
    pt2 = [157.6, 72.3]
    im.set_baseline(pt1, pt2)
    # Simple display
    im.display()
    plt.title('Raw image')
    # binarize
    im1 = im._binarize_simple(threshold=100)
    im2 = im._binarize_adaptative(size=400)
    im3 = im._binarize_otsu()
    plt.figure()
    im1.display()
    plt.title(f'Simple threshold at {im1.used_threshold}')
    plt.figure()
    im2.display()
    plt.title(f'Adaptative threshold of size 400')
    plt.figure()
    im3.display()
    plt.title(f'Otsu optimized threshold at {im3.used_threshold}')
    plt.show()
