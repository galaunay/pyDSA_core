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

    def import_from_file(self, filepath):
        """
        Import greyscale image from an image file.
        """
        self.path = filepath
        data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if data.shape == [0, 0]:
            raise IOError()
        self.data = np.array(data, dtype=np.uint8)

    def import_from_array(self, data):
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

    def display(self):
        plt.imshow(self.data, cmap='gray', interpolation='nearest')
        plt.xticks([]), plt.yticks([])

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
        ret_val, new_img = cv2.threshold(self.data,
                                         threshold,
                                         255,
                                         cv2.THRESH_BINARY)
        if inplace:
            self.data = new_img
            self.used_threshold = int(ret_val)
        tmp_im = Image()
        tmp_im.import_from_array(new_img)
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
        new_img = cv2.adaptiveThreshold(self.data,
                                        255,
                                        kind,
                                        cv2.THRESH_BINARY,
                                        size,
                                        1)
        if inplace:
            self.data = new_img
            self.used_threshold = 'adapt'
        tmp_im = Image()
        tmp_im.import_from_array(new_img)
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
        ret_val, new_img = cv2.threshold(self.data,
                                         0,
                                         255,
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if inplace:
            self.data = new_img
            self.used_threshold = int(ret_val)
        tmp_im = Image()
        tmp_im.import_from_array(new_img)
        tmp_im.used_threshold = int(ret_val)
        return tmp_im


class ImageSequence(object):
    pass


class FeatureExtractor(object):
    pass


class ContactAngleExtractor(object):
    pass


class ContactAngleHysteresisExtractor(object):
    pass


if True:
    # simple display
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/data/Test Sample 2.bmp"
    # # TEMP
    # img = cv2.imread(path, 0)
    # ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # bug
    # # TEMP - End
    im = Image()
    im.import_from_file(path)
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
