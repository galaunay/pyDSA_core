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
from IMTreatment import ScalarField, Profile, Points, TemporalScalarFields,\
    TemporalPoints
import IMTreatment.file_operation as imtio


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

    def edge_detection(self, threshold1=None, threshold2=None, verbose=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        if verbose:
            plt.figure()
            self.display()
            plt.title('Initial image')
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
        im_edges = cv2.Canny(np.array(self.values, dtype=np.uint8),
                             threshold1, threshold2)
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(self.axe_x, self.axe_y,
                                  im_edges, mask=self.mask,
                                  unit_x=self.unit_x, unit_y=self.unit_y)
            im.display()
            plt.title('Canny edge detection')
        # Keep only the bigger edge
        labels, nmb = spim.label(im_edges, np.ones((3, 3)))
        sizes = [np.sum(labels == label) for label in np.arange(1, nmb + 1)]
        for i, size in enumerate(sizes):
            if size <= .5*np.max(sizes):
                im_edges[labels == i] = 0
        if verbose:
            plt.figure()
            im = Image()
            im.import_from_arrays(self.axe_x, self.axe_y,
                                  im_edges, mask=self.mask,
                                  unit_x=self.unit_x, unit_y=self.unit_y)
            im.display()
            plt.title('Canny edge detection + blob selection')
        # Get points coordinates
        xs, ys = np.where(im_edges)
        if verbose:
            plt.figure()
            self.display()
            plt.plot(xs, ys, ".k")
            plt.title('Initial image + edge points')
        return Points(xy=list(zip(xs, ys)), unit_x=self.unit_x,
                      unit_y=self.unit_y)


class TemporalImages(TemporalScalarFields):
    def __init__(self):
        super().__init__()
        self.baseline = None
        self.field_type = Image

    def set_baseline(self, pt1, pt2):
        """
        Set the drop baseline.
        """
        self.baseline = [pt1, pt2]
        for i in range(len(self.fields)):
            self.fields[i].set_baseline(pt1=pt1, pt2=pt2)

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

    def edge_detection(self, threshold1=None, threshold2=None, verbose=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        pts = TemporalPoints()
        for i in range(len(self.fields)):
            pt = self.fields[i].edge_detection(threshold1=threshold1,
                                               threshold2=threshold2)
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
        return pts


class DropEdges(Points):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        self.drop_edges = self._separate_drop_edges()

    def _separate_drop_edges(self):
        """
        Separate the two sides of the drop.
        """
        ind_sort = np.argsort(self.xy[:, 0])
        xs = self.xy[:, 0][ind_sort]
        ys = self.xy[:, 1][ind_sort]
        dxs = xs[1::] - xs[0:-1]
        dxs_sorted = np.sort(dxs)
        if dxs_sorted[-1] > 10*dxs_sorted[-2]:
            # ind of the needle center (needle)
            ind_cut = np.argmax(dxs) + 1
        else:
            # ind of the higher point (no needle)
            ind_cut = np.argmax(ys)
        xs1 = xs[0:ind_cut]
        ys1 = ys[0:ind_cut]
        xs2 = xs[ind_cut::]
        ys2 = ys[ind_cut::]
        de1 = Points(list(zip(xs1, ys1)), unit_x=self.unit_x,
                     unit_y=self.unit_y)
        de2 = Points(list(zip(xs2, ys2)), unit_x=self.unit_x,
                     unit_y=self.unit_y)
        return de1, de2

    def get_fitting(self, k=5, s=None, verbose=False):
        """
        Get a fitting for the droplet shape.

        Parameters
        ----------
        kind : string, optional
            The kind of fitting used. Can be 'polynomial' or 'ellipse'.
        order : integer
            Approximation order for the fitting.
        """
        print('TODO: to clean')
        # Prepare drop edge for interpolation
        de1, de2 = self._separate_drop_edges()
        x1 = de1.xy[:, 0]
        y1 = de1.xy[:, 1]
        new_y1 = np.sort(list(set(y1)))
        new_x1 = [np.mean(x1[y == y1]) for y in new_y1]
        x2 = de2.xy[:, 0]
        y2 = de2.xy[:, 1]
        new_y2 = np.sort(list(set(y2)))
        new_x2 = [np.mean(x2[y == y2]) for y in new_y2]
        # spline interpolation
        s = s or len(new_y1)/10
        spline1 = spint.UnivariateSpline(new_y1, new_x1, k=k, s=s)
        spline2 = spint.UnivariateSpline(new_y2, new_x2, k=k, s=s)
        if verbose:
            plt.figure()
            de1.display()
            plt.plot(spline1(new_y1), new_y1, 'r')
            de2.display()
            plt.plot(spline2(new_y2), new_y2, 'r')
            plt.axis('equal')
            plt.show()
        return spline1, spline2

    def get_contact_angle(self, k=5, s=None, verbose=False):
        print('TODO: to clean')
        # get spline
        de1, de2 = self._separate_drop_edges()
        y1 = de1.xy[:, 1]
        y2 = de2.xy[:, 1]
        spline1, spline2 = self.get_fitting(k=k, s=s, verbose=verbose)
        y1 = np.linspace(np.min(y1), np.max(y1), len(y1)*10)
        x1 = spline1(y1)
        y2 = np.linspace(np.min(y2), np.max(y2), len(y2)*10)
        x2 = spline2(y2)
        # Get gradients
        print('TODO: better way to estimate gradients...')
        grad1 = np.gradient(x1, y1[1] - y1[0])
        grad2 = np.gradient(x2, y2[1] - y2[0])
        if verbose:
            plt.figure()
            plt.plot(y1, grad1)
            plt.plot(y2, grad2)
            plt.title('gradients')
            plt.show()
        return grad1[0], grad2[0]



class ImageSequence(object):
    pass


class FeatureExtractor(object):
    pass


class ContactAngleExtractor(object):
    pass


class ContactAngleHysteresisExtractor(object):
    pass


def import_from_image(path, dx=1, dy=1, unit_x="", unit_y=""):
    """
    Import an image into an Image object.

    Parameters
    ==========
    path: string
        Path to the image file.
    dx, dy: numbers
        Real distance between two pixels.
    unit_x, unit_y: strings
        Unities of dx and dy.

    Returns
    =======
    img: Image object
        Image
    """
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).transpose()
    data = data[:, ::-1]
    axe_x = np.arange(0, data.shape[0]*dx, dx)
    axe_y = np.arange(0, data.shape[1]*dy, dy)
    img = Image()
    img.import_from_arrays(axe_x, axe_y, data,
                           unit_x=unit_x, unit_y=unit_y)
    return img


def import_from_video(path, dx=1, dy=1, dt=1, unit_x="", unit_y="", unit_t="",
                      frame_inds=None):
    """
    Import a images from a video file.

    Parameters
    ==========
    path: string
        Path to the video file.
    dx, dy: numbers
        Real distance between two pixels.
    dt: number
        Time interval between two frames.
    unit_x, unit_y, unit_y: strings
        Unities of dx, dy and dt.
    frame_inds: 2x1 array of integers
        Range of frame to import (default to all).

    Returns
    =======
    imgs: TemporalImages object
        Images
    """
    frame_inds = frame_inds or [0, np.inf]
    # open video
    vid = cv2.VideoCapture()
    vid.open(path)
    ti = TemporalImages()
    i = 0
    print("TODO: read video directly in grayscale")
    while True:
        if i < frame_inds[0]:
            i += 1
            vid.grab()
            continue
        success, im = vid.read()
        if not success:
            break
        im = np.mean(im, axis=2)
        im = im.transpose()[:, ::-1]
        axe_x = np.arange(0, im.shape[0]*dx, dx)
        axe_y = np.arange(0, im.shape[1]*dy, dy)
        sf = Image()
        sf.import_from_arrays(axe_x, axe_y, im, mask=False,
                              unit_x=unit_x, unit_y=unit_y)
        ti.add_field(sf, time=i*dt, unit_times=unit_t)
        i += 1
        if i >= frame_inds[1]:
            break
    return ti


if __name__ == '__main__':

    #==========================================================================
    # Video
    #==========================================================================
    # Create image from video
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
           "data/CAH Sample 2 Test.avi"
    ims = import_from_video(path, dx=1, dy=1, unit_x="um", unit_y="um",
                            frame_inds=[80, 380])
                            # frame_inds=[80, 90])
    ims.crop(intervy=[74.6, 500], inplace=True)
    # Edge detection
    edges = ims.edge_detection()
    # fitting
    theta1 = []
    theta2 = []
    verbose = False
    for i in range(len(edges.point_sets)):
        edge = edges.point_sets[i]
        edge = DropEdges(edge.xy, unit_x=edge.unit_x, unit_y=edge.unit_y)
        t1, t2 = edge.get_contact_angle(k=5, verbose=verbose)
        theta1.append(t1)
        theta2.append(t2)
    plt.figure()
    plt.plot(edges.times, theta1)
    plt.plot(edges.times, theta2)
    plt.show()

    bug

    # #==========================================================================
    # # Image
    # #==========================================================================
    # # Create image
    # path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/data/Test"\
    #        " Sample 2.bmp"
    # im = import_from_image(path, dx=1, dy=1, unit_x="um", unit_y="um")
    # # set baseline
    # pt1 = [604.8, 68.6]
    # pt2 = [157.6, 72.3]
    # im.set_baseline(pt1, pt2)
    # # Simple display
    # im.display()
    # plt.title('Raw image')
    # # make Canny edge detection
    # edge_pts = im.edge_detection(verbose=False, inplace=True)
    # plt.figure()
    # # edge_pts.display()
    # edge_pts.display()
    # plt.title('Canny edge detection + area selection')
    # plt.show()
    # # Fit
    # raise Exception('Todo')
    # spl = spint.UnivariateSpline(xs, ys, k=3)
    # order = 5
    # zx = np.polyfit(range(len(xs)), xs, order)
    # fx = np.poly1d(zx)
    # zy = np.polyfit(range(len(ys)), ys, order)
    # fy = np.poly1d(zy)
    # plt.plot(fx(range(len(xs))), fy(range(len(ys))), "--k")
    # plt.show()
