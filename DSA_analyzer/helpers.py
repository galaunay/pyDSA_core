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
from .image import Image
from .temporalimages import TemporalImages
from IMTreatment.utils import ProgressCounter
import warnings

"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


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
                      frame_inds=None, verbose=False):
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
    if frame_inds is None:
        frame_inds = [0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT))]
    # logs
    if verbose:
        nmb_frames = frame_inds[1] - frame_inds[0]
        pg = ProgressCounter("Decoding video", "Done", nmb_frames, 'frames', 5)
    while True:
        if i < frame_inds[0]:
            i += 1
            vid.grab()
            continue
        success, im = vid.read()
        if not success:
            warnings.warn(f"Can't decode frame number {i}")
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # im = np.mean(im, axis=2)
        im = im.transpose()[:, ::-1]
        axe_x = np.arange(0, im.shape[0]*dx, dx)
        axe_y = np.arange(0, im.shape[1]*dy, dy)
        sf = Image()
        sf.import_from_arrays(axe_x, axe_y, im, mask=False,
                              unit_x=unit_x, unit_y=unit_y)
        ti.add_field(sf, time=i*dt, unit_times=unit_t, copy=False)
        i += 1
        if verbose:
            pg.print_progress()
        if i >= frame_inds[1]:
            break
    return ti