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
import os
import numpy as np
from .image import Image
from .temporalimages import TemporalImages
from IMTreatment.utils import ProgressCounter
import IMTreatment.file_operation as imtio
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
    img = Image(filepath=path)
    img.import_from_arrays(axe_x, axe_y, data,
                           unit_x=unit_x, unit_y=unit_y)
    return img


def import_from_video(path, dx=1, dy=1, dt=1, unit_x="", unit_y="", unit_t="",
                      frame_inds=None, incr=1, dtype=np.uint8, verbose=False):
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
    incr: integer
        Number of frame to import.
        (ex: with a value of 2, only 1/2 frames will be imported).
    dtype: type
        Numerical type used for the stored data.
        Should be a type supported by numpy arrays.
        Default to 8 bit unsigned integers (np.uint8) to optimize memory usage.

    Returns
    =======
    imgs: TemporalImages object
        Images
    """
    frame_inds = frame_inds or [0, np.inf]
    # Check if file exist
    imtio.check_path(path)
    # open video
    vid = cv2.VideoCapture()
    vid.open(path)
    ti = TemporalImages(filepath=path)
    i = 0
    max_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_inds is None:
        frame_inds = [0, max_frame]
    if frame_inds[1] > max_frame:
        frame_inds[1] = max_frame
    # logs
    if verbose:
        nmb_frames = int((frame_inds[1] - frame_inds[0])/incr)
        pg = ProgressCounter(init_mess="Decoding video",
                             nmb_max=nmb_frames,
                             name_things='frames',
                             perc_interv=5)
    t = 0
    for i in np.arange(0, frame_inds[1], 1):
        if i < frame_inds[0] or i % incr != 0:
            t += dt
            vid.grab()
            continue
        success, im = vid.read()
        if not success:
            if frame_inds[1] != np.inf:
                warnings.warn(f"Can't decode frame number {i}")
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # im = np.mean(im, axis=2)
        im = im.transpose()[:, ::-1]
        axe_x = np.arange(0, im.shape[0]*dx - 0.1*dx, dx)
        axe_y = np.arange(0, im.shape[1]*dy - 0.1*dy, dy)
        sf = Image()
        sf.import_from_arrays(axe_x, axe_y, im, mask=False,
                              unit_x=unit_x, unit_y=unit_y,
                              dtype=dtype)
        ti.add_field(sf, time=t, unit_times=unit_t, copy=False)
        t += dt
        if verbose:
            pg.print_progress()

    # Try to import infos if the infofile exist
    if os.path.isfile(ti.infofile_path):
        ti._import_infos()
    # cache axis
    ti._dump_infos()
    #
    if verbose and frame_inds[1] == np.inf:
        pg._print_end()
    if len(ti.fields) == 0:
        raise Exception("Something goes wrong during video import, "
                        "the video does not contain any image...")
    return ti
