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
from glob import glob
import numpy as np
from . import temporalimages as tis
from . import image
from IMTreatment.utils import ProgressCounter, make_unit
import IMTreatment.file_operation as imtio
import warnings
import scipy.optimize as spopt

"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


def import_from_image(path, dx=1, dy=1, unit_x="", unit_y="",
                      cache_infos=True, dtype=np.uint8):
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
    imtio.check_path(path)
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise IOError(f"{path} is not a valid image.")
    data = data.transpose()
    data = data[:, ::-1]
    axe_x = np.arange(0, data.shape[0]*dx, dx)
    axe_y = np.arange(0, data.shape[1]*dy, dy)
    img = image.Image(filepath=path, cache_infos=cache_infos)
    img.import_from_arrays(axe_x, axe_y, data,
                           unit_x=unit_x, unit_y=unit_y,
                           dtype=dtype)
    if cache_infos:
        # Try to import infos if the infofile exist
        if os.path.isfile(img.infofile_path):
            img._import_infos()
        # cache axis
        img._dump_infos()
    return img


def import_from_images(path, dx=1, dy=1, dt=1, unit_x="", unit_y="",
                       unit_times="", dtype=np.uint8,
                       iteration_hook=None, verbose=False):
    """
    Import a set of images into an TemporalImages object.

    Parameters
    ==========
    path: string
        Path to the images directory.
    dx, dy: numbers
        Real distance between two pixels.
    dt: numbers
        Time interval between two images.
    unit_x, unit_y, unit_times: strings
        Unities of dx, dy and time.

    Returns
    =======
    imgs: TemporalImages object
        Images
    """
    dirname = os.path.dirname(path)
    imtio.check_path(dirname)
    paths = glob(path)
    paths = sorted(paths)
    ims = tis.TemporalImages(dirname, cache_infos=True)
    t = 0
    # verbose
    if verbose:
        pg = ProgressCounter("Importing images", len(paths),
                             end_mess="Done", name_things="images")
    for i, path in enumerate(paths):
        im = import_from_image(path=path, dx=dx, dy=dy, unit_y=unit_y,
                               unit_x=unit_x, cache_infos=False,
                               dtype=dtype)
        ims.add_field(im, time=t, unit_times=unit_times)
        t += dt
        if verbose:
            pg.print_progress()
        if iteration_hook is not None:
            iteration_hook(i, len(paths))
    # load saved things if necessary
    ims._import_infos()
    return ims


def import_from_video(path, dx=1, dy=1, dt=None,
                      unit_x="", unit_y="", unit_t="",
                      frame_inds=None, frame_range=None,
                      incr=1, nmb_frame_to_import=None,
                      intervx=None, intervy=None,
                      cache_infos=True, dtype=np.uint8,
                      iteration_hook=None,
                      verbose=False):
    """
    Import a images from a video file.

    Parameters
    ==========
    path: string
        Path to the video file.
    dx, dy: numbers
        Real distance between two pixels.
    dt: number, optional
        Time interval between two frames.
        If not specified, the time interval is taken from the video metadata
    unit_x, unit_y, unit_y: strings
        Unities of dx, dy and dt.
    frame_range: 2x1 array of integers
        Range of frame to import (default to all).
    frame_inds: array of integers
        Indices of frames to import (default to all).
    incr: integer
        Number of frame to import.
        (ex: with a value of 2, only 1/2 frames will be imported).
    nmb_frame_to_import: integer
        Number of evenly distributed frames to import
        Will overwrite any values of 'incr'
    cache_infos: boolean
        If True (default) an infofile is created alongside the video file
        To keep track of scaling preoperties and baseline.
    intervx, intervy: 2x1 list of numbers
        Cropping dimensions applied on each frames.
    dtype: type
        Numerical type used for the stored data.
        Should be a type supported by numpy arrays.
        Default to 8 bit unsigned integers (np.uint8) to optimize memory usage.

    Returns
    =======
    imgs: TemporalImages object
        Images
    """
    # check
    if frame_inds is not None and frame_range is not None:
        raise Exception()
    if frame_inds is not None and incr != 1:
        raise Exception("You cannot specify both the 'frame_inds' and an "
                        "'incr' value")
    if frame_inds is not None and nmb_frame_to_import is not None:
        raise Exception("You cannot specify both the 'frame_inds' and an "
                        "'nmb_frame_to_import' value")
    # Check if file exist
    imtio.check_path(path)
    # convert units (upfront for optimization purpose)
    unit_t = make_unit(unit_t)
    unit_x = make_unit(unit_x)
    unit_y = make_unit(unit_y)
    # open video
    vid = cv2.VideoCapture()
    vid.open(path)
    ti = tis.TemporalImages(filepath=path)
    i = 0
    # Get indices to import
    max_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame == 0:
        raise ImportError("Couldn't get the number of frame from the video file")
    if frame_inds is not None:
        if frame_inds[-1] > max_frame:
            raise Exception()
    else:
        if frame_range is None:
            frame_max = max_frame - 1
            frame_min = 0
        else:
            if frame_range[0] < 0:
                frame_min = 0
            else:
                frame_min = frame_range[0]
            if frame_range[1] > max_frame:
                frame_max = max_frame
            else:
                frame_max = frame_range[1]
        if nmb_frame_to_import is not None:
            incr = int((frame_max - frame_min)/nmb_frame_to_import)
            if incr == 0:
                incr = 1
        frame_inds = np.arange(frame_min, frame_max, incr, dtype=int)
    # Get fps
    if dt is None:
        fps = float(vid.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 1
        dt = 1/fps
    # logs
    if verbose:
        nmb_frames = len(frame_inds)
        if nmb_frames <= 1:
            raise Exception("No frames selected")
        pg = ProgressCounter(init_mess="Decoding video",
                             nmb_max=nmb_frames,
                             name_things='frames',
                             perc_interv=5)
    t = 0
    for i in np.arange(0, frame_inds[-1] + 1, 1):
        if i not in frame_inds:
            t += dt
            continue
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
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
        sf = image.Image()
        sf.import_from_arrays(axe_x, axe_y, im, mask=False,
                              unit_x=unit_x, unit_y=unit_y,
                              dontchecknans=True,
                              dontcheckunits=True,
                              dtype=dtype)
        sf.crop(intervx=intervx, intervy=intervy, inplace=True)
        ti.add_field(sf, time=t, unit_times=unit_t, copy=False)
        t += dt
        if verbose:
            pg.print_progress()
        if iteration_hook is not None:
            iteration_hook(i, frame_inds[-1] + 1)

    if cache_infos:
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


def circle_from_three_points(pt1, pt2, pt3):
    """
    Give the center and radius of a circle from three points
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3
    # ensure no division by zero
    if (x2 - x1) == 0:
        ma = (y3 - y1)/(x3 - x1)
        mb = (y3 - y2)/(x3 - x2)
    elif (x3 - x2) == 0:
        ma = (y2 - y1)/(x2 - x1)
        mb = (y3 - y1)/(x3 - x1)
    else:
        ma = (y2 - y1)/(x2 - x1)
        mb = (y3 - y2)/(x3 - x2)
    if (mb - ma) == 0:
        x = np.mean([x1, x2, x3])
    else:
        x = (ma*mb*(y1 - y3) + mb*(x1 + x2) - ma*(x2 + x3))/(2*(mb - ma))
    if ma == 0:
        y = np.mean([y1, y2, y3])
    else:
        y = -(1/ma)*(x - (x1 + x2)/2) + (y1 + y2)/2
    R = ((y1 - y)**2 + (x1 - x)**2)**.5
    return (x, y), R


def fit_circle(xs, ys, baseline=None, tangent_circ=None, sigma_max=None,
               soft_constr=False):
    """
    Fit a circle to the given points.

    Parameters
    ==========
    xs, ys: arrays
        Coordinates of the points to fit.
    baseline: Baseline object
        If specified, the fitting will try to not pass through the baseline.
    tangent_circ: ((x, y), R) tuple
        Circle to be tangent to the fitting
    sigma_max: number
        If specified, points too far from the fit are iteratively removed
        until they all fall in the range R +- sigma*R
    soft_constr: boolean
        If True, constraints are fitted instead of being imposed.
    """
    # Init some params
    if baseline is not None:
        basefun = baseline.get_baseline_fun()
    if tangent_circ is not None:
        (xd, yd), Rd = tangent_circ
        if xd > np.mean(xs):
            side = -1
        else:
            side = +1
    # # Compute weight from points density
    # # (to not give too much power to high density regions)
    # # DOES NOT WORK BECAUSE THE POINTS ARE NOT SORTED !
    # if np.all(ys[1::] - ys[:-1] > 0):
    #     indsort = np.argsort(ys)
    #     ys = ys[indsort]
    #     xs = xs[indsort]
    # weights = []
    # for i in range(len(ys)):
    #     if i == 0:
    #         weights.append(((ys[i] - ys[i+1])**2 + (xs[i] - xs[i+1])**2)**.5)
    #     elif i == len(ys) - 1:
    #         weights.append(((ys[i] - ys[i-1])**2 + (xs[i] - xs[i-1])**2)**.5)
    #     else:
    #         d1 = ((ys[i] - ys[i+1])**2 + (xs[i] - xs[i+1])**2)**.5
    #         d2 = ((ys[i] - ys[i-1])**2 + (xs[i] - xs[i-1])**2)**.5
    #         weights.append((d1 + d2)/2)
    # weights = np.asarray(weights)
    # weights /= np.sum(weights)/len(weights)
    # # TEMP
    # weights = np.ones(len(weights))
    # # TEMP - End

    def calc_R(x, y, xc, yc):
        """
        Calculate the distance of each data points from the center (xc, yc)
        """
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_circle_soft_constr(args, x, y):
        """
        Calculate the algebraic distance between the 2D points and the mean
        circle centered at c=(xc, yc)
        """
        xc, yc = args
        Ri = calc_R(x, y, xc, yc)
        residu = (Ri - Ri.mean())
        if baseline is not None:
            yb = basefun(xc)
            residu += (Ri.mean() - (yc - yb))
        if tangent_circ is not None:
            (xo, yo), Ro = tangent_circ
            dist_circ = ((xo - xc)**2 + (yo - yc)**2)**.5
            residu += (dist_circ - (Ro + Ri.mean()))
        return residu

    def f_circle_constr(args, x, y):
        """
        Calculate the algebraic distance between the 2D points and the mean
        circle centered at c=(xc, yc)
        """
        R = abs(args[0])
        (xd, yd), Rd = tangent_circ
        # first y guess
        xc, yc = center_from_R(R)
        # return residu
        Ri = calc_R(x, y, xc, yc)
        residu = (Ri - R)
        return residu

    def f_circle(args, x, y):
        """
        Calculate the algebraic distance between the 2D points and the mean
        circle centered at c=(xc, yc)
        """
        xc, yc = args
        Ri = calc_R(x, y, xc, yc)
        residu = (Ri - Ri.mean())
        return residu

    def center_from_R(R):
        """
        Return the position of the fitting center for a
        constrained circle of diameter R
        """
        (xd, yd), Rd = tangent_circ
        # first y guess
        yc = basefun(xd - Rd - R) + R
        # first x guess
        xc = xd + side*((R + Rd)**2 - (yc - yd)**2)**.5
        # corrections
        old_yc = np.inf
        while abs(old_yc - yc)/R > 1e-6:
            old_yc = yc
            yc = basefun(xc) + R
            xc = xd + side*((R + Rd)**2 - (yc - yd)**2)**.5
        return xc, yc

    # First guess from three points
    (xc, yc), R = circle_from_three_points([xs[0], ys[0]],
                                           [xs[int(len(xs)/2)],
                                            ys[int(len(xs)/2)]],
                                           [xs[-1], ys[-1]])
    # Choose adequate fitting function
    if baseline is None and tangent_circ is None:
        fit_function = f_circle
        init_guess = (xc, yc)
    elif baseline is not None and tangent_circ is not None:
        # checking side
        if soft_constr:
            fit_function = f_circle_soft_constr
            init_guess = (xc, yc)
        else:
            fit_function = f_circle_constr
            init_guess = (R,)
    else:
        raise Exception("Not implemented yet")
    # Iterative fit (removing marginal values each time)
    tmp_xs = xs.copy()
    tmp_ys = ys.copy()
    if sigma_max is not None:
        while True:
            args, ier = spopt.leastsq(
                fit_function, init_guess,
                args=(tmp_xs, tmp_ys))
            if len(args) == 1:
                R = abs(args[0])
                center = center_from_R(R)
                Rs = calc_R(tmp_xs, tmp_ys, *center)
            else:
                center = args
                Rs = calc_R(tmp_xs, tmp_ys, *center)
                R = np.mean(Rs)
            R_std = np.std(Rs)
            if R_std < R*sigma_max or len(tmp_xs) <= 10:
                break
            filt = np.logical_and(Rs < R + R_std*2,
                                  Rs > R - R_std*2)
            if np.all(filt):
                filt[np.argmax(abs(Rs - R))] = False
            tmp_xs = tmp_xs[filt]
            tmp_ys = tmp_ys[filt]
    # Classical fit
    else:
        args, ier = spopt.leastsq(
            fit_function, init_guess,
            args=(tmp_xs, tmp_ys))
        if len(args) == 1:
            R = abs(args[0])
            center = center_from_R(R)
        else:
            center = args
            R = np.mean(calc_R(tmp_xs, tmp_ys, *center))
    # return
    return center, R


def fit_ellipse(xs, ys):
    """
    Fit an Ellispe to the given set of points.

    Parameters
    ----------
    xs, ys: arrays
        Coordinates of the points to fit.

    Returns
    -------
    center: 2x1 array of numbers
        Position of the ellipse center.
    R1, R2: numbers
        Main and secondary ellipse radii.
    theta: angle
        Angle of the ellispe main axis (in radian).
    """

    # Taken from: http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    # Get a
    xs = np.asarray(xs)[:, np.newaxis]
    ys = np.asarray(ys)[:, np.newaxis]
    D = np.hstack((xs*xs, xs*ys, ys*ys, xs, ys, np.ones_like(xs)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(E)
    lambd = V[:, n]
    b, c, d, f, g, a = (lambd[1]/2, lambd[2], lambd[3]/2,
                        lambd[4]/2, lambd[5], lambd[0])
    # Get ellipse center
    num = b*b - a*c
    x0 = (c*d - b*f) / num
    y0 = (a*f - b*d) / num
    xc = np.array([x0, y0])
    # Get ellispe rotation
    theta = 0.5*np.arctan(2*b / (a - c))
    # Get ellipse radii
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b - a*c)*((c - a)*np.sqrt(1 + 4*b*b / (
        (a - c)*(a - c))) - (c + a))
    down2 = (b*b - a*c)*((a - c)*np.sqrt(1 + 4*b*b / (
        (a - c)*(a - c))) - (c + a))
    R1 = np.sqrt(up / down1)
    R2 = np.sqrt(up / down2)
    # Return
    return xc, R1, R2, theta


def get_ellipse_points(xc, yc, R1, R2, theta, res=100):
    """
    Return points on the coresponding ellipse.
    (For plotting purposes)
    """
    thetas = np.linspace(0, np.pi*2, res)
    xs = xc + R1*np.cos(thetas)
    ys = yc + R2*np.sin(thetas)
    rot_xs = xc + np.cos(theta)*(xs - xc) - np.sin(theta)*(ys - yc)
    rot_ys = yc + np.cos(theta)*(ys - yc) + np.sin(theta)*(xs - xc)
    return rot_xs, rot_ys
