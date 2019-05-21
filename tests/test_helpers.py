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

import os
import sys
try:
    dirname = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dirname)
    os.chdir('..')
    sys.path.append(os.getcwd())
except:
    pass

from helper import sane_parameters
import numpy as np
from pyDSA_core.helpers import import_from_image, import_from_images, \
    import_from_video, circle_from_three_points, fit_circle, fit_ellipse
from pyDSA_core import Image
from IMTreatment.utils import make_unit


class TestImportFromImage(object):
    """ Done """

    def setup(self):
        sane_parameters()

    def test_import_from_image(self):
        im = import_from_image("./tests/res/image.bmp",
                               dx=0.12, dy=5.14,
                               unit_x="mm",
                               unit_y="um",
                               cache_infos=False,
                               dtype=np.uint8)
        assert im.dx == 0.12
        assert im.dy == 5.14
        assert im.unit_x == make_unit('mm')
        assert im.unit_y == make_unit('um')
        assert im.values.dtype == np.uint8
        assert im.shape == (780, 580)
        assert np.all(im.values[100:110, 100:110]
                      == [[31, 33, 34, 38, 42, 46, 52, 49, 58, 61],
                          [34, 34, 36, 41, 46, 47, 49, 53, 57, 65],
                          [30, 33, 37, 38, 44, 46, 48, 54, 58, 59],
                          [30, 34, 35, 38, 42, 44, 48, 51, 59, 66],
                          [30, 33, 36, 42, 42, 45, 50, 53, 56, 65],
                          [30, 33, 36, 42, 42, 46, 49, 56, 57, 66],
                          [29, 34, 38, 40, 41, 48, 49, 52, 61, 63],
                          [29, 34, 38, 39, 41, 47, 46, 55, 58, 61],
                          [31, 34, 39, 43, 44, 49, 49, 55, 58, 60],
                          [31, 34, 37, 40, 44, 48, 48, 56, 58, 63]])

    def test_import_from_image_should_cache_info(self):
        im = import_from_image("./tests/res/image.bmp",
                               unit_x="mm",
                               unit_y="um",
                               cache_infos=True,
                               dtype=np.uint8)
        im.set_baseline([1, 2], [0.1, 123.2])
        im.scale(scalex=0.1, scaley=14, inplace=True)
        im2 = import_from_image("./tests/res/image.bmp",
                                cache_infos=True,
                                dtype=np.uint8)
        os.remove("./tests/res/image.info")
        assert im == im2


class TestImportFromVideo(object):
    """ Done """

    def setup(self):
        sane_parameters()

    def test_import_from_video(self):
        ims = import_from_video("./tests/res/video.mp4",
                                dx=0.12, dy=4,
                                dt=0.1,
                                unit_x="km",
                                unit_y="px",
                                unit_t="ms",
                                nmb_frame_to_import=20,
                                dtype=np.uint8,
                                cache_infos=False)
        assert ims.field_type == Image
        assert ims.unit_times == make_unit('ms')
        assert np.allclose(ims.times, np.arange(0, 37, 0.1*18))
        assert isinstance(ims.fields[0], Image)
        assert ims.fields[0].dx == 0.12
        assert ims.fields[0].dy == 4
        assert ims.fields[0].unit_x == make_unit('km')
        assert ims.fields[0].unit_y == make_unit('px')
        assert ims.fields[0].values.dtype == np.uint8
        assert ims.fields[0].shape == (784, 592)

    def test_import_from_video_should_import_only_certain_frames(self):
        ims = import_from_video("./tests/res/video.mp4",
                                frame_range=[0, 20],
                                cache_infos=False)
        ims2 = import_from_video("./tests/res/video.mp4",
                                 frame_range=[10, 14],
                                 cache_infos=False)
        assert ims[10] == ims2[0]

    def test_import_from_video_should_use_incr(self):
        ims = import_from_video("./tests/res/video.mp4",
                                incr=10,
                                cache_infos=False)
        ims2 = import_from_video("./tests/res/video.mp4",
                                 frame_range=[0, 20],
                                 cache_infos=False)
        assert ims[1] == ims2[10]

    def test_import_from_video_should_use_nmb_frame_to_import(self):
        ims = import_from_video("./tests/res/video.mp4",
                                nmb_frame_to_import=10,
                                frame_range=[0, 100],
                                cache_infos=False)
        ims2 = import_from_video("./tests/res/video.mp4",
                                 frame_range=[0, 20],
                                 cache_infos=False)
        assert ims[1] == ims2[10]

    def test_import_from_video_should_crop_images(self):
        ims = import_from_video("./tests/res/video.mp4",
                                frame_range=[0, 10],
                                cache_infos=False)
        ims2 = import_from_video("./tests/res/video.mp4",
                                 frame_range=[0, 10],
                                 intervx=[100, 200],
                                 intervy=[100, 200],
                                 cache_infos=False)
        assert np.allclose(ims2[0].values, ims[0].values[100:201, 100:201])


class TestImportFromImages(object):
    """ Done """

    def setup(self):
        sane_parameters()

    def test_import_from_images(self):
        ims = import_from_images("./tests/res/imageset/*",
                                 dx=9.3, dy=.344, dt=0.12,
                                 unit_x="dm", unit_y="nm",
                                 unit_times="")
        assert ims.field_type == Image
        assert ims.unit_times == make_unit('')
        assert np.allclose(ims.times, np.arange(0, len(ims)*0.12, 0.12))
        assert isinstance(ims.fields[0], Image)
        assert ims.fields[0].dx == 9.3
        assert ims.fields[0].dy == .344
        assert ims.fields[0].unit_x == make_unit('dm')
        assert ims.fields[0].unit_y == make_unit('nm')
        assert ims.fields[0].values.dtype == np.uint8
        assert ims.fields[0].shape == (784, 592)


class TestFittingFunctions(object):
    def setup(self):
        sane_parameters()

    def test_circle_from_three_points(self):
        pt1 = [0, 1]
        pt2 = [1, 0]
        pt3 = [0, -1]
        cent, R = circle_from_three_points(pt1, pt2, pt3)
        assert np.allclose(cent, [0, 0])

    def test_fit_circle(self):
        center, R = fit_circle([0, 0, 1], [1, -1, 0])
        assert np.allclose(center, [0, 0])
        assert np.isclose(R, 1)

    def test_fit_ellipse(self):
        xs = [0, 1, 2, -2, -1.5, 0]
        ys = [-1, -0.5, 0, 0, 0.75, 1]
        center, R1, R2, theta = fit_ellipse(xs, ys)
        assert np.allclose(center, [-0.10407845,  0.05203922])
        assert np.isclose(R1, 1.9760989377)
        assert np.isclose(R2, 0.979127582)
        assert np.isclose(theta, 0.0690882025)
