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
import mock
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
    import_from_video
from pyDSA_core import import_from_image
import json
import pytest


class TestImage(object):

    def setup(self):
        sane_parameters()
        self.im = import_from_image("./tests/res/image.bmp",
                                    dx=0.12, dy=0.12,
                                    unit_x="mm",
                                    unit_y="um",
                                    cache_infos=False,
                                    dtype=np.uint8)
        self.basept1 = np.array([13, 27.3*12/14])
        self.basept2 = np.array([83.88, 26.88*12/14])
        self.im.set_baseline(self.basept1, self.basept2)

    def test_eq(self):
        im2 = self.im.copy()
        im3 = self.im.copy()
        assert im2 == self.im
        im2.values[10, 10] = 9999
        assert im2 != self.im
        im3.baseline.pt1 = [3, 4]
        assert im3 != self.im

    def test_set_baseline(self):
        im2 = self.im.copy()
        im2.set_baseline([0, 4], [im2.axe_x[-1], 5])
        assert np.allclose(im2.baseline.pt1, [0, 4])
        assert np.allclose(im2.baseline.pt2, [im2.axe_x[-1], 5])

    def test_scale(self):
        im2 = self.im.copy()
        assert im2 == self.im
        im3 = im2.scale(scalex=2, scaley=0.2, scalev=3, inplace=False)
        assert im2 == self.im
        assert np.isclose(im3.dx, 2*im2.dx)
        assert np.isclose(im3.dy, 0.2*im2.dy)
        assert np.isclose(im3.max, 255)
        assert np.isclose(im3.min, im2.min*3)
        assert np.allclose(im3.baseline.pt1, im2.baseline.pt1*[2, 0.2])
        assert np.allclose(im3.baseline.pt2, im2.baseline.pt2*[2, 0.2])

    def test_infofile(self):
        im = import_from_image("./tests/res/image.bmp",
                               dx=0.12, dy=5.14,
                               unit_x="mm",
                               unit_y="um",
                               cache_infos=True,
                               dtype=np.uint8)
        im.set_baseline(self.basept1, self.basept2)
        # Check the baseline is recorded
        with open(im.infofile_path, 'r') as f:
            infos = json.load(f)
        assert np.allclose(infos['baseline_pt1'], im.baseline.pt1)
        assert np.allclose(infos['baseline_pt2'], im.baseline.pt2)
        # check the scaling is recorded
        dx = im.dx.copy()
        dy = im.dy.copy()
        base1 = im.baseline.pt1.copy()
        base2 = im.baseline.pt2.copy()
        im.scale(scalex=2, scaley=0.2, inplace=True)
        with open(im.infofile_path, 'r') as f:
            infos = json.load(f)
        assert np.allclose(infos['baseline_pt1'], base1*[2, 0.2])
        assert np.allclose(infos['baseline_pt2'], base2*[2, 0.2])
        assert infos['dx'] == dx*2
        assert infos['dy'] == dy*0.2
        os.remove(im.infofile_path)

    def test_edge_detection_canny(self):
        # nothing without baseline
        im2 = self.im.copy()
        im2.baseline = None
        with pytest.raises(Exception):
            im2.edge_detection_canny()
        # test1
        edge = self.im.edge_detection_canny(nmb_edges=1)
        assert np.isclose(np.max(edge.xy[:, 1]), 36.36)
        # test2
        edge2 = self.im.edge_detection_canny(
            threshold1=100,
            threshold2=200,
            base_max_dist=0,
            size_ratio=.2,
            nmb_edges=2,
            smooth_size=4)
        # test3
        with pytest.raises(Exception):
            edge2 = self.im.edge_detection_canny(
                threshold1=100,
                threshold2=200,
                base_max_dist=0,
                size_ratio=.2,
                nmb_edges=1,
                smooth_size=4)

    def test_edge_detection_contour(self):
        # test1
        edge = self.im.edge_detection_contour(
            nmb_edges=2,
            ignored_pixels=2,
            level=0.75,
            size_ratio=0.5)
        assert np.isclose(np.max(edge.xy[:, 1]), 36.46894736)
        # test2
        edge2 = self.im.edge_detection_contour(
            nmb_edges=1,
            ignored_pixels=2,
            level=0.25,
            size_ratio=0.5)
        assert np.isclose(np.max(edge2.xy[:, 1]), 36.211499999)

    @mock.patch('matplotlib.pyplot.show')
    def test_display(self, mocked):
        # Just check that it is not raising an error...
        self.im.display()
        self.im.choose_baseline()
        self.im.scale_interactive()
        self.im.choose_tp_interactive()
