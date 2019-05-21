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
    import_from_video
import mock


class TestDropEdges(object):

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
        self.edges = self.im.edge_detection_canny(nmb_edges=1)

    def test_rotate(self):
        edges2 = self.edges.copy()
        edges3 = edges2.rotate(np.pi/2, inplace=False)
        assert edges2 == self.edges
        assert np.allclose(self.edges.xy[:, 0], edges3.xy[:, 1])
        assert np.allclose(self.edges.xy[:, 1], -edges3.xy[:, 0])

    def test_fit_spline(self):
        fit = self.edges.fit_spline()
        assert np.isclose(fit.fits[0][0](0.5), 34.33066782)
        assert np.isclose(fit.fits[0][1](0.5), 31.99779506)
        assert np.isclose(fit.fits[1][0](0.5), 55.55784286)
        assert np.isclose(fit.fits[1][1](0.5), 32.63825691)

    def test_fit_circle(self):
        fit = self.edges.fit_circle()
        assert np.allclose(fit.fits[0], [45.29229029, 20.7177082])
        assert np.isclose(fit.fits[1], 15.632425588460043)

    def test_fit_circle_with_tp(self):
        fit = self.edges.fit_circle([[0, 30], [0, 30]])
        assert np.allclose(fit.fits[0], [45.29702127, 20.05322552])
        assert np.isclose(fit.fits[1], 16.204308562817413)

    def test_fit_ellipse(self):
        fit = self.edges.fit_ellipse()
        assert np.allclose(fit.fits[0], [45.29482370716114, 22.492606884654457])
        assert np.isclose(fit.fits[1], 15.186512415884058)
        assert np.isclose(fit.fits[2], 13.737885470721212)
        assert np.isclose(fit.fits[3], 0.0008047328743570523)

    def test_fit_ellipse_with_tp(self):
        fit = self.edges.fit_ellipse(triple_pts=[[0, 30], [0, 30]])
        assert np.allclose(fit.fits[0], [45.36841082948874, 22.82189945252258])
        assert np.isclose(fit.fits[1], 15.04754519152597)
        assert np.isclose(fit.fits[2], 13.40293764551634)
        assert np.isclose(fit.fits[3], -0.028131229179985305)

    def test_fit_circles(self):
        fit = self.edges.fit_circles([[0, 30], [0, 31]])
        print(fit.fits)
        assert np.allclose(fit.fits[0][0],
                           [45.30343097, 20.01766113])
        assert np.allclose(fit.fits[1][0],
                           [24.7037305265706, 29.990566677882825])
        assert np.allclose(fit.fits[2][0],
                           [67.18009092332332, 31.678309466667272])
        assert np.isclose(fit.fits[0][1],
                          16.23680941556352)
        assert np.isclose(fit.fits[1][1],
                          6.650010004658293)
        assert np.isclose(fit.fits[2][1],
                          8.553490510623169)

    @mock.patch('matplotlib.pyplot.show')
    def test_display(self, mocked):
        # Just check that it is not raising an error...
        self.edges.display()
