# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA
# Version: 0.1

# This file is part of pyDSA

# pyDSA is distributed in the hope that it will be useful,
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

import numpy as np
from helper import sane_parameters
from pyDSA.helpers import import_from_image, import_from_images, \
    import_from_video
from pyDSA import import_from_image
from IMTreatment.utils import make_unit
import json
import pytest


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
        assert np.isclose(fit.fits[0](30), 32.60263012)
        assert np.isclose(fit.fits[1](30), 58.02943488)

    def test_fit_circle(self):
        fit = self.edges.fit_circle()
        assert np.allclose(fit.fits[0], [45.31111377, 44.12578495])
        assert np.isclose(fit.fits[1], 18.4417563803888)

    def test_fit_ellipse(self):
        fit = self.edges.fit_ellipse()
        assert np.allclose(fit.fits[0], [45.28924487844782, 22.44230674570451])
        assert np.isclose(fit.fits[1], 15.195167689658016)
        assert np.isclose(fit.fits[2], 13.803113982495326)
        assert np.isclose(fit.fits[3], 0.006127974297680402)

    def test_fit_circles(self):
        fit = self.edges.fit_circles([[0, 30], [0, 31]])
        print(fit.fits)
        assert np.allclose(fit.fits[0][0],
                           [45.30608443, 20.12977316])
        assert np.allclose(fit.fits[1][0],
                           [24.78903693094256, 29.925640074689813])
        assert np.allclose(fit.fits[2][0],
                           [66.87895666665062, 31.24366199918333])
        assert np.isclose(fit.fits[0][1], 16.150099586208466)
        assert np.isclose(fit.fits[1][1], 6.585516673258724)
        assert np.isclose(fit.fits[2][1], 8.117313580746385)
