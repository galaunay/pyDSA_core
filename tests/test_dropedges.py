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
        assert np.isclose(fit.fits[0][0](0.5), 34.31274988)
        assert np.isclose(fit.fits[0][1](0.5), 31.980125)
        assert np.isclose(fit.fits[1][0](0.5), 55.56954598)
        assert np.isclose(fit.fits[1][1](0.5), 32.6269985)

    def test_fit_circle(self):
        fit = self.edges.fit_circle()
        assert np.allclose(fit.fits[0], [45.29168927, 20.71102873])
        assert np.isclose(fit.fits[1], 15.637520870474857)

    def test_fit_ellipse(self):
        fit = self.edges.fit_ellipse()
        assert np.allclose(fit.fits[0], [45.29323007783934, 22.531376233580968])
        assert np.isclose(fit.fits[1], 15.176520926201963)
        assert np.isclose(fit.fits[2], 13.69615877921492)
        assert np.isclose(fit.fits[3], 0.002431001542474775)

    def test_fit_circles(self):
        fit = self.edges.fit_circles([[0, 30], [0, 31]])
        assert np.allclose(fit.fits[0][0],
                           [45.30486171, 20.00029723])
        assert np.allclose(fit.fits[1][0],
                           [24.603718004843614, 30.13988575468891])
        assert np.allclose(fit.fits[2][0],
                           [67.15249900266305, 31.61664272423973])
        assert np.isclose(fit.fits[0][1], 16.252178998805576)
        assert np.isclose(fit.fits[1][1], 6.798821117200337)
        assert np.isclose(fit.fits[2][1], 8.491683628686477)
