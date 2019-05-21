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

import numpy as np
from helper import sane_parameters
from pyDSA_core.helpers import import_from_image, import_from_images, \
    import_from_video
from pyDSA_core.baseline import Baseline
from IMTreatment.utils import make_unit
import json
import pytest


class TestBaseline(object):

    def setup(self):
        sane_parameters()
        self.blpt1 = np.array([13, 27.3*12/14])
        self.blpt2 = np.array([83.88, 26.88*12/14])
        self.bl = Baseline([self.blpt1, self.blpt2])

    def test_init_from_2_pts(self):
        bl = Baseline([self.blpt1, self.blpt2])
        assert np.allclose(bl.pt1, self.blpt1)
        assert np.allclose(bl.pt2, self.blpt2)

    def test_init_from_more_pts(self):
        bl = Baseline([self.blpt1,
                       self.blpt1+[1.1, 0],
                       self.blpt1+[0.9, 0],
                       self.blpt1+[0, 1.1],
                       self.blpt1+[0, 0.9],
                       self.blpt2,
                       self.blpt2+[1.1, 0],
                       self.blpt2+[0.9, 0],
                       self.blpt2+[0, 1.1],
                       self.blpt2+[0, 0.9]])
        assert np.allclose(bl.pt1, [13, 23.8065])
        assert np.allclose(bl.pt2, [84.98, 23.4318774])

    def test_init_with_boundaries(self):
        bl = Baseline([self.blpt1, self.blpt2],
                      xmin=0, xmax=100)
        assert bl.pt1[0] == 0
        assert bl.pt2[0] == 100
        assert bl.pt1[1] != self.blpt1[1]
        assert bl.pt2[1] != self.blpt2[1]

    def test_eq(self):
        bl = Baseline([self.blpt1, self.blpt2])
        bl2 = bl.copy()
        assert bl2 == bl
        bl2.pt1 = [0, 1]
        assert bl2 != bl
        bl3 = bl.copy()
        assert bl3 == bl
        bl3.pt2 = [0, 1]
        assert bl3 != bl

    def test_get_baseline_fun(self):
        fun = self.bl.get_baseline_fun()
        assert np.isclose(fun(self.blpt1[0]), self.blpt1[1])
        assert np.isclose(fun(self.blpt2[0]), self.blpt2[1])
        fun2 = self.bl.get_baseline_fun(along_y=True)
        assert np.isclose(fun2(self.blpt1[1]), self.blpt1[0])
        assert np.isclose(fun2(self.blpt2[1]), self.blpt2[0])

    def test_get_projection_to_baseline(self):
        pt2 = self.bl.get_projection_to_baseline([40, 100])
        assert np.allclose(pt2, [39.61026163, 23.2648463])

    def test_get_distance_to_baseline(self):
        dist = self.bl.get_distance_to_baseline([40, 100])
        assert np.allclose(dist, 76.7361434345631)

    def test_scale(self):
        pass
        # Done in Image tests

    def test_set_origin(self):
        bl2 = self.bl.copy()
        bl2.set_origin(20, 40)
        assert np.allclose(bl2.pt1, self.blpt1 - [20, 40])
        assert np.allclose(bl2.pt2, self.blpt2 - [20, 40])

    def test_rotate(self):
        bl2 = self.bl.copy()
        bl2.rotate(np.pi/2)
        assert np.allclose(bl2.pt1, [-self.blpt1[1], self.blpt1[0]])
        assert np.allclose(bl2.pt2, [-self.blpt2[1], self.blpt2[0]])
