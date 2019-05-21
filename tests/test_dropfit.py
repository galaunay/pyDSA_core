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


class TestDropFit(object):

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
        self.sfit = self.edges.fit_spline()
        self.cfit = self.edges.fit_circle()
        self.efit = self.edges.fit_ellipse()
        self.tefit = self.edges.fit_ellipses()
        self.csfit = self.edges.fit_circles([[0, 30], [0, 31]])
        self.pfit = self.edges.fit_polyline()

    def test_compute_contact_angle(self):
        self.sfit.compute_contact_angle()
        assert np.allclose(self.sfit.thetas, [85.12249622, 94.08458956])
        self.cfit.compute_contact_angle()
        assert np.allclose(self.cfit.thetas, [80.72975961, 99.27024039])
        self.efit.compute_contact_angle()
        assert np.allclose(self.efit.thetas, [86.49734474, 93.35350819])
        self.tefit.compute_contact_angle()
        assert np.allclose(self.tefit.thetas, [86.35520992, 92.90359146])
        self.csfit.compute_contact_angle()
        assert np.allclose(self.csfit.thetas, [78.56794776, 101.43205224])
        self.pfit.compute_contact_angle()
        assert np.allclose(self.pfit.thetas, [85.12249622, 94.08458956])

    @mock.patch('matplotlib.pyplot.show')
    def test_display(self, mocked):
        # Just check that it is not raising an error...
        self.sfit.display()
        self.cfit.display()
        self.efit.display()
        self.csfit.display()
        self.pfit.display()
