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
from pyDSA_core import import_from_image
import mock


class TestTemporalImages(object):

    def setup(self):
        sane_parameters()
        ims = import_from_video("./tests/res/video.mp4",
                                dx=0.12, dy=0.12,
                                dt=0.1,
                                unit_x="km",
                                unit_y="px",
                                unit_t="ms",
                                nmb_frame_to_import=20,
                                dtype=np.uint8,
                                cache_infos=False)
        self.ims = ims

    def test_set_baseline(self):
        self.ims.set_baseline([0., 7.07504895], [93.96, 6.85475044])
        assert np.allclose(self.ims[0].baseline.pt1, [0., 7.07504895])
        assert np.allclose(self.ims[10].baseline.pt2, [93.96, 6.85475044])
        assert np.allclose(self.ims.baseline.pt1, [0., 7.07504895])
        assert np.allclose(self.ims.baseline.pt2, [93.96, 6.85475044])

    def test_edge_detection(self):
        self.ims.set_baseline([0., 7.07504895], [93.96, 6.85475044])
        self.edges = self.ims.edge_detection_canny(nmb_edges=1)
        assert np.isclose(np.max(self.edges.point_sets[0].xy), 70.92)
        assert np.isclose(np.min(self.edges.point_sets[10].xy), 7.08)
        self.edges2 = self.ims.edge_detection_contour(nmb_edges=1)
        assert np.isclose(np.max(self.edges2.point_sets[0].xy), 70.92)
        assert np.isclose(np.min(self.edges2.point_sets[10].xy), 7.199999)

    @mock.patch('matplotlib.pyplot.show')
    def test_display(self, mocked):
        # Just check that it is not raising an error...
        self.ims.display()
