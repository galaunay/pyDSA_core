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
        self.edges = self.im.edge_detection_canny()
