# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/DSA_analyzer
# Version: 0.1

# This file is part of DSA_analyzer

# DSA_analyzer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import matplotlib.pyplot as plt
from IMTreatment import TemporalScalarFields, TemporalPoints
from .image import Image


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class TemporalImages(TemporalScalarFields):
    def __init__(self):
        super().__init__()
        self.baseline = None
        self.field_type = Image

    def set_baseline(self, pt1, pt2):
        """
        Set the drop baseline.
        """
        self.baseline = [pt1, pt2]
        for i in range(len(self.fields)):
            self.fields[i].set_baseline(pt1=pt1, pt2=pt2)

    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        if self.baseline is not None:
            pt1, pt2 = self.baseline
            plt.plot([pt1[0], pt2[0]],
                     [pt1[1], pt2[1]],
                     color='g',
                     ls='none',
                     marker='o')
            plt.plot([pt1[0], pt2[0]],
                     [pt1[1], pt2[1]],
                     color='g',
                     ls='-')

    def edge_detection(self, threshold1=None, threshold2=None, verbose=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        pts = TemporalPoints()
        for i in range(len(self.fields)):
            pt = self.fields[i].edge_detection(threshold1=threshold1,
                                               threshold2=threshold2)
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
        return pts
