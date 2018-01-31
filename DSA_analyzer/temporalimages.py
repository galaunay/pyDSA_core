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

import numpy as np
import matplotlib.pyplot as plt
from IMTreatment import TemporalScalarFields, TemporalPoints
from IMTreatment.utils import ProgressCounter
from .image import Image
from .temporaldropedges import TemporalDropEdges
import scipy.optimize as spopt



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
        for i in range(len(self.fields)):
            self.fields[i].set_baseline(pt1=pt1, pt2=pt2)
        self.baseline = self.fields[0].baseline

    def choose_baseline(self, ind_image=0):
        """
        Choose baseline position interactively.

        Select as many points as you want (click on points to delete them),
        and close the figure when done.
        """
        self.baseline = self.fields[ind_image].choose_baseline()
        for i in range(len(self.fields)):
            self.fields[i].baseline = self.baseline
        return self.baseline

    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        if self.baseline is not None:
            self.baseline.display()

    def edge_detection(self, threshold1=None, threshold2=None, verbose=False):
        """
        Make Canny edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        """
        pts = TemporalDropEdges()
        pts.baseline = self.baseline
        if verbose:
            pg = ProgressCounter("Detecting drop edges", "Done",
                                 len(self.fields), 'images', 5)
        for i in range(len(self.fields)):
            pt = self.fields[i].edge_detection(threshold1=threshold1,
                                               threshold2=threshold2)
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
            if verbose:
                pg.print_progress()
        return pts
