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
import IMTreatment.plotlib as pplt
from .image import Image
from .baseline import Baseline
from .temporaldropedges import TemporalDropEdges
from .dropedges import DropEdges



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

    def set_evolving_baseline(self, baseline1, baseline2):
        """
        Set a linearly evolving baseline.
        """
        base1_pt1 = np.asarray(baseline1[0])
        base1_pt2 = np.asarray(baseline1[1])
        base2_pt1 = np.asarray(baseline2[0])
        base2_pt2 = np.asarray(baseline2[1])
        imax = len(self.fields) - 1
        for i in range(len(self.fields)):
            ratio = i/imax
            pt1 = base1_pt1*(1 - ratio) + base2_pt1*ratio
            pt2 = base1_pt2*(1 - ratio) + base2_pt2*ratio
            self.fields[i].set_baseline(pt1, pt2)
        self.baseline = "evolving"

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
        if isinstance(self.baseline, Baseline):
            self.baseline.display()
        elif self.baseline == "evolving":
            # Display evolving baseline
            xs = []
            ys = []
            for field in self.fields:
                bl = field.baseline
                xs.append([bl.pt1[0], bl.pt2[0]])
                ys.append([bl.pt1[1], bl.pt2[1]])
            db = pplt.Displayer(xs, ys, color=self[0].colors[0])
            pplt.ButtonManager(db)
        else:
            pass

    def edge_detection(self, threshold1=None, threshold2=None,
                       base_max_dist=15, size_ratio=.5,
                       keep_exterior=True, verbose=False):
        """
        Perform edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        base_max_dist: integers
            Maximal distance (in pixel) between the baseline and
            the beginning of the drop edge (default to 15).
        size_ratio: number
            Minimum size of edges, regarding the bigger detected one
            (default to 0.5).
        keep_exterior: boolean
            If True (default), only keep the exterior edges.
        """
        pts = TemporalDropEdges()
        pts.baseline = self.baseline
        if verbose:
            pg = ProgressCounter("Detecting drop edges", "Done",
                                 len(self.fields), 'images', 5)
        for i in range(len(self.fields)):
            try:
                pt = self.fields[i].edge_detection(threshold1=threshold1,
                                                   threshold2=threshold2)
            except Exception:
                pt = DropEdges(xy=[], unit_x=self.unit_x, unit_y=self.unit_y,
                               baseline=self.baseline, dx=self.dx, dy=self.dy)
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
            if verbose:
                pg.print_progress()
        return pts
