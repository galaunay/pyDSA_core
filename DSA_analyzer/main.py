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
from DSA_analyzer import Image, DropEdges, import_from_video


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__license__ = ""
__version__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


#==========================================================================
# Create image from video
#==========================================================================
dx = .5
dy = .5
dt = .1

# Set 1
path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
       "data/CAH Sample 2 Test.avi"
frame_inds = [0, np.inf]
incr = 5
pt1 = [0.0, 34.16334950964495]
pt2 = [389.5, 34.16334950964497]

# Set 2
path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
       "data/ridges_case.avi"
frame_inds = [832, np.inf]
incr = 5
pt1 = [0.0, 24.449893541093864]
pt2 = [389.5, 22.5907049969411]

ims = import_from_video(path, dx=dx, dy=dy, unit_x="um", unit_y="um",
                        dt=dt, frame_inds=frame_inds, incr=incr,
                        verbose=True)
display_ind = 10

#==============================================================================
# Crop images
#==============================================================================
ims.crop(intervy=[0, 150], inplace=True)

# #==============================================================================
# # Display video
# #==============================================================================
# ims.display()
# plt.show()


#==============================================================================
# Choosing baseline
#==============================================================================
# ims.choose_baseline(ind_image=50)
# print(ims.baseline.pt1, ims.baseline.pt2)
ims.set_baseline(pt1, pt2)
# ims.display()
# plt.show()


#==============================================================================
# Detect drop edges
#==============================================================================
edges = ims.edge_detection(verbose=True)
# ims.display()
# edges.display()
# plt.show()


#==============================================================================
# Fit the drop edges
#==============================================================================
edges.fit(verbose=True, s=10)
# plt.figure()
# ims.fields[display_ind].display()
# edges.point_sets[display_ind].display()
# plt.show()

#==============================================================================
# Detect triple points
#==============================================================================
edges.detect_triple_points(verbose=True)
# plt.figure()
# ims[display_ind].display()
# edges[display_ind].display()
# plt.show()

#==============================================================================
# Get contact angles
#==============================================================================
edges.compute_contact_angle(smooth=10, verbose=True)

#==============================================================================
# Display the whole thing
#==============================================================================
ims.display()
edges.display()

#==============================================================================
# Get the drop base evolution
#==============================================================================
edges.display_summary()
plt.show()
bug
