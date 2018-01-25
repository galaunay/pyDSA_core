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
path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
       "data/CAH Sample 2 Test.avi"
ims = import_from_video(path, dx=1, dy=1, unit_x="um", unit_y="um",
                        # frame_inds=[80, 375], verbose=True)
                        frame_inds=[80, 100], verbose=True)
ims.crop(intervy=[0, 421], inplace=True)


#==============================================================================
# Display video
#==============================================================================
# ims.display()
# plt.show()


#==============================================================================
# Choosing baseline
#==============================================================================
# ims.choose_baseline()
# ims.display()
# plt.show()
pt1 = [604.8, 68.6]
pt2 = [157.6, 72.3]
ims.set_baseline(pt1, pt2)


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
edges.fit(verbose=True)
edges.point_sets[1].display_fit()


# #==============================================================================
# # Get the drop base evolution
# #==============================================================================
# bdp = edges.get_drop_base()
# # plt.figure()
# # plt.plot(bdp[:, 0])
# # plt.plot(bdp[:, 1])
# # plt.show()
