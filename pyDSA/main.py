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
set_ind = 1


if set_ind == 1:
    # Set 1
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
        "data/CAH Sample 2 Test.avi"
    frame_inds = [0, np.inf]
    incr = 5
    pt1 = [0.0, 34.16334950964495]
    pt2 = [389.5, 34.16334950964497]
    cropy = [0, 180]
    cropx = [0, np.inf]
elif set_ind == 2:
    # Set 2
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
        "data/ridges_case.avi"
    frame_inds = [832, np.inf]
    incr = 5
    pt1 = [0.0, 24.449893541093864]
    pt2 = [389.5, 22.5907049969411]
    cropy = [0, np.inf]
    cropx = [0, np.inf]
elif set_ind == 3:
    # Set 3
    path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
        "data/CAH_2.avi"
    frame_inds = [0, np.inf]
    incr = 1
    pt1 = [0.0, 35.24610550093472]
    pt2 = [389.5, 34.17310274611377]
    cropy = [20, 84]
    cropx = [60, 220]
else:
    raise ValueError('Unknown set')

ims = import_from_video(path, dx=dx, dy=dy, unit_x="um", unit_y="um",
                        dt=dt, frame_inds=frame_inds, incr=incr,
                        verbose=True)

#==============================================================================
# Crop images
#==============================================================================
ims.crop(intervy=cropy, intervx=cropx, inplace=True)

# #==============================================================================
# # Display video
# #==============================================================================
# ims.display()
# plt.show()


#==============================================================================
# Choosing baseline
#==============================================================================
# ims.choose_baseline(ind_image=10)
# print(ims.baseline.pt1, ims.baseline.pt2)
ims.set_baseline(pt1, pt2)
# ims.display()
# plt.show()

# #==============================================================================
# # TMP
# #==============================================================================
# im = ims[120]
# im.edge_detection(verbose=True, debug=True, base_max_dist=20)
# plt.show()
# bug

#==============================================================================
# Detect drop edges
#==============================================================================
edges = ims.edge_detection(verbose=True)
# ims.display()
# edges.display()
# plt.show()

#==============================================================================
# TMP
#==============================================================================
edge = edges[0]
edge.fit_ellipse()
plt.show()
bug

#==============================================================================
# Fit the drop edges
#==============================================================================
edges.fit(verbose=True, s=10)
# plt.figure()
# ims.display()
# edges.display()
# plt.show()


#==============================================================================
# Detect triple points
#==============================================================================
edges.detect_triple_points(smooth=1, verbose=True)
# plt.figure()
# ims.display()
# edges.display()
# plt.show()

#==============================================================================
# Get contact angles
#==============================================================================
edges.compute_contact_angle(smooth=.5, verbose=True)

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
