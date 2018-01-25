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
# Video
#==========================================================================
# Create image from video
path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/"\
        "data/CAH Sample 2 Test.avi"
ims = import_from_video(path, dx=1, dy=1, unit_x="um", unit_y="um",
                        # frame_inds=[80, 380])
                        frame_inds=[80, 90])
ims.crop(intervy=[74.6, 500], inplace=True)
# Edge detection
edges = ims.edge_detection()
# fitting
theta1 = []
theta2 = []
verbose = False
for i in range(len(edges.point_sets)):
    edge = edges.point_sets[i]
    edge = DropEdges(edge.xy, unit_x=edge.unit_x, unit_y=edge.unit_y)
    t1, t2 = edge.get_contact_angle(k=5, verbose=verbose)
    theta1.append(t1)
    theta2.append(t2)
plt.figure()
plt.plot(edges.times, theta1)
plt.plot(edges.times, theta2)
plt.show()

bug

# #==========================================================================
# # Image
# #==========================================================================
# # Create image
# path = "/home/muahah/Postdoc_GSLIPS/180112-Test_DSA_Images/data/Test"\
#        " Sample 2.bmp"
# im = import_from_image(path, dx=1, dy=1, unit_x="um", unit_y="um")
# # set baseline
# pt1 = [604.8, 68.6]
# pt2 = [157.6, 72.3]
# im.set_baseline(pt1, pt2)
# # Simple display
# im.display()
# plt.title('Raw image')
# # make Canny edge detection
# edge_pts = im.edge_detection(verbose=False, inplace=True)
# plt.figure()
# # edge_pts.display()
# edge_pts.display()
# plt.title('Canny edge detection + area selection')
# plt.show()
# # Fit
# raise Exception('Todo')
# spl = spint.UnivariateSpline(xs, ys, k=3)
# order = 5
# zx = np.polyfit(range(len(xs)), xs, order)
# fx = np.poly1d(zx)
# zy = np.polyfit(range(len(ys)), ys, order)
# fy = np.poly1d(zy)
# plt.plot(fx(range(len(xs))), fy(range(len(ys))), "--k")
# plt.show()
