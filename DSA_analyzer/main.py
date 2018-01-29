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
dx = .5
dy = .5
dt = .1
ims = import_from_video(path, dx=dx, dy=dy, unit_x="um", unit_y="um",
                        dt=dt, frame_inds=[80, 375], verbose=True)
                        # dt=dt, frame_inds=[80, 100], verbose=True)
display_ind = 194


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
pt1 = [604.8*dx, 68.6*dy]
pt2 = [157.6*dx, 72.3*dy]
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
plt.figure()
edges.point_sets[display_ind].display_fit()
ims.fields[display_ind].display()

#==============================================================================
# Get contact angles
#==============================================================================
thetas = edges.get_contact_angle(smooth=10)
plt.figure()
edges.point_sets[display_ind].get_contact_angle(verbose=True)
ims.fields[display_ind].display()

#==============================================================================
# Get the drop base evolution
#==============================================================================
bdp = edges.get_drop_base()
ts = np.arange(0, len(ims)*dt, dt)
plt.figure()
plt.plot(ts, bdp[:, 0], label="Contact (left)")
plt.plot(ts, bdp[:, 1], label="Contact (right)")
plt.plot(ts, abs(bdp[:, 1] - bdp[:, 0]), label="Radius")
plt.ylabel('[um]')
plt.legend(loc=2)
ax1 = plt.gca()
ax2 = plt.twinx()
plt.sca(ax2)
plt.plot(ts, thetas[:, 0], label="Angle (left)", ls='--')
plt.plot(ts, thetas[:, 1], label="Angle (right)", ls='--')
plt.axvline(display_ind*dt, ls="--", color="r")
plt.xlabel('Time [s]')
plt.ylabel('[Deg]')
plt.legend(loc=1)
plt.show()
