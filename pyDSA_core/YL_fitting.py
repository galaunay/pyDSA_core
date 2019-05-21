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


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spopt
from IMTreatment import Profile


def fit_YL(s, x, z, delta_rho, gamma_0=40e-3):
    """
    Fit the Young-Laplace equation to the given points, returning the
    surface tension.

    Parameters
    ----------
    s, x, z: arrays of numbers
        Parametrized interface of the pendant drop.
        s = 0 should correspond to the drop apex
    delta_rho: number
        Difference between the two fluid densities
    gamma_0: number
        Guess for the surface tension (helps for numerical convergence)

    Returns
    -------
    gamma: number
        Surface tension.

    Notes
    -----
    Inspired by https://www.dataphysics-instruments.com/knowledge/
    understanding-interfaces/pendant-drop-method/
    """
    # sort (just in case)
    indsort = np.argsort(s)
    s = s[indsort]
    x = x[indsort]
    z = z[indsort]
    # Angle between the horizontal and the local tangent of the surface
    theta1 = np.arcsin(np.gradient(x, s))
    theta2 = np.arccos(np.gradient(z, s))
    filter_theta1 = np.isnan(theta1)
    filter_theta2 = np.isnan(theta2)
    theta1[filter_theta1] = theta2[filter_theta1]
    theta2[filter_theta2] = theta1[filter_theta2]
    theta = np.mean([theta1, theta2], axis=0)
    theta[np.isnan(theta)] = 0
    print(theta1)
    print(theta2)
    print(theta)
    # Radius of curvature at the apex
    zp = np.gradient(z[0:5], s[0:5])
    zpp = np.gradient(zp, s[0:5])[0]
    zp = zp[0]
    xp = np.gradient(x[0:5], s[0:5])
    xpp = np.gradient(xp, s[0:5])[0]
    xp = xp[0]
    R0 = np.abs((xp**2 + zp**2)**(3/2)/(xp*zpp - zp*xpp))
    #
    thetap = np.gradient(theta, s)[1::]
    K1 = -np.sin(theta[1::])/x[1::] + 2/R0
    g = 9.81
    def minfun(gamma):
        return thetap - K1 - np.abs(delta_rho*g*z[1::]/gamma)
    gamma, res = spopt.leastsq(minfun, (gamma_0, ))
    if res > 4:
        raise Exception(res)
    return gamma




s = np.linspace(0, 1, 100)
x = s
z = 1 - (1 - s**2)**.5
dists = ((x[1::] - x[0:-1])**2 + (z[1::] - z[0:-1])**2)**.5
cumdists = np.cumsum(np.concatenate(([0], dists)))
s = cumdists/cumdists[-1]
x = Profile(s, x)
x.evenly_space(inplace=True)
z = Profile(s, z)
z.evenly_space(inplace=True)
s = z.x
z = z.y
x = x.y
plt.figure()
plt.plot(s, x, 'o')
plt.figure()
plt.plot(s, z, 'o')
plt.figure()
plt.plot(x, z, 'o')

gamma = fit_YL(s, x, z, 1000 - 1.25)
