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

from setuptools import setup, find_packages

setup(
    name='DSA_analyzer',
    version='0.1',
    description='Tools to analyze DSA drop images',
    author='Gaby Launay',
    author_email='gaby.launay@tutanota.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='DSA drop shape contact angle hysteresis',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'samples']),
    install_requires=['numpy', 'matplotlib'],
    extras_require={},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
