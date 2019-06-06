# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA_core


# This file is part of pyDSA_core

# pyDSA_core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name='pyDSA_core',
    version='1.2.2',
    description='Python Drop Shape Analyzer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://framagit.org/gabylaunay/pyDSA_core',
    author='Gaby Launay',
    author_email='gaby.launay@protonmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='DSA drop shape contact angle hysteresis',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'samples']),
    install_requires=['numpy', 'matplotlib', 'opencv-python', 'scipy',
                      'imageio', 'scikit-image', 'IMTreatment==1.2.0'],
    extras_require={},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
