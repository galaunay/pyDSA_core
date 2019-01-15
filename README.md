# pyDSA
[![Build Status](https://travis-ci.org/gabylaunay/pyDSA.svg?branch=master)](https://travis-ci.org/gabylaunay/pyDSA)
[![Coverage Status](https://coveralls.io/repos/github/gabylaunay/pyDSA/badge.svg?branch=master)](https://coveralls.io/github/gabylaunay/pyDSA?branch=master)

PyDSA is a python3 package for drop shape analysis.

## Dependencies

pyDSA relies on matplotlib, numpy, scipy, IMTreatment and OpenCV.

## Installation

To install IMTreatment (not on pipy), run:

``pip install 'git+https://framagit.org/gabylaunay/IMTreatment.git#egg=IMTreatment'``.

Then install pyDSA with:

``pip install 'git+https://framagit.org/gabylaunay/pyDSA.git#egg=pyDSA'``.

## Features

pyDSA allow to import videos or images of drops and to get their properties, including

  - Drop edges
  - Contact angles
  - Base radius
  - Triple points (for SLIPS surfaces)

A tutorial is available [here](https://gabylaunay.github.io/Python-cookbook/).
