# pyDSA
[![Build Status](https://travis-ci.org/gabylaunay/pyDSA.svg?branch=master)](https://travis-ci.org/gabylaunay/pyDSA)
[![Coverage Status](https://coveralls.io/repos/github/gabylaunay/pyDSA/badge.svg?branch=master)](https://coveralls.io/github/gabylaunay/pyDSA?branch=master)

PyDSA is a python3 package for drop shape analysis.

## Dependencies

pyDSA relies on matplotlib, numpy, scipy, IMTreatment and OpenCV.

## Installation

Download and install [IMTreatment](https://framagit.org/gabylaunay/IMTreatment).
The other dependencies will be installed automatically during the package installation.

Download pyDSA and install it with:
```bash
python setup.py install
```

## Features

pyDSA allow to import videos or images of drops and to get their properties, including

  - Drop edges
  - Contact angles
  - Base radius
  - Triple points (for SLIPS surfaces)

A tutorial is available [here](https://gabylaunay.github.io/Python-cookbook/).
