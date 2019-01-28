# pyDSA
[![pipeline status](https://framagit.org/gabylaunay/pyDSA/badges/master/pipeline.svg)](https://framagit.org/gabylaunay/pyDSA/commits/master)
[![coverage report](https://framagit.org/gabylaunay/pyDSA/badges/master/coverage.svg)](https://framagit.org/gabylaunay/pyDSA/commits/master)

PyDSA is a python3 package for drop shape analysis.

## Features

PyDSA allow to import videos and images of droplets and to get their properties, including

  - Drop edges positions
  - Contact angles
  - Contact angle hysteresis
  - Radius
  - Volume
  - Triple point positions (for SLIPS surfaces)
  - ...

A Graphical interface is also available [here](https://framagit.org/gabylaunay/pyDSAqt5).

A tutorial is available [here](https://gabylaunay.github.io/Python-cookbook/).

## Dependencies

pyDSA relies on matplotlib, numpy, scipy, IMTreatment and OpenCV.

## Installation

### Linux

You will need git to be installed.

To install IMTreatment (not on pipy), run:

``pip install 'git+https://framagit.org/gabylaunay/IMTreatment.git#egg=IMTreatment'``.

Then install pyDSA with:

``pip install 'git+https://framagit.org/gabylaunay/pyDSA.git#egg=pyDSA'``.

### Anaconda

Install git with:

``conda install git``

Install the dependencies:

``pip install git+https://framagit.org/gabylaunay/IMTreatment.git#egg=IMTreatment``.

And finally install pyDSA:

``pip install git+https://framagit.org/gabylaunay/pyDSA.git#egg=pyDSA``.


