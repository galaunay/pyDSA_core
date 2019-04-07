<div align="center">
  <img width=500px" src="https://framagit.org/gabylaunay/pyDSA_core/raw/master/branding/pyDSA_logo_python_text.svg"><br><br>
</div>

# PyDSA_core: python3 package for drop shape analysis.

## Features

PyDSA_core allow to import videos and images of droplets and to get their properties, including

  - Drop edges positions
  - Contact angles
  - Contact angle hysteresis
  - Radius
  - Volume
  - Triple point positions (for SLIPS surfaces)
  - ...

A Graphical interface is also available [here](https://framagit.org/gabylaunay/pyDSA_gui).

A tutorial is available [here](https://gabylaunay.github.io/Python-cookbook/).

## Dependencies

pyDSA_core relies on matplotlib, numpy, scipy, IMTreatment and OpenCV.

## Installation

### From Pypi

``pip install pydsa_core``

### Manually (Linux)

You will need git to be installed.

``pip install 'git+https://framagit.org/gabylaunay/pyDSA_core.git#egg=pyDSA_core'``.

### Manually (Anaconda)

Install git with:

``conda install git``

And install pyDSA_core:

``pip install git+https://framagit.org/gabylaunay/pyDSA_core.git#egg=pyDSA_core``.

## Citing this software

If PyDSA_core have been usefull for you, please consider citing it:
```
Launay G. pyDSA_core: Drop shape analysis in Python, 2018-, https://framagit.org/gabylaunay/pyDSA_core [Online; accessed <today>].
```

bibtex entry:
``` bibtex
@Misc{,
  author =    {Gaby Launay},
  title =     {{pyDSA_core}: Drop shape analysis in {Python}},
  year =      {2018--},
  url = "https://framagit.org/gabylaunay/pyDSA_core",
  note = {[Online; accessed <today>]}
}
```
