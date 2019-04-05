<div align="center">
  <img width=500px" src="https://framagit.org/gabylaunay/pyDSA/raw/master/branding/pyDSA_logo_python_text.svg"><br><br>
</div>

# PyDSA: python3 package for drop shape analysis.

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

## Citing this software

If PyDSA have been usefull for you, please consider citing it:
```
Launay G. pyDSA: Drop shape analysis in Python, 2018-, https://framagit.org/gabylaunay/pyDSA [Online; accessed <today>].
```

bibtex entry:
``` bibtex
@Misc{,
  author =    {Gaby Launay},
  title =     {{pyDSA}: Drop shape analysis in {Python}},
  year =      {2018--},
  url = "https://framagit.org/gabylaunay/pyDSA",
  note = {[Online; accessed <today>]}
}
```
