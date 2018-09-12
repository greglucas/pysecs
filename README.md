pySECS
======

pySECS is an open source library for calculating Spherical Elementary Current Systems (SECS).

## Features
- Based upon the [scikit-learn](http://scikit-learn.org/) framework with `fit()` and `predict()` methods.
- The only dependency is [NumPy](http://www.numpy.org/). It is built using fast broadcasting techniques to provide highly scalable calculations.

## Examples

Example notebooks can be found in [notebooks/](./notebooks/)

## Install

1. clone the git repository

    ```bash
    $ git clone https://github.com/greglucas/pySECS
    ```

2. Build and install the package

    ```bash
    $ python setup.py build
    $ python setup.py install
    ```

## License
The code is released under an MIT license
[License described in LICENSE.md](./LICENSE.md)

## References
This package has been developed from different publications. Please consider citing the papers
that are relevant to the work you are doing if you are utilizing this code.

### [Original Paper](https://doi.org/10.5636/jgg.49.947)
```
Amm, O. "Ionospheric Elementary Current Systems in Spherical Coordinates and Their Application."
Journal of geomagnestism and geoelectricity 49.7 (1997): 947-955. doi:10.5636/jgg.49.947
```

### [Applications Paper](https://doi.org/10.1186/BF03352247)

```
Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic field continuation
from the ground to the ionosphere using spherical elementary current systems."
Earth, Planets and Space 51.6 (1999): 431-440. doi:10.1186/BF03352247
```

## Problems or Questions?

- [Report an issue using the GitHub issue tracker](http://github.com/greglucas/pySECS/issues)
