"""Spherical Elementary Current System (SECS) package.

This package provides a Python implementation of the Spherical Elementary Current
System (SECS) model. It can be used to interpolate between observations
of the magnetic field at Earth's surface. This can be the case when the magnetic
field is assumed to arise from a current sheet in the ionosphere or within the
surface of the Earth.
"""

import importlib.metadata

from pysecs.secs import *  # noqa


__version__ = importlib.metadata.version("pysecs")
