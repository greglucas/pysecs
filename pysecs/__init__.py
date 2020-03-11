# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", )
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
        "Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

from .secs import *
__all__ = ["SECS"]
__version__ = '0.0.4'
