from setuptools import setup
import sys

__version__ = "1.0"

if sys.version_info < (3, 7):
    sys.exit("TopoNetX requires Python 3.7 or later.")

