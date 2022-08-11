from setuptools import setup
import sys

__version__ = "1.0"

if sys.version_info < (3, 7):
    sys.exit("TopoNetX requires Python 3.7 or later.")
    
    

from setuptools import find_packages, setup

# Package meta-data.
NAME = "stnets"
DESCRIPTION = "Python module integrating higher order deep learning."
URL = "https://github.com/mhajij/stnets"
VERSION = 0.2
REQUIRED = ["numpy", "torch>=1.9.0", "scipy", "scikit-learn"]


here = path.abspath(path.dirname(__file__))


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=NAME,
    version=VERSION,
    description="TopoNetX : Higher order networks for Python.",
    long_description="TopoNetX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of higher order networks.",
    url=URL,
    download_url=URL,
    license="MIT",
    author="pyt-team Authors",
    contact_email="mustafahajij@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "cell complexes",
        "higher order networks",
        "simplicial complexes",
        "hypergraphs",
        "multiway networks",
        "combinatorial complex",
        "Cellular complexes",
        "CW complex",
    ],
    python_requires=">=3.6",
    install_requires=REQUIRED,    

