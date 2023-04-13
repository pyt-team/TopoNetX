from codecs import open
from os import path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "TopoNetX"
DESCRIPTION = "TNX provides classes and methods for modeling simplicial, cellular, CW and combinatorial complexes."
URL = "https://github.com/pyt-team/TopoNetX"
VERSION = 0.1

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "gudhi",
    "decorator",
    "networkx",
    "hypernetx",
    "numpy",
    "pre-commit",
    "scipy",
    "trimesh",
]

full_requires = []

test_requires = ["pytest", "pytest-cov", "jupyter"]

dev_requires = test_requires + [
    "pre-commit",
    "flake8",
    "yapf",
    "black==22.6.0",
    "black[jupyter]",
    "isort==5.10.1",
    "coverage",
]

setup(
    name=NAME,
    version=VERSION,
    description="TNX provides classes and methods for modeling simplicial, cellular, CW and combinatorial complexes.",
    long_description="The TNX library provides classes and methods for modeling the entities and relationships found in higher order networks such as simplicial, cellular, CW and combinatorial complexes.",
    url=URL,
    download_url=URL,
    license="MIT",
    author="PyT-Team Authors",
    contact_email="mustafahajij@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "higher order networks",
        "Simplicial Complexes",
        "Simplicial Complex Neural Networks",
        "Cell Complex Neural Networks",
        "Cell Complex Networks",
        "Cubical Complexes",
        "Cellular complexes",
        "Cell Complex",
        "Topological Data Analysis",
        "Topological Machine Learning",
        "Topological Deep Learning",
        "Combinatorial complexes",
        "CW Complex",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "full": full_requires,
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
)
