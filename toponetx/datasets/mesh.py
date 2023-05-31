"""Various examples of named meshes represented as complexes."""

import os
import os.path

from toponetx import CellComplex, CombinatorialComplex, SimplicialComplex

__all__ = ["stanford_bunny"]

#: the absolute path repr the directory containing this file
# DIR = os.path.abspath(os.getcwd())
DIR = os.path.dirname(__file__)


def stanford_bunny(domain="simplicial"):
    """Load the Stanford Bunny mesh as a complex.

    Parameters
    ----------
    domain : str, optional
        The type of complex to load. Supported values are
        "simplicial" and "cellular".
        The default is "simplicial complex".

    Returns
    -------
    SimplicialComplex or CellComplex
        The loaded complex of the specified type.

    Raises
    ------
    ValueError
        If domain is not one of the supported values.
    """
    if domain == "simplicial":
        cpx = SimplicialComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    elif domain == "cellular":
        cpx = CellComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    else:
        raise ValueError("domain must be 'simplicial complex' or 'cellular'")
