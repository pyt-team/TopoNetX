"""Various examples of named meshes represented as complexes."""

import os
import os.path

from toponetx import CellComplex, CombinatorialComplex, SimplicialComplex

__all__ = ["stanford_bunny"]

#: the absolute path repr the directory containing this file
# DIR = os.path.abspath(os.getcwd())
DIR = os.path.dirname(__file__)


def stanford_bunny(cmplex_type="simplicial complex"):
    """Load the Stanford Bunny mesh as a complex.

    Parameters
    ----------
    cmplex_type : str, optional
        The type of complex to load. Supported values are
        "simplicial complex" and "cell complex".
        The default is "simplicial complex".

    Returns
    -------
    SimplicialComplex or CellComplex
        The loaded complex of the specified type.

    Raises
    ------
    ValueError
        If cmplex_type is not one of the supported values.
    """
    if cmplex_type == "simplicial complex":
        cpx = SimplicialComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    elif cmplex_type == "cell complex":
        cpx = CellComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    else:
        raise ValueError("cmplex_type must be 'simplicial complex' or 'cell complex'")
