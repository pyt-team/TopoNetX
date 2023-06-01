"""Various examples of named meshes represented as complexes."""

import os
import os.path
import urllib.request

import numpy as np

from toponetx import CellComplex, CombinatorialComplex, SimplicialComplex

__all__ = ["stanford_bunny", "shrec_16"]

#: the absolute path repr the directory containing this file
# DIR = os.path.abspath(os.getcwd())
DIR = os.path.dirname(__file__)


def stanford_bunny(complex_type="simplicial complex"):
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
    if complex_type == "simplicial complex":
        cpx = SimplicialComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    elif complex_type == "cell complex":
        cpx = CellComplex.load_mesh(DIR + "/bunny.obj")
        return cpx
    else:
        raise ValueError("cmplex_type must be 'simplicial complex' or 'cell complex'")


def shrec_16():
    """Load training/testing shrec 16 datasets".

    Returns
    -------
    tuple of length 2 npz files
        The npz files store the training/testing complexes of shrec 16 dataset along
        with thier nodes, edges and faces features.

    Notes
    -----
    Each npz file stores 5 keys
    "complexes",label","node_feat","edge_feat" and "face_feat".

    Example
    -------
    shrec_training, shrec_testing = shrec_16()

    training_complexes = shrec_training["complexes"]
    training_labels = shrec_training["label"]
    node_feat = shrec_training["node_feat"]
    edge_feat = shrec_training["edge_feat"]
    face_feat = shrec_training["face_feat"]
    """
    if os.path.isfile(DIR + "/shrec_training.npz") and os.path.isfile(
        DIR + "/shrec_testing.npz"
    ):
        print("Loading dataset..")
        shrec_training = np.load(DIR + "/shrec_training.npz", allow_pickle=True)
        shrec_testing = np.load(DIR + "/shrec_testing.npz", allow_pickle=True)
        print("done!")
        return shrec_training, shrec_testing
    else:
        raise ValueError("Files are on the HD, fail to load the dataset.")
