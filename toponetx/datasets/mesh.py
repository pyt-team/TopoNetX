"""Various examples of named meshes represented as complexes."""

import os
import os.path
import zipfile

import numpy as np
import wget

from toponetx import CellComplex, CombinatorialComplex, SimplicialComplex

__all__ = ["stanford_bunny", "shrec_16"]

#: the absolute path repr the directory containing this file
# DIR = os.path.abspath(os.getcwd())
DIR = os.path.dirname(__file__)


def stanford_bunny(complex_type="simplicial complex"):
    """Load the Stanford Bunny mesh as a complex.

    Parameters
    ----------
    complex_type : str, optional
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
        If complex_type is not one of the supported values.
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
        with their nodes, edges and faces features.

    Notes
    -----
    Each npz file stores 5 keys
    "complexes",label","node_feat","edge_feat" and "face_feat".

    Example
    -------
    >>> shrec_training, shrec_testing = shrec16()
    >>> # training dataset
    >>> training_complexes = shrec_training["complexes"]
    >>> training_labels = shrec_training["label"]
    >>> training_node_feat = shrec_training["node_feat"]
    >>> training_edge_feat = shrec_training["edge_feat"]
    >>> training_face_feat = shrec_training["face_feat"]


    >>> # testing dataset
    >>> testing_complexes = shrec_testing["complexes"]
    >>> testing_labels = shrec_testing["label"]
    >>> testing_node_feat = shrec_testing["node_feat"]
    >>> testing_edge_feat = shrec_testing["edge_feat"]
    >>> testing_face_feat = shrec_testing["face_feat"]

    """
    url = "https://github.com/mhajij/shrec_16/raw/main/shrec.zip"

    if not os.path.isfile(DIR + "/shrec.zip"):
        print("downloading dataset...\n")
        wget.download(url, DIR + "/shrec.zip")
    print("unzipping the files...\n")
    with zipfile.ZipFile(DIR + "/shrec.zip", "r") as zip_ref:
        zip_ref.extractall(DIR)
    # if not os.path.isfile(DIR + "/shrec_testing.npz"):
    #    wget.download(url_testing, DIR + "/shrec_testing.npz")
    print("done!")

    if os.path.isfile(DIR + "/shrec_training.npz") and os.path.isfile(
        DIR + "/shrec_testing.npz"
    ):
        print("Loading dataset...\n")
        shrec_training = np.load(DIR + "/shrec_training.npz", allow_pickle=True)
        shrec_testing = np.load(DIR + "/shrec_testing.npz", allow_pickle=True)
        print("done!")
        return shrec_training, shrec_testing
    else:
        raise ValueError(
            f"Files couldn't be found in folder {DIR}, fail to load the dataset."
        )
