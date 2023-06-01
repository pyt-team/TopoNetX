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


def shrec_16(size="full"):
    """Load training/testing shrec 16 datasets".

    Parameters
    ----------
    size : str, optional
        options are "full" or "small"

    Returns
    -------
    tuple of length 2 npz files
        The npz files store the training/testing complexes of shrec 16 dataset along
        with their nodes, edges and faces features.

    Notes
    -----
    Each npz file stores 5 keys
    "complexes",label","node_feat","edge_feat" and "face_feat".
    complex : stores the simplicial complex of the mesh
    label :  stores the label of the mesh
    node_feat : stores position and normal of the each node in the mesh
    edge_feat : stores 10 edge features: diheral angle, edge span, 2 edge angle in the triangle, 6 edge ratios.
    face_feat : face area, face normal, face angle

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
    url_small = "https://github.com/mhajij/shrec_16/raw/main/small_shrec.zip"
    if size == "full":
        if not os.path.isfile(DIR + "/shrec.zip"):
            print("downloading dataset...\n")
            wget.download(url, DIR + "/shrec.zip")
        print("unzipping the files...\n")
        with zipfile.ZipFile(DIR + "/shrec.zip", "r") as zip_ref:
            zip_ref.extractall(DIR)
        print("done!")
    elif size == "small":
        if not os.path.isfile(DIR + "/small_shrec.zip"):
            print("downloading dataset...\n")
            wget.download(url_small, DIR + "/small_shrec.zip")
        print("unzipping the files...\n")
        with zipfile.ZipFile(DIR + "/small_shrec.zip", "r") as zip_ref:
            zip_ref.extractall(DIR)
    else:
        raise ValueError(f"size must be 'full' or 'small' got {size}.")
    if size == "full":
        training = DIR + "/shrec_training.npz"
        testing = DIR + "/shrec_testing.npz"

    elif size == "small":
        training = DIR + "/small_shrec_training.npz"
        testing = DIR + "/small_shrec_testing.npz"

    if os.path.isfile(training) and os.path.isfile(testing):
        print("Loading dataset...\n")
        shrec_training = np.load(training, allow_pickle=True)
        shrec_testing = np.load(testing, allow_pickle=True)
        print("done!")
        return shrec_training, shrec_testing
    else:
        raise ValueError(
            f"Files couldn't be found in folder {DIR}, fail to load the dataset."
        )
