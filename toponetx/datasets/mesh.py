"""Various examples of named meshes represented as complexes."""

import os
import os.path
import zipfile
from typing import Literal, Union, overload

import numpy as np
import wget

from toponetx import CellComplex, SimplicialComplex

__all__ = ["stanford_bunny", "shrec_16"]

DIR = os.path.dirname(__file__)
DS_MAP = {
    "full": ("shrec", "https://github.com/mhajij/shrec_16/raw/main/shrec.zip"),
    "small": (
        "small_shrec",
        "https://github.com/mhajij/shrec_16/raw/main/small_shrec.zip",
    ),
}


@overload
def stanford_bunny(complex_type: Literal["cell"] = ...) -> CellComplex:
    ...


@overload
def stanford_bunny(complex_type: Literal["simplicial"] = ...) -> SimplicialComplex:
    ...


def stanford_bunny(
    complex_type: Literal["cell", "simplicial"] = "simplicial"
) -> Union[CellComplex, SimplicialComplex]:
    """Load the Stanford Bunny mesh as a complex.

    Parameters
    ----------
    complex_type : {'cell', 'simplicial'}, default='simplicial'
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
    if complex_type == "simplicial":
        return SimplicialComplex.load_mesh(os.path.join(DIR, "bunny.obj"))
    if complex_type == "cell":
        return CellComplex.load_mesh(os.path.join(DIR, "bunny.obj"))

    raise ValueError("complex_type must be 'simplicial' or 'cell'")


def shrec_16(size: Literal["full", "small"] = "full"):
    """Load training/testing shrec 16 datasets".

    Parameters
    ----------
    size : {'full', 'small'}, default='full'
        Dataset size. Options are "full" or "small".

    Returns
    -------
    tuple of length 2 npz files
        The npz files store the training/testing complexes of shrec 16 dataset along
        with their nodes, edges and faces features.

    Notes
    -----
    Each npz file stores 5 keys:
    "complexes",label","node_feat","edge_feat" and "face_feat".
    complex : stores the simplicial complex of the mesh
    label :  stores the label of the mesh
    node_feat : stores 6 dim node feature vector: position and normal of the each node in the mesh
    edge_feat : stores 10 dim edge feature vector: diheral angle, edge span, 2 edge angle in the triangle, 6 edge ratios.
    face_feat : face area, face normal, face angle

    Example
    -------
    >>> shrec_training, shrec_testing = shrec_16()
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
    if size not in DS_MAP:
        raise ValueError(f"size must be 'full' or 'small' got {size}.")
    ds_name, url = DS_MAP[size]

    zip_file = os.path.join(DIR, f"{ds_name}.zip")
    training = os.path.join(DIR, f"{ds_name}_training.npz")
    testing = os.path.join(DIR, f"{ds_name}_testing.npz")

    if not os.path.isfile(training) or not os.path.isfile(testing):
        print("downloading dataset...\n")
        wget.download(url, DIR)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(DIR)
        print("done!")

    if os.path.isfile(training) and os.path.isfile(testing):
        print("Loading dataset...\n")
        shrec_training = np.load(training, allow_pickle=True)
        shrec_testing = np.load(testing, allow_pickle=True)
        print("done!")
        return shrec_training, shrec_testing

    raise ValueError(
        f"Files couldn't be found in folder {DIR}, fail to load the dataset."
    )
