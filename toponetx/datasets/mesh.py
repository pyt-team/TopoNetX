"""Various examples of named meshes represented as complexes."""

import zipfile
from pathlib import Path
from typing import Literal, overload

import numpy as np
import requests

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.readwrite.atomlist import load_from_atomlist

__all__ = ["coseg", "shrec_16", "stanford_bunny"]

DIR = Path(__file__).parent
SHREC_DS_MAP = {
    "full": "https://github.com/pyt-team/topological-datasets/raw/aa3ec28b1166f7713e872011df1ff9e1136d92ae/resources/shrec16_full.zip",
    "small": "https://github.com/pyt-team/topological-datasets/raw/aa3ec28b1166f7713e872011df1ff9e1136d92ae/resources/shrec16_small.zip",
}

COSEG_DS_MAP = {
    "alien": "https://github.com/pyt-team/topological-datasets/raw/aa3ec28b1166f7713e872011df1ff9e1136d92ae/resources/coseg_alien.zip",
    "chair": "https://github.com/pyt-team/topological-datasets/raw/aa3ec28b1166f7713e872011df1ff9e1136d92ae/resources/coseg_chair.zip",
    "vase": "https://github.com/pyt-team/topological-datasets/raw/aa3ec28b1166f7713e872011df1ff9e1136d92ae/resources/coseg_vase.zip",
}


@overload
def stanford_bunny(
    complex_type: Literal["cell"],
) -> CellComplex:  # numpydoc ignore=GL08
    ...


@overload
def stanford_bunny(
    complex_type: Literal["simplicial"],
) -> SimplicialComplex:  # numpydoc ignore=GL08
    ...


def stanford_bunny(
    complex_type: Literal["cell", "simplicial"],
) -> CellComplex | SimplicialComplex:
    """Load the Stanford Bunny mesh as a complex.

    Parameters
    ----------
    complex_type : {'cell', 'simplicial'}
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
    dataset_file = DIR / "bunny.obj"

    if not dataset_file.exists():
        r = requests.get(
            "https://github.com/pyt-team/topological-datasets/raw/main/resources/bunny.obj",
            timeout=10,
        )
        with dataset_file.open("wb") as f:
            f.write(r.content)

    if complex_type == "simplicial":
        return SimplicialComplex.load_mesh(dataset_file)
    if complex_type == "cell":
        return CellComplex.load_mesh(dataset_file)

    raise ValueError("complex_type must be 'simplicial' or 'cell'")


def shrec_16(size: Literal["full", "small"] = "full"):
    """Load training/testing shrec 16 datasets.

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
    "complexes","node_feat","edge_feat", "face_feat" and mesh label".
    complex : stores the simplicial complex of the mesh
    node_feat : stores a 6 dim node feature vector: position and normal of the each node in the mesh
    edge_feat : stores a 10 dim edge feature vector: dihedral angle, edge span, 2 edge angle in the triangle, 6 edge ratios.
    face_feat : stores a 7-dimensional face feature vector: face area (1 feat), face normal (3 feat), face angles (3 feat)
    mesh label : stores the label of the mesh

    Raises
    ------
    RuntimeError
        If dataset is not found on in DIR.

    Examples
    --------
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
    if size not in SHREC_DS_MAP:
        raise ValueError(
            f"Parameter `size` must be one of {list(SHREC_DS_MAP.keys())}, got {size}."
        )
    url = SHREC_DS_MAP[size]

    cache_file = DIR / f"shrec_{size}.zip"
    if not cache_file.exists():
        r = requests.get(url, timeout=10)
        with cache_file.open("wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(cache_file) as zip_ref:
        train_complexes = []
        test_complexes = []

        for file in zip_ref.namelist():
            if file.endswith(".atomlist"):
                i = int(file.split("/")[1].split(".")[0])
                SC = load_from_atomlist(zip_ref.open(file), complex_type="simplicial")
                if "train" in file:
                    train_complexes.append((i, SC))
                if "test" in file:
                    test_complexes.append((i, SC))

        # `zip_ref.namelist` does not sort the files in the expected order
        train_complexes = [x[1] for x in sorted(train_complexes)]
        test_complexes = [x[1] for x in sorted(test_complexes)]

        train_data = {
            "complexes": train_complexes,
            "label": np.load(zip_ref.open("train/labels.npy")),
            "node_feat": np.load(
                zip_ref.open("train/node_feat.npy"), allow_pickle=True
            ),
            "edge_feat": np.load(zip_ref.open("train/edge_feat.npy")),
            "face_feat": np.load(zip_ref.open("train/face_feat.npy")),
        }
        test_data = {
            "complexes": test_complexes,
            "label": np.load(zip_ref.open("test/labels.npy")),
            "node_feat": np.load(zip_ref.open("test/node_feat.npy"), allow_pickle=True),
            "edge_feat": np.load(zip_ref.open("test/edge_feat.npy")),
            "face_feat": np.load(zip_ref.open("test/face_feat.npy")),
        }

    return train_data, test_data


def coseg(data: Literal["alien", "vase", "chair"] = "alien"):
    """Load coseg mesh segmentation datasets.

    The coseg dataset was downloaded and processed from the repo:
    https://github.com/Ideas-Laboratory/shape-coseg-dataset

    Parameters
    ----------
    data : {'alien', 'vase', 'chair'}, default='alien'
        The name of the coseg dataset to be loaded. Options are 'alien', 'vase', or 'chair'.

    Returns
    -------
    npz file
        The npz files store the complexes of coseg segmentation dataset along
        with their nodes, edges, and faces features.

    Raises
    ------
    RuntimeError
        If the dataset is not found in DIR.

    Notes
    -----
    Each npz file stores 5 keys:
    "complexes", "label", "node_feat", "edge_feat", and "face_feat".
    complex : stores the simplicial complex of the mesh
    node_feat : stores a 6-dimensional node feature vector: position and normal of each node in the mesh
    edge_feat : stores a 10-dimensional edge feature vector: dihedral angle, edge span, 2 edge angles in the triangle, 6 edge ratios.
    face_feat : stores a 7-dimensional face feature vector: face area (1 feat), face normal (3 feat), face angles (3 feat)
    face_label : stores the label of mesh segmentation as a face label

    Examples
    --------
    >>> coseg_data = coseg("alien")
    >>> complexes = coseg_data["complexes"]
    >>> node_feat = coseg_data["node_feat"]
    >>> edge_feat = coseg_data["edge_feat"]
    >>> face_feat = coseg_data["face_feat"]
    >>> face_label = coseg_data["face_label"]
    """
    if data not in COSEG_DS_MAP:
        raise ValueError(f"data must be 'alien', 'vase', or 'chair' got {data}.")
    url = COSEG_DS_MAP[data]

    cache_file = DIR / f"coseg_{data}.zip"
    if not cache_file.exists():
        r = requests.get(url, timeout=10)
        with cache_file.open("wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(cache_file) as zip_ref:
        SCs = [
            (
                int(file.split(".")[0]),
                load_from_atomlist(zip_ref.open(file), complex_type="simplicial"),
            )
            for file in zip_ref.namelist()
            if file.endswith(".atomlist")
        ]

        SCs = [x[1] for x in sorted(SCs)]

        return {
            "complexes": SCs,
            "face_label": np.load(zip_ref.open("face_label.npy"), allow_pickle=True),
            "node_feat": np.load(zip_ref.open("node_feat.npy"), allow_pickle=True),
            "edge_feat": np.load(zip_ref.open("edge_feat.npy"), allow_pickle=True),
            "face_feat": np.load(zip_ref.open("face_feat.npy"), allow_pickle=True),
        }
