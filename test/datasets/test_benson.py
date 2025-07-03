"""Test the data loading functions for Benson's datasets."""

from pathlib import Path

import pytest

from toponetx.classes.simplex import Simplex
from toponetx.datasets.benson import load_benson_hyperedges, load_benson_simplices


def test_load_benson_hyperedges_folder_error() -> None:
    """Test that the `load_benson_hyperedges` function raises an error for an invalid folder."""
    with pytest.raises(ValueError):
        load_benson_hyperedges("./")
    with pytest.raises(ValueError):
        load_benson_hyperedges("./nonexistent_folder")
    with pytest.raises(ValueError):
        load_benson_hyperedges("./test_benson.py")


def test_load_benson_hyperedges_dataset_2() -> None:
    """Test the `load_benson_hyperedges` function."""
    nodes, hyperedges = load_benson_hyperedges(
        Path(__file__).parent / "benson_sample_dataset-2"
    )
    assert len(nodes) == 7
    assert len(hyperedges) == 6

    for node in nodes:
        assert isinstance(node, Simplex)
        assert node["name"] == f"Node {next(iter(node))}"
        assert node["label"] in ["Label 1", "Label 2"]

    for simplex in hyperedges:
        assert isinstance(simplex, Simplex)


def test_load_benson_hyperedges_dataset_3() -> None:
    """Test the `load_benson_hyperedges` function."""
    nodes, hyperedges = load_benson_hyperedges(
        Path(__file__).parent / "benson_sample_dataset-3"
    )
    assert len(nodes) == 7
    assert len(hyperedges) == 6

    for node in nodes:
        assert isinstance(node, Simplex)
        assert node["name"] == f"Node {next(iter(node))}"
        assert isinstance(node["label"], list)
        assert all(
            label in ["Label 1", "Label 2", "Label 3"] for label in node["label"]
        )

    for simplex in hyperedges:
        assert isinstance(simplex, Simplex)


def test_load_benson_simplices_folder_error() -> None:
    """Test that the `load_benson_simplices` function raises an error for an invalid folder."""
    with pytest.raises(ValueError):
        load_benson_simplices("./")
    with pytest.raises(ValueError):
        load_benson_simplices("./nonexistent_folder")
    with pytest.raises(ValueError):
        load_benson_simplices("./test_benson.py")


def test_load_benson_simplices() -> None:
    """Test the `load_benson_simplices` function."""
    simplices = load_benson_simplices(Path(__file__).parent / "benson_sample_dataset")
    assert len(simplices) == 8

    for simplex in simplices:
        assert isinstance(simplex, Simplex)
        assert simplex["time"] in [1, 2]
