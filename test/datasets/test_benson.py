"""Test the data loading functions for Benson's datasets."""

from pathlib import Path

import pytest

from toponetx.classes.simplex import Simplex
from toponetx.datasets.benson import load_benson_simplices


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
