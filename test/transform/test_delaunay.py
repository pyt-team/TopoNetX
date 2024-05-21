"""Test delaunay triangulation."""

import numpy as np

from toponetx.transform import delaunay_triangulation


def test_delaunay_triangulation_simple():
    """Test the Delaunay triangulation of a simple set of points."""
    points = np.array([[0, 0], [1, 0], [0, 1]])
    SC = delaunay_triangulation(points)

    assert set(SC.simplices) == {
        frozenset([0]),
        frozenset([1]),
        frozenset([2]),
        frozenset([0, 1]),
        frozenset([0, 2]),
        frozenset([1, 2]),
        frozenset([0, 1, 2]),
    }


def test_delaunay_triangulation():
    """Test the Delaunay triangulation of a set of points."""
    points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
    SC = delaunay_triangulation(points)

    assert set(SC.simplices) == {
        frozenset([0]),
        frozenset([1]),
        frozenset([2]),
        frozenset([3]),
        frozenset([0, 1]),
        frozenset([0, 2]),
        frozenset([1, 2]),
        frozenset([1, 3]),
        frozenset([2, 3]),
        frozenset([0, 1, 2]),
        frozenset([1, 2, 3]),
    }
