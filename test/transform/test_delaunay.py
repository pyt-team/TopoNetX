"""Test delaunay triangulation."""

import numpy as np

from toponetx.classes.simplex import Simplex
from toponetx.transform import delaunay_triangulation


def test_delaunay_triangulation_simple():
    """Test the Delaunay triangulation of a simple set of points."""
    points = np.array([[0, 0], [1, 0], [0, 1]])
    SC = delaunay_triangulation(points)

    assert set(SC.simplices) == {
        Simplex((0,)),
        Simplex((1,)),
        Simplex((2,)),
        Simplex((0, 1)),
        Simplex((0, 2)),
        Simplex((1, 2)),
        Simplex((0, 1, 2)),
    }


def test_delaunay_triangulation():
    """Test the Delaunay triangulation of a set of points."""
    points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
    SC = delaunay_triangulation(points)

    assert set(SC.simplices) == {
        Simplex((0,)),
        Simplex((1,)),
        Simplex((2,)),
        Simplex((3,)),
        Simplex((0, 1)),
        Simplex((0, 2)),
        Simplex((1, 2)),
        Simplex((1, 3)),
        Simplex((2, 3)),
        Simplex((0, 1, 2)),
        Simplex((1, 2, 3)),
    }
