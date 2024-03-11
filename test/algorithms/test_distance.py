"""Test distance module."""

import pytest

from toponetx.algorithms.distance import cell_distance, distance
from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.exception import TopoNetXNoPath


class TestDistance:
    """Test distance module."""

    def test_distance(self):
        """Test for the distance method."""
        CC = CellComplex()

        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        assert distance(CC, 2, 3) == 1

    def test_distance_edgecases(self):
        """Test the distance method to check exceptions."""
        CCC = CombinatorialComplex()

        CCC.add_cell([2, 3, 4], rank=2)
        CCC.add_cell([5, 6, 7], rank=2)

        with pytest.raises(TopoNetXNoPath):
            distance(CCC, 2, 5)

        with pytest.raises(KeyError):
            distance(CCC, [2, 3, 4], [5, 6, 7])

        with pytest.raises(KeyError):
            distance(CCC, Cell([2, 3, 4]), Cell([5, 6, 7]))

        with pytest.raises(TypeError):
            distance(1, 2, 3)

        with pytest.raises(TypeError):
            distance(SimplicialComplex(), 2, 3)

    def test_cell_distance(self):
        """Test for the cell_distance method."""
        CC = CellComplex()

        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)
        CC.add_cell([2, 5], rank=1)

        assert cell_distance(CC, (2, 3, 4), (5, 6, 7)) == 2

    def test_cell_distance_edgecases(self):
        """Test for cell_distance with exceptions."""
        CCC = CombinatorialComplex()

        CCC.add_cell([2, 3, 4], rank=2)
        CCC.add_cell([5, 6, 7], rank=2)

        with pytest.raises(TopoNetXNoPath):
            cell_distance(CCC, (2, 3, 4), (5, 6, 7))

        with pytest.raises(KeyError):
            cell_distance(CCC, [3, 4, 5], [5, 6, 7])

        with pytest.raises(KeyError):
            cell_distance(CCC, HyperEdge([3, 4, 5]), HyperEdge([5, 6, 7]))

        with pytest.raises(TypeError):
            cell_distance(1, 2, 3)

        with pytest.raises(TypeError):
            cell_distance(SimplicialComplex(), 2, 3)
