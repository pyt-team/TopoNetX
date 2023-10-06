"""Test distance module."""

import pytest
import scipy.sparse as sparse

from toponetx.algorithms.distance import cell_distance, distance
from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestDistance:
    """Test distance module."""

    def test_distance(self):
        """Test for the distance method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function
        result = distance(CC, 2, 3)
        expected_result = 1
        assert result == expected_result

    def test_distance_edgecases(self):
        """Test the distance method to check exceptions."""
        CCC = CombinatorialComplex()

        CCC.add_cell([2, 3, 4], rank=2)
        CCC.add_cell([5, 6, 7], rank=2)

        with pytest.warns():
            distance(CCC, 2, 3)

        with pytest.warns():
            distance(CCC, [2, 3, 4], [5, 6, 7])

        with pytest.warns():
            distance(CCC, Cell([2, 3, 4]), Cell([5, 6, 7]))

        with pytest.raises(ValueError):
            distance(1, 2, 3)

        with pytest.raises(ValueError):
            distance(SimplicialComplex(), 2, 3)

    def test_cell_distance(self):
        """Test for the cell_distance method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)
        CC.add_cell([2, 5], rank=1)

        # Test the function
        result = cell_distance(CC, (2, 3, 4), (5, 6, 7))
        expected_result = 2
        assert result == expected_result

    def test_cell_distance_edgecases(self):
        """Test for cell_distance with exceptions."""
        CCC = CombinatorialComplex()

        CCC.add_cell([2, 3, 4], rank=2)
        CCC.add_cell([5, 6, 7], rank=2)

        with pytest.warns():
            cell_distance(CCC, 2, 3)

        with pytest.warns():
            cell_distance(CCC, [2, 3, 4], [5, 6, 7])

        with pytest.warns():
            cell_distance(CCC, HyperEdge([2, 3, 4]), HyperEdge([5, 6, 7]))

        with pytest.raises(ValueError):
            cell_distance(1, 2, 3)

        with pytest.raises(ValueError):
            cell_distance(SimplicialComplex(), 2, 3)
