"""Test distance module."""

import pytest
import scipy.sparse as sparse

from toponetx.algorithms.distance import cell_distance, distance
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
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
