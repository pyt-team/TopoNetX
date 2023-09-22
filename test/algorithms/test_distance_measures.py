"""Test distance measures module."""

import pytest
import scipy.sparse as sparse

from toponetx.algorithms.distance import cell_distance, distance
from toponetx.algorithms.distance_measures import (
    cell_diameter,
    cell_diameters,
    diameter,
    node_diameters,
)
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestDistanceMeasures:
    """Test distance measures module."""

    def test_node_diameters(self):
        """Test for the node_diameters method."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function
        result = list(node_diameters(CX))
        expected_result = [[1, 1], [{2, 3, 4}, {5, 6, 7}]]
        assert result == expected_result

    def test_cell_diameters(self):
        """Test for the cell_diameters method."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function
        result = list(cell_diameters(CX))
        expected_result = [
            [1, 1],
            [{(2, 3), (2, 3, 4), (2, 4), (3, 4)}, {(5, 6), (5, 6, 7), (5, 7), (6, 7)}],
        ]
        assert result == expected_result

    def test_diameter(self):
        """Test for the diameter method."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)
        CX.add_cell([2, 5], rank=1)
        # Test the function
        result = diameter(CX)
        expected_result = 3
        assert result == expected_result

    def test_cell_diameter(self):
        """Test for the cell_diameter method."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)
        CX.add_cell([2, 5], rank=1)

        # Test the function
        result = cell_diameter(CX)
        expected_result = 4
        assert result == expected_result
