"""Test distance measures module."""

import pytest

from toponetx.algorithms.distance_measures import (
    cell_diameter,
    cell_diameters,
    diameter,
    node_diameters,
)
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestDistanceMeasures:
    """Test distance measures module."""

    def test_node_diameters(self):
        """Test for the node_diameters method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function
        result = list(node_diameters(CC))
        expected_result = [[1, 1], [{2, 3, 4}, {5, 6, 7}]]
        assert result == expected_result

        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.add_cell([8, 9, 10], rank=2)
        result = list(node_diameters(CHG))
        expected_result = [
            [2, 1],
            [
                {frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4})},
                {frozenset({8}), frozenset({9}), frozenset({10})},
            ],
        ]
        assert result == expected_result

    def test_cell_diameters(self):
        """Test for the cell_diameters method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function
        result = list(cell_diameters(CC))
        expected_result = [
            [1, 1],
            [{(2, 3), (2, 3, 4), (2, 4), (3, 4)}, {(5, 6), (5, 6, 7), (5, 7), (6, 7)}],
        ]
        assert result == expected_result

        # invalid input
        with pytest.raises(TypeError):
            cell_diameters(SimplicialComplex())

    def test_diameter(self):
        """Test for the diameter method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)
        CC.add_cell([2, 5], rank=1)
        # Test the function
        result = diameter(CC)
        expected_result = 3
        assert result == expected_result

        # invalid input
        with pytest.raises(TypeError):
            diameter(SimplicialComplex())

        # diameter is not defined for non-connected complexes
        CC.add_cell([10, 22], rank=1)
        with pytest.raises(RuntimeError):
            _ = diameter(CC)

    def test_cell_diameter(self):
        """Test for the cell_diameter method."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)
        CC.add_cell([2, 5], rank=1)

        # Test the function
        result = cell_diameter(CC)
        expected_result = 4
        assert result == expected_result

        # invalid input
        with pytest.raises(TypeError):
            cell_diameter(SimplicialComplex())

        # cell diameter is not defined for non-connected complexes
        CC.add_cell([10, 22], rank=1)
        with pytest.raises(RuntimeError):
            _ = cell_diameter(CC)
