"""Test components module."""

import pytest
import scipy.sparse as sparse

from toponetx.algorithms.components import (
    connected_component_subcomplexes,
    connected_components,
    s_component_subcomplexes,
    s_connected_components,
)
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestComponents:
    """Test components module."""

    def test_s_connected_components(self):
        """Test_s_connected_components."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=True
        result = list(s_connected_components(CX, s=1, cells=True))
        expected_result = [
            {(2, 3), (2, 3, 4), (2, 4), (3, 4)},
            {(5, 6), (5, 6, 7), (5, 7), (6, 7)},
        ]
        assert result == expected_result

        # Test the function with cells=False
        result = list(s_connected_components(CX, s=1, cells=False))
        expected_result = [{2, 3, 4}, {5, 6, 7}]
        assert result == expected_result

        # Test the function with return_singletons=True
        result = list(
            s_connected_components(CX, s=1, cells=False, return_singletons=True)
        )
        expected_result = [{2, 3, 4}, {5, 6, 7}]
        assert result == expected_result

        # Test the function with return_singletons=False
        result = list(
            s_connected_components(CX, s=2, cells=False, return_singletons=False)
        )
        expected_result = [{2, 3, 4}, {5, 6, 7}]
        assert result == expected_result

    def test_s_component_subcomplexes(self):
        """Test_s_component_subcomplexes."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=True
        result = list(s_component_subcomplexes(CX, s=1, cells=True))
        expected_result = [CellComplex(), CellComplex()]
        expected_result[0].add_cell([2, 3, 4], rank=2)
        expected_result[1].add_cell([5, 6, 7], rank=2)
        assert len(result[0].cells) == len(expected_result[0].cells)
        assert len(result[1].cells) == len(expected_result[1].cells)

        CX = CellComplex()
        # Test the function with return_singletons=False
        result = list(
            s_component_subcomplexes(CX, s=1, cells=False, return_singletons=False)
        )
        expected_result = []
        assert result == expected_result

    def test_connected_components(self):
        """Test_connected_components."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=False and return_singletons=True
        result = list(connected_components(CX, cells=True, return_singletons=True))
        expected_result = [
            {(2, 3), (2, 3, 4), (2, 4), (3, 4)},
            {(5, 6), (5, 6, 7), (5, 7), (6, 7)},
        ]
        assert result == expected_result

        CX = CellComplex()
        # Test the function with cells=False and return_singletons=False
        result = list(connected_components(CX, cells=False, return_singletons=False))
        expected_result = []
        assert result == expected_result

    def test_connected_component_subcomplexes(self):
        """Test_connected_component_subcomplexes."""
        CX = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([5, 6, 7], rank=2)

        # Test the function with return_singletons=True
        result = list(connected_component_subcomplexes(CX, return_singletons=True))
        expected_result = [CellComplex(), CellComplex()]
        expected_result[0].add_cell([2, 3, 4], rank=2)
        expected_result[1].add_cell([5, 6, 7], rank=2)
        assert len(result[0].cells) == len(expected_result[0].cells)
        assert len(result[1].cells) == len(expected_result[1].cells)

        # Test the function with return_singletons=False
        CX = CellComplex()  # Initialize your class object
        result = list(connected_component_subcomplexes(CX, return_singletons=False))
        expected_result = []
        assert result == expected_result
