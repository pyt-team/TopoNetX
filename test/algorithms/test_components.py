"""Test components module."""

import pytest

from toponetx.algorithms.components import (
    connected_component_subcomplexes,
    connected_components,
    s_component_subcomplexes,
    s_connected_components,
)
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestComponents:
    """Test components module."""

    def test_s_connected_components(self):
        """Test_s_connected_components."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=True
        result = list(s_connected_components(CC, s=1, cells=True))
        expected_result = [
            {(2, 3), (2, 3, 4), (2, 4), (3, 4)},
            {(5, 6), (5, 6, 7), (5, 7), (6, 7)},
        ]
        assert result == expected_result

        # Test the function with cells=False
        result = list(s_connected_components(CC, s=1, cells=False))
        expected_result = [{2, 3, 4}, {5, 6, 7}]
        assert result == expected_result

        # test with ColoredHyperGraph
        CHG = CC.to_colored_hypergraph()
        result = list(s_connected_components(CHG, s=1, cells=True))
        expected_result = [
            {
                (frozenset({3, 4}), 0),
                (frozenset({2, 4}), 0),
                (frozenset({2, 3}), 0),
                (frozenset({2, 3, 4}), 0),
            },
            {
                (frozenset({5, 7}), 0),
                (frozenset({5, 6, 7}), 0),
                (frozenset({5, 6}), 0),
                (frozenset({6, 7}), 0),
            },
        ]
        assert result == expected_result

        result = list(s_connected_components(CHG, s=1, cells=False))
        expected_result = [{2, 3, 4}, {5, 6, 7}]
        assert result == expected_result

        # test cells=True, return_singletons=False
        CC.add_node(10)
        result = list(s_connected_components(CC, s=1, cells=True))
        expected_result = [
            {(2, 3), (2, 3, 4), (2, 4), (3, 4)},
            {(5, 6), (5, 6, 7), (5, 7), (6, 7)},
        ]
        assert result == expected_result

        # Test the function with return_singletons=True
        CC2 = CellComplex()
        CC2.add_cell([1, 2, 3, 4], rank=2)
        CC2.add_cell([1, 2, 3, 4], rank=2)
        CC2.add_cell([1, 2, 3, 4], rank=2)
        CC2.add_node(0)
        CC2.add_node(10)

        result = list(
            s_connected_components(CC2, s=1, cells=False, return_singletons=True)
        )
        expected_result = [{0}, {1, 2, 3, 4}, {10}]

        # Test the function with return_singletons=False
        result = list(
            s_connected_components(CC2, s=2, cells=False, return_singletons=False)
        )
        expected_result = [{1, 2, 3, 4}]
        assert result == expected_result

        # invalid input
        with pytest.raises(TypeError):
            list(s_connected_components(SimplicialComplex()))

    def test_s_component_subcomplexes(self):
        """Test_s_component_subcomplexes."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=True
        result = list(s_component_subcomplexes(CC, s=1, cells=True))
        expected_result = [CellComplex(), CellComplex()]
        expected_result[0].add_cell([2, 3, 4], rank=2)
        expected_result[1].add_cell([5, 6, 7], rank=2)
        assert len(result[0].cells) == len(expected_result[0].cells)
        assert len(result[1].cells) == len(expected_result[1].cells)

        CC = CellComplex()
        # Test the function with return_singletons=False
        result = list(
            s_component_subcomplexes(CC, s=1, cells=False, return_singletons=False)
        )
        expected_result = []
        assert result == expected_result

    def test_connected_components(self):
        """Test_connected_components."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function with cells=False and return_singletons=True
        result = list(connected_components(CC, cells=True, return_singletons=True))
        expected_result = [
            {(2, 3), (2, 3, 4), (2, 4), (3, 4)},
            {(5, 6), (5, 6, 7), (5, 7), (6, 7)},
        ]
        assert result == expected_result

        CC = CellComplex()
        # Test the function with cells=False and return_singletons=False
        result = list(connected_components(CC, cells=False, return_singletons=False))
        expected_result = []
        assert result == expected_result

    def test_connected_component_subcomplexes(self):
        """Test_connected_component_subcomplexes."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([5, 6, 7], rank=2)

        # Test the function with return_singletons=True
        result = list(connected_component_subcomplexes(CC, return_singletons=True))
        expected_result = [CellComplex(), CellComplex()]
        expected_result[0].add_cell([2, 3, 4], rank=2)
        expected_result[1].add_cell([5, 6, 7], rank=2)
        assert len(result[0].cells) == len(expected_result[0].cells)
        assert len(result[1].cells) == len(expected_result[1].cells)

        # Test the function with return_singletons=False
        CC = CellComplex()  # Initialize your class object
        result = list(connected_component_subcomplexes(CC, return_singletons=False))
        expected_result = []
        assert result == expected_result
