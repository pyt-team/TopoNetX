"""Test hyperedge class."""

import unittest
from collections.abc import Hashable, Iterable

from toponetx.classes.hyperedge import HyperEdge


class HyperEdgeTestCase(unittest.TestCase):
    """Test case for the HyperEdge class."""

    def test_hyperedge_creation(self):
        """Test creating a HyperEdge object."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        self.assertEqual(len(hyperedge), 3)
        self.assertEqual(tuple(hyperedge), elements)

    def test_hyperedge_with_rank_and_attributes(self):
        """Test creating a HyperEdge object with rank and attributes."""
        elements = ("a", "b", "c")
        rank = 10
        attributes = {"color": "red", "weight": 2.5}
        hyperedge = HyperEdge(elements, rank=rank, **attributes)
        self.assertEqual(hyperedge.rank, rank)
        self.assertEqual(hyperedge["color"], attributes["color"])
        self.assertEqual(hyperedge["weight"], attributes["weight"])

    def test_hyperedge_contains(self):
        """Test checking element containment in a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        self.assertIn(1, hyperedge)
        self.assertNotIn(4, hyperedge)

    def test_hyperedge_iteration(self):
        """Test iterating over the elements of a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        self.assertCountEqual(hyperedge, elements)

    def test_hyperedge_representation(self):
        """Test the string representation of a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        expected_repr = f"HyperEdge{elements}"
        self.assertEqual(repr(hyperedge), expected_repr)

    def test_hyperedge_str(self):
        """Test the string conversion of a HyperEdge."""
        elements = (1, 2, 3)
        attributes = {"color": "red", "weight": 2.5}
        hyperedge = HyperEdge(elements, **attributes)
        expected_str = f"Nodes set:{elements}, attrs:{attributes}"
        self.assertEqual(str(hyperedge), expected_str)


if __name__ == "__main__":
    unittest.main()
