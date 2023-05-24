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

        elements = (1,)
        hyperedge = HyperEdge(elements)
        assert len(hyperedge) == 1
        assert tuple(hyperedge) == elements

        hyperedge = HyperEdge(1)
        assert len(hyperedge) == 1
        assert tuple(hyperedge) == (1,)

    def test_getitem_(self):
        """Test getitem method."""
        hyperedge = HyperEdge(1)
        hyperedge["weight"] = 1
        with self.assertRaises(KeyError):
            hyperedge["weightss"]

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
        # expected_repr = f"HyperEdge{elements}"
        assert isinstance(repr(hyperedge), str)

    def test_hyperedge_str(self):
        """Test the string conversion of a HyperEdge."""
        elements = (1, 2, 3)
        attributes = {"color": "red", "weight": 2.5}
        hyperedge = HyperEdge(elements, **attributes)
        # expected_str = f"Nodes set:{elements}, attrs:{attributes}"
        assert isinstance(str(hyperedge), str)

    def test_set_get_method(self):
        """Test set get methods."""
        hyperedge = HyperEdge((1, 2, 3))
        hyperedge["weight"] = 1
        assert hyperedge["weight"] == 1

    def test_rank(self):
        """Test rank."""
        he = HyperEdge([1, 4, 2], rank=4)
        assert he.rank == 4
        he = HyperEdge([1, 5, 2])
        assert he.rank is None

    def test_name(self):
        """Test name."""
        he = HyperEdge([1, 4, 2], rank=4)
        assert he.name == ""

        he = HyperEdge([1, 4, 2], name="A")
        assert he.name == "A"

    def test_inite_method_non_hashabe(self):
        """Test non hashable."""
        with self.assertRaises(TypeError):
            HyperEdge([1, [1], 2], rank=4)

    def test_duplicates_elements(self):
        """Test duplicates_elements."""
        with self.assertRaises(ValueError):
            HyperEdge([1, 1, 2], rank=4)


if __name__ == "__main__":
    unittest.main()
