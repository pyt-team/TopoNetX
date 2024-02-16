"""Test hyperedge class."""

import pytest

from toponetx.classes.hyperedge import HyperEdge


class TestHyperEdgeCases:
    """Test case for the HyperEdge class."""

    def test_hyperedge_creation(self):
        """Test creating a HyperEdge object."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        assert len(hyperedge) == 3
        assert tuple(hyperedge) == elements

        elements = (1,)
        hyperedge = HyperEdge(elements)
        assert len(hyperedge) == 1
        assert tuple(hyperedge) == elements

        hyperedge = HyperEdge([1])
        assert len(hyperedge) == 1
        assert tuple(hyperedge) == (1,)

    def test_getitem_(self):
        """Test getitem method."""
        hyperedge = HyperEdge([1])
        hyperedge["weight"] = 1
        with pytest.raises(KeyError):
            _ = hyperedge["weightss"]

    def test_hyperedge_with_rank_and_attributes(self):
        """Test creating a HyperEdge object with rank and attributes."""
        elements = ("a", "b", "c")
        rank = 10
        attributes = {"color": "red", "weight": 2.5}
        hyperedge = HyperEdge(elements, rank=rank, **attributes)
        assert hyperedge.rank == rank
        assert hyperedge["color"] == attributes["color"]
        assert hyperedge["weight"] == attributes["weight"]

    def test_hyperedge_contains(self):
        """Test checking element containment in a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        assert 1 in hyperedge
        assert 4 not in hyperedge

    def test_hyperedge_iteration(self):
        """Test iterating over the elements of a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        assert tuple(hyperedge) == elements

    def test_hyperedge_representation(self):
        """Test the string representation of a HyperEdge."""
        elements = (1, 2, 3)
        hyperedge = HyperEdge(elements)
        expected_repr = (
            f"Nodes set: {tuple(hyperedge.elements)}, rank: {hyperedge.rank}"
        )
        assert isinstance(repr(hyperedge), str)
        assert repr(hyperedge) == expected_repr

    def test_hyperedge_str(self):
        """Test the string conversion of a HyperEdge."""
        elements = (1, 2, 3)
        attributes = {"color": "red", "weight": 2.5}
        hyperedge = HyperEdge(elements, **attributes)
        expected_str = (
            f"Nodes set: {tuple(hyperedge.elements)}, attrs: {hyperedge._attributes}"
        )
        assert isinstance(str(hyperedge), str)
        assert str(hyperedge) == expected_str

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

    def test_eq(self):
        """Test eq."""
        he1 = HyperEdge((1, 4, 2), rank=4)
        he2 = HyperEdge((1, 4, 2), rank=4)
        assert he1 == he2

        he1 = HyperEdge([1, 4, 2], rank=4)
        he2 = HyperEdge([1, 4, 2], rank=5)
        assert he1 != he2

        he1 = HyperEdge([1, 4, 2], rank=4)
        he2 = HyperEdge([1, 4, 2])
        assert he1 != he2

        he1 = HyperEdge([1, 4, 2], rank=4)
        he2 = HyperEdge([1, 4, 2], name="A")
        assert he1 != he2

        assert he1 != 1

    def test_hash(self):
        """Test HyperEdge hash."""
        he1 = HyperEdge((1, 4, 2), rank=4)
        he2 = HyperEdge((1, 4, 2), rank=4)
        assert hash(he1) == hash(he2)

        he1 = HyperEdge([1, 4, 2], rank=4)
        he2 = HyperEdge([1, 4, 2], rank=5)
        assert hash(he1) != hash(he2)

    def test_inite_method_non_hashabe(self):
        """Test non hashable."""
        with pytest.raises(TypeError) as exp_exception:
            HyperEdge([1, [1], 2], rank=4)
        assert (
            str(exp_exception.value) == "Every element of HyperEdge must be hashable."
        )

    def test_duplicates_elements(self):
        """Test duplicates_elements."""
        with pytest.raises(ValueError) as exp_exception:
            HyperEdge([1, 1, 2], rank=4)

        assert str(exp_exception.value) == "A hyperedge cannot contain duplicate nodes."
