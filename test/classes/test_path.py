"""Test Path class."""

import pytest

from toponetx.classes.path import Path


class TestPath:
    """Test the Path class."""

    def test_path_creation(self):
        """Test path creation."""
        s = Path((1,))
        assert len(s) == 1
        assert tuple(s) == (1,)
        assert s.construct_boundaries is False
        assert s._attributes == {}

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        with pytest.raises(ValueError):
            _ = Path([[1], 2, 3])

        with pytest.raises(ValueError):
            _ = Path(Path((1, 2, 3)))

        with pytest.raises(ValueError):
            _ = Path(frozenset((1, 2, 3)))

    def test_duplicate_nodes(self):
        """Test creation of path with duplicate nodes."""
        with pytest.raises(ValueError):
            Path((1, 1, 2))

    def test_contains(self):
        """Test the __contains__ method of the path."""
        s = Path((1, 2))
        assert 1 in s
        assert 3 not in s

    def test_boundary(self):
        """Test the boundary property of the path."""
        s = Path((1, 2, 3))
        boundary = s.boundary
        assert len(boundary) == 0

    def test_representation(self):
        """Test the string representation of the path."""
        s = Path((1, 2, 3))
        assert repr(s), "Path((1, 2, 3))"
        assert str(s) == "Node set: (1, 2, 3), Boundaries: [], Attributes: {}"

    def test_construct_boundaries(self):
        """Test the construct_tree property of the path."""
        s = Path((1, 2, 3), construct_boundaries=True)
        assert len(s.boundary) == 3

        s = Path(
            (1, 2, 3),
            construct_boundaries=True,
            allowed_paths=[(1,), (2,), (3,), (1, 2), (2, 3)],
        )
        assert len(s.boundary) == 2

        with pytest.raises(ValueError):
            s = Path((3, 2, 1), reserve_sequence_order=False)

        s = Path((1, 3, 2), reserve_sequence_order=False, construct_boundaries=True)
        assert (2, 3) in s.boundary
        assert (1, 2) in s.boundary
        assert (1, 3) in s.boundary

        s = Path((1, 2, 3), construct_boundaries=False)
        assert len(s.boundary) == 0

    def test_getting_and_setting_items(self):
        """Test getting and setting items in the path."""
        s = Path((1, 2, 3))
        s["weight"] = 1
        assert s["weight"] == 1

    def test_clone(self):
        """Test clone method."""
        s = Path((1, 2, 3))
        s1 = s.clone()

        assert 1 in s1
        assert 2 in s1
        assert 3 in s1
