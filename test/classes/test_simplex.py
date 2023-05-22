"""Test Cell class."""

import unittest

from toponetx.classes.simplex import Simplex


class TestSimplex(unittest.TestCase):
    """Test the Simplex class."""

    def test_simplex_creation(self):
        """Test simplex creation."""
        s = Simplex((1,))
        self.assertEqual(len(s), 1)
        self.assertEqual(tuple(s), (1,))
        self.assertEqual(s.construct_tree, True)
        self.assertEqual(s.name, "")
        self.assertEqual(s.properties, {})

    def test_duplicate_nodes(self):
        """Test creation of simplex with duplicate nodes."""
        with self.assertRaises(ValueError):
            Simplex((1, 1, 2))

    def test_contains(self):
        """Test the __contains__ method of the simplex."""
        s = Simplex((1, 2))
        self.assertIn(1, s)
        self.assertIn((1,), s)
        self.assertNotIn(3, s)
        self.assertIn((1, 2), s)

    def test_boundary(self):
        """Test the boundary property of the simplex."""
        s = Simplex((1, 2, 3))
        boundary = s.boundary
        self.assertEqual(len(boundary), 3)

    def test_representation(self):
        """Test the string representation of the simplex."""
        s = Simplex((1, 2, 3))
        assert repr(s), "Simplex(1, 2, 3)"
        assert str(s) == "Nodes set: (1, 2, 3), attrs: {}"

    def test__contains__(self):
        """Test the __contains__ method of the simplex."""
        s = Simplex([1, 2, 3])
        assert (1, 2) in s
        assert (1, 3) in s
        assert frozenset((1, 3)) in s
        assert (1, 3, 4, 5, 6, 7) not in s

    def test_construct_tree(self):
        """Test the construct_tree property of the simplex."""
        s = Simplex((1, 2, 3), construct_tree=True)
        assert len(s.boundary) == 3

        s = Simplex((1, 2, 3), construct_tree=False)
        assert len(s.boundary) == 3

    def test_getting_and_setting_items(self):
        """Test getting and setting items in the simplex."""
        s = Simplex((1, 2, 3), construct_tree=True)
        s["weight"] = 1
        assert s["weight"] == 1


if __name__ == "__main__":
    unittest.main()
