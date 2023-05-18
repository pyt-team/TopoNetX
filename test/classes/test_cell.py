"""Test Cell class."""

import unittest

from toponetx.classes.cell import Cell


class TestCell(unittest.TestCase):
    def test_elements_and_nodes(self):
        """Test elements attributes of a cell."""
        cell = Cell([1, 2, 3])
        assert cell.elements == (1, 2, 3)

    def test_is_regular(self):
        """Test is_regular property."""
        cell = Cell([1, 2, 3])
        assert cell.is_regular
        cell = Cell([1, 2, 3, 2], regular=False)
        assert not cell.is_regular

    def test_reverse(self):
        """Test reverse."""
        cell = Cell([1, 2, 3])
        cell_reverse = cell.reverse()
        expected = Cell([3, 2, 1])
        assert cell_reverse.elements == expected.elements

    def test_self_loop_irregular(self):
        """Test loop irregular"""
        with self.assertRaises(ValueError):
            Cell([1, 1, 2], regular=False)

    def test___setitem__(self):
        """Test __setitem__"""
        c = Cell([1, 2, 3])
        c["weight"] = 1
        assert c.properties["weight"] == 1

    def test___iter__(self):
        """Test __iter__"""
        c = Cell([1, 2, 3])
        for i in c:
            pass

    def test___repr__(self):
        """Test __repr__"""
        c = Cell([1, 2, 3])
        print(c)

    def test_is_homotopic_to(self):
        """Test is_homotopic_to"""
        c = Cell([1, 2, 3])

        assert c.is_homotopic_to(Cell([1, 2, 3])) is True

        assert c.is_homotopic_to(Cell([2, 4, 1])) is False

    def test_are_homotopic(self):
        """Test _are_homotopic"""
        c1 = Cell([1, 2, 3])
        c2 = Cell([2, 3, 1])

        Cell._are_homotopic(c1, c2)


if __name__ == "__main__":
    unittest.main()
