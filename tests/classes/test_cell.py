"""Test cell class."""

import unittest

from toponetx.classes.cell import Cell


class TestCell(unittest.TestCase):
    def test_elements_and_nodes(self):
        """Test elements and nodes attributes of a cell."""
        cell = Cell([1, 2, 3])
        assert cell.elements == (1, 2, 3)
        assert cell.nodes == (1, 2, 3)

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
        assert cell_reverse.nodes == expected.nodes
        assert cell_reverse.elements == expected.elements


if __name__ == "__main__":
    unittest.main()
