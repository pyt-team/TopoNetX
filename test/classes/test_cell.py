"""Test Cell class."""

from collections.abc import Iterable

import pytest

from toponetx.classes.cell import Cell


class TestCell:
    """Test Cell class."""

    def test_create_cell_with_less_than_2_edges(self):
        """Test creating a cell with less than 2 edges."""
        elements = [1]
        with pytest.raises(ValueError):
            Cell(elements)

    def test_create_cell_with_self_loops(self):
        """Test creating a cell with self-loops."""
        elements = [1, 2, 3, 2]
        with pytest.raises(ValueError):
            Cell(elements)

    def test_sign_with_invalid_input(self):
        """Test signing with invalid input."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        with pytest.raises(TypeError):
            cell.sign(123)
        with pytest.raises(ValueError):
            cell.sign((1,))
        with pytest.raises(KeyError):
            cell.sign((1, 4))

    def test_get_item_with_invalid_key(self):
        """Test getting an item with an invalid key."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        with pytest.raises(KeyError):
            cell["invalid_key"]

    def test_are_homotopic_with_invalid_input(self):
        """Test checking if cells are homotopic with invalid input."""
        elements = [1, 2, 3]
        cell1 = Cell(elements)
        with pytest.raises(TypeError):
            Cell._are_homotopic(cell1, 123)

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
        """Test loop irregular."""
        with pytest.raises(ValueError):
            Cell([1, 1, 2], regular=False)

    def test___setitem__(self):
        """Test __setitem__."""
        c = Cell([1, 2, 3])
        c["weight"] = 1
        assert c["weight"] == 1

    def test___iter__(self):
        """Test __iter__."""
        c = Cell([1, 2, 3])
        for _ in c:
            pass

    def test___repr__(self):
        """Test __repr__."""
        c = Cell([1, 2, 3])
        print(c)

    def test_is_homotopic_to(self):
        """Test is_homotopic_to."""
        c = Cell([1, 2, 3])

        assert c.is_homotopic_to(Cell([1, 2, 3]))
        assert not c.is_homotopic_to(Cell([2, 4, 1]))

    def test_are_homotopic(self):
        """Test _are_homotopic."""
        c1 = Cell([1, 2, 3])
        c2 = Cell([2, 3, 1])

        assert Cell._are_homotopic(c1, c2)
        assert Cell._are_homotopic(c1, (1, 2, 3))

        with pytest.raises(TypeError):
            assert Cell._are_homotopic((1, 2, 3), c1)

    def test_get_item(self):
        """Test the __getitem__ method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        cell["name"] = "cell1"
        assert cell["name"] == "cell1"
        with pytest.raises(KeyError):
            _ = cell["invalid_key"]

    def test_set_item(self):
        """Test the __setitem__ method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        cell["name"] = "cell1"
        assert cell["name"] == "cell1"

    def test_is_regular_property(self):
        """Test the is_regular property of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert cell.is_regular

    def test_len(self):
        """Test the __len__ method of Cell."""
        elements = [1, 2, 3, 4]
        cell = Cell(elements)
        assert len(cell) == 4

    def test_iter(self):
        """Test the __iter__ method of Cell."""
        elements = [1, 2, 3, 4]
        cell = Cell(elements)
        assert isinstance(iter(cell), Iterable)
        assert list(cell) == elements

    def test_sign(self):
        """Test the sign method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert cell.sign((1, 2)) == 1
        assert cell.sign((2, 1)) == -1
        with pytest.raises(KeyError):
            cell.sign((2, 4))
        with pytest.raises(ValueError):
            cell.sign((1,))

    def test_contains(self):
        """Test the __contains__ method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert 1 in cell
        assert 4 not in cell

    def test_repr(self):
        """Test the __repr__ method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert repr(cell) == f"Cell({cell.elements})"

    def test_boundary(self):
        """Test the boundary property of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert list(cell.boundary) == [(1, 2), (2, 3), (3, 1)]

    def test_elements(self):
        """Test the elements property of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        assert cell.elements == tuple(elements)

    def test_reverse2(self):
        """Test the reverse method of Cell."""
        elements = [1, 2, 3]
        cell = Cell(elements)
        reversed_cell = cell.reverse()
        assert reversed_cell.elements == tuple(elements[::-1])

    def test_valid_regular_cell(self):
        """Test a valid regular cell."""
        elements = [1, 2, 3]
        assert Cell.is_valid_cell(elements, regular=True)

    def test_valid_irregular_cell(self):
        """Test a valid irregular cell."""
        elements = [1, 2, 3, 4, 1, 2, 3]
        assert Cell.is_valid_cell(elements, regular=False)

    def test_invalid_single_element_cell(self):
        """Test an invalid cell with a single element."""
        elements = [1]
        assert not Cell.is_valid_cell(elements, regular=True)

    def test_invalid_repeated_element_cell(self):
        """Test an invalid cell with repeated elements."""
        elements = [1, 1, 1, 1]
        assert not Cell.is_valid_cell(elements, regular=False)

    def test_invalid_regular_cell(self):
        """Test an invalid regular cell."""
        elements = [1, 2, 3, 2]
        assert not Cell.is_valid_cell(elements, regular=True)

    def test_invalid_irregular_cell(self):
        """Test an invalid irregular cell."""
        elements = [1, 2, 2, 4]
        assert not Cell.is_valid_cell(elements, regular=False)
