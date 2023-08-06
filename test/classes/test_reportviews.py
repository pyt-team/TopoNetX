"""Test ReportViews module."""

import pytest

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.reportviews import CellView, HyperEdgeView, NodeView, SimplexView


class TestReportViews_CellView:
    """Test Cell View Class of the ReportViews module."""

    CX = CellComplex()
    CX._insert_cell((1, 2, 3, 4), color="red")
    CX._insert_cell((1, 2, 3, 4), color="green")
    CX._insert_cell((1, 2, 3, 4), shape="square")
    CX._insert_cell((2, 3, 4, 5))

    CV = CX._cells
    cell_1 = Cell((1, 2, 3, 4))
    cell_2 = Cell((1, 2, 3, 8))

    def test_cell_view_report_getitem_method(self):
        """Test the __getitem__ method of CellView class."""
        with pytest.raises(KeyError) as exp_exception:
            self.CV.__getitem__(self.cell_2)

        assert (
            str(exp_exception.value)
            == "'cell Cell((1, 2, 3, 8)) is not in the cell dictionary'"
        )

        assert list(self.CV.__getitem__(self.cell_1)) == [
            {"color": "red"},
            {"color": "green"},
            {"shape": "square"},
        ]

        with pytest.raises(KeyError) as exp_exception:
            self.CV.__getitem__((1, 2, 3, 8))

        assert (
            str(exp_exception.value)
            == "'cell (1, 2, 3, 8) is not in the cell dictionary'"
        )

        with pytest.raises(TypeError) as exp_exception:
            self.CV.__getitem__(1)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_1(self):
        """Test the raw method for the CellView class."""
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw(1)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_2(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is int
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw(2)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_3(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is string
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw("testing string")

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_4(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is set
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw({1})

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_5(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is dict
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw({"1": "some_string"})

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_6(self):
        """Test the raw method for the CellView class."""
        # Raw Method should raise a KeyError exception when cell is not present
        with pytest.raises(KeyError) as exp_exception:
            self.CV.raw(self.cell_2)

        assert (
            str(exp_exception.value)
            == "'cell Cell((1, 2, 3, 8)) is not in the cell dictionary'"
        )

    def test_cell_view_raw_method_7(self):
        """Test the raw method for the CellView class."""
        assert str(self.CV.raw(self.cell_1)) == str(
            [Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4))]
        )

    def test_cell_view_raw_method_8(self):
        """Test the raw method for the CellView class."""
        with pytest.raises(KeyError) as exp_exception:
            self.CV.raw((1, 2, 3, 8))

        assert (
            str(exp_exception.value)
            == "'cell (1, 2, 3, 8) is not in the cell dictionary'"
        )

    def test_cell_view_contains_method(self):
        """Test the contains method for the CellView class."""
        CX = CellComplex()
        # Inserting some cells in CX
        CX._insert_cell((1, 2, 3, 4), color="red")
        CX._insert_cell((1, 2, 3, 4), color="green")
        CX._insert_cell((7, 8, 9, 1), shape="square")
        CX._insert_cell((2, 3, 4, 5))

        # Creating Cell View from Cell Complex
        CV = CX._cells

        # To test we are using three cells that are present in the cell complex
        # these are the three cases of multiple homotopic cells present (with properties)
        # and single cell (with and without properties attribute)

        cell_1 = Cell((1, 2, 3, 4))
        cell_2 = Cell((2, 3, 4, 1))
        cell_3 = Cell((1, 3, 2, 4))
        cell_4 = Cell((7, 8, 9, 1))
        cell_5 = Cell((1, 7, 8, 9))
        cell_6 = Cell((7, 1, 8, 9))
        cell_7 = Cell((2, 3, 4, 5))
        cell_8 = Cell((3, 4, 5, 2))
        cell_9 = Cell((3, 2, 4, 5))
        cell_10 = Cell((1, 2, 3, 8))

        assert CV.__contains__(cell_1) is True
        assert CV.__contains__(cell_2) is True
        assert CV.__contains__(cell_3) is False

        assert CV.__contains__(cell_4) is True
        assert CV.__contains__(cell_5) is True
        assert CV.__contains__(cell_6) is False

        assert CV.__contains__(cell_7) is True
        assert CV.__contains__(cell_8) is True
        assert CV.__contains__(cell_9) is False

        assert CV.__contains__(cell_10) is False

        # Now testing the same with list and tuples as inputs

        cell_1_tp = (1, 2, 3, 4)
        cell_2_tp = (2, 3, 4, 1)
        cell_3_tp = (1, 3, 2, 4)
        cell_4_tp = (7, 8, 9, 1)
        cell_5_tp = (1, 7, 8, 9)
        cell_6_tp = (7, 1, 8, 9)
        cell_7_tp = (2, 3, 4, 5)
        cell_8_tp = (3, 4, 5, 2)
        cell_9_tp = (3, 2, 4, 5)
        cell_10_tp = (1, 2, 3, 8)

        assert CV.__contains__(cell_1_tp) is True
        assert CV.__contains__(cell_2_tp) is True
        assert CV.__contains__(cell_3_tp) is False

        assert CV.__contains__(cell_4_tp) is True
        assert CV.__contains__(cell_5_tp) is True
        assert CV.__contains__(cell_6_tp) is False

        assert CV.__contains__(cell_7_tp) is True
        assert CV.__contains__(cell_8_tp) is True
        assert CV.__contains__(cell_9_tp) is False

        assert CV.__contains__(cell_10_tp) is False

        assert CV.__contains__(list(cell_1_tp)) is True
        assert CV.__contains__(list(cell_2_tp)) is True
        assert CV.__contains__(list(cell_3_tp)) is False

        assert CV.__contains__(list(cell_4_tp)) is True
        assert CV.__contains__(list(cell_5_tp)) is True
        assert CV.__contains__(list(cell_6_tp)) is False

        assert CV.__contains__(list(cell_7_tp)) is True
        assert CV.__contains__(list(cell_8_tp)) is True
        assert CV.__contains__(list(cell_9_tp)) is False

        assert CV.__contains__(list(cell_10_tp)) is False

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__({1, 2, 3, 4})

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__(1)

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__("1234")

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__({"1": True, "2": True, "3": True, "4": True})

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

    def test_cell_view_repr_method(self):
        """Test the __repr__ method for the CellView class."""
        assert (
            self.CV.__repr__()
            == "CellView([Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))])"
        )

    def test_cell_view_str_method(self):
        """Test the __str__ method for the CellView class."""
        assert (
            self.CV.__str__()
            == "CellView([Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))])"
        )


class TestReportViews_HyperEdgeView:
    """Test the HyperEdgeView class of the ReportView class."""
