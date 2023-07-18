"""Test ReportViews module."""

import pytest

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.reportviews import CellView, HyperEdgeView, NodeView, SimplexView


class TestReportViews:
    """Test ReportViews module."""

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
