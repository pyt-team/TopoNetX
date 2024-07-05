"""Test the classic cell complex generators."""

from toponetx.generators.classic_cell_complexes import (
    pyrmaid_complex,
    single_cell_complex,
)


class TestSingleCellComplex:
    """Test the `single_cell_complex` function."""

    def test_single_cell(self):
        """Test `single_cell_complex` for zero nodes."""
        CC = single_cell_complex(6)
        assert len(CC.cells) == 1
        assert (1, 2, 3, 4, 5) in CC.cells


class TestPyrmaidComplex:
    """Test the `pyrmaid_complex` function."""

    def test_pyrmaid_complex(self):
        """Test `pyrmaid_complex` for zero nodes."""
        CC = pyrmaid_complex(6)
        assert len(CC.cells) == 6
        assert (1, 2, 3, 4, 5) in CC.cells
