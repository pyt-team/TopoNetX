"""Test the random cell complex generators."""

from toponetx.generators.random_cell_complexes import np_cell_complex


class TestNPCellComplex:
    """Test the `np_cell_complex` function."""

    def test_zero_nodes(self):
        """Test `np_cell_complex` for zero nodes."""
        cc = np_cell_complex(0, 0.1)
        assert cc.shape == (0, 0, 0)

    def test_non_zero_nodes(self):
        """Test `np_cell_complex` for nonzero nodes."""
        cc = np_cell_complex(5, 0.1)
        assert cc.shape[0] == 5

    def test_zero_probability(self):
        """Test `np_cell_complex` for zero probability."""
        cc = np_cell_complex(40, 0)
        assert cc.shape[0] == 40
        assert cc.shape[-1] == 0
