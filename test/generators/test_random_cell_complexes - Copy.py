"""Test the random cell complex generators."""
from toponetx.generators.random_cell_complexes import cc_np_complex


class TestCcNpComplex:
    """Test the `cc_np_complex` function."""

    def test_zero_nodes(self):
        """Test `cc_np_complex` for zero nodes."""
        cc = cc_np_complex(0, 0.1)
        assert cc.shape == (0,0,0)

    def test_non_zero_nodes(self):
        """Test `cc_np_complex` for nonzero nodes."""
        cc = cc_np_complex(5, 0.1)
        assert cc.shape[0] == 5
        
    def test_non_zero_nodes(self):
        """Test `cc_np_complex` for zero probability."""
        cc = cc_np_complex(40, 0)
        assert cc.shape[0] == 40

        assert cc.shape[-1] == 0
