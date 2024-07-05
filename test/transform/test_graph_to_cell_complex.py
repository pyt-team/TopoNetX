"""Test graph to cell complex transformation."""

import networkx as nx

from toponetx.transform.graph_to_cell_complex import homology_cycle_cell_complex


class TestGraphToCellComplex:
    """Test graph to cell complex transformation."""

    def test_homology_cycle_cell_complex(self):
        """Test homology_cycle_cell_complex."""
        cx = homology_cycle_cell_complex(nx.karate_club_graph())
        assert len(cx.cells) != 0
