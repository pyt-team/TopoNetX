"""Test graph to simplicial complex transformation."""

import unittest

import networkx as nx

from toponetx.transform.graph_to_cell_complex import homology_cycle_cell_complex


class TestGraphToCellComplex(unittest.TestCase):
    """Test graph to simplicial complex transformation."""

    def test_homology_cycle_cell_complex(self):
        """Test homology_cycle_cell_complex."""
        cx = homology_cycle_cell_complex(nx.karate_club_graph())
        assert len(cx.cells) != 0


if __name__ == "__main__":
    unittest.main()
