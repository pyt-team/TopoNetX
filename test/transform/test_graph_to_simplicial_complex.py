"""Test graph to simplicial complex transformation."""

import unittest

import networkx as nx

from toponetx.transform.graph_to_simplicial_complex import (
    graph_2_clique_complex,
    graph_2_neighbor_complex,
)


class TestGraphToSimplicialComplex(unittest.TestCase):
    """Test graph to simplicial complex transformation."""

    def test_graph_2_neighbor_complex(self):
        """Test graph_2_neighbor_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_2_neighbor_complex(G)

        assert sc.dim == 2
        assert (0, 1) in sc
        assert (0, 2) in sc

    def test_graph_2_clique_complex(self):
        """Test graph_2_clique_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_2_clique_complex(G)

        assert sc.dim == 2
        assert (0, 2, 3) in sc
        assert (0, 1, 2) in sc

        sc = graph_2_clique_complex(G, max_dim=2)

        assert sc.dim == 1
        assert (0, 2, 3) not in sc
        assert (0, 1, 2) not in sc

        return


if __name__ == "__main__":
    unittest.main()
