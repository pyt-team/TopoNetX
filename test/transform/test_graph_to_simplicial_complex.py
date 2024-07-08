"""Test graph to simplicial complex transformation."""

import networkx as nx
import pytest

from toponetx.transform.graph_to_simplicial_complex import (
    graph_2_clique_complex,
    graph_2_neighbor_complex,
    graph_to_clique_complex,
    graph_to_neighbor_complex,
    weighted_graph_to_vietoris_rips_complex,
)


class TestGraphToSimplicialComplex:
    """Test graph to simplicial complex transformation."""

    def test_graph_to_neighbor_complex(self):
        """Test graph_2_neighbor_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_to_neighbor_complex(G)

        assert sc.dim == 2
        assert (0, 1) in sc
        assert (0, 2) in sc

    def test_graph_to_clique_complex(self):
        """Test graph_2_clique_complex."""
        G = nx.Graph()
        G.graph["label"] = 12

        G.add_node(0, label=5)
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_to_clique_complex(G)

        assert sc.dim == 2
        assert (0, 2, 3) in sc
        assert (0, 1, 2) in sc

        # attributes are copied
        assert sc.complex["label"] == 12
        assert sc[(0,)]["label"] == 5
        assert sc[(0, 1)]["weight"] == 10

        sc = graph_to_clique_complex(G, max_rank=1)

        assert sc.dim == 1
        assert (0, 2, 3) not in sc
        assert (0, 1, 2) not in sc

    def test_graph_2_neighbor_complex(self):
        """Test graph_2_neighbor_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        with pytest.deprecated_call():
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

        with pytest.deprecated_call():
            sc = graph_2_clique_complex(G)

        assert sc.dim == 2
        assert (0, 2, 3) in sc
        assert (0, 1, 2) in sc

        with pytest.deprecated_call():
            sc = graph_2_clique_complex(G, max_rank=1)

        assert sc.dim == 1
        assert (0, 2, 3) not in sc
        assert (0, 1, 2) not in sc

    def test_weighted_graph_to_vietoris_rips_complex(self):
        """Test weighted_graph_to_vietoris_rips_complex."""

        def generate_weighted_graph_for_vietoris_rips():
            """Create a weighted graph in networkx to test the Vietoris-Rips simplicial complex lift.

            Returns
            -------
            networkx graph
                A undirected weighted graph.
            """
            G = nx.Graph()
            # We add a 3-clique to the graph with pairwise weights 1.0
            G.add_edge(0, 1, weight=1.0)
            G.add_edge(0, 2, weight=1.0)
            G.add_edge(0, 3, weight=1.0)
            G.add_edge(1, 2, weight=1.0)
            G.add_edge(1, 3, weight=1.0)
            G.add_edge(2, 3, weight=1.0)
            # We add a 2-clique to the graph with pairwise weights 2.0
            G.add_edge(4, 5, weight=2.0)
            G.add_edge(4, 6, weight=2.0)
            G.add_edge(5, 6, weight=2.0)
            # We add an edge between the two cliques with weight 3.0
            G.add_edge(3, 4, weight=3.0)
            # We add a new vertex 7 to the graph that forms a 3 clique with the previous
            # vertices 4, 5, and 6 if we take r > 4.0
            G.add_edge(4, 7, weight=4.0)
            G.add_edge(5, 7, weight=4.0)
            G.add_edge(6, 7, weight=4.0)
            return G

        def generate_expected_simplices_for_vietoris_rips_complex(r):
            """Generate expected and unexpected simplices for the Vietoris-Rips complex of the test.

            This function returns a pair of lists of tuples containing the expected and unexpected simplices of the
            Vietoris-Rips persistence diagram of the graph generated with the function
            generate_weighted_graph_for_vietoris_rips depending on the radius r.

            Parameters
            ----------
            r : float
                Radius of the Vietoris-Rips complex.

            Returns
            -------
            expected_and_unexpected_simplices: tuple[list[tuple], list[tuple]]
                A tuple containing the lists of the expected and unexpected simplices (first and second lists,
                respectively) for the Vietoris-Rips complex.
            """
            expected_vertices = list(range(7))
            expected_edges = []
            unexpected_edges = []
            expected_triangles = []
            unexpected_triangles = []
            expected_tetrahedra = []
            unexpected_tetrahedra = []
            # Now, for each radius, we include or exclude simplices depending on its value.
            # Radius 1
            edges = expected_edges if r >= 1 else unexpected_edges
            triangles = expected_triangles if r >= 1 else unexpected_triangles
            tetrahedra = expected_tetrahedra if r >= 1 else unexpected_tetrahedra
            edges.extend([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
            triangles.extend([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
            tetrahedra.append((0, 1, 2, 3))
            # Radius 2
            edges = expected_edges if r >= 2 else unexpected_edges
            triangles = expected_triangles if r >= 2 else unexpected_triangles
            edges.extend([(4, 5), (4, 6), (5, 6)])
            triangles.append((4, 5, 6))
            # Radius 3
            edges = expected_edges if r >= 3 else unexpected_edges
            edges.append((3, 4))
            # Radius 4
            edges = expected_edges if r >= 4 else unexpected_edges
            triangles = expected_triangles if r >= 4 else unexpected_triangles
            tetrahedra = expected_tetrahedra if r >= 4 else unexpected_tetrahedra
            edges.extend([(4, 7), (5, 7), (6, 7)])
            triangles.extend([(4, 5, 7), (4, 6, 7), (5, 6, 7)])
            tetrahedra.append((4, 5, 6, 7))
            expected_simplices = (
                expected_vertices
                + expected_edges
                + expected_triangles
                + expected_tetrahedra
            )
            unexpected_simplices = (
                unexpected_edges + unexpected_triangles + unexpected_tetrahedra
            )
            return expected_simplices, unexpected_simplices

        radii_to_check = [
            0,
            1,
            2,
            3,
            4,
        ]  # Four possible different configurations for the Vietoris-Rips complex of the
        # associated graph.
        weighted_graph = generate_weighted_graph_for_vietoris_rips()
        for radius in radii_to_check:
            sc = weighted_graph_to_vietoris_rips_complex(weighted_graph, radius)
            (
                expected_simplices,
                unexpected_simplices,
            ) = generate_expected_simplices_for_vietoris_rips_complex(radius)
            for simplex in expected_simplices:
                assert simplex in sc
            for simplex in unexpected_simplices:
                assert simplex not in sc
