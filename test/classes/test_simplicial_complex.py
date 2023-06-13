"""Test simplicial complex class."""

import unittest

import hypernetx as hnx
import networkx as nx
import numpy as np
import scipy
import trimesh
from gudhi import SimplexTree

from toponetx import (
    CombinatorialComplex,
    Simplex,
    SimplicialComplex,
    TopoNetXError,
    stanford_bunny,
)


class TestSimplicialComplex(unittest.TestCase):
    """Test SimplicialComplex class."""

    def test_iterable_simplices(self):
        """Test TypeError for simplices not iterable."""
        with self.assertRaises(TypeError):
            SimplicialComplex(simplices=1)

    def test_shape_property(self):
        """Test shape property."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.shape, (5, 6, 2))

        sc = SimplicialComplex()
        self.assertEqual(sc.shape, tuple())

    def test_dim_property(self):
        """Test dim property."""
        # Test the dim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.dim, 2)

    def test_maxdim_property(self):
        """Test maxdim property."""
        # Test the maxdim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.dim, 2)

    def test_nodes_property(self):
        """Test nodes property."""
        # Test the nodes property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        nodes = sc.nodes
        self.assertEqual(len(nodes), 5)
        self.assertIn([1], nodes)
        self.assertNotIn([8], nodes)

    def test_simplices_property(self):
        """Test simplices property."""
        # Test the simplices property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        simplices = sc.simplices
        self.assertEqual(len(simplices), 13)
        self.assertIn([1, 2, 3], simplices)
        self.assertIn([2, 3, 4], simplices)
        self.assertIn([0, 1], simplices)
        # ... add more assertions based on the expected simplices

    def test_is_maximal(self):
        """Test is_maximal method."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertTrue(sc.is_maximal([1, 2, 3]))

        with self.assertRaises(ValueError):
            sc.is_maximal([1, 2, 3, 4])

    def test_contructor_using_graph(self):
        """Test input a networkx graph in the constructor."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G)

        assert (0, 1) in SC.simplices
        assert (2, 5) in SC.simplices
        assert (5, 4) in SC.simplices
        assert SC.simplices[(5, 4)]["weight"] == 5

    def test_skeleton_raise_errors(self):
        """Test skeleton raises."""
        with self.assertRaises(ValueError):
            G = nx.Graph()
            G.add_edge(0, 1)
            G.add_edge(2, 5)
            G.add_edge(5, 4, weight=5)
            SC = SimplicialComplex(G)
            SC.skeleton(-2)

        with self.assertRaises(ValueError):
            G = nx.Graph()
            G.add_edge(0, 1)
            G.add_edge(2, 5)
            G.add_edge(5, 4, weight=5)
            SC = SimplicialComplex(G)
            SC.skeleton(2)

    def test_str(self):
        """Test str method."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        assert (
            str(SC)
        ) == f"Simplicial Complex with shape {SC.shape} and dimension {SC.dim}"

    def test_rep_str(self):
        """Test repr string."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        assert (repr(SC)) == "SimplicialComplex(name=graph complex)"
        assert SC.name == "graph complex"

    def test_getittem__(self):
        """Test __getitem__ and __setitem__ methods."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        SC.add_simplex((1, 2, 3), heat=5)
        # with self.assertRaises(ValueError):
        assert SC[(1, 2, 3)]["heat"] == 5
        with self.assertRaises(KeyError):
            SC[(1, 2, 3, 4, 5)]["heat"]

    def test_setitem__(self):
        """Test __getitem__ and __setitem__ methods."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        SC.add_simplex((1, 2, 3), heat=5)
        # with self.assertRaises(ValueError):

        SC[(1, 2, 3)]["heat"] = 6

        assert SC[(1, 2, 3)]["heat"] == 6

        SC[(2, 5)]["heat"] = 1

        assert SC[(2, 5)]["heat"] == 1

        s = Simplex((1, 2, 3, 4), heat=1)
        SC.add_simplex(s)
        assert SC[(1, 2, 3, 4)]["heat"] == 1

        s = Simplex(("A"), heat=1)
        SC.add_simplex(s)
        assert SC["A"]["heat"] == 1

    def add_simplices_from(self):
        """Test add simplices from."""
        with self.assertRaises(ValueError):
            SC = SimplicialComplex()
            SC._add_simplices_from(4)

    def test__insert_node(self):
        """Test _insert_node."""
        SC = SimplicialComplex()
        SC._insert_node(9)
        assert 9 in SC
        with self.assertRaises(ValueError):
            SC._insert_node((1, 2))

        with self.assertRaises(ValueError):
            s = Simplex((1, 2, 3, 4))
            SC._insert_node(s)

        SC = SimplicialComplex()
        assert SC.dim == -1
        SC._insert_node(9)
        assert SC.dim == 0
        assert SC[9]["is_maximal"] is True

    def test_add_node(self):
        """Test add node."""
        SC = SimplicialComplex()
        SC.add_node(9)
        assert 9 in SC
        with self.assertRaises(ValueError):
            SC.add_node((1, 2))

        with self.assertRaises(ValueError):
            s = Simplex((1, 2, 3, 4))
            SC.add_node(s)

    def test_add_simplex(self):
        """Test add_simplex method."""
        # create a SimplicialComplex object with no simplices
        SC = SimplicialComplex()

        # add a simplex using the add_simplex() method
        SC.add_simplex([1, 2, 3])

        # assert that the simplex was added correctly
        assert (1, 2, 3) in SC.simplices
        assert [1, 2, 3] in SC.simplices
        assert (1, 2) in SC.simplices
        assert (1, 3) in SC.simplices
        assert (2, 3) in SC.simplices
        assert (1,) in SC.simplices
        assert (2,) in SC.simplices
        assert (3,) in SC.simplices

        SC.add_simplex((4, 5, 6), heat=5)
        assert (4, 5, 6) in SC.simplices
        assert [4, 5, 6] in SC.simplices
        assert (4, 5) in SC.simplices
        assert (4, 6) in SC.simplices
        assert (5, 6) in SC.simplices
        assert (4,) in SC.simplices
        assert (5,) in SC.simplices
        assert (6,) in SC.simplices

        # simplex cannot contain unhashable elements
        with self.assertRaises(TypeError):
            SC.add_simplex([[1, 2], [2, 3]])

        # simplex cannot contain duplicate nodes
        with self.assertRaises(ValueError):
            SC.add_simplex((1, 2, 2))

    def test_remove_maximal_simplex(self):
        """Test remove_maximal_simplex method."""
        # create a SimplicialComplex object with a few simplices
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # remove a maximal simplex using the remove_maximal_simplex() method
        SC.remove_maximal_simplex([1, 2, 3])

        # check that the simplex was removed correctly (tuple)
        assert (1, 2, 3) not in SC.simplices

        # create a SimplicialComplex object with a few simplices
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4]])

        # remove a maximal simplex from the complex
        SC.remove_maximal_simplex([2, 3, 4])

        # check that the simplex was removed correctly (list)
        assert [2, 3, 4] not in SC.simplices

        # check after the add_simplex method
        SC = SimplicialComplex()
        SC.add_simplex((1, 2, 3, 4), weight=1)
        c1 = Simplex((1, 2, 3, 4, 5))
        SC.add_simplex(c1)
        SC.remove_maximal_simplex((1, 2, 3, 4, 5))
        self.assertNotIn((1, 2, 3, 4, 5), SC)

        # check removal with Simplex
        SC = SimplicialComplex()
        SC.add_simplex((1, 2, 3, 4), weight=1)
        c1 = Simplex((1, 2, 3, 4, 5))
        SC.add_simplex(c1)
        SC.remove_maximal_simplex(c1)
        self.assertNotIn((1, 2, 3, 4, 5), SC)

        # check error when simplex not in complex
        with self.assertRaises(KeyError):
            SC = SimplicialComplex()
            SC.add_simplex((1, 2, 3, 4), weight=1)
            SC.remove_maximal_simplex([5, 6, 7])

        # only maximal simplices can be removed
        with self.assertRaises(ValueError):
            SC = SimplicialComplex()
            SC.add_simplex((1, 2, 3, 4), weight=1)
            SC.remove_maximal_simplex((1, 2, 3))

    def test_skeleton_and_cliques(self):
        """Test skeleton and cliques methods."""
        G = nx.karate_club_graph()
        cliques = list(nx.enumerate_all_cliques(G))

        nodes = [i for i in cliques if len(i) == 1]
        edges = [i for i in cliques if len(i) == 2]
        faces = [i for i in cliques if len(i) == 3]
        threefaces = [i for i in cliques if len(i) == 4]
        fourfaces = [i for i in cliques if len(i) == 5]

        SC = SimplicialComplex(cliques)
        assert len(SC.skeleton(rank=0)) == len(nodes)
        assert len(SC.skeleton(rank=1)) == len(edges)
        assert len(SC.skeleton(rank=2)) == len(faces)
        assert len(SC.skeleton(rank=3)) == len(threefaces)
        assert len(SC.skeleton(rank=4)) == len(fourfaces)

    def test_incidence_matrix_1(self):
        """Test incidence_matrix shape and values."""
        # create a SimplicialComplex object with a few simplices
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the incidence matrix using the boundary_matrix() method
        B2 = SC.incidence_matrix(rank=2)

        assert B2.shape == (6, 2)

        # assert that the incidence matrix is correct
        np.testing.assert_array_equal(
            B2.toarray(),
            np.array([[0, 1, -1, 1, 0, 0], [0, 0, 0, 1, -1, 1]]).T,
        )

    def test_hodge_laplacian_matrix(self):
        """Test hodge_laplacian_matrix shape and values."""
        # create a SimplicialComplex object with a few simplices
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the Hodge Laplacian using the hodge_laplacian_matrix() method
        L_hodge = SC.hodge_laplacian_matrix(rank=0)

        assert L_hodge.shape == (5, 5)

        D = np.diag([1, 3, 3, 3, 2])
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(L_hodge.toarray(), D - A)

    def test_adjacency_matrix(self):
        """Test adjacency_matrix shape and values."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        A = SC.adjacency_matrix(rank=0)

        assert A.shape == (5, 5)

        np.testing.assert_array_equal(
            A.toarray(),
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ),
        )

    def test_get_boundaries(self):
        """Test the get_boundaries method."""
        simplices = [(1, 2, 3), (2, 3, 4), (0, 1)]
        with self.assertRaises(TypeError):
            SimplicialComplex.get_boundaries(1)

        boundaries = SimplicialComplex.get_boundaries(simplices)
        self.assertIn(frozenset((1, 2)), boundaries)
        self.assertIn(frozenset((1, 3)), boundaries)
        self.assertIn(frozenset((2, 3)), boundaries)

        # test for min dim/max dim combinations
        boundaries = SimplicialComplex.get_boundaries(simplices, min_dim=2)
        assert boundaries == {frozenset({1, 2, 3}), frozenset({2, 3, 4})}

        boundaries = SimplicialComplex.get_boundaries(simplices, min_dim=1, max_dim=1)
        assert boundaries == {
            frozenset({3, 4}),
            frozenset({2, 3}),
            frozenset({1, 2}),
            frozenset({0, 1}),
            frozenset({2, 4}),
            frozenset({1, 3}),
        }

        boundaries = SimplicialComplex.get_boundaries(simplices, max_dim=0)
        assert boundaries == {
            frozenset({2}),
            frozenset({3}),
            frozenset({1}),
            frozenset({4}),
            frozenset({0}),
        }

    def test_get_cofaces(self):
        """Test the get_cofaces method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        cofaces = SC.get_cofaces([1, 2, 4], codimension=1)
        self.assertIn(frozenset({1, 2, 3, 4}), cofaces)
        self.assertNotIn((3, 4, 8), cofaces)
        # ... add more assertions based on the expected cofaces

    def test_get_star(self):
        """Test the get_star method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        star = SC.get_star([1, 2, 4])
        self.assertIn(frozenset({1, 2, 4}), star)
        self.assertIn(frozenset({1, 2, 3, 4}), star)
        # ... add more assertions based on the expected star

    def test_set_simplex_attributes(self):
        """Test the set_simplex_attributes method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        d = {(1, 2, 3): "red", (1, 2, 4): "blue"}
        SC.set_simplex_attributes(d, name="color")
        self.assertEqual(SC[(1, 2, 3)]["color"], "red")

        # test for non-existing simplex
        d = {(3, 4, 5): "Nope"}
        SC.set_simplex_attributes(d, name="color")  # should not raise an error
        SC.set_simplex_attributes(d)  # should not raise an error

    def test_get_edges_from_matrix(self):
        """Test the get_edges_from_matrix method."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1)]

        edges = SimplicialComplex().get_edges_from_matrix(matrix)

        self.assertEqual(set(edges), set(expected_edges))

    def test_incidence_matrix(self):
        """Test incidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 5])
        with self.assertRaises(ValueError):
            SC.incidence_matrix(10)
        with self.assertRaises(ValueError):
            SC.incidence_matrix(-1)

        B1 = SC.incidence_matrix(1)
        B2 = SC.incidence_matrix(2)
        assert B1.shape == tuple(SC.shape[:2])
        assert B2.shape == tuple(SC.shape[1:3])
        # B1 : 1->0
        # B2 : 2->1
        assert np.sum(abs(B1.dot(B2))) == 0  # boundary of boundary = 0
        # for the last to be true, order of bases must be canonical in both matrices

        row, col, B1 = SC.incidence_matrix(1, index=True)

        assert (len(row), len(col)) == B1.shape

        B0 = SC.incidence_matrix(0)
        assert B0.shape[-1] == len(SC.skeleton(0))

        assert np.sum(B0.dot(B1)) == 0  # boundary of boundary = 0

    def test_coincidence_matrix_2(self):
        """Test coincidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        B1T = SC.coincidence_matrix(1)
        B2T = SC.coincidence_matrix(2)
        assert B1T.shape == tuple(SC.shape[:2][::-1])
        assert B2T.shape == tuple(SC.shape[1:3][::-1])
        # B1T : 0->1
        # B2T : 1->2
        assert np.sum(abs(B2T.dot(B1T))) == 0  # coboundary of coboundary = 0
        # for the last to be true, order of bases must be canonical in both matrices

        row, col, B1T = SC.coincidence_matrix(1, index=True)

        assert (len(col), len(row)) == B1T.shape

        B0 = SC.coincidence_matrix(0)
        assert B0.shape[0] == len(SC.skeleton(0))

    def test_is_triangular_mesh(self):
        """Test is_triangular_mesh."""
        SC = stanford_bunny()
        self.assertTrue(SC.is_triangular_mesh())

        # test for non triangular mesh
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        self.assertFalse(SC.is_triangular_mesh())

    def test_to_trimesh(self):
        """Test to_trimesh."""
        SC = stanford_bunny()
        trimesh_obj = SC.to_trimesh()
        assert len(trimesh_obj.vertices) == len(SC.skeleton(0))

        # test for non triangular mesh
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        with self.assertRaises(TopoNetXError):
            SC.to_trimesh()

    def test_laplace_beltrami_operator(self):
        """Test laplace_beltrami_operator."""
        SC = stanford_bunny()

        laplacian_matrix = SC.laplace_beltrami_operator()

        self.assertIsInstance(laplacian_matrix, np.ndarray)

    def test_from_nx_graph(self):
        """Test from networkx graph."""
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(3, 4, weight=4)
        SC = SimplicialComplex.from_nx_graph(G)
        self.assertEqual(SC[(1, 2)]["weight"], 2)

    def test_is_connected(self):
        """Test is connected."""
        SC = stanford_bunny()
        self.assertTrue(SC.is_connected())

    def test_simplicial_closure_of_hypergraph(self):
        """Test simplicial_closure_of_hypergraph."""
        hg = hnx.Hypergraph([[1, 2, 3, 4], [1, 2, 3]], static=True)
        sc = SimplicialComplex.simplicial_closure_of_hypergraph(hg)
        expected_simplices = [
            (1,),
            (2,),
            (3,),
            (4,),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (1, 2, 3, 4),
        ]
        assert len(sc.simplices) == len(expected_simplices)

    def test_to_hypergraph(self):
        """Convert a SimplicialComplex to a hypergraph and compare the number of edges."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        expected_result = hnx.Hypergraph(
            {
                "e0": [1, 2],
                "e1": [1, 3],
                "e2": [1, 4],
                "e3": [2, 3],
                "e4": [2, 4],
                "e5": [2, 5],
                "e6": [1, 2, 3],
                "e7": [1, 2, 4],
            },
            name="",
        )
        result = SC.to_hypergraph()
        assert len(result.edges) == len(expected_result.edges)

    def test_to_combinatorial_complex(self):
        """Convert a SimplicialComplex to a CombinatorialComplex and compare the number of cells and nodes."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        expected_result = CombinatorialComplex()
        expected_result.add_cell((1, 2, 3), rank=2)
        expected_result.add_cell((1, 2, 4), rank=2)
        expected_result.add_cell((2, 5), rank=1)
        expected_result.add_cell((1, 2), rank=1)
        expected_result.add_cell((1, 3), rank=1)
        expected_result.add_cell((1, 4), rank=1)
        expected_result.add_cell((2, 3), rank=1)
        expected_result.add_cell((2, 4), rank=1)
        expected_result.add_cell((2, 5), rank=1)
        result = SC.to_combinatorial_complex()
        assert len(result.cells) == len(expected_result.cells)
        assert len(result.nodes) == len(expected_result.nodes)

    def test_from_gudhi(self):
        """Create a SimplicialComplex from a Gudhi SimplexTree and compare the number of simplices."""
        tree = SimplexTree()
        tree.insert([1, 2, 3, 5])
        expected_result = SimplicialComplex()
        expected_result.add_simplex((1, 2, 3, 5))
        result = SimplicialComplex.from_gudhi(tree)
        assert len(result.simplices) == len(expected_result.simplices)

    def test_add_elements_from_nx_graph(self):
        """Add elements from a networkx graph to a SimplicialComplex and compare the number of simplices."""
        c1 = Simplex((1, 2, 3))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c3])
        G = nx.Graph()
        G.add_edge(4, 5)
        expected_result = SimplicialComplex([c1, c3, Simplex((4, 5))])
        SC.add_elements_from_nx_graph(G)
        assert len(SC.simplices) == len(expected_result.simplices)

    def test_restrict_to_nodes(self):
        """Restrict a SimplicialComplex to the specified nodes and compare the result with the expected SimplicialComplex."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        node_set = [1, 2, 3, 4]
        expected_result = SimplicialComplex(
            [
                Simplex((1,)),
                Simplex((2,)),
                Simplex((3,)),
                Simplex((4,)),
                Simplex((1, 2)),
                Simplex((1, 3)),
                Simplex((1, 4)),
                Simplex((2, 3)),
                Simplex((2, 4)),
                Simplex((1, 2, 3)),
                Simplex((1, 2, 4)),
            ]
        )
        result = SC.restrict_to_nodes(node_set)
        assert len(result.simplices) == len(expected_result.simplices)

    def test_get_all_maximal_simplices(self):
        """Retrieve all maximal simplices from a SimplicialComplex and compare the number of simplices."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        result = SC.get_all_maximal_simplices()
        assert len(result) == 3

    def test_coincidence_matrix(self):
        """Test for coincidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([0, 1, 2])

        row, col, B1 = SC.coincidence_matrix(1, index=True)

        assert B1.shape == (3, 3)
        assert np.allclose(
            B1.toarray(),
            np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]]),
        )

        B2 = SC.coincidence_matrix(2)

        assert B2.shape == (1, 3)
        assert np.allclose(B2.toarray(), np.array([[1.0, -1.0, 1.0]]))

    def test_down_laplacian_matrix(self):
        """Test the down_laplacian_matrix method of SimplicialComplex."""
        # Test case 1: Rank is within valid range
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])
        rank = 1
        signed = True
        weight = None
        index = False

        result = SC.down_laplacian_matrix(rank, signed, weight, index)

        # Assert the result is of type scipy.sparse.csr.csr_matrix
        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        rank = 0

        with self.assertRaises(ValueError):
            SC.down_laplacian_matrix(rank, signed, weight, index)

        # Test case 3: Rank is above the valid range
        rank = 5

        with self.assertRaises(ValueError):
            SC.down_laplacian_matrix(rank, signed, weight, index)

    def test_adjacency_matrix2(self):
        """Test the adjacency_matrix method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])

        # Test case 1: Rank is within valid range
        rank = 1
        signed = False
        weight = None
        index = False

        result = SC.adjacency_matrix(rank, signed, weight, index)

        # Assert the result is of type scipy.sparse.csr.csr_matrix
        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        rank = -1

        with self.assertRaises(ValueError):
            SC.adjacency_matrix(rank, signed, weight, index)

        # Test case 3: Rank is above the valid range
        rank = 5

        with self.assertRaises(ValueError):
            SC.adjacency_matrix(rank, signed, weight, index)

    def test_coadjacency_matrix(self):
        """Test the coadjacency_matrix method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])
        # Test case 1: Rank is within valid range
        rank = 1
        signed = False
        weight = None
        index = False

        result = SC.coadjacency_matrix(rank, signed, weight, index)

        # Assert the result is of type scipy.sparse.csr.csr_matrix
        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        rank = 0

        with self.assertRaises(ValueError):
            SC.coadjacency_matrix(rank, signed, weight, index)

        # Test case 3: Rank is above the valid range
        rank = 5

        with self.assertRaises(ValueError):
            SC.coadjacency_matrix(rank, signed, weight, index)

    def test_restrict_to_simplices(self):
        """Test the restrict_to_simplices method of SimplicialComplex."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        SC1 = SC.restrict_to_simplices([c1, (2, 4)])
        assert len(SC1.simplices) == 9
        assert Simplex((1, 2, 3)) in SC1.simplices
        assert Simplex((2, 4)) in SC1.simplices
        assert Simplex((1, 2)) in SC1.simplices
        assert Simplex((1, 3)) in SC1.simplices
        assert Simplex((2, 3)) in SC1.simplices
        assert Simplex((1,)) in SC1.simplices
        assert Simplex((2,)) in SC1.simplices
        assert Simplex((3,)) in SC1.simplices
        assert Simplex((4,)) in SC1.simplices
        assert Simplex((1, 2, 3, 4)) not in SC1.simplices
        assert Simplex((1, 2, 5)) not in SC1.simplices

    def test_clone(self):
        """Test the clone method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3], color="red")
        SC2 = SC.clone()
        assert SC2 is not SC
        assert SC2[(1, 2, 3)] is not SC[(1, 2, 3)]
        SC2.remove_maximal_simplex([1, 2, 3])
        assert 1 in SC
        assert (1, 2, 3) in SC


if __name__ == "__main__":
    unittest.main()
