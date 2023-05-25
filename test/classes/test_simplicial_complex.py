"""Test simplicial complex class."""

import unittest

import networkx as nx
import numpy as np

from toponetx import Simplex, SimplicialComplex


class TestSimplicialComplex(unittest.TestCase):
    """Test SimplicialComplex class."""

    def test_shape_property(self):
        """Test shape property."""
        # Test the shape property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.shape, [5, 6, 2])

    def test_dim_property(self):
        """Test dim property."""
        # Test the dim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.dim, 2)

    def test_maxdim_property(self):
        """Test maxdim property."""
        # Test the maxdim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.maxdim, 2)

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
        # Test the is_maximal method of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        is_maximal = sc.is_maximal([1, 2, 3])
        self.assertTrue(is_maximal)

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
        assert SC.maxdim == -1
        SC._insert_node(9)
        assert SC.maxdim == 0
        assert SC[9]["is_maximal"] is True

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
        boundaries = SimplicialComplex.get_boundaries(simplices)
        self.assertIn(frozenset((1, 2)), boundaries)
        self.assertIn(frozenset((1, 3)), boundaries)
        self.assertIn(frozenset((2, 3)), boundaries)
        # ... add more assertions based on the expected boundaries

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
        # ... add more assertions based on the expected simplex attributes

    def test_incidence_matrix(self):
        """Test incidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 5])
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


if __name__ == "__main__":
    unittest.main()
