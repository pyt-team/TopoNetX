"""Test simplicial complex class."""


import unittest

import networkx as nx
import numpy as np

from toponetx import Simplex, SimplicialComplex


class TestSimplicialComplex(unittest.TestCase):
    def test_shape_property(self):
        # Test the shape property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.shape, [5, 6, 2])

    def test_dim_property(self):
        # Test the dim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.dim, 2)

    def test_maxdim_property(self):
        # Test the maxdim property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        self.assertEqual(sc.maxdim, 2)

    def test_nodes_property(self):
        # Test the nodes property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        nodes = sc.nodes
        self.assertEqual(len(nodes), 5)
        self.assertIn([1], nodes)
        self.assertNotIn([8], nodes)

    def test_simplices_property(self):
        # Test the simplices property of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        simplices = sc.simplices
        self.assertEqual(len(simplices), 13)
        self.assertIn([1, 2, 3], simplices)
        self.assertIn([2, 3, 4], simplices)
        self.assertIn([0, 1], simplices)
        # ... add more assertions based on the expected simplices

    def test_is_maximal(self):
        # Test the is_maximal method of the SimplicialComplex class
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        is_maximal = sc.is_maximal([1, 2, 3])
        self.assertTrue(is_maximal)

    def test_add_simplex(self):
        """ "Test add_simplex method."""
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
        """ "Test remove_maximal_simplex method."""
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

    def test_skeleton_and_cliques(self):
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

    def test_incidence_matrix(self):
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

    def test_remove_maximal_simplex(self):
        # Test the _remove_maximal_simplex method of the SimplicialComplex class
        SC = SimplicialComplex()
        SC.add_simplex((1, 2, 3, 4), weight=1)
        c1 = Simplex((1, 2, 3, 4, 5))
        SC.add_simplex(c1)
        SC.remove_maximal_simplex((1, 2, 3, 4, 5))
        self.assertNotIn((1, 2, 3, 4, 5), SC)

    def test_get_boundaries(self):
        # Test the get_boundaries method of the SimplicialComplex class
        simplices = [(1, 2, 3), (2, 3, 4), (0, 1)]
        boundaries = SimplicialComplex.get_boundaries(simplices)
        self.assertIn(frozenset((1, 2)), boundaries)
        self.assertIn(frozenset((1, 3)), boundaries)
        self.assertIn(frozenset((2, 3)), boundaries)
        # ... add more assertions based on the expected boundaries

    def test_get_cofaces(self):
        # Test the get_cofaces method of the SimplicialComplex class
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        cofaces = SC.get_cofaces([1, 2, 4], codimension=1)
        self.assertIn(frozenset({1, 2, 3, 4}), cofaces)
        self.assertNotIn((3, 4, 8), cofaces)
        # ... add more assertions based on the expected cofaces

    def test_get_star(self):
        # Test the get_star method of the SimplicialComplex class
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        star = SC.get_star([1, 2, 4])
        self.assertIn(frozenset({1, 2, 4}), star)
        self.assertIn(frozenset({1, 2, 3, 4}), star)
        # ... add more assertions based on the expected star

    def test_set_simplex_attributes(self):
        # Test the set_simplex_attributes method of the SimplicialComplex class
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        d = {(1, 2, 3): "red", (1, 2, 4): "blue"}
        SC.set_simplex_attributes(d, name="color")
        self.assertEqual(SC[(1, 2, 3)]["color"], "red")
        # ... add more assertions based on the expected simplex attributes


if __name__ == "__main__":
    unittest.main()
