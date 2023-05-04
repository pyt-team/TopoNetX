"""Test simplicial complex class."""


import unittest

import networkx as nx
import numpy as np

from toponetx import SimplicialComplex


class TestSimplicialComplex(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
