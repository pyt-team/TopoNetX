"""Test simplicial complex class."""


import unittest

import numpy as np

from toponetx import SimplicialComplex


class TestSimplicialComplex(unittest.TestCase):
    def test_add_simplex(self):
        """ "Test add_simplex method."""
        # create a SimplicialComplex object with no simplices
        sc = SimplicialComplex()

        # add a simplex using the add_simplex() method
        sc.add_simplex([1, 2, 3])

        # assert that the simplex was added correctly
        assert (1, 2, 3) in sc.simplices
        assert [1, 2, 3] in sc.simplices
        assert (1, 2) in sc.simplices
        assert (1, 3) in sc.simplices
        assert (2, 3) in sc.simplices
        assert (1,) in sc.simplices
        assert (2,) in sc.simplices
        assert (3,) in sc.simplices

    def test_remove_maximal_simplex(self):
        """ "Test remove_maximal_simplex method."""
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # remove a maximal simplex using the remove_maximal_simplex() method
        sc.remove_maximal_simplex([1, 2, 3])

        # check that the simplex was removed correctly (tuple)
        assert (1, 2, 3) not in sc.simplices

        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4]])

        # remove a maximal simplex from the complex
        sc.remove_maximal_simplex([2, 3, 4])

        # check that the simplex was removed correctly (list)
        assert [2, 3, 4] not in sc.simplices

    def test_incidence_matrix(self):
        """Test incidence_matrix shape and values."""
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the incidence matrix using the boundary_matrix() method
        inc_2 = sc.incidence_matrix(rank=2)

        assert inc_2.shape == (6, 2)

        # assert that the incidence matrix is correct
        np.testing.assert_array_equal(
            inc_2.toarray(),
            np.array([[0, 1, -1, 1, 0, 0], [0, 0, 0, 1, -1, 1]]).T,
        )

    def test_hodge_laplacian_matrix(self):
        """Test hodge_laplacian_matrix shape and values."""
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the Hodge Laplacian using the hodge_laplacian_matrix() method
        hl = sc.hodge_laplacian_matrix(rank=0)

        assert hl.shape == (5, 5)

        deg = np.diag([1, 3, 3, 3, 2])
        adj = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        # assert that the Hodge Laplacian is correct
        np.testing.assert_array_equal(hl.toarray(), deg - adj)

    def test_adjacency_matrix(self):
        """Test adjacency_matrix shape and values."""
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        adj_matrix = sc.adjacency_matrix(rank=0)

        assert adj_matrix.shape == (5, 5)

        # assert that the higher-order adjacency matrix is correct
        np.testing.assert_array_equal(
            adj_matrix.toarray(),
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
