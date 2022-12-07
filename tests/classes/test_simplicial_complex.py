import unittest

import numpy as np

from toponetx import SimplicialComplex


class TestSimplicialComplex(unittest.TestCase):
    def test_add_maximal_simplex(self):
        # create a SimplicialComplex object with no simplices
        sc = SimplicialComplex()

        # add a maximal simplex using the add_maximal_simplex() method
        sc.add_simplex([1, 2, 3])

        # assert that the simplex was added correctly
        assert (1, 2, 3) in sc.simplices
        assert (1, 2) in sc.simplices
        assert (1, 3) in sc.simplices
        assert (2, 3) in sc.simplices
        assert (1,) in sc.simplices
        assert (2,) in sc.simplices
        assert (3,) in sc.simplices

    def test_remove_maximal_simplex(self):
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # remove a maximal simplex using the remove_maximal_simplex() method
        sc.remove_maximal_simplex([1, 2, 3])

        assert (1, 2, 3) not in sc.simplices

    def test_add_maximal_simplex_2(self):
        # create a SimplicialComplex object
        sc = SimplicialComplex()

        # add a maximal simplex to the complex
        sc.add_simplex([1, 2, 3])

        # check that the simplex was added correctly
        assert [1, 2, 3] in sc.simplices

    def test_remove_maximal_simplex_2(self):
        # create a SimplicialComplex object
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4]])

        # remove a maximal simplex from the complex
        sc.remove_maximal_simplex([2, 3, 4])

        # check that the simplex was removed correctly
        assert [2, 3, 4] not in sc.simplices

    def test_boundary_matrix(self):
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the incidence matrix using the boundary_matrix() method
        B2 = sc.incidence_matrix(2)

        # assert that the incidence matrix is correct
        np.testing.assert_array_equal(
            B2.todense(), np.array([[1, -1, 1, 0, 0, 0], [0, 0, 1, -1, 1, 0]]).T
        )

    def test_hodge_laplacian(self):
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        # compute the Hodge Laplacian using the hodge_laplacian_matrix() method
        hl = sc.hodge_laplacian_matrix(0)

        # assert that the Hodge Laplacian is correct
        np.testing.assert_array_equal(
            hl.todense(),
            np.diag([3, 3, 3, 2, 1])
            - np.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_adjacency(self):
        # create a SimplicialComplex object with a few simplices
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        hoa = sc.adjacency_matrix(0)

        # assert that the higher-order adjacency matrix is correct
        np.testing.assert_array_equal(
            hoa.todense(),
            np.array(
                [
                    [0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
