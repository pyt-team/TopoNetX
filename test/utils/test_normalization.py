"""Test normalization."""

import numpy as np
from scipy.sparse import csr_matrix

from toponetx.utils.normalization import (
    compute_bunch_normalized_matrices,
    compute_kipf_adjacency_normalized_matrix,
    compute_laplacian_normalized_matrix,
    compute_x_laplacian_normalized_matrix,
)


def test_compute_laplacian_normalized_matrix():
    """Test normalize laplacian."""
    adjacency_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    # Calculate the degree matrix
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))

    # Calculate the Laplacian matrix
    L = np.array(degree_matrix - adjacency_matrix)

    L = csr_matrix(L).asfptype()
    normalized_L = compute_laplacian_normalized_matrix(L)
    expected_result = [
        [0.5, -0.25, 0.0, 0.0, 0.0, -0.25],
        [-0.25, 0.5, -0.25, 0.0, 0.0, 0.0],
        [0.0, -0.25, 0.5, -0.25, 0.0, 0.0],
        [0.0, 0.0, -0.25, 0.5, -0.25, 0.0],
        [0.0, 0.0, 0.0, -0.25, 0.5, -0.25],
        [-0.25, 0.0, 0.0, 0.0, -0.25, 0.5],
    ]

    assert np.allclose(normalized_L.toarray(), expected_result)


def test_compute_x_laplacian_normalized_matrix():
    """Test normalize up or down laplacian."""
    L = csr_matrix([[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]])
    Lx = csr_matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    normalized_Lx = compute_x_laplacian_normalized_matrix(L, Lx)
    expected_result = [[0.25, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.25, 0.0]]
    assert np.allclose(normalized_Lx.toarray(), expected_result)

    # Test case 2
    L = csr_matrix([[4.0, 0], [0.0, 4.0]])
    Lx = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    normalized_Lx = compute_x_laplacian_normalized_matrix(L, Lx)
    expected_result = [[0.0, 0.25], [0.25, 0.0]]
    assert np.allclose(normalized_Lx.toarray(), expected_result)


def test_compute_kipf_adjacency_normalized_matrix():
    """Test kipf_adjacency_matrix_normalization."""
    # Test case 1
    A_opt = [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ]

    normalized_A_opt = compute_kipf_adjacency_normalized_matrix(csr_matrix(A_opt))
    expected_result = [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    ]

    assert np.allclose(normalized_A_opt.toarray(), expected_result)

    normalized_A_opt = compute_kipf_adjacency_normalized_matrix(
        csr_matrix(A_opt), add_identity=True
    )
    expected_result = [
        [0.33333333, 0.33333333, 0.0, 0.0, 0.0, 0.33333333],
        [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
        [0.0, 0.33333333, 0.33333333, 0.33333333, 0.0, 0.0],
        [0.0, 0.0, 0.33333333, 0.33333333, 0.33333333, 0.0],
        [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
        [0.33333333, 0.0, 0.0, 0.0, 0.33333333, 0.33333333],
    ]

    assert np.allclose(normalized_A_opt.toarray(), expected_result)


def test_compute_bunch_normalized_matrice_numpy_arrays():
    """Unit tests for bunch_normalization function numpy arrays."""
    B1 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    B2 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    B1N, B1TN, B2N, B2TN = compute_bunch_normalized_matrices(B1, B2)

    assert np.allclose(
        B1N,
        np.array([[0.1, 0.0, 0.1], [0.0, 0.125, 0.125], [0.16666667, 0.16666667, 0.0]]),
    )

    assert np.allclose(
        B1TN,
        np.array([[0.2, 0.0, 0.33333333], [0.0, 0.125, 0.16666667], [0.3, 0.375, 0.0]]),
    )

    assert np.allclose(
        B2N,
        np.array(
            [
                [0.33333333, 0.0, 0.33333333],
                [0.0, 0.33333333, 0.0],
                [0.33333333, 0.33333333, 0.33333333],
            ]
        ),
    )

    assert np.allclose(
        B2TN,
        np.array(
            [[0.5, 0.0, 0.33333333], [0.0, 1.0, 0.33333333], [0.5, 0.0, 0.33333333]]
        ),
    )


def test_compute_bunch_normalized_matrice_scipy_arrays():
    """Unit tests for bunch_normalization function scipy arrays."""
    B1 = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    B2 = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    B1N, B1TN, B2N, B2TN = compute_bunch_normalized_matrices(B1, B2)

    assert np.allclose(
        B1N.toarray(),
        [[0.1, 0.0, 0.1], [0.0, 0.125, 0.125], [0.16666667, 0.16666667, 0.0]],
    )

    assert np.allclose(
        B1TN.toarray(),
        [[0.2, 0.0, 0.33333333], [0.0, 0.125, 0.16666667], [0.3, 0.375, 0.0]],
    )

    assert np.allclose(
        B2N.toarray(),
        [
            [0.33333333, 0.0, 0.33333333],
            [0.0, 0.33333333, 0.0],
            [0.33333333, 0.33333333, 0.33333333],
        ],
    )

    assert np.allclose(
        B2TN.toarray(),
        np.array(
            [[0.5, 0.0, 0.33333333], [0.0, 1.0, 0.33333333], [0.5, 0.0, 0.33333333]]
        ),
    )
