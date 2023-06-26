"""Test normalization."""

import numpy as np
import scipy.sparse.linalg as spl
from scipy.sparse import coo_matrix, csr_matrix, diags

from toponetx.utils.normalization import (
    _compute_B1_normalized,
    _compute_B1T_normalized,
    _compute_B2_normalized,
    _compute_B2T_normalized,
    _compute_D1,
    _compute_D2,
    _compute_D3,
    _compute_D5,
    asymmetric_kipf_normalization,
    asymmetric_xu_normalization,
    bunch_normalization,
    kipf_adjacency_matrix_normalization,
    normalize_laplacian,
    normalize_x_laplacian,
)


def test_normalize_laplacian():
    """Test normalize laplacian."""
    adjacency_matrix = np.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )

    # Calculate the degree matrix
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))

    # Calculate the Laplacian matrix
    L = np.array(degree_matrix - adjacency_matrix)

    L = csr_matrix(L).asfptype()
    normalized_L = normalize_laplacian(L)
    expected_result = csr_matrix(
        [
            [0.5, -0.25, 0.0, 0.0, 0.0, -0.25],
            [-0.25, 0.5, -0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, -0.25, 0.0, 0.0],
            [0.0, 0.0, -0.25, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, -0.25, 0.5, -0.25],
            [-0.25, 0.0, 0.0, 0.0, -0.25, 0.5],
        ]
    )
    assert np.allclose(normalized_L.toarray(), expected_result.toarray())


def test_normalize_x_laplacian():
    """Test normalize up or down laplacian."""
    L = csr_matrix([[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]])
    Lx = csr_matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    normalized_Lx = normalize_x_laplacian(L, Lx)
    expected_result = csr_matrix([[0.25, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.25, 0.0]])
    assert np.allclose(normalized_Lx.toarray(), expected_result.toarray())

    # Test case 2
    L = csr_matrix([[4.0, 0], [0.0, 4.0]])
    Lx = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    normalized_Lx = normalize_x_laplacian(L, Lx)
    expected_result = csr_matrix([[0.0, 0.25], [0.25, 0.0]])
    assert np.allclose(normalized_Lx.toarray(), expected_result.toarray())


def test_kipf_adjacency_matrix_normalization():
    """Test kipf_adjacency_matrix_normalization."""
    # Test case 1
    A_opt = np.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )
    normalized_A_opt = kipf_adjacency_matrix_normalization(coo_matrix(A_opt))
    expected_result = coo_matrix(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]
    )

    assert np.allclose(normalized_A_opt.toarray(), expected_result.toarray())

    normalized_A_opt = kipf_adjacency_matrix_normalization(
        coo_matrix(A_opt), add_identity=True
    )
    expected_result = coo_matrix(
        [
            [0.33333333, 0.33333333, 0.0, 0.0, 0.0, 0.33333333],
            [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0],
            [0.0, 0.33333333, 0.33333333, 0.33333333, 0.0, 0.0],
            [0.0, 0.0, 0.33333333, 0.33333333, 0.33333333, 0.0],
            [0.0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
            [0.33333333, 0.0, 0.0, 0.0, 0.33333333, 0.33333333],
        ]
    )
    assert np.allclose(normalized_A_opt.toarray(), expected_result.toarray())


def test_asymmetric_kipf_normalization():
    """Unit test for asymmetric_kipf_normalization function."""
    # Test case 1: Sparse matrix input
    A = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]])
    expected_result = coo_matrix(
        [[0.0, 0.70710678, 0.70710678], [0.70710678, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    normalized_matrix = asymmetric_kipf_normalization(A, is_sparse=True)
    assert np.allclose(normalized_matrix.toarray(), expected_result.toarray())

    # Test case 2: Dense matrix input
    B = np.array([[0, 2, 1], [1, 0, 0], [0, 1, 0]])
    expected_result = np.array(
        [[0.0, 0.89442719, 0.4472136], [0.4472136, 0.0, 0.0], [0.0, 0.89442719, 0.0]]
    )
    normalized_matrix = asymmetric_kipf_normalization(B, is_sparse=False)
    assert np.allclose(normalized_matrix, expected_result)

    # Test case 3: Empty matrix
    C = np.array([])
    expected_result = np.array([])
    normalized_matrix = asymmetric_kipf_normalization(C)
    assert np.allclose(normalized_matrix, expected_result)

    # Test case 4: Single-row matrix
    D = np.array([[0, 1, 0]])
    expected_result = np.array([[0.0, 1.0, 0.0]])
    normalized_matrix = asymmetric_kipf_normalization(D)
    assert np.allclose(normalized_matrix, expected_result)

    # Test case 5: Single-column matrix
    E = np.array([[0], [1], [0]])
    expected_result = np.array([[0.0], [1.0], [0.0]])
    normalized_matrix = asymmetric_kipf_normalization(E)
    assert np.allclose(normalized_matrix, expected_result)
