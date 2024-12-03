"""Normalize of Laplacians, (co)adjacency, boundary matrices of complexes."""

import numpy as np
import scipy.sparse.linalg as spl
from numpy.linalg import pinv
from scipy.sparse import csr_matrix, diags

__all__ = [
    "_compute_B1T_normalized_matrix",
    "_compute_B1_normalized_matrix",
    "_compute_B2T_normalized_matrix",
    "_compute_B2_normalized_matrix",
    "_compute_D1",
    "_compute_D2",
    "_compute_D3",
    "_compute_D5",
    "compute_bunch_normalized_matrices",
    "compute_kipf_adjacency_normalized_matrix",
    "compute_laplacian_normalized_matrix",
    "compute_x_laplacian_normalized_matrix",
    "compute_xu_asymmetric_normalized_matrix",
]


def compute_laplacian_normalized_matrix(L: csr_matrix) -> csr_matrix:
    """Normalize the Laplacian matrix.

    Parameters
    ----------
    L : csr_matrix
        The Laplacian matrix.

    Returns
    -------
    csr_matrix
        The normalized Laplacian matrix.

    Notes
    -----
    This function normalizes the Laplacian matrix by dividing it by the largest eigenvalue.
    """
    topeigen_val = spl.eigsh(L.asfptype(), k=1, which="LM", return_eigenvectors=False)[
        0
    ]
    return L * (1.0 / topeigen_val)


def compute_x_laplacian_normalized_matrix(L: csr_matrix, Lx: csr_matrix) -> csr_matrix:
    """Normalize the up or down Laplacians.

    Parameters
    ----------
    L : csr_matrix
        The Laplacian matrix.
    Lx : csr_matrix
        The up or down Laplacian matrix.

    Returns
    -------
    csr_matrix
        The normalized up or down Laplacian matrix.

    Notes
    -----
    This function normalizes the up or down Laplacian matrices by dividing them
    by the largest eigenvalue of the Laplacian matrix.
    """
    if not L.shape[0] == L.shape[1]:
        raise ValueError("Laplacian matrix must be square.")

    topeig = spl.eigsh(L.asfptype(), k=1, which="LM", return_eigenvectors=False)[0]
    return Lx * (1.0 / topeig)


def compute_kipf_adjacency_normalized_matrix(
    A_opt: np.ndarray, add_identity: bool = False, identity_multiplier: float = 1.0
) -> csr_matrix:
    """Normalize the adjacency matrix using Kipf's normalization.

    Typically used to normalize adjacency matrices.

    Parameters
    ----------
    A_opt : np.ndarray
        The adjacency matrix.
    add_identity : bool, default=False
        Determines if the identity matrix is to be added to the adjacency matrix.
    identity_multiplier : float, default=1.0
        A multiplier of the identity. This parameter is helpful for higher order
        (co)adjacency matrices where the neighbor of a cell is obtained from multiple sources.

    Returns
    -------
    csr_matrix
        The normalized adjacency matrix.

    Notes
    -----
    This normalization is based on Kipf's formulation, which computes the row-sums,
    constructs a diagonal matrix D^(-0.5), and applies the normalization as D^(-0.5) * A_opt * D^(-0.5).
    """
    if add_identity:
        size = A_opt.shape[0]
        eye = diags(size * [identity_multiplier])
        A_opt = np.abs(A_opt) + eye
    else:
        A_opt = np.abs(A_opt)
    rowsum = np.array(A_opt.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = diags(r_inv_sqrt)
    return A_opt.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def compute_xu_asymmetric_normalized_matrix(
    B: np.ndarray | csr_matrix, is_sparse: bool = True
) -> np.ndarray | csr_matrix:
    """Compute Xu's normalized asymmetric matrix.

    Typically used to normalize boundary operators.

    Parameters
    ----------
    B : np.ndarray or csr_matrix
        The asymmetric matrix.
    is_sparse : bool, default=True
        If True, treat B as a sparse matrix.

    Returns
    -------
    np.ndarray or csr_matrix
        The normalized asymmetric matrix.

    Notes
    -----
    This normalization is based on Xu's formulation, which computes the diagonal matrix D^(-1)
    and multiplies it with the asymmetric matrix B.
    """
    if is_sparse:
        rowsum = np.array(np.abs(B).sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = diags(r_inv)
        return r_mat_inv.dot(B)

    Di = np.sum(np.abs(B), axis=1)
    Di2 = np.diag(np.power(Di, -1))
    Di2[np.isinf(Di2)] = 0.0
    B = np.array(B)
    return Di2.dot(B)


def compute_bunch_normalized_matrices(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> tuple[
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
]:
    """Get Bunch normalization.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1 of a complex.

    Returns
    -------
    B1 : np.ndarray or csr_matrix
        Normalized B1: C1->C0.
    B1T : np.ndarray or csr_matrix
        Normalized B1T: C0->C1.
    B2 : np.ndarray or csr_matrix
        Normalized B2: C2->C1.
    B2T : np.ndarray or csr_matrix
        Normalized B2T: C1->C2.

    References
    ----------
    .. [1] Schaub, Benson, Horn, Lippner, Jadbabaie.
        Random walks on simplicial complexes and the normalized hodge 1-laplacian.
        Paper: https://epubs.siam.org/doi/10.1137/18M1201019

    .. [2] Bunch, You, Fung, Singh.
        Simplicial 2-Complex Convolutional Neural Networks.
        Paper: https://openreview.net/forum?id=TLbnsKrt6J-
    """
    B1N = _compute_B1_normalized_matrix(B1, B2)
    B1TN = _compute_B1T_normalized_matrix(B1, B2)
    B2N = _compute_B2_normalized_matrix(B2)
    B2TN = _compute_B2T_normalized_matrix(B2)
    return B1N, B1TN, B2N, B2TN


def _compute_B1_normalized_matrix(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> np.ndarray | csr_matrix:
    """Compute normalized boundary matrix B1.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1 of a complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized B1: C1->C0.
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, np.ndarray):
        D1_pinv = pinv(D1)
    elif isinstance(B1, csr_matrix):
        D1_pinv = csr_matrix(pinv(D1.toarray()))
    return D1_pinv @ B1


def _compute_B1T_normalized_matrix(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> np.ndarray | csr_matrix:
    """Compute normalized transpose boundary matrix B1T.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0  of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1  of a complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized transpose boundary matrix B1T: C0->C1.
        This is the coboundary C0->C1.
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, np.ndarray):
        D1_pinv = pinv(D1)
    elif isinstance(B1, csr_matrix):
        D1_pinv = csr_matrix(pinv(D1.toarray()))
    else:
        raise TypeError("input type must be either np.ndarray or csr_matrix")
    return D2 @ B1.T @ D1_pinv


def _compute_B2_normalized_matrix(
    B2: np.ndarray | csr_matrix,
) -> np.ndarray | csr_matrix:
    """Compute normalized boundary matrix B2.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized boundary matrix B2: C2 -> C1.
    """
    D3 = _compute_D3(B2)
    return B2 @ D3


def _compute_B2T_normalized_matrix(
    B2: np.ndarray | csr_matrix,
) -> np.ndarray | csr_matrix:
    """Compute normalized transpose matrix operator B2T.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized transpose matrix operator B2T: C1->C2.
        This is the coboundary matrix: C1->C2.
    """
    D5 = _compute_D5(B2)
    if isinstance(B2, np.ndarray):
        D5_pinv = pinv(D5)
        return B2.T @ D5_pinv
    if isinstance(B2, csr_matrix):
        D5_pinv = csr_matrix(pinv(D5.toarray()))
        return B2.T @ D5_pinv
    raise TypeError("input type must be either np.ndarray or csr_matrix")


def _compute_D1(B1: np.ndarray | csr_matrix, D2: diags) -> diags:
    """Compute the degree matrix D1.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    D2 : diags
        The degree matrix D2.

    Returns
    -------
    diags
        Normalized degree matrix D1.
    """
    if isinstance(B1, csr_matrix):
        rowsum = np.array((np.abs(B1) @ D2).sum(axis=1)).flatten()
        return 2 * diags(rowsum)
    if isinstance(B1, np.ndarray):
        rowsum = (np.abs(B1) @ D2).sum(axis=1)
        return 2 * np.diag(rowsum)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D2(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D2.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Normalized degree matrix D2.
    """
    if isinstance(B2, csr_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        return diags(np.maximum(rowsum, 1))
    if isinstance(B2, np.ndarray):
        rowsum = np.abs(B2).sum(axis=1)
        return np.diag(np.maximum(rowsum, 1))
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D3(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D3.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Degree matrix D3.
    """
    if isinstance(B2, csr_matrix):
        return diags(np.ones(B2.shape[1]) / 3)
    if isinstance(B2, np.ndarray):
        return np.diag(np.ones(B2.shape[1]) / 3)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D5(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D5.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Normalized degree matrix D5.
    """
    if isinstance(B2, csr_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        return diags(rowsum)
    if isinstance(B2, np.ndarray):
        rowsum = np.abs(B2).sum(axis=1)
        return np.diag(rowsum)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")
