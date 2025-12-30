"""Test exterior calculus operators module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

import toponetx as tnx
from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators


def solve_dirichlet(A, b, bd_idx, u_bd):
    """Solve a sparse linear system with Dirichlet boundary conditions.

    This helper solves ``A u = b`` while enforcing fixed values ``u[bd_idx] = u_bd``.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Square sparse matrix.
    b : np.ndarray
        Right-hand side vector.
    bd_idx : np.ndarray
        Indices of boundary degrees of freedom.
    u_bd : np.ndarray
        Dirichlet values at boundary indices, same length as ``bd_idx``.

    Returns
    -------
    np.ndarray
        Solution vector ``u`` with Dirichlet values imposed.
    """
    n = A.shape[0]
    is_bd = np.zeros(n, dtype=bool)
    is_bd[bd_idx] = True
    interior = np.where(~is_bd)[0]

    A_ii = A[interior][:, interior].tocsr()
    rhs = b[interior] - A[interior][:, bd_idx] @ u_bd
    u_i = spsolve(A_ii, rhs)

    u = np.zeros(n, dtype=float)
    u[interior] = u_i
    u[bd_idx] = u_bd
    return u


def build_square_grid_sc(n: int = 10) -> tuple[tnx.SimplicialComplex, np.ndarray]:
    """Build a triangulated unit square grid embedded in R^3.

    Parameters
    ----------
    n : int
        Number of vertices per side.

    Returns
    -------
    tuple[tnx.SimplicialComplex, np.ndarray]
        The simplicial complex and vertex positions of shape (n^2, 3).
    """
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)

    def vid(i: int, j: int) -> int:
        """Return the vertex index for grid coordinates (i, j).

        Parameters
        ----------
        i : int
            Grid index in the x-direction.
        j : int
            Grid index in the y-direction.

        Returns
        -------
        int
            Global vertex index in the flattened grid.
        """
        return j * n + i

    P2 = np.array([[x, y] for y in ys for x in xs], dtype=float)
    P = np.column_stack([P2, np.zeros((P2.shape[0],), dtype=float)])

    faces: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i, j + 1)
            d = vid(i + 1, j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])

    sc = tnx.SimplicialComplex(faces)
    sc.set_simplex_attributes(
        {k: P[k].tolist() for k in range(P.shape[0])},
        name="position",
    )
    return sc, P


def boundary_vertex_indices(P: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return boundary vertex indices for a unit-square mesh.

    Parameters
    ----------
    P : np.ndarray
        Vertex positions of shape (n_vertices, 3) or (n_vertices, 2).
    tol : float
        Tolerance for boundary detection.

    Returns
    -------
    np.ndarray
        1D array of boundary vertex indices.
    """
    x = P[:, 0]
    y = P[:, 1]
    is_bd = (
        (np.abs(x - 0.0) < tol)
        | (np.abs(x - 1.0) < tol)
        | (np.abs(y - 0.0) < tol)
        | (np.abs(y - 1.0) < tol)
    )
    return np.where(is_bd)[0]


class TestExteriorCalculusOperators:
    """Test ExteriorCalculusOperators behavior."""

    def test_dim_and_d_matrix_shapes(self):
        """Check dimension and coboundary matrix shapes on a triangle mesh."""
        sc, _ = build_square_grid_sc(n=6)
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        assert ops.dim == 2

        d0 = ops.d_matrix(0, signed=True)
        d1 = ops.d_matrix(1, signed=True)

        n0 = len(list(sc.skeleton(0)))
        n1 = len(list(sc.skeleton(1)))
        n2 = len(list(sc.skeleton(2)))

        assert d0.shape == (n1, n0)
        assert d1.shape == (n2, n1)

    def test_hodge_star_identity_when_metric_none(self):
        """Hodge star should be identity when metric is None."""
        sc, _ = build_square_grid_sc(n=5)
        ops = ExteriorCalculusOperators(sc, metric=None, pos_name="position")

        S0 = ops.hodge_star(0)
        n0 = len(list(sc.skeleton(0)))
        assert np.allclose(S0.diagonal(), np.ones(n0))

    def test_codifferential_requires_k_ge_1(self):
        """Codifferential should reject k <= 0."""
        sc, _ = build_square_grid_sc(n=5)
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        with pytest.raises(ValueError):
            ops.codifferential(0)

    def test_dec_laplacian_shapes(self):
        """DEC Laplacians should have correct shapes for k=0,1,2."""
        sc, _ = build_square_grid_sc(n=6)
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        n0 = len(list(sc.skeleton(0)))
        n1 = len(list(sc.skeleton(1)))
        n2 = len(list(sc.skeleton(2)))

        assert ops.dec_hodge_laplacian(0).shape == (n0, n0)
        assert ops.dec_hodge_laplacian(1).shape == (n1, n1)
        assert ops.dec_hodge_laplacian(2).shape == (n2, n2)

    def test_poisson_dirichlet_exact_solution_smoke(self):
        """Solve a Poisson problem and verify the solution is reasonable."""
        sc, P = build_square_grid_sc(n=18)
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        Delta0 = ops.dec_hodge_laplacian(0, signed=True).tocsr()

        x = P[:, 0]
        y = P[:, 1]
        u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        f = 2.0 * (np.pi**2) * u_exact

        bd = boundary_vertex_indices(P)
        u_bd = u_exact[bd]

        u_dec = solve_dirichlet((-Delta0).tocsr(), f, bd, u_bd)

        rel = np.linalg.norm(u_dec - u_exact) / (np.linalg.norm(u_exact) + 1e-30)
        assert rel < 0.1

    def test_triangle_mesh_backend_requires_positions(self):
        """Triangle-mesh presets should fail if vertex positions are missing."""
        sc = tnx.SimplicialComplex([[0, 1, 2]])
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        with pytest.raises((KeyError, ValueError)):
            ops.hodge_star(0)
