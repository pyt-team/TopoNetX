"""
Test exterior calculus operators.

This module contains unit tests for the user-facing
`ExteriorCalculusOperators` class, verifying:
- correct construction of DEC operators,
- valid shapes and symmetry,
- basic error handling.

These tests are intentionally lightweight and do not attempt
to verify PDE convergence or numerical accuracy beyond sanity checks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import spmatrix

import toponetx as tnx
from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators


def build_single_triangle_sc() -> tnx.SimplicialComplex:
    """Build a single-triangle simplicial complex embedded in R^3.

    Returns
    -------
    tnx.SimplicialComplex
        A simplicial complex with one triangle and vertex positions.
    """
    faces = [[0, 1, 2]]
    sc = tnx.SimplicialComplex(faces)
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
    }
    sc.set_simplex_attributes(pos, name="position")
    return sc


class TestExteriorCalculusOperators:
    """Test the ExteriorCalculusOperators interface."""

    def test_dim_property(self):
        """Return correct topological dimension."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)
        assert ops.dim == 2

    def test_d_matrix_shapes(self):
        """Return coboundary matrices with consistent shapes."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)

        d0 = ops.d_matrix(0)
        d1 = ops.d_matrix(1)

        # TopoNetX may return CSR or CSC; both are valid SciPy sparse matrices.
        assert isinstance(d0, spmatrix)
        assert isinstance(d1, spmatrix)

        # C^0 -> C^1 and C^1 -> C^2 must align on the edge dimension.
        assert d0.shape[0] == d1.shape[1]  # number of edges

        # Single triangle complex has exactly one 2-simplex.
        assert d1.shape[0] == 1

    def test_identity_metric_hodge_star(self):
        """Return identity Hodge stars when metric is identity."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        S0 = ops.hodge_star(0)
        S1 = ops.hodge_star(1)
        S2 = ops.hodge_star(2)

        assert S0.shape == (3, 3)
        assert S1.shape[0] == S1.shape[1]
        assert S2.shape == (1, 1)

        assert np.allclose(S0.diagonal(), 1.0)
        assert np.allclose(S2.diagonal(), 1.0)

    def test_codifferential_shape(self):
        """Return codifferential with correct shape."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        delta1 = ops.codifferential(1)
        assert isinstance(delta1, spmatrix)
        assert delta1.shape == (3, 3)  # C^1 -> C^0: edges -> vertices

    def test_dec_laplacian_symmetry(self):
        """Return symmetric DEC Laplacian for identity metric."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        L0 = ops.dec_hodge_laplacian(0)
        assert isinstance(L0, spmatrix)

        # For identity stars: Delta0 = d0^T d0, which is symmetric.
        diff = L0 - L0.T
        assert diff.nnz == 0

    def test_invalid_k_raises(self):
        """Raise ValueError for invalid cochain degree."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)

        with pytest.raises(ValueError):
            ops.d_matrix(-1)

        with pytest.raises(ValueError):
            ops.dec_hodge_laplacian(5)

    def test_triangle_mesh_backend_basic_star(self):
        """Return valid circumcentric Hodge star on a minimal triangle mesh."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        S0 = ops.hodge_star(0)
        S1 = ops.hodge_star(1)
        S2 = ops.hodge_star(2)

        assert S0.shape == (3, 3)
        assert S2.shape == (1, 1)

        # Circumcentric dual areas/lengths should be nonnegative in this simple case.
        assert np.all(S0.diagonal() >= 0.0)
        assert np.all(S1.diagonal() >= 0.0)
        assert np.all(S2.diagonal() > 0.0)
