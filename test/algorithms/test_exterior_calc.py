"""
Test exterior calculus operators.

This module contains unit tests for the user-facing
`ExteriorCalculusOperators` class, verifying:

- correct construction of DEC operators (coboundaries, Hodge stars, codifferentials),
- valid shapes, basic algebraic identities, and symmetry where expected,
- error handling for invalid inputs and unsupported configurations,
- triangle-mesh metric presets route to the geometry backend and expose FEM helpers,
- additional operators: up/down Laplacians, Hodge--Dirac, inner products, norms,
  and Hodge decomposition utilities.

These tests are intentionally lightweight and avoid asserting PDE convergence.
They aim to validate correctness of API behavior and fundamental operator properties.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, spmatrix

import toponetx as tnx
from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators
from toponetx.algorithms.exterior_calculus.metric import (
    DiagonalHodgeStar,
    MetricSpec,
    TriangleMesh3DBackend,
)


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


def build_single_triangle_sc_no_positions() -> tnx.SimplicialComplex:
    """Build a single-triangle simplicial complex without vertex positions.

    Returns
    -------
    tnx.SimplicialComplex
        A simplicial complex with one triangle and no `position` attribute.
    """
    return tnx.SimplicialComplex([[0, 1, 2]])


def build_two_triangle_square_sc() -> tnx.SimplicialComplex:
    """Build a two-triangle square patch embedded in R^3.

    This is the smallest nontrivial mesh with an interior edge shared by two
    triangles. It is useful to exercise triangle-mesh backends where some
    circumcentric dual quantities differ from a single-triangle boundary mesh.

    Returns
    -------
    tnx.SimplicialComplex
        A simplicial complex with two triangles and vertex positions.
    """
    faces = [[0, 1, 2], [0, 2, 3]]
    sc = tnx.SimplicialComplex(faces)
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [1.0, 1.0, 0.0],
        3: [0.0, 1.0, 0.0],
    }
    sc.set_simplex_attributes(pos, name="position")
    return sc


def _counts(sc: tnx.SimplicialComplex) -> tuple[int, int, int]:
    """Return (n0, n1, n2) counts for a simplicial complex.

    Parameters
    ----------
    sc : tnx.SimplicialComplex
        The simplicial complex.

    Returns
    -------
    tuple[int, int, int]
        Counts of vertices, edges, and triangles.
    """
    n0 = len(list(sc.skeleton(0)))
    n1 = len(list(sc.skeleton(1)))
    n2 = len(list(sc.skeleton(2)))
    return n0, n1, n2


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

        assert isinstance(d0, spmatrix)
        assert isinstance(d1, spmatrix)

        n0, n1, n2 = _counts(sc)
        assert d0.shape == (n1, n0)  # C^0 -> C^1
        assert d1.shape == (n2, n1)  # C^1 -> C^2
        assert n2 == 1

    def test_d_matrix_unsigned_is_abs_signed(self):
        """Return unsigned coboundary equal to abs(signed coboundary)."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)

        d0_signed = ops.d_matrix(0, signed=True)
        d0_unsigned = ops.d_matrix(0, signed=False)

        assert isinstance(d0_signed, spmatrix)
        assert isinstance(d0_unsigned, spmatrix)

        # For TopoNetX, unsigned incidence should correspond to absolute values.
        assert (abs(d0_signed) != d0_unsigned).nnz == 0

    def test_d0_d1_composition_is_zero(self):
        """Satisfy the cochain complex identity d1 @ d0 = 0."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc)

        d0 = ops.d_matrix(0, signed=True)
        d1 = ops.d_matrix(1, signed=True)

        comp = (d1 @ d0).tocsr()
        assert comp.nnz == 0

    def test_identity_metric_hodge_star(self):
        """Return identity Hodge stars when metric is identity."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        S0 = ops.hodge_star(0)
        S1 = ops.hodge_star(1)
        S2 = ops.hodge_star(2)

        n0, n1, n2 = _counts(sc)
        assert S0.shape == (n0, n0)
        assert S1.shape == (n1, n1)
        assert S2.shape == (n2, n2)

        assert np.allclose(S0.diagonal(), 1.0)
        assert np.allclose(S1.diagonal(), 1.0)
        assert np.allclose(S2.diagonal(), 1.0)

    def test_hodge_star_inverse_matches_identity_metric(self):
        """Return inverse star equal to star for identity preset."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        S0 = ops.hodge_star(0, inverse=False)
        S0_inv = ops.hodge_star(0, inverse=True)

        assert np.allclose(S0.diagonal(), S0_inv.diagonal())

    def test_codifferential_shape_and_type(self):
        """Return codifferentials with correct shapes."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, n1, n2 = _counts(sc)

        delta1 = ops.codifferential(1)
        delta2 = ops.codifferential(2)

        assert isinstance(delta1, spmatrix)
        assert isinstance(delta2, spmatrix)

        assert delta1.shape == (n0, n1)  # C^1 -> C^0
        assert delta2.shape == (n1, n2)  # C^2 -> C^1

    def test_codifferential_invalid_k_raises(self):
        """Raise ValueError for codifferential with k < 1."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        with pytest.raises(ValueError):
            ops.codifferential(0)

        with pytest.raises(ValueError):
            ops.codifferential(-1)

    def test_laplace_up_and_down_shapes_and_boundary_behavior(self):
        """Return up/down Laplacians with correct shapes and boundary zeros."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, n1, n2 = _counts(sc)

        Lup0 = ops.laplace_up(0)
        Ldn0 = ops.laplace_down(0)
        assert isinstance(Lup0, spmatrix)
        assert isinstance(Ldn0, spmatrix)
        assert Lup0.shape == (n0, n0)
        assert Ldn0.shape == (n0, n0)
        # Boundary: down term at k=0 must be zero.
        assert Ldn0.tocsr().nnz == 0

        Lup2 = ops.laplace_up(2)
        Ldn2 = ops.laplace_down(2)
        assert isinstance(Lup2, spmatrix)
        assert isinstance(Ldn2, spmatrix)
        assert Lup2.shape == (n2, n2)
        assert Ldn2.shape == (n2, n2)
        # Boundary: up term at k=dim must be zero.
        assert Lup2.tocsr().nnz == 0

        Lup1 = ops.laplace_up(1)
        Ldn1 = ops.laplace_down(1)
        assert Lup1.shape == (n1, n1)
        assert Ldn1.shape == (n1, n1)

    def test_dec_laplacian_equals_up_plus_down(self):
        """Return Laplacian equal to sum of up/down pieces."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        for k in (0, 1, 2):
            L = ops.dec_hodge_laplacian(k).tocsr()
            Lup = ops.laplace_up(k).tocsr()
            Ldn = ops.laplace_down(k).tocsr()
            diff = (L - (Lup + Ldn)).tocsr()
            assert diff.nnz == 0

    def test_dec_laplacian_shapes(self):
        """Return Laplacians with correct shapes for k=0,1,2."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, n1, n2 = _counts(sc)

        L0 = ops.dec_hodge_laplacian(0)
        L1 = ops.dec_hodge_laplacian(1)
        L2 = ops.dec_hodge_laplacian(2)

        assert isinstance(L0, spmatrix)
        assert isinstance(L1, spmatrix)
        assert isinstance(L2, spmatrix)

        assert L0.shape == (n0, n0)
        assert L1.shape == (n1, n1)
        assert L2.shape == (n2, n2)

    def test_dec_laplacian_symmetry_identity_metric(self):
        """Return symmetric DEC Laplacians for identity metric."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        for k in (0, 1, 2):
            Lk = ops.dec_hodge_laplacian(k)
            diff = (Lk - Lk.T).tocsr()
            assert diff.nnz == 0

    def test_dec_laplacian_psd_energy_identity_metric(self):
        """Produce nonnegative quadratic energy u^T Delta u for identity metric."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        L0 = ops.dec_hodge_laplacian(0).tocsr()
        n0 = L0.shape[0]
        u = np.arange(1, n0 + 1, dtype=float)

        energy = float(u @ (L0 @ u))
        assert energy >= -1e-12

    def test_hodge_dirac_shapes(self):
        """Return Hodge--Dirac stacked operator with expected shape."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, n1, n2 = _counts(sc)

        D0 = ops.hodge_dirac(0)
        assert isinstance(D0, spmatrix)
        assert D0.shape == (n1, n0)  # only d0 block

        D1 = ops.hodge_dirac(1)
        assert isinstance(D1, spmatrix)
        assert D1.shape == (n0 + n2, n1)  # delta1 stacked with d1

        D2 = ops.hodge_dirac(2)
        assert isinstance(D2, spmatrix)
        assert D2.shape == (n1, n2)  # only delta2 block

    def test_inner_product_and_norm_identity_metric(self):
        """Match Euclidean dot product and L2 norm for identity stars."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, _, _ = _counts(sc)
        a = np.arange(1, n0 + 1, dtype=float)
        b = np.arange(2, n0 + 2, dtype=float)

        ip = ops.inner_product(0, a, b)
        assert np.isclose(ip, float(a @ b))

        na = ops.norm(0, a)
        assert np.isclose(na, float(np.linalg.norm(a)))

    def test_inner_product_shape_mismatch_raises(self):
        """Raise ValueError when inner product inputs have wrong length."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        a = np.ones(3, dtype=float)
        b = np.ones(4, dtype=float)
        with pytest.raises(ValueError):
            _ = ops.inner_product(0, a, b)

    def test_hodge_decomposition_reconstruction(self):
        """Return components that reconstruct the input cochain."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        _, n1, _ = _counts(sc)
        x = np.linspace(0.1, 1.0, n1)

        x_exact, x_coexact, x_harm = ops.hodge_decomposition(1, x, solver="spsolve")
        assert x_exact.shape == x.shape
        assert x_coexact.shape == x.shape
        assert x_harm.shape == x.shape

        recon = x_exact + x_coexact + x_harm
        assert np.allclose(recon, x, atol=1e-8, rtol=1e-8)

    def test_hodge_decomposition_invalid_k_raises(self):
        """Raise ValueError for invalid k in hodge_decomposition."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        x0 = np.ones(len(list(sc.skeleton(0))), dtype=float)
        with pytest.raises(ValueError):
            _ = ops.hodge_decomposition(-1, x0)

        with pytest.raises(ValueError):
            _ = ops.hodge_decomposition(99, x0)

    def test_invalid_k_raises(self):
        """Raise ValueError for invalid cochain degree."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)

        with pytest.raises(ValueError):
            ops.d_matrix(-1)

        with pytest.raises(ValueError):
            ops.d_matrix(3)

        with pytest.raises(ValueError):
            ops.dec_hodge_laplacian(-1)

        with pytest.raises(ValueError):
            ops.dec_hodge_laplacian(5)

        with pytest.raises(ValueError):
            ops.laplace_up(-1)

        with pytest.raises(ValueError):
            ops.laplace_down(99)

        with pytest.raises(ValueError):
            ops.hodge_dirac(123)

    def test_triangle_mesh_backend_basic_star(self):
        """Return valid circumcentric Hodge stars on a minimal triangle mesh."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        S0 = ops.hodge_star(0)
        S1 = ops.hodge_star(1)
        S2 = ops.hodge_star(2)

        n0, n1, n2 = _counts(sc)
        assert S0.shape == (n0, n0)
        assert S1.shape == (n1, n1)
        assert S2.shape == (n2, n2)

        # Circumcentric dual measures are nonnegative; triangle-area reciprocal is positive.
        assert np.all(S0.diagonal() >= 0.0)
        assert np.all(S1.diagonal() >= 0.0)
        assert np.all(S2.diagonal() > 0.0)

    def test_triangle_mesh_metric_requires_positions_raises(self):
        """Raise when triangle-mesh metric is used without vertex positions."""
        sc = build_single_triangle_sc_no_positions()

        # Constructing the backend occurs during ops init; missing position labels should fail.
        with pytest.raises((KeyError, ValueError, AttributeError)):
            _ = ExteriorCalculusOperators(
                sc, metric="circumcentric", pos_name="position"
            )

    def test_triangle_mesh_backend_exposes_fem_helpers(self):
        """Expose FEM helpers only for triangle-mesh presets."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        M0 = ops.fem_mass_matrix_0()
        K0 = ops.cotan_stiffness_0()

        assert isinstance(M0, spmatrix)
        assert isinstance(K0, spmatrix)
        assert M0.shape == (4, 4)
        assert K0.shape == (4, 4)

        # Mass and stiffness should be symmetric.
        assert (M0 - M0.T).tocsr().nnz == 0
        assert (K0 - K0.T).tocsr().nnz == 0

    def test_fem_helpers_raise_without_triangle_mesh_backend(self):
        """Raise RuntimeError for FEM helpers when backend is not triangle-mesh."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity", pos_name="position")

        with pytest.raises(RuntimeError):
            _ = ops.fem_mass_matrix_0()

        with pytest.raises(RuntimeError):
            _ = ops.cotan_stiffness_0()

        with pytest.raises(RuntimeError):
            _ = ops.riemannian_stiffness_0()

        with pytest.raises(RuntimeError):
            _ = ops.cotan_laplacian_0()

        with pytest.raises(RuntimeError):
            _ = ops.riemannian_laplacian_0()

    def test_cotan_laplacian_lumped_and_unlumped_shapes(self):
        """Return cotan Laplacian with correct shapes for lumped/unlumped modes."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        L_lumped = ops.cotan_laplacian_0(lumped=True)
        L_unlumped = ops.cotan_laplacian_0(lumped=False)

        assert isinstance(L_lumped, spmatrix)
        assert isinstance(L_unlumped, spmatrix)
        assert L_lumped.shape == (4, 4)
        assert L_unlumped.shape == (4, 4)

    def test_riemannian_laplacian_defaults_to_isotropic(self):
        """Return anisotropic Laplacian even when no tensors are provided."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        L = ops.riemannian_laplacian_0(lumped=True)
        assert isinstance(L, spmatrix)
        assert L.shape == (4, 4)

    def test_custom_star_backend_overrides_metric_selection(self):
        """Allow explicit star_backend to override automatic metric backend."""
        sc = build_single_triangle_sc_no_positions()

        # If a custom backend is provided, ops should not attempt to build the
        # triangle-mesh backend (and thus should not require positions).
        custom_backend = DiagonalHodgeStar(metric=MetricSpec(preset="identity"))
        ops = ExteriorCalculusOperators(
            sc,
            metric="circumcentric",
            pos_name="position",
            star_backend=custom_backend,
        )

        S0 = ops.hodge_star(0)
        assert isinstance(S0, spmatrix)
        assert np.allclose(S0.diagonal(), 1.0)

    def test_metric_none_gives_identity_stars(self):
        """Return identity stars when metric support is disabled (metric=None)."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric=None)

        S0 = ops.hodge_star(0)
        assert isinstance(S0, spmatrix)
        assert np.allclose(S0.diagonal(), 1.0)

    def test_hodge_star_returns_sparse_matrix_for_all_k(self):
        """Return sparse matrices for all valid cochain degrees."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        for k in (0, 1, 2):
            Sk = ops.hodge_star(k)
            assert isinstance(Sk, spmatrix)

    def test_identity_metric_returns_csr_for_d_matrix_if_backend_is_toponetx(self):
        """Return SciPy sparse matrix type from TopoNetX incidence (not necessarily CSR)."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        d0 = ops.d_matrix(0)
        assert isinstance(d0, spmatrix)
        d0_csr = d0.tocsr()
        assert isinstance(d0_csr, csr_matrix)

    # ------------------------------------------------------------------
    # Extra coverage to hit remaining branches in ExteriorCalculusOperators
    # ------------------------------------------------------------------

    def test_normalize_metric_accepts_metricspec(self):
        """Accept MetricSpec inputs without modification."""
        ms = MetricSpec(preset="identity")
        out = ExteriorCalculusOperators._normalize_metric(ms)
        assert isinstance(out, MetricSpec)
        assert out.preset == "identity"

    def test_normalize_metric_accepts_preset_string(self):
        """Accept preset strings and convert to MetricSpec."""
        out = ExteriorCalculusOperators._normalize_metric("diagonal")
        assert isinstance(out, MetricSpec)
        assert out.preset == "diagonal"

    def test_normalize_metric_type_error(self):
        """Raise TypeError for unsupported metric types."""
        with pytest.raises(TypeError):
            _ = ExteriorCalculusOperators._normalize_metric(123)  # type: ignore[arg-type]

    def test_select_backend_diagonal_builds_diagonal_backend(self):
        """Diagonal preset builds DiagonalHodgeStar backend when weights are provided."""
        sc = build_single_triangle_sc()

        metric = MetricSpec(
            preset="diagonal",
            diagonal_weights={0: np.ones(3)},
        )

        ops = ExteriorCalculusOperators(sc, metric=metric)

        S0 = ops.hodge_star(0)
        assert isinstance(S0, csr_matrix)
        assert np.allclose(S0.diagonal(), 1.0)

    def test_select_backend_triangle_presets_build_triangle_backend(self):
        """Route triangle-mesh presets to TriangleMesh3DBackend."""
        sc = build_single_triangle_sc()
        for preset in ("barycentric_lumped", "circumcentric", "voronoi"):
            ops = ExteriorCalculusOperators(sc, metric=preset, pos_name="position")
            assert isinstance(ops.star_backend, TriangleMesh3DBackend)

    def test_select_backend_unsupported_preset_raises(self):
        """Raise ValueError for unsupported preset name."""
        sc = build_single_triangle_sc()
        with pytest.raises(ValueError):
            _ = ExteriorCalculusOperators(sc, metric="not-a-real-preset")  # type: ignore[arg-type]

    def test_euclidean_preset_constructs_triangle_backend_and_star_raises(self):
        """Construct triangle backend for 'euclidean' and raise for unsupported star preset."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="euclidean", pos_name="position")

        # Selection path coverage:
        assert isinstance(ops.star_backend, TriangleMesh3DBackend)

        # TriangleMesh3DBackend.star does not implement "euclidean" for stars.
        with pytest.raises(ValueError):
            _ = ops.hodge_star(0)

    # ------------------------------------------------------------------
    # Optional coverage for "_matrix" naming convention (if you add aliases)
    # ------------------------------------------------------------------

    def test_matrix_aliases_match_primary_methods_if_present(self):
        """Match alias *_matrix methods to primary methods when available."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        if hasattr(ops, "hodge_star_matrix"):
            assert (ops.hodge_star_matrix(1) != ops.hodge_star(1)).nnz == 0

        if hasattr(ops, "codifferential_matrix"):
            assert (ops.codifferential_matrix(1) != ops.codifferential(1)).nnz == 0

        if hasattr(ops, "laplace_up_matrix"):
            assert (ops.laplace_up_matrix(1) != ops.laplace_up(1)).nnz == 0

        if hasattr(ops, "laplace_down_matrix"):
            assert (ops.laplace_down_matrix(1) != ops.laplace_down(1)).nnz == 0

        if hasattr(ops, "dec_hodge_laplacian_matrix"):
            assert (
                ops.dec_hodge_laplacian_matrix(1) != ops.dec_hodge_laplacian(1)
            ).nnz == 0

        if hasattr(ops, "hodge_dirac_matrix"):
            assert (ops.hodge_dirac_matrix(1) != ops.hodge_dirac(1)).nnz == 0
