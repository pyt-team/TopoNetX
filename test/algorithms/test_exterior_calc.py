"""
Test exterior calculus operators.

This module contains unit tests for the user-facing
`ExteriorCalculusOperators` class, verifying:

- correct construction of DEC operators (coboundaries, Hodge stars, codifferentials),
- valid shapes, basic algebraic identities, and symmetry where expected,
- error handling for invalid inputs and unsupported configurations,
- triangle-mesh metric presets route to the geometry backend and expose FEM helpers.

These tests are intentionally lightweight and avoid asserting PDE convergence.
They aim to validate correctness of API behavior and fundamental operator properties.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, spmatrix

from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators
from toponetx.algorithms.exterior_calculus.metric import (
    DiagonalHodgeStar,
    MetricSpec,
    TriangleMesh3DBackend,
)
from toponetx.classes.simplicial_complex import SimplicialComplex


def build_single_triangle_sc() -> SimplicialComplex:
    """Build a single-triangle simplicial complex embedded in R^3.

    Returns
    -------
    SimplicialComplex
        A simplicial complex with one triangle and vertex positions.
    """
    sc = SimplicialComplex([[0, 1, 2]])
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
    }
    sc.set_simplex_attributes(pos, name="position")
    return sc


def build_single_triangle_sc_no_positions() -> SimplicialComplex:
    """Build a single-triangle simplicial complex without vertex positions.

    Returns
    -------
    SimplicialComplex
        A simplicial complex with one triangle and no `position` attribute.
    """
    return SimplicialComplex([[0, 1, 2]])


def build_two_triangle_square_sc() -> SimplicialComplex:
    """Build a two-triangle square patch embedded in R^3.

    This is the smallest nontrivial mesh with an interior edge shared by two
    triangles. It is useful to exercise triangle-mesh backends where some
    circumcentric dual quantities differ from a single-triangle boundary mesh.

    Returns
    -------
    SimplicialComplex
        A simplicial complex with two triangles and vertex positions.
    """
    sc = SimplicialComplex([[0, 1, 2], [0, 2, 3]])
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [1.0, 1.0, 0.0],
        3: [0.0, 1.0, 0.0],
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

        assert isinstance(d0, spmatrix)
        assert isinstance(d1, spmatrix)

        n0, n1, n2 = sc.shape
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

        n0, n1, n2 = sc.shape
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

        n0, n1, n2 = sc.shape

        delta1 = ops.codifferential_matrix(1)
        delta2 = ops.codifferential_matrix(2)

        assert isinstance(delta1, spmatrix)
        assert isinstance(delta2, spmatrix)

        assert delta1.shape == (n0, n1)  # C^1 -> C^0
        assert delta2.shape == (n1, n2)  # C^2 -> C^1

    def test_codifferential_invalid_k_raises(self):
        """Raise ValueError for codifferential with k < 1."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        with pytest.raises(ValueError):
            ops.codifferential_matrix(0)

        with pytest.raises(ValueError):
            ops.codifferential_matrix(-1)

    def test_dec_laplacian_shapes(self):
        """Return Laplacians with correct shapes for k=0,1,2."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        n0, n1, n2 = sc.shape

        L0 = ops.dec_hodge_laplacian_matrix(0)
        L1 = ops.dec_hodge_laplacian_matrix(1)
        L2 = ops.dec_hodge_laplacian_matrix(2)

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
            Lk = ops.dec_hodge_laplacian_matrix(k)
            diff = (Lk - Lk.T).tocsr()
            assert diff.nnz == 0

    def test_dec_laplacian_psd_energy_identity_metric(self):
        """Produce nonnegative quadratic energy u^T Delta u for identity metric."""
        sc = build_two_triangle_square_sc()
        ops = ExteriorCalculusOperators(sc, metric="identity")

        L0 = ops.dec_hodge_laplacian_matrix(0).tocsr()
        n0 = L0.shape[0]
        u = np.arange(1, n0 + 1, dtype=float)

        energy = float(u @ (L0 @ u))
        assert energy >= -1e-12

    def test_invalid_k_raises(self):
        """Raise ValueError for invalid cochain degree."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc)

        with pytest.raises(ValueError):
            ops.d_matrix(-1)

        with pytest.raises(ValueError):
            ops.d_matrix(3)

        with pytest.raises(ValueError):
            ops.dec_hodge_laplacian_matrix(-1)

        with pytest.raises(ValueError):
            ops.dec_hodge_laplacian_matrix(5)

    def test_triangle_mesh_backend_basic_star(self):
        """Return valid circumcentric Hodge stars on a minimal triangle mesh."""
        sc = build_single_triangle_sc()
        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")

        S0 = ops.hodge_star(0)
        S1 = ops.hodge_star(1)
        S2 = ops.hodge_star(2)

        n0, n1, n2 = sc.shape
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
        with pytest.raises(KeyError):
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
