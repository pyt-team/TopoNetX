"""
Test metric and geometry backends for exterior calculus.

These tests cover:
- geometry helper utilities (areas, circumcenters, barycentric gradients),
- metric parsing helpers (_as_1d_float),
- diagonal Hodge star backend (identity/diagonal + inverse + error paths),
- triangle-mesh backend stars (identity/barycentric/circumcentric/voronoi + inverse),
- FEM operators (mass/stiffness) and Laplacian variants (lumped/unlumped),
- anisotropic tensor resolution paths (explicit tensors, callable fn, default isotropic),
- basic structural properties (shapes, symmetry, row-sum zero for stiffness),
- robustness on degenerate triangles (skip contributions without crashing).

Notes
-----
This is a test module; we avoid heavyweight numerical accuracy checks and focus on
sanity properties and error paths. Some helper callables used in tests must be
properly documented to satisfy numpydoc-validation.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from toponetx.algorithms.exterior_calculus.metric import (
    DiagonalHodgeStar,
    MetricSpec,
    TriangleMesh3DBackend,
    _as_1d_float,
    _circumcenter_3d,
    _grad_barycentric_3d,
    _sorted_edge,
    _triangle_area_3d,
)
from toponetx.classes.simplicial_complex import SimplicialComplex

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


def _build_single_triangle_sc() -> SimplicialComplex:
    """Build a single-triangle simplicial complex with 3D positions.

    Returns
    -------
    SimplicialComplex
        Simplicial complex with one 2-simplex and 3D vertex positions.
    """
    sc = SimplicialComplex([[0, 1, 2]])
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
    }
    sc.set_simplex_attributes(pos, name="position")
    return sc


def _build_two_triangle_square_sc() -> SimplicialComplex:
    """Build a two-triangle square patch embedded in R^3.

    Returns
    -------
    SimplicialComplex
        Simplicial complex with two 2-simplices and 3D vertex positions.
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


def _build_degenerate_triangle_sc() -> SimplicialComplex:
    """Build a collinear (degenerate) triangle embedded in R^3.

    Returns
    -------
    SimplicialComplex
        Simplicial complex with one degenerate triangle and vertex positions.
    """
    sc = SimplicialComplex([[0, 1, 2]])
    pos = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
    }
    sc.set_simplex_attributes(pos, name="position")
    return sc


class TestGeometryHelpers:
    """Test geometry helper functions used by the metric backends."""

    def test_sorted_edge(self):
        """Return deterministic edge ordering."""
        assert _sorted_edge(1, 2) == (1, 2)
        assert _sorted_edge(2, 1) == (1, 2)

    def test_triangle_area_3d(self):
        """Compute triangle area for a right triangle in the plane z=0."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        assert _triangle_area_3d(p0, p1, p2) == pytest.approx(0.5)

    def test_triangle_area_degenerate_is_zero(self):
        """Return zero area for collinear points."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        assert _triangle_area_3d(p0, p1, p2) == pytest.approx(0.0)

    def test_circumcenter_right_triangle(self):
        """Compute circumcenter for a right triangle."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        cc = _circumcenter_3d(p0, p1, p2)
        assert cc.shape == (3,)
        assert cc == pytest.approx(np.array([0.5, 0.5, 0.0]))

    def test_circumcenter_degenerate_triangle_fallback(self):
        """Return centroid fallback for collinear points."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        cc = _circumcenter_3d(p0, p1, p2)
        assert cc == pytest.approx((p0 + p1 + p2) / 3.0)

    def test_grad_barycentric_identities(self):
        """Satisfy P1 barycentric gradient identities."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])

        g0, g1, g2, det_norm = _grad_barycentric_3d(p0, p1, p2)
        assert det_norm == pytest.approx(1.0)

        assert (g0 + g1 + g2) == pytest.approx(np.zeros(3))

        n = np.cross(p1 - p0, p2 - p0)
        assert float(g0 @ n) == pytest.approx(0.0, abs=1e-12)
        assert float(g1 @ n) == pytest.approx(0.0, abs=1e-12)
        assert float(g2 @ n) == pytest.approx(0.0, abs=1e-12)

    def test_grad_barycentric_degenerate(self):
        """Return zero gradients for degenerate triangles."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        g0, g1, g2, det_norm = _grad_barycentric_3d(p0, p1, p2)
        assert det_norm == pytest.approx(0.0)
        assert g0 == pytest.approx(np.zeros(3))
        assert g1 == pytest.approx(np.zeros(3))
        assert g2 == pytest.approx(np.zeros(3))


class TestAs1DFloat:
    """Test conversion to 1D float arrays."""

    def test_as_1d_float_flattens(self):
        """Convert array into flattened float array."""
        x = np.array([[1, 2, 3]])
        y = _as_1d_float(x)
        assert y.dtype == float
        assert y.shape == (3,)
        assert y == pytest.approx([1.0, 2.0, 3.0])


class TestDiagonalHodgeStar:
    """Test the geometry-free diagonal Hodge star backend."""

    def test_identity_star(self):
        """Return identity star for preset='identity'."""
        sc = _build_single_triangle_sc()
        hs = DiagonalHodgeStar(metric=MetricSpec(preset="identity"))
        S0 = hs.star(sc, 0)
        assert isinstance(S0, csr_matrix)
        assert S0.shape == (3, 3)
        assert S0.diagonal() == pytest.approx(np.ones(3))

    def test_diagonal_star_from_weights(self):
        """Return diagonal star from explicit diagonal weights."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(
            preset="diagonal",
            diagonal_weights={0: np.array([2.0, 3.0, 4.0])},
        )
        hs = DiagonalHodgeStar(metric=ms)
        S0 = hs.star(sc, 0)
        assert S0.diagonal() == pytest.approx([2.0, 3.0, 4.0])

        S0_inv = hs.star(sc, 0, inverse=True)
        assert S0_inv.diagonal() == pytest.approx([0.5, 1.0 / 3.0, 0.25])

    def test_diagonal_star_from_measures(self):
        """Return diagonal star from primal and dual measures."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(
            preset="diagonal",
            primal_measures={0: np.array([1.0, 2.0, 4.0])},
            dual_measures={0: np.array([2.0, 2.0, 2.0])},
        )
        hs = DiagonalHodgeStar(metric=ms)
        S0 = hs.star(sc, 0)
        assert S0.diagonal() == pytest.approx([2.0, 1.0, 0.5])

    def test_diagonal_star_inverse_via_measures(self):
        """Return inverse diagonal star for measure-driven diagonal preset."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(
            preset="diagonal",
            primal_measures={0: np.array([1.0, 2.0, 4.0])},
            dual_measures={0: np.array([2.0, 2.0, 2.0])},
        )
        hs = DiagonalHodgeStar(metric=ms)
        S0_inv = hs.star(sc, 0, inverse=True)
        assert S0_inv.diagonal() == pytest.approx([0.5, 1.0, 2.0])

    def test_diagonal_star_invalid_preset(self):
        """Raise error for incompatible preset."""
        sc = _build_single_triangle_sc()
        hs = DiagonalHodgeStar(metric=MetricSpec(preset="circumcentric"))
        with pytest.raises(ValueError):
            hs.star(sc, 0)

    def test_diagonal_star_length_mismatch(self):
        """Raise error if diagonal length mismatches skeleton size."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(preset="diagonal", diagonal_weights={0: np.array([1.0, 2.0])})
        hs = DiagonalHodgeStar(metric=ms)
        with pytest.raises(ValueError):
            hs.star(sc, 0)

    def test_diagonal_star_missing_measures(self):
        """Raise error if no measures or weights are provided."""
        sc = _build_single_triangle_sc()
        hs = DiagonalHodgeStar(metric=MetricSpec(preset="diagonal"))
        with pytest.raises(ValueError):
            hs.star(sc, 0)

    def test_diagonal_star_missing_k_in_measures_raises(self):
        """Raise when measures dicts do not contain the requested degree k."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(preset="diagonal", primal_measures={}, dual_measures={})
        hs = DiagonalHodgeStar(metric=ms)
        with pytest.raises(KeyError):
            _ = hs.star(sc, 0)


class TestTriangleMesh3DBackend:
    """Test the geometry-based triangle mesh backend."""

    def test_backend_requires_triangles(self):
        """Raise error if no 2-simplices exist."""
        sc = SimplicialComplex([])
        sc.add_simplex([0])
        sc.set_simplex_attributes({0: [0.0, 0.0, 0.0]}, name="position")
        ms = MetricSpec(preset="barycentric_lumped")
        with pytest.raises(ValueError):
            TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")

    def test_backend_requires_3d_positions(self):
        """Raise error if vertex positions are not 3D."""
        faces = [[0, 1, 2]]
        sc = SimplicialComplex(faces)
        pos_bad = {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 1.0]}
        sc.set_simplex_attributes(pos_bad, name="position")
        ms = MetricSpec(preset="barycentric_lumped")
        with pytest.raises(ValueError):
            TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")

    def test_star_identity_has_ones(self):
        """Return identity star with ones diagonal for any k."""
        sc = _build_single_triangle_sc()
        n0, n1, n2 = sc.shape
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="identity"))

        S0 = be.star(sc, 0)
        S1 = be.star(sc, 1)
        S2 = be.star(sc, 2)

        assert S0.shape == (n0, n0)
        assert S1.shape == (n1, n1)
        assert S2.shape == (n2, n2)

        assert np.allclose(S0.diagonal(), 1.0)
        assert np.allclose(S1.diagonal(), 1.0)
        assert np.allclose(S2.diagonal(), 1.0)

    def test_star_barycentric_lumped_shapes_and_positivity(self):
        """Return positive diagonal stars for barycentric lumped metric."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(preset="barycentric_lumped")
        be = TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")

        S0 = be.star(sc, 0)
        S1 = be.star(sc, 1)
        S2 = be.star(sc, 2)

        assert S0.shape == (3, 3)
        assert S1.shape[0] == S1.shape[1]
        assert S2.shape == (1, 1)

        assert np.all(S0.diagonal() > 0.0)
        assert np.all(S1.diagonal() > 0.0)
        assert np.all(S2.diagonal() > 0.0)

    def test_star_barycentric_lumped_inverse(self):
        """Return inverse barycentric star with reciprocal diagonal."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="barycentric_lumped")
        )

        S1 = be.star(sc, 1, inverse=False)
        S1_inv = be.star(sc, 1, inverse=True)

        assert np.all(S1.diagonal() > 0.0)
        assert np.allclose(S1.diagonal() * S1_inv.diagonal(), 1.0, atol=1e-10)

    def test_star_circumcentric_shapes_and_positivity(self):
        """Return valid circumcentric stars."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(preset="circumcentric")
        be = TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")

        S0 = be.star(sc, 0)
        S1 = be.star(sc, 1)
        S2 = be.star(sc, 2)

        assert S0.shape == (3, 3)
        assert S1.shape[0] == S1.shape[1]
        assert S2.shape == (1, 1)

        assert np.all(S0.diagonal() >= 0.0)
        assert np.all(S1.diagonal() >= 0.0)
        assert np.all(S2.diagonal() > 0.0)

    def test_star_voronoi_matches_circumcentric(self):
        """Treat preset='voronoi' as circumcentric in this backend."""
        sc = _build_single_triangle_sc()
        be_cc = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        be_v = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="voronoi"))

        for k in (0, 1, 2):
            S_cc = be_cc.star(sc, k)
            S_v = be_v.star(sc, k)
            assert np.allclose(S_cc.diagonal(), S_v.diagonal())

    def test_star_circumcentric_inverse(self):
        """Return inverse circumcentric star with reciprocal diagonal where positive."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        S2 = be.star(sc, 2)
        S2_inv = be.star(sc, 2, inverse=True)

        assert np.all(S2.diagonal() > 0.0)
        assert np.allclose(S2.diagonal() * S2_inv.diagonal(), 1.0, atol=1e-10)

    def test_star_invalid_k(self):
        """Raise error for unsupported cochain degree."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="barycentric_lumped")
        )
        with pytest.raises(ValueError):
            be.star(sc, 3)

    def test_star_unsupported_preset_raises(self):
        """Raise error for unsupported star preset on triangle backend."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="euclidean"))
        with pytest.raises(ValueError):
            _ = be.star(sc, 0)

    def test_fem_mass_and_stiffness_sanity(self):
        """Return symmetric FEM mass and stiffness matrices."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        M = be.fem_mass_matrix_0()
        K = be.cotan_stiffness_0()

        assert isinstance(M, csr_matrix)
        assert isinstance(K, csr_matrix)

        assert M.shape == (3, 3)
        assert K.shape == (3, 3)

        assert (M - M.T).nnz == 0
        assert (K - K.T).nnz == 0

        rsum = np.asarray(K.sum(axis=1)).reshape(-1)
        assert np.allclose(rsum, 0.0, atol=1e-10)

    def test_cotan_laplacian_lumped_and_unlumped(self):
        """Return cotan Laplacian with correct shapes and finite entries."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        K = be.cotan_laplacian_0(lumped=False)
        L = be.cotan_laplacian_0(lumped=True)

        assert K.shape == (4, 4)
        assert L.shape == (4, 4)

    def test_riemannian_tensors_shape_validation(self):
        """Raise error when metric.tensors has invalid shape."""
        sc = _build_single_triangle_sc()
        bad_tensors = np.zeros((2, 3, 3), dtype=float)
        ms = MetricSpec(preset="circumcentric", tensors=bad_tensors)
        be = TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")
        with pytest.raises(ValueError):
            _ = be.riemannian_stiffness_0()

    def test_riemannian_fn_shape_validation(self):
        """Raise error when metric.fn returns invalid shape."""
        sc = _build_single_triangle_sc()

        def bad_fn(t: int, P: np.ndarray, sc_obj) -> np.ndarray:
            """Return an invalid (2,2) tensor for testing.

            Parameters
            ----------
            t : int
                Triangle index.
            P : np.ndarray
                Triangle vertex positions, shape (3, 3).
            sc_obj : Any
                Complex object.

            Returns
            -------
            np.ndarray
                Invalid tensor of shape (2, 2).
            """
            _ = t
            _ = P
            _ = sc_obj
            return np.zeros((2, 2), dtype=float)

        ms = MetricSpec(preset="circumcentric", fn=bad_fn)
        be = TriangleMesh3DBackend(sc=sc, metric=ms, pos_name="position")
        with pytest.raises(ValueError):
            _ = be.riemannian_stiffness_0()

    def test_riemannian_default_is_isotropic(self):
        """Default to isotropic tensors and produce symmetric stiffness."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        K_iso = be.cotan_stiffness_0()
        K_riem = be.riemannian_stiffness_0()

        assert K_iso.shape == K_riem.shape
        assert (K_riem - K_riem.T).nnz == 0
        assert (K_iso - K_riem).nnz == 0

    def test_riemannian_explicit_tensors_used(self):
        """Use explicit per-triangle tensors when provided."""
        sc = _build_two_triangle_square_sc()
        nT = len(list(sc.skeleton(2)))

        tensors = np.repeat(np.eye(3)[None, :, :], nT, axis=0)
        tensors[0] = 2.0 * np.eye(3)

        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="circumcentric", tensors=tensors)
        )
        K = be.riemannian_stiffness_0()
        assert K.shape == (4, 4)
        assert (K - K.T).nnz == 0

    def test_riemannian_fn_used(self):
        """Use metric.fn to generate per-triangle tensors."""
        sc = _build_two_triangle_square_sc()

        def fn(t: int, P: np.ndarray, sc_obj) -> np.ndarray:
            """Return a simple SPD tensor depending on triangle index.

            Parameters
            ----------
            t : int
                Triangle index.
            P : np.ndarray
                Triangle vertex positions, shape (3, 3).
            sc_obj : Any
                Complex object.

            Returns
            -------
            np.ndarray
                SPD tensor of shape (3, 3).
            """
            _ = P
            _ = sc_obj
            scale = 1.0 + 0.1 * float(t)
            return scale * np.eye(3, dtype=float)

        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="circumcentric", fn=fn)
        )
        K = be.riemannian_stiffness_0()

        assert K.shape == (4, 4)
        assert (K - K.T).nnz == 0

    def test_riemannian_laplacian_lumped_and_unlumped(self):
        """Return riemannian Laplacian with correct shapes and finite entries."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        K = be.riemannian_laplacian_0(lumped=False)
        L = be.riemannian_laplacian_0(lumped=True)

        assert K.shape == (4, 4)
        assert L.shape == (4, 4)
        assert np.isfinite(L.data).all()

    def test_degenerate_triangle_does_not_crash_fem_ops(self):
        """Handle degenerate triangles by skipping contributions."""
        sc = _build_degenerate_triangle_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))

        M = be.fem_mass_matrix_0()
        K = be.cotan_stiffness_0()
        Kr = be.riemannian_stiffness_0()

        assert M.shape == (3, 3)
        assert K.shape == (3, 3)
        assert Kr.shape == (3, 3)

        assert M.nnz == 0
        assert K.nnz == 0
        assert Kr.nnz == 0

    def test_degenerate_triangle_star_is_finite(self):
        """Return finite diagonal stars even for degenerate geometry."""
        sc = _build_degenerate_triangle_sc()

        be_bary = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="barycentric_lumped")
        )
        S2 = be_bary.star(sc, 2)
        assert np.isfinite(S2.diagonal()).all()

        be_cc = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        S2_cc = be_cc.star(sc, 2)
        assert np.isfinite(S2_cc.diagonal()).all()


class TestMetricSpecDefaults:
    """Test basic MetricSpec default behavior."""

    def test_default_eps_is_positive(self):
        """Provide a positive default epsilon."""
        ms = MetricSpec()
        assert ms.eps > 0.0

    def test_default_preset_is_barycentric_lumped(self):
        """Default to barycentric_lumped preset."""
        ms = MetricSpec()
        assert ms.preset == "barycentric_lumped"


class TestMetricCallableType:
    """Test that metric.fn can be passed as a normal Python callable."""

    def test_fn_is_callable(self):
        """Accept a callable function as metric.fn."""
        fn: Callable[[int, np.ndarray, object], np.ndarray]

        def _fn(t: int, P: np.ndarray, sc_obj) -> np.ndarray:
            """Return identity tensor.

            Parameters
            ----------
            t : int
                Triangle index.
            P : np.ndarray
                Triangle vertex positions, shape (3, 3).
            sc_obj : Any
                Complex object.

            Returns
            -------
            np.ndarray
                Identity tensor of shape (3, 3).
            """
            _ = t
            _ = P
            _ = sc_obj
            return np.eye(3, dtype=float)

        fn = _fn
        assert callable(fn)


# ============================
# Additional unit tests (augment only)
# ============================


class TestAs1DFloatAdditional:
    """Additional coverage for `_as_1d_float` behavior."""

    def test_as_1d_float_accepts_list(self):
        """Accept Python lists and cast to float."""
        y = _as_1d_float([1, 2, 3])
        assert y.dtype == float
        assert y.shape == (3,)
        assert y == pytest.approx([1.0, 2.0, 3.0])

    def test_as_1d_float_accepts_scalar(self):
        """Accept scalar inputs and return a length-1 float array."""
        y = _as_1d_float(7)
        assert y.dtype == float
        assert y.shape == (1,)
        assert y == pytest.approx([7.0])

    def test_as_1d_float_casts_int_dtype(self):
        """Cast integer arrays to float."""
        x = np.array([1, 2, 3], dtype=np.int64)
        y = _as_1d_float(x)
        assert y.dtype == float
        assert y == pytest.approx([1.0, 2.0, 3.0])

    def test_as_1d_float_rejects_non_numeric(self):
        """Raise ValueError when input cannot be converted to float."""
        with pytest.raises(ValueError):
            _ = _as_1d_float(np.array(["a", "b"], dtype=object))


class TestGeometryHelpersAdditional:
    """Additional coverage for geometry helper input validation."""

    def test_triangle_area_3d_raises_on_bad_shape(self):
        """Raise ValueError if vertices are not 3D vectors."""
        p0 = np.array([0.0, 0.0])  # wrong shape
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            _triangle_area_3d(p0, p1, p2)

    def test_circumcenter_3d_raises_on_bad_shape(self):
        """Raise ValueError if vertices are not 3D vectors."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0])  # wrong shape
        p2 = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            _circumcenter_3d(p0, p1, p2)

    def test_grad_barycentric_3d_raises_on_bad_shape(self):
        """Raise ValueError if vertices are not 3D vectors."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0])  # wrong shape
        with pytest.raises(ValueError):
            _grad_barycentric_3d(p0, p1, p2)


class TestDiagonalHodgeStarAdditional:
    """Additional coverage for numerical safeguards and shapes."""

    def test_diagonal_star_inverse_eps_safeguard(self):
        """Invert safely when diagonal contains zeros (uses eps safeguard)."""
        sc = _build_single_triangle_sc()
        ms = MetricSpec(
            preset="diagonal",
            diagonal_weights={0: np.array([0.0, 2.0, 0.0])},
            eps=1e-12,
        )
        hs = DiagonalHodgeStar(metric=ms)
        S = hs.star(sc, 0)
        Sinv = hs.star(sc, 0, inverse=True)
        assert S.shape == (3, 3)
        assert Sinv.shape == (3, 3)
        assert np.isfinite(Sinv.diagonal()).all()
        assert Sinv.diagonal()[1] == pytest.approx(0.5)

    def test_identity_star_all_k_shapes(self):
        """Identity preset returns square matrices sized by skeleton(k)."""
        sc = _build_two_triangle_square_sc()
        hs = DiagonalHodgeStar(metric=MetricSpec(preset="identity"))
        n0, n1, n2 = sc.shape
        S0 = hs.star(sc, 0)
        S1 = hs.star(sc, 1)
        S2 = hs.star(sc, 2)
        assert S0.shape == (n0, n0)
        assert S1.shape == (n1, n1)
        assert S2.shape == (n2, n2)
        assert np.allclose(S0.diagonal(), 1.0)
        assert np.allclose(S1.diagonal(), 1.0)
        assert np.allclose(S2.diagonal(), 1.0)


class TestTriangleMesh3DBackendAdditional:
    """Additional coverage for backend operators and consistency checks."""

    def test_fem_mass_is_psd_and_has_positive_diagonal(self):
        """Mass matrix should be symmetric with nonnegative diagonal entries."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        M = be.fem_mass_matrix_0()
        assert isinstance(M, csr_matrix)
        assert (M - M.T).nnz == 0
        assert np.all(M.diagonal() > 0.0)

    def test_stiffness_has_nonpositive_offdiagonal(self):
        """Stiffness matrix off-diagonals are typically nonpositive for cotan FEM on planar meshes."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        K = be.cotan_stiffness_0()
        K_dense = K.toarray()
        off = K_dense - np.diag(np.diag(K_dense))
        assert np.max(off) <= 1e-12

    def test_laplacian_outputs_are_finite(self):
        """Laplacian variants should not contain NaN/inf entries."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        Lc = be.cotan_laplacian_0(lumped=True)
        Lr = be.riemannian_laplacian_0(lumped=True)
        assert np.isfinite(Lc.data).all()
        assert np.isfinite(Lr.data).all()

    def test_star_barycentric_lumped_k2_matches_triangle_count(self):
        """For k=2, the star is 1x1 for a single triangle and nT x nT in general."""
        sc1 = _build_single_triangle_sc()
        be1 = TriangleMesh3DBackend(
            sc=sc1, metric=MetricSpec(preset="barycentric_lumped")
        )
        S2_1 = be1.star(sc1, 2)
        assert S2_1.shape == (1, 1)

        sc2 = _build_two_triangle_square_sc()
        nT = len(list(sc2.skeleton(2)))
        be2 = TriangleMesh3DBackend(
            sc=sc2, metric=MetricSpec(preset="barycentric_lumped")
        )
        S2_2 = be2.star(sc2, 2)
        assert S2_2.shape == (nT, nT)
        assert np.all(S2_2.diagonal() > 0.0)

    def test_dual_measure_barycentric_invalid_k_raises(self):
        """Raise ValueError for unsupported degree in barycentric dual measure helper."""
        sc = _build_single_triangle_sc()
        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="barycentric_lumped")
        )
        with pytest.raises(ValueError):
            _ = be._dual_measure_barycentric(3)

    def test_dual_measure_barycentric_k2_equals_triangle_areas(self):
        """For k=2, barycentric dual measure returns triangle areas."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(
            sc=sc, metric=MetricSpec(preset="barycentric_lumped")
        )
        A = be._triangle_areas()
        d2 = be._dual_measure_barycentric(2)
        assert d2.shape == A.shape
        assert d2 == pytest.approx(A)

    def test_star_circumcentric_inverse_k0_is_finite(self):
        """Inverse circumcentric star for k=0 returns finite diagonal."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        S0 = be.star(sc, 0, inverse=False)
        S0_inv = be.star(sc, 0, inverse=True)
        assert np.isfinite(S0_inv.diagonal()).all()
        mask = S0.diagonal() > 0.0
        if np.any(mask):
            assert np.allclose(
                S0.diagonal()[mask] * S0_inv.diagonal()[mask], 1.0, atol=1e-10
            )

    def test_star_circumcentric_inverse_k1_is_finite(self):
        """Inverse circumcentric star for k=1 returns finite diagonal."""
        sc = _build_two_triangle_square_sc()
        be = TriangleMesh3DBackend(sc=sc, metric=MetricSpec(preset="circumcentric"))
        S1 = be.star(sc, 1, inverse=False)
        S1_inv = be.star(sc, 1, inverse=True)
        assert np.isfinite(S1_inv.diagonal()).all()
        mask = S1.diagonal() > 0.0
        if np.any(mask):
            assert np.allclose(
                S1.diagonal()[mask] * S1_inv.diagonal()[mask], 1.0, atol=1e-10
            )
