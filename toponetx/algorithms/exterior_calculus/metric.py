"""
Metric and geometry backends for exterior calculus on triangle meshes embedded in R^3.

Notes
-----
This module implements the *metric/geometry* ingredients needed by (Discrete) Exterior Calculus (DEC)
and closely related FEM discretizations on triangle meshes.

High-level picture:
- A triangle mesh can be viewed as a 2D simplicial complex K embedded in R^3.
- Cochains C^k(K) assign scalars to oriented k-simplices (0: vertices, 1: edges, 2: triangles).
- Topology enters via boundary/coboundary operators; geometry enters via the Hodge star.

Continuous Hodge star (motivation):
- On an oriented Riemannian manifold (M, g), the Hodge star is an isomorphism
  * : Omega^k(M) -> Omega^{n-k}(M), determined by the metric g and the orientation.
- It is characterized by the identity
  a wedge (*b) = <a, b>_g vol_g
  for differential forms a, b of the same degree.

Discrete (diagonal) Hodge star (DEC):
- In DEC, one approximates the Hodge star by a sparse operator
  *_k : C^k(K) -> C^{n-k}(K^*),
  mapping primal k-cochains to dual (n-k)-cochains on a dual complex K^*.
- A common choice is a diagonal approximation:
      *_k ~= diag( |dual_k| / |primal_k| )
  where |primal_k| denotes a primal k-simplex measure and |dual_k| is the corresponding dual-cell
  measure (constructed e.g. using barycentric or circumcentric/Voronoi duals).

Robust Option A behavior for Voronoi/circumcentric presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Circumcentric/Voronoi duals can produce negative dual measures on meshes with obtuse triangles.
To keep the API simple and robust, the geometry backend uses the following default behavior
for presets "voronoi" and "circumcentric":

- For k=0 (vertex dual areas), we use a "mixed Voronoi area" construction:
  - non-obtuse triangles use Voronoi (cotangent) contributions,
  - obtuse triangles use an area split (A/2 to the obtuse vertex, A/4 to the others).
  This yields non-negative dual areas for non-degenerate triangles.
- For k=1 (dual edge lengths), we fall back to the barycentric dual edge length to avoid
  negative cotangent contributions on obtuse triangles.
- For k=2, the standard area relationship is used.

This module provides:
- Geometry utilities for triangles embedded in R^3.
- A user-facing `MetricSpec` describing diagonal Hodge stars and anisotropic FEM tensors.
- Backends implementing diagonal Hodge stars:
  - `DiagonalHodgeStar` (geometry-free).
  - `TriangleMesh3DBackend` (geometry-based).

Conventions:
- The mesh is treated as a 2D complex embedded in R^3.
- The Hodge stars returned here are diagonal (fast, standard in many DEC pipelines).
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from toponetx.classes.simplicial_complex import SimplicialComplex

MetricPreset = Literal[
    "identity",
    "diagonal",
    "barycentric_lumped",
    "circumcentric",
    "voronoi",
    "euclidean",
]

MetricCallable = Callable[[int, np.ndarray, SimplicialComplex], np.ndarray]


def _sorted_edge(u: Any, v: Any) -> tuple[Any, Any]:
    """Return an edge tuple with deterministic ordering.

    Parameters
    ----------
    u : Any
        First vertex label.
    v : Any
        Second vertex label.

    Returns
    -------
    tuple[Any, Any]
        Ordered pair (min(u, v), max(u, v)) under Python ordering.
    """
    return (u, v) if u <= v else (v, u)


def _triangle_area_3d(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the area of a triangle embedded in R^3.

    Parameters
    ----------
    p0 : np.ndarray
        First vertex position, shape (3,).
    p1 : np.ndarray
        Second vertex position, shape (3,).
    p2 : np.ndarray
        Third vertex position, shape (3,).

    Returns
    -------
    float
        Triangle area.

    Raises
    ------
    ValueError
        If any input does not represent a 3D vector.
    """
    p0a = np.asarray(p0, dtype=float).reshape(-1)
    p1a = np.asarray(p1, dtype=float).reshape(-1)
    p2a = np.asarray(p2, dtype=float).reshape(-1)
    if p0a.shape != (3,) or p1a.shape != (3,) or p2a.shape != (3,):
        raise ValueError(
            "Triangle vertices must be 3D vectors with shape (3,). "
            f"Got shapes p0={p0a.shape}, p1={p1a.shape}, p2={p2a.shape}."
        )
    n = np.cross(p1a - p0a, p2a - p0a)
    return 0.5 * float(np.linalg.norm(n))


def _circumcenter_3d(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute the circumcenter of a triangle embedded in R^3.

    Parameters
    ----------
    p0 : np.ndarray
        First vertex position, shape (3,).
    p1 : np.ndarray
        Second vertex position, shape (3,).
    p2 : np.ndarray
        Third vertex position, shape (3,).

    Returns
    -------
    np.ndarray
        Circumcenter position, shape (3,).

    Raises
    ------
    ValueError
        If any input does not represent a 3D vector.
    """
    a = np.asarray(p0, dtype=float).reshape(-1)
    b = np.asarray(p1, dtype=float).reshape(-1)
    c = np.asarray(p2, dtype=float).reshape(-1)
    if a.shape != (3,) or b.shape != (3,) or c.shape != (3,):
        raise ValueError(
            "Triangle vertices must be 3D vectors with shape (3,). "
            f"Got shapes p0={a.shape}, p1={b.shape}, p2={c.shape}."
        )
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn < 1e-28:
        return (a + b + c) / 3.0
    ab2 = float(ab @ ab)
    ac2 = float(ac @ ac)
    u = (ac2 * np.cross(n, ab) + ab2 * np.cross(ac, n)) / (2.0 * nn)
    return a + u


def _grad_barycentric_3d(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute gradients of P1 basis functions on an R^3 triangle.

    Parameters
    ----------
    p0 : np.ndarray
        First vertex position, shape (3,).
    p1 : np.ndarray
        Second vertex position, shape (3,).
    p2 : np.ndarray
        Third vertex position, shape (3,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        Gradients (g0, g1, g2) each shape (3,), and det_norm = ||(p1-p0)x(p2-p0)||.
        The triangle area is A = 0.5 * det_norm.

    Raises
    ------
    ValueError
        If any input does not represent a 3D vector.
    """
    p0a = np.asarray(p0, dtype=float).reshape(-1)
    p1a = np.asarray(p1, dtype=float).reshape(-1)
    p2a = np.asarray(p2, dtype=float).reshape(-1)
    if p0a.shape != (3,) or p1a.shape != (3,) or p2a.shape != (3,):
        raise ValueError(
            "Triangle vertices must be 3D vectors with shape (3,). "
            f"Got shapes p0={p0a.shape}, p1={p1a.shape}, p2={p2a.shape}."
        )
    e01 = p1a - p0a
    e02 = p2a - p0a
    n = np.cross(e01, e02)
    nn = float(n @ n)
    det_norm = float(np.linalg.norm(n))
    if nn < 1e-28:
        z = np.zeros(3, dtype=float)
        return z, z, z, det_norm
    g0 = np.cross(n, p2a - p1a) / nn
    g1 = np.cross(n, p0a - p2a) / nn
    g2 = np.cross(n, p1a - p0a) / nn
    return g0.astype(float), g1.astype(float), g2.astype(float), det_norm


@dataclass(frozen=True)
class MetricSpec:
    """Describe how to build diagonal Hodge stars and anisotropic FEM tensors.

    Notes
    -----
    In DEC and related discretizations, the metric is represented by the discrete Hodge star.
    This class stores a *specification* for how diagonal stars (and optional anisotropic tensors)
    should be built.

    A common diagonal DEC choice is:
        *_k = diag(w_k), where w_k ~ |dual_k| / |primal_k|.

    For anisotropic diffusion / Riemannian metrics on the surface, one can also provide a
    per-triangle SPD tensor field G (a 3x3 matrix per triangle in embedding coordinates) that
    defines a stiffness operator assembled from P1 basis gradients.

    Attributes
    ----------
    preset : {"identity", "diagonal", "barycentric_lumped", "circumcentric", "voronoi", "euclidean"}
        Preset name controlling the backend.
    diagonal_weights : dict[int, np.ndarray], optional
        Explicit diagonal entries for the Hodge star at each degree k.
        Each entry must have length equal to the number of k-simplices.
    primal_measures : dict[int, np.ndarray], optional
        Primal measures |primal_k| for generic diagonal DEC stars.
    dual_measures : dict[int, np.ndarray], optional
        Dual measures |dual_k| for generic diagonal DEC stars.
    tensors : np.ndarray, optional
        Per-triangle SPD tensors for anisotropic stiffness, shape (nT, 3, 3).
    fn : callable, optional
        Per-triangle tensor function fn(t, P, sc) -> (3, 3), where P has shape (3, 3).
    eps : float
        Numerical safeguard used for divisions/inversions.
    """

    preset: MetricPreset = "barycentric_lumped"
    diagonal_weights: dict[int, np.ndarray] | None = None
    primal_measures: dict[int, np.ndarray] | None = None
    dual_measures: dict[int, np.ndarray] | None = None
    tensors: np.ndarray | None = None
    fn: MetricCallable | None = None
    eps: float = 1e-12


class HodgeStarBackend(Protocol):
    """Protocol for Hodge star backends."""

    def star(
        self, sc: SimplicialComplex, k: int, *, inverse: bool = False
    ) -> csr_matrix:
        """Return *_k (or its inverse).

        Parameters
        ----------
        sc : SimplicialComplex
            Complex object.
        k : int
            Cochain degree.
        inverse : bool, default=False
            If True, return (*_k)^{-1}.

        Returns
        -------
        csr_matrix
            Sparse star matrix.
        """
        ...


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    """Convert an array-like to a 1D float array.

    Parameters
    ----------
    x : np.ndarray
        Input array-like.

    Returns
    -------
    np.ndarray
        1D float array.

    Raises
    ------
    ValueError
        If the input cannot be viewed as a 1D array after reshape.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array after reshape, got shape {arr.shape}.")
    return arr


@dataclass(frozen=True)
class DiagonalHodgeStar:
    """Geometry-free diagonal Hodge star backend.

    Notes
    -----
    This backend does not use vertex positions. It builds diagonal stars from:
    - the identity preset, or
    - user-supplied diagonal weights / primal+dual measures.

    Attributes
    ----------
    metric : MetricSpec
        Metric specification.
    """

    metric: MetricSpec

    def star(
        self, sc: SimplicialComplex, k: int, *, inverse: bool = False
    ) -> csr_matrix:
        """Return a diagonal Hodge star matrix.

        Parameters
        ----------
        sc : SimplicialComplex
            Complex providing `skeleton(k)`.
        k : int
            Cochain degree.
        inverse : bool
            If True, return the inverse diagonal star.

        Returns
        -------
        csr_matrix
            Diagonal Hodge star matrix.
        """
        n_k = len(list(sc.skeleton(k)))
        eps = float(self.metric.eps)
        if self.metric.preset == "identity":
            diag = np.ones(n_k, dtype=float)
            return diags(diag, 0, format="csr")
        if self.metric.preset != "diagonal":
            raise ValueError(
                f"DiagonalHodgeStar requires preset='diagonal', got {self.metric.preset!r}."
            )
        if (
            self.metric.diagonal_weights is not None
            and k in self.metric.diagonal_weights
        ):
            diag = _as_1d_float(self.metric.diagonal_weights[k])
            if diag.shape[0] != n_k:
                raise ValueError(
                    f"diagonal_weights[{k}] must have length {n_k}, got {diag.shape[0]}."
                )
        else:
            if self.metric.primal_measures is None or self.metric.dual_measures is None:
                raise ValueError(
                    "preset='diagonal' requires diagonal_weights or both primal_measures and dual_measures."
                )
            pk = _as_1d_float(self.metric.primal_measures[k])
            dk = _as_1d_float(self.metric.dual_measures[k])
            if pk.shape[0] != n_k or dk.shape[0] != n_k:
                raise ValueError(
                    f"Measures for k={k} must have length {n_k}; got primal {pk.shape[0]}, "
                    f"dual {dk.shape[0]}."
                )
            diag = dk / np.maximum(pk, eps)
        if inverse:
            diag = 1.0 / np.maximum(diag, eps)
        return diags(diag, 0, format="csr")


@dataclass
class TriangleMesh3DBackend:
    """Geometry backend for triangle meshes embedded in R^3.

    Notes
    -----
    This backend precomputes mesh-aligned arrays (vertices, edges, triangles) from a
    `SimplicialComplex` with 3D vertex positions, and provides:
    - diagonal Hodge stars for k in {0, 1, 2} using barycentric-lumped or robust Voronoi-like
      constructions;
    - FEM operators on vertices (0-forms): mass, stiffness, and Laplacian-like operators;
    - anisotropic stiffness/Laplacian using per-triangle SPD tensors.

    Attributes
    ----------
    sc : SimplicialComplex
        Complex providing `skeleton(0/1/2)` and `get_node_attributes(pos_name)`.
    metric : MetricSpec
        Metric specification.
    pos_name : str
        Node attribute name for vertex positions in R^3.
    """

    sc: SimplicialComplex
    metric: MetricSpec
    pos_name: str = "position"

    def __post_init__(self) -> None:
        """Build the internal mesh cache."""
        self.eps = float(self.metric.eps)
        self._cache_ready = False
        self._build_cache()

    def _build_cache(self) -> None:
        """Build internal mesh cache."""
        tri = [tuple(s.elements) for s in self.sc.skeleton(2)]
        if not tri:
            raise ValueError("TriangleMesh3DBackend requires 2-simplices (triangles).")
        vlabels = [next(iter(s)) for s in self.sc.skeleton(0)]
        self._vid = {v: i for i, v in enumerate(vlabels)}
        pos = self.sc.get_node_attributes(self.pos_name)
        V = np.stack(
            [np.asarray(pos[v], dtype=float).reshape(-1) for v in vlabels], axis=0
        )
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(
                "Vertex positions must be 3D vectors (R^3) with shape (nV, 3). "
                f"Got shape {V.shape}."
            )
        self._V = V
        self._vlabels = vlabels
        edges = [tuple(s.elements) for s in self.sc.skeleton(1)]
        edges_sorted = [_sorted_edge(u, v) for (u, v) in edges]
        self._E = np.array(edges_sorted, dtype=object)
        self._T = np.array(tri, dtype=object)

        # Precompute per-vertex incident triangles (1-ring).
        nV = self._V.shape[0]
        self._incident: list[list[tuple[int, int, int]]] = [[] for _ in range(nV)]
        for t, (a, b, c) in enumerate(self._T):
            ia = self._vid[a]
            ib = self._vid[b]
            ic = self._vid[c]
            self._incident[ia].append((t, ib, ic))
            self._incident[ib].append((t, ia, ic))
            self._incident[ic].append((t, ia, ib))

        self._cache_ready = True

    def star(
        self, sc: SimplicialComplex, k: int, *, inverse: bool = False
    ) -> csr_matrix:
        """Return a diagonal Hodge star for triangle meshes.

        Parameters
        ----------
        sc : SimplicialComplex
            Complex (unused except for sizing consistency).
        k : int
            Cochain degree.
        inverse : bool, default=False
            If True, return (*_k)^{-1}.

        Returns
        -------
        csr_matrix
            Diagonal star matrix.
        """
        preset = self.metric.preset
        if preset == "identity":
            n = len(list(sc.skeleton(k)))
            return diags(np.ones(n, dtype=float), 0, format="csr")
        if preset == "barycentric_lumped":
            return self._star_barycentric_lumped(k, inverse=inverse)
        if preset in ("circumcentric", "voronoi"):
            # Robust Option A behavior (see module Notes).
            return self._star_voronoi_like_option_a(k, inverse=inverse)
        raise ValueError(
            f"TriangleMesh3DBackend does not support preset={preset!r} for stars."
        )

    def _star_barycentric_lumped(self, k: int, inverse: bool) -> csr_matrix:
        """Compute a barycentric-lumped diagonal star.

        Notes
        -----
        This constructs diagonal weights w_k = |dual_k| / |primal_k| using a barycentric dual.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool, default=False
            If True, invert the diagonal star.

        Returns
        -------
        csr_matrix
            Diagonal star.
        """
        if k not in (0, 1, 2):
            raise ValueError(
                "barycentric_lumped supports only k in {0, 1, 2} for triangle meshes."
            )
        if k == 0:
            primal = np.ones(len(list(self.sc.skeleton(0))), dtype=float)
        elif k == 1:
            primal = self._primal_edge_lengths()
        else:
            primal = self._triangle_areas()
        dual = self._dual_measure_barycentric(k)
        w = dual / np.maximum(primal, self.eps)
        if inverse:
            w = 1.0 / np.maximum(w, self.eps)
        return diags(w, 0, format="csr")

    def _dual_measure_barycentric(self, k: int) -> np.ndarray:
        """Compute barycentric dual measures.

        Parameters
        ----------
        k : int
            Cochain degree.

        Returns
        -------
        np.ndarray
            Dual measures array for degree k.
        """
        A = self._triangle_areas()

        if k == 2:
            return A.copy()

        if k == 0:
            acc: dict[Any, float] = {}
            for (a, b, c), area in zip(self._T, A, strict=True):
                acc[a] = acc.get(a, 0.0) + area / 3.0
                acc[b] = acc.get(b, 0.0) + area / 3.0
                acc[c] = acc.get(c, 0.0) + area / 3.0
            return np.array([acc.get(v, 0.0) for v in self._vlabels], dtype=float)

        if k == 1:
            acc_e: dict[tuple[Any, Any], float] = {}

            # For barycentric dual on edges in 2D:
            # |e*| is approximated by summing, over incident triangles,
            # the segment length from the triangle barycenter to the edge midpoint.
            for (a, b, c), _area in zip(self._T, A, strict=True):
                ia = self._vid[a]
                ib = self._vid[b]
                ic = self._vid[c]

                pa = self._V[ia]
                pb = self._V[ib]
                pc = self._V[ic]

                bary = (pa + pb + pc) / 3.0  # triangle barycenter

                # edge (a, b)
                e = _sorted_edge(a, b)
                mid = (pa + pb) / 2.0
                acc_e[e] = acc_e.get(e, 0.0) + float(np.linalg.norm(bary - mid))

                # edge (a, c)
                e = _sorted_edge(a, c)
                mid = (pa + pc) / 2.0
                acc_e[e] = acc_e.get(e, 0.0) + float(np.linalg.norm(bary - mid))

                # edge (b, c)
                e = _sorted_edge(b, c)
                mid = (pb + pc) / 2.0
                acc_e[e] = acc_e.get(e, 0.0) + float(np.linalg.norm(bary - mid))

            edges = [tuple(e) for e in self._E.tolist()]
            return np.array([acc_e.get(tuple(e), 0.0) for e in edges], dtype=float)

        raise ValueError("k must be 0, 1, or 2.")

    def _star_voronoi_like_option_a(self, k: int, inverse: bool = False) -> csr_matrix:
        """Compute a robust Voronoi-like diagonal star (Option A).

        Notes
        -----
        This method is used for presets "voronoi" and "circumcentric" to avoid failures
        due to obtuse triangles. See the module Notes for the behavior.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool, default=False
            If True, invert the diagonal star.

        Returns
        -------
        csr_matrix
            Diagonal star.

        Raises
        ------
        ValueError
            If k is not in {0, 1, 2}.
        """
        if k == 0:
            w = self._dual_area_vertices_voronoi_mixed()
            if inverse:
                w = 1.0 / np.maximum(w, self.eps)
            return diags(w, 0, format="csr")

        if k == 1:
            # Robust fallback: barycentric dual edge length, then star1 = |e*| / |e|.
            lstar = self._dual_measure_barycentric(1)
            le = self._primal_edge_lengths()
            w = lstar / np.maximum(le, self.eps)
            if inverse:
                w = 1.0 / np.maximum(w, self.eps)
            return diags(w, 0, format="csr")

        if k == 2:
            A = self._triangle_areas()
            w = 1.0 / np.maximum(A, self.eps)
            if inverse:
                w = A / np.maximum(1.0, self.eps)
            return diags(w, 0, format="csr")

        raise ValueError(
            "voronoi/circumcentric supports only k in {0, 1, 2} for triangle meshes."
        )

    def fem_mass_matrix_0(self) -> csr_matrix:
        """Return the consistent P1 FEM mass matrix on vertices.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex mass matrix.
        """
        nV = self._V.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for a, b, c in self._T:
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            A = _triangle_area_3d(self._V[ia], self._V[ib], self._V[ic])
            if A <= 1e-14:
                continue
            m = (A / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
            idx = [ia, ib, ic]
            for i_local in range(3):
                for j_local in range(3):
                    rows.append(idx[i_local])
                    cols.append(idx[j_local])
                    data.append(float(m[i_local, j_local]))
        return coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

    def cotan_stiffness_0(self) -> csr_matrix:
        """Return the surface FEM stiffness matrix on vertices.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex stiffness matrix.
        """
        nV = self._V.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for a, b, c in self._T:
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            p0, p1, p2 = self._V[ia], self._V[ib], self._V[ic]
            g0, g1, g2, det_norm = _grad_barycentric_3d(p0, p1, p2)
            A = 0.5 * det_norm
            if A <= 1e-14:
                continue
            grads = [g0, g1, g2]
            idx = [ia, ib, ic]
            for i_local in range(3):
                for j_local in range(3):
                    kij = A * float(grads[i_local] @ grads[j_local])
                    rows.append(idx[i_local])
                    cols.append(idx[j_local])
                    data.append(kij)
        return coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

    def riemannian_stiffness_0(self) -> csr_matrix:
        """Return anisotropic surface stiffness matrix on vertices.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex anisotropic stiffness matrix.
        """
        G = self._resolve_metric_tensors_per_triangle()
        nV = self._V.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for t, (a, b, c) in enumerate(self._T):
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            p0, p1, p2 = self._V[ia], self._V[ib], self._V[ic]
            g0, g1, g2, det_norm = _grad_barycentric_3d(p0, p1, p2)
            A = 0.5 * det_norm
            if A <= 1e-14:
                continue
            GT = np.asarray(G[t], dtype=float)
            if GT.shape != (3, 3):
                raise ValueError(
                    f"Per-triangle tensor must have shape (3, 3), got {GT.shape}."
                )
            grads = [g0, g1, g2]
            idx = [ia, ib, ic]
            for i_local in range(3):
                for j_local in range(3):
                    kij = A * float(grads[i_local].T @ (GT @ grads[j_local]))
                    rows.append(idx[i_local])
                    cols.append(idx[j_local])
                    data.append(kij)
        return coo_matrix((data, (rows, cols)), shape=(nV, nV)).tocsr()

    def cotan_laplacian_0(self, *, lumped: bool = True) -> csr_matrix:
        """Return the cotangent Laplacian on vertices.

        Parameters
        ----------
        lumped : bool, default=True
            If True, return M_lumped^{-1} K. Otherwise return K.

        Returns
        -------
        csr_matrix
            Laplacian-like operator matrix.
        """
        K = self.cotan_stiffness_0()
        if not lumped:
            return K
        M0 = self.fem_mass_matrix_0()
        d = np.asarray(M0.sum(axis=1)).reshape(-1)
        Minv = diags(1.0 / np.maximum(d, self.eps), 0, format="csr")
        return Minv @ K

    def riemannian_laplacian_0(self, *, lumped: bool = True) -> csr_matrix:
        """Return anisotropic Laplacian on vertices.

        Parameters
        ----------
        lumped : bool, default=True
            If True, return M_lumped^{-1} K(G). Otherwise return K(G).

        Returns
        -------
        csr_matrix
            Laplacian-like operator matrix.
        """
        K = self.riemannian_stiffness_0()
        if not lumped:
            return K
        M0 = self.fem_mass_matrix_0()
        d = np.asarray(M0.sum(axis=1)).reshape(-1)
        Minv = diags(1.0 / np.maximum(d, self.eps), 0, format="csr")
        return Minv @ K

    def _triangle_areas(self) -> np.ndarray:
        """Compute triangle areas.

        Returns
        -------
        np.ndarray
            Areas of all triangles.
        """
        A = np.zeros((len(self._T),), dtype=float)
        for t, (a, b, c) in enumerate(self._T):
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            A[t] = _triangle_area_3d(self._V[ia], self._V[ib], self._V[ic])
        return A

    def _primal_edge_lengths(self) -> np.ndarray:
        """Compute primal edge lengths.

        Returns
        -------
        np.ndarray
            Lengths of all edges.
        """
        le = np.zeros((len(self._E),), dtype=float)
        for i, (u, v) in enumerate(self._E):
            pu = self._V[self._vid[u]]
            pv = self._V[self._vid[v]]
            le[i] = float(np.linalg.norm(pu - pv))
        return le

    def _cot_from_squared_lengths(self, a2: float, b2: float, c2: float) -> float:
        """Compute cot(angle opposite side with squared length c2) from squared lengths.

        Parameters
        ----------
        a2 : float
            Squared length of one adjacent side.
        b2 : float
            Squared length of the other adjacent side.
        c2 : float
            Squared length of the opposite side.

        Returns
        -------
        float
            Cotangent of the angle opposite the side with squared length c2.
        """
        # cos(theta) = (a2 + b2 - c2) / (2ab)
        denom = 2.0 * np.sqrt(max(a2 * b2, 0.0)) + self.eps
        cos_t = (a2 + b2 - c2) / denom
        sin2 = max(1.0 - cos_t * cos_t, 0.0)
        sin_t = np.sqrt(max(sin2, self.eps))
        return float(cos_t / sin_t)

    def _dual_area_vertices_voronoi_mixed(self) -> np.ndarray:
        """Compute mixed Voronoi (circumcentric) dual area per vertex.

        Notes
        -----
        This is the standard "mixed area" construction used to avoid negative Voronoi areas
        on meshes with obtuse triangles:
        - If a triangle is non-obtuse, use Voronoi (cotangent) contributions.
        - If a triangle is obtuse, split its area: A/2 to the obtuse vertex, A/4 to the others.

        Returns
        -------
        np.ndarray
            Array of dual (mixed Voronoi) areas for each vertex, shape (n_vertices,).
        """
        nV = self._V.shape[0]
        Astar = np.zeros(nV, dtype=float)

        for a, b, c in self._T:
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            pa, pb, pc = self._V[ia], self._V[ib], self._V[ic]

            # Squared edge lengths opposite vertices a, b, c.
            lab2 = float(np.dot(pa - pb, pa - pb))
            lbc2 = float(np.dot(pb - pc, pb - pc))
            lca2 = float(np.dot(pc - pa, pc - pa))

            # Angle obtuse test via squared lengths: angle at a is obtuse iff opposite edge bc is longest
            # and lbc2 > lab2 + lca2.
            obt_a = lbc2 > lab2 + lca2
            obt_b = lca2 > lab2 + lbc2
            obt_c = lab2 > lbc2 + lca2

            A = _triangle_area_3d(pa, pb, pc)
            if A <= 1e-14:
                continue

            if obt_a or obt_b or obt_c:
                # Mixed area split for obtuse triangles.
                if obt_a:
                    Astar[ia] += 0.5 * A
                    Astar[ib] += 0.25 * A
                    Astar[ic] += 0.25 * A
                elif obt_b:
                    Astar[ib] += 0.5 * A
                    Astar[ia] += 0.25 * A
                    Astar[ic] += 0.25 * A
                else:
                    Astar[ic] += 0.5 * A
                    Astar[ia] += 0.25 * A
                    Astar[ib] += 0.25 * A
                continue

            # Non-obtuse: Voronoi area formula using cotangents.
            # Contribution at vertex a: (|ab|^2 cot(gamma) + |ac|^2 cot(beta)) / 8
            # where beta is angle at b? careful:
            # - cot at b is opposite edge ac => uses lengths (ab, bc, ac)
            # - cot at c is opposite edge ab => uses lengths (bc, ca, ab)
            cot_b = self._cot_from_squared_lengths(
                lab2, lbc2, lca2
            )  # angle at b, opposite ca
            cot_c = self._cot_from_squared_lengths(
                lca2, lbc2, lab2
            )  # angle at c, opposite ab
            cot_a = self._cot_from_squared_lengths(
                lab2, lca2, lbc2
            )  # angle at a, opposite bc

            # Map squared lengths:
            # |ab|^2 = lab2, |ac|^2 = lca2, |bc|^2 = lbc2
            Astar[ia] += (lab2 * cot_c + lca2 * cot_b) / 8.0
            Astar[ib] += (lab2 * cot_c + lbc2 * cot_a) / 8.0
            Astar[ic] += (lca2 * cot_b + lbc2 * cot_a) / 8.0

        # Numerical guard: clip tiny negatives due to floating error.
        return np.maximum(Astar, 0.0)

    def _dual_area_vertices_circumcentric(self) -> np.ndarray:
        """Compute circumcentric/Voronoi dual areas per vertex using a cotangent formula.

        Notes
        -----
        This method is retained for reference and diagnostics. For robust default behavior on
        obtuse meshes, use `_dual_area_vertices_voronoi_mixed()` (Option A).

        Returns
        -------
        np.ndarray
            Array of dual (Voronoi) areas for each vertex, shape (n_vertices,).
        """
        nV = self._V.shape[0]
        Astar = np.zeros(nV, dtype=float)
        for v in range(nV):
            pv = self._V[v]
            for _, j, k in self._incident[v]:
                pj = self._V[j]
                pk = self._V[k]
                vj = pj - pv
                vk = pk - pv
                jk = pk - pj
                lj2 = float(np.dot(vj, vj))
                lk2 = float(np.dot(vk, vk))
                ljk2 = float(np.dot(jk, jk))

                denom_j = 2.0 * np.sqrt(lj2 * ljk2) + self.eps
                cos_j = (lj2 + ljk2 - lk2) / denom_j
                denom_k = 2.0 * np.sqrt(lk2 * ljk2) + self.eps
                cos_k = (lk2 + ljk2 - lj2) / denom_k

                sin_j_sq = max(1.0 - cos_j * cos_j, 0.0)
                sin_j = np.sqrt(max(sin_j_sq, self.eps))
                cot_j = cos_j / sin_j

                sin_k_sq = max(1.0 - cos_k * cos_k, 0.0)
                sin_k = np.sqrt(max(sin_k_sq, self.eps))
                cot_k = cos_k / sin_k

                Astar[v] += (lk2 * cot_j + lj2 * cot_k) / 8.0
        return Astar

    def _dual_length_edges_circumcentric(self) -> np.ndarray:
        """Compute circumcentric dual edge lengths using a cotangent formula.

        Notes
        -----
        This method is retained for reference and diagnostics. Under Option A, the default
        star for k=1 uses barycentric dual edge lengths to avoid negative cotangent sums.

        Returns
        -------
        np.ndarray
            Array of dual edge lengths, shape (n_edges,).
        """
        nE = len(self._E)
        lstar = np.zeros(nE, dtype=float)
        primal_len = self._primal_edge_lengths()

        edge_to_opp = defaultdict(list)
        for a, b, c in self._T:
            edge_to_opp[_sorted_edge(a, b)].append(c)
            edge_to_opp[_sorted_edge(a, c)].append(b)
            edge_to_opp[_sorted_edge(b, c)].append(a)

        for ei, (u, v) in enumerate(self._E):
            key = _sorted_edge(u, v)
            pu = self._V[self._vid[u]]
            pv = self._V[self._vid[v]]
            le = primal_len[ei]

            cot_sum = 0.0
            for opp_label in edge_to_opp[key]:
                opp_idx = self._vid[opp_label]
                po = self._V[opp_idx]

                ou = pu - po
                ov = pv - po
                lu2 = float(np.dot(ou, ou))
                lv2 = float(np.dot(ov, ov))
                uv2 = float(np.dot(pu - pv, pu - pv))

                denom = 2.0 * np.sqrt(lu2 * lv2) + self.eps
                cos_opp = (lu2 + lv2 - uv2) / denom

                sin_sq = max(1.0 - cos_opp * cos_opp, 0.0)
                sin_val = np.sqrt(max(sin_sq, self.eps))
                cot_opp = cos_opp / sin_val

                cot_sum += cot_opp

            lstar[ei] = (le / 2.0) * cot_sum

        return lstar

    def _resolve_metric_tensors_per_triangle(self) -> np.ndarray:
        """Resolve per-triangle SPD tensors.

        Returns
        -------
        np.ndarray
            Tensor array of shape (nT, 3, 3).

        Raises
        ------
        ValueError
            If `tensors` has the wrong shape or `fn` does not return a (3, 3) matrix.
        """
        nT = len(self._T)
        if self.metric.tensors is not None:
            G = np.asarray(self.metric.tensors, dtype=float)
            if G.shape != (nT, 3, 3):
                raise ValueError(
                    f"metric.tensors must have shape {(nT, 3, 3)}, got {G.shape}."
                )
            return G
        if self.metric.fn is not None:
            G = np.zeros((nT, 3, 3), dtype=float)
            for t, (a, b, c) in enumerate(self._T):
                P = np.stack(
                    [
                        self._V[self._vid[a]],
                        self._V[self._vid[b]],
                        self._V[self._vid[c]],
                    ],
                    axis=0,
                )
                GT = np.asarray(self.metric.fn(t, P, self.sc), dtype=float)
                if GT.shape != (3, 3):
                    raise ValueError("metric.fn must return a (3, 3) matrix.")
                G[t] = GT
            return G
        return np.repeat(np.eye(3, dtype=float)[None, :, :], nT, axis=0)
