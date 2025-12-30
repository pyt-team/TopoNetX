"""
Metric and geometry backends for exterior calculus on triangle meshes embedded in R^3.

This module provides:
- Geometry utilities for triangles embedded in R^3.
- A user-facing `MetricSpec` describing diagonal Hodge stars and anisotropic tensors.
- Backends implementing diagonal Hodge stars:
  - `DiagonalHodgeStar` (geometry-free).
  - `TriangleMesh3DBackend` (geometry-based).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

MetricPreset = Literal[
    "identity",
    "diagonal",
    "barycentric_lumped",
    "circumcentric",
    "voronoi",
    "euclidean",
]

MetricCallable = Callable[[int, np.ndarray, Any], np.ndarray]


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
    """
    n = np.cross(p1 - p0, p2 - p0)
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
    """
    a = p0
    b = p1
    c = p2
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
    """
    e01 = p1 - p0
    e02 = p2 - p0
    n = np.cross(e01, e02)
    nn = float(n @ n)
    det_norm = float(np.linalg.norm(n))
    if nn < 1e-28:
        z = np.zeros(3, dtype=float)
        return z, z, z, det_norm

    g0 = np.cross(n, p2 - p1) / nn
    g1 = np.cross(n, p0 - p2) / nn
    g2 = np.cross(n, p1 - p0) / nn
    return g0.astype(float), g1.astype(float), g2.astype(float), det_norm


@dataclass(frozen=True)
class MetricSpec:
    """Describe how to build diagonal Hodge stars and anisotropic FEM tensors.

    Attributes
    ----------
    preset : {"identity","diagonal","barycentric_lumped","circumcentric","voronoi","euclidean"}
        Preset name controlling the backend.
    diagonal_weights : dict[int, np.ndarray] | None
        Explicit diagonal entries for the Hodge star at each degree k.
        Each entry must have length equal to the number of k-simplices.
    primal_measures : dict[int, np.ndarray] | None
        Primal measures for generic diagonal DEC stars.
    dual_measures : dict[int, np.ndarray] | None
        Dual measures for generic diagonal DEC stars.
    tensors : np.ndarray | None
        Per-triangle SPD tensors for anisotropic stiffness, shape (nT, 3, 3).
    fn : callable | None
        Per-triangle tensor function fn(t, P, sc) -> (3,3), where P has shape (3,3).
    eps : float
        Numerical safeguard used for divisions/inversions.
    """

    preset: MetricPreset = "identity"
    diagonal_weights: dict[int, np.ndarray] | None = None
    primal_measures: dict[int, np.ndarray] | None = None
    dual_measures: dict[int, np.ndarray] | None = None
    tensors: np.ndarray | None = None
    fn: MetricCallable | None = None
    eps: float = 1e-12


class HodgeStarBackend(Protocol):
    """Protocol for Hodge star backends."""

    def star(self, sc: Any, k: int, *, inverse: bool = False) -> csr_matrix:
        """Return *_k (or its inverse).

        Parameters
        ----------
        sc : Any
            Complex object.
        k : int
            Cochain degree.
        inverse : bool
            If True, return (*_k)^{-1}.

        Returns
        -------
        csr_matrix
            Sparse star matrix.
        """


def _as_1d_float(x: np.ndarray, *, name: str) -> np.ndarray:
    """Convert an array to a 1D float array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        1D float array.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape, got shape {arr.shape}.")
    return arr


@dataclass(frozen=True)
class DiagonalHodgeStar:
    """Geometry-free diagonal Hodge star backend.

    Attributes
    ----------
    metric : MetricSpec
        Metric specification.
    """

    metric: MetricSpec

    def star(self, sc: Any, k: int, *, inverse: bool = False) -> csr_matrix:
        """Return a diagonal Hodge star matrix.

        Parameters
        ----------
        sc : Any
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
            diag = _as_1d_float(
                self.metric.diagonal_weights[k], name=f"diagonal_weights[{k}]"
            )
            if diag.shape[0] != n_k:
                raise ValueError(
                    f"diagonal_weights[{k}] must have length {n_k}, got {diag.shape[0]}."
                )
        else:
            if self.metric.primal_measures is None or self.metric.dual_measures is None:
                raise ValueError(
                    "preset='diagonal' requires diagonal_weights or both primal_measures and dual_measures."
                )
            pk = _as_1d_float(
                self.metric.primal_measures[k], name=f"primal_measures[{k}]"
            )
            dk = _as_1d_float(self.metric.dual_measures[k], name=f"dual_measures[{k}]")
            if pk.shape[0] != n_k or dk.shape[0] != n_k:
                raise ValueError(
                    f"Measures for k={k} must have length {n_k}; got primal {pk.shape[0]}, dual {dk.shape[0]}."
                )
            diag = dk / np.maximum(pk, eps)

        if inverse:
            diag = 1.0 / np.maximum(diag, eps)

        return diags(diag, 0, format="csr")


@dataclass
class TriangleMesh3DBackend:
    """Geometry backend for triangle meshes embedded in R^3.

    Attributes
    ----------
    sc : Any
        Complex providing `skeleton(0/1/2)` and `get_node_attributes(pos_name)`.
    metric : MetricSpec
        Metric specification.
    pos_name : str
        Node attribute name for vertex positions in R^3.
    """

    sc: Any
    metric: MetricSpec
    pos_name: str = "position"

    def __post_init__(self) -> None:
        """Build the internal mesh cache.

        Returns
        -------
        None
            This method prepares internal arrays for vertices, edges, and triangles.
        """
        self.eps = float(self.metric.eps)
        self._cache_ready = False
        self._build_cache()

    def _build_cache(self) -> None:
        """Build internal arrays aligned with the complex skeleton ordering.

        Returns
        -------
        None
            Populates internal mesh data.
        """
        tri = [tuple(s.elements) for s in self.sc.skeleton(2)]
        if not tri:
            raise ValueError("TriangleMesh3DBackend requires 2-simplices (triangles).")

        vlabels = [next(iter(s)) for s in self.sc.skeleton(0)]
        self._vid = {v: i for i, v in enumerate(vlabels)}

        pos = self.sc.get_node_attributes(self.pos_name)
        V = np.stack([np.asarray(pos[v], dtype=float) for v in vlabels], axis=0)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("Vertex positions must be 3D vectors (R^3).")

        self._V = V
        self._vlabels = vlabels

        edges = [tuple(s.elements) for s in self.sc.skeleton(1)]
        edges_sorted = [_sorted_edge(u, v) for (u, v) in edges]
        self._E = np.array(edges_sorted, dtype=object)

        self._T = np.array(tri, dtype=object)
        self._cache_ready = True

    def star(self, sc: Any, k: int, *, inverse: bool = False) -> csr_matrix:
        """Return a diagonal Hodge star for triangle meshes.

        Parameters
        ----------
        sc : Any
            Complex (unused except for sizing consistency).
        k : int
            Cochain degree.
        inverse : bool
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
            return self._star_circumcentric(k, inverse=inverse)

        raise ValueError(
            f"TriangleMesh3DBackend does not support preset={preset!r} for stars."
        )

    def _star_barycentric_lumped(self, k: int, inverse: bool) -> csr_matrix:
        """Compute a barycentric-lumped diagonal star.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool
            If True, invert the diagonal star.

        Returns
        -------
        csr_matrix
            Diagonal star.
        """
        if k not in (0, 1, 2):
            raise ValueError(
                "barycentric_lumped supports only k in {0,1,2} for triangle meshes."
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
            for (a, b, c), area in zip(self._T, A, strict=True):
                for e in (_sorted_edge(a, b), _sorted_edge(a, c), _sorted_edge(b, c)):
                    acc_e[e] = acc_e.get(e, 0.0) + area / 3.0
            edges = [tuple(e) for e in self._E.tolist()]
            return np.array([acc_e.get(tuple(e), 0.0) for e in edges], dtype=float)

        raise ValueError("k must be 0, 1, or 2.")

    def _star_circumcentric(self, k: int, inverse: bool) -> csr_matrix:
        """Compute a circumcentric (Voronoi) diagonal star.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool
            If True, invert the diagonal star.

        Returns
        -------
        csr_matrix
            Diagonal star.
        """
        if k == 0:
            w = self._dual_area_vertices_circumcentric()
            if inverse:
                w = 1.0 / np.maximum(w, self.eps)
            return diags(w, 0, format="csr")

        if k == 1:
            lstar = self._dual_length_edges_circumcentric()
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
            "circumcentric/voronoi supports only k in {0,1,2} for triangle meshes."
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
        lumped : bool
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
        lumped : bool
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

    def _dual_area_vertices_circumcentric(self) -> np.ndarray:
        """Compute circumcentric dual areas per vertex.

        Returns
        -------
        np.ndarray
            Dual areas per vertex.
        """
        nV = self._V.shape[0]
        Astar = np.zeros((nV,), dtype=float)

        C = np.zeros((len(self._T), 3), dtype=float)
        for t, (a, b, c) in enumerate(self._T):
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            C[t] = _circumcenter_3d(self._V[ia], self._V[ib], self._V[ic])

        for t, (a, b, c) in enumerate(self._T):
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            pa, pb, pc = self._V[ia], self._V[ib], self._V[ic]
            cc = C[t]

            mab = 0.5 * (pa + pb)
            mac = 0.5 * (pa + pc)
            mbc = 0.5 * (pb + pc)

            Astar[ia] += _triangle_area_3d(pa, mab, cc) + _triangle_area_3d(pa, cc, mac)
            Astar[ib] += _triangle_area_3d(pb, mbc, cc) + _triangle_area_3d(pb, cc, mab)
            Astar[ic] += _triangle_area_3d(pc, mac, cc) + _triangle_area_3d(pc, cc, mbc)

        return Astar

    def _dual_length_edges_circumcentric(self) -> np.ndarray:
        """Compute circumcentric dual lengths per edge.

        Returns
        -------
        np.ndarray
            Dual lengths per edge.
        """
        C = np.zeros((len(self._T), 3), dtype=float)
        for t, (a, b, c) in enumerate(self._T):
            ia, ib, ic = self._vid[a], self._vid[b], self._vid[c]
            C[t] = _circumcenter_3d(self._V[ia], self._V[ib], self._V[ic])

        inc: dict[tuple[Any, Any], list[int]] = {tuple(e): [] for e in self._E.tolist()}
        for t, (a, b, c) in enumerate(self._T):
            for e in (_sorted_edge(a, b), _sorted_edge(a, c), _sorted_edge(b, c)):
                if e in inc:
                    inc[e].append(t)

        lstar = np.zeros((len(self._E),), dtype=float)
        for ei, (u, v) in enumerate(self._E):
            ts = inc[(u, v)]
            pu = self._V[self._vid[u]]
            pv = self._V[self._vid[v]]
            mid = 0.5 * (pu + pv)

            if len(ts) >= 2:
                lstar[ei] = float(np.linalg.norm(C[ts[0]] - C[ts[1]]))
            elif len(ts) == 1:
                lstar[ei] = float(np.linalg.norm(C[ts[0]] - mid))
            else:
                lstar[ei] = 0.0

        return lstar

    def _resolve_metric_tensors_per_triangle(self) -> np.ndarray:
        """Resolve per-triangle SPD tensors.

        Returns
        -------
        np.ndarray
            Tensor array of shape (nT, 3, 3).
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
                    raise ValueError("metric.fn must return a (3,3) matrix.")
                G[t] = GT
            return G

        return np.repeat(np.eye(3, dtype=float)[None, :, :], nT, axis=0)
