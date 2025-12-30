"""
Exterior calculus operators built on TopoNetX coincidence matrices.

This module exposes `ExteriorCalculusOperators`, a user-facing wrapper around:
- topological coboundaries `d_k`,
- diagonal Hodge stars `*_k`,
- codifferentials `delta_k`,
- DEC Hodge Laplacians `Delta_k`.

For triangle meshes embedded in R^3, additional surface FEM operators on 0-forms
are available through the chosen metric backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.sparse import csr_matrix, identity

from .metric import (
    DiagonalHodgeStar,
    HodgeStarBackend,
    MetricPreset,
    MetricSpec,
    TriangleMesh3DBackend,
)


@dataclass
class ExteriorCalculusOperators:
    """Compute metric-aware DEC operators on a TopoNetX complex.

    Attributes
    ----------
    sc : object
        Complex providing `dim`, `skeleton(k)`, and `coincidence_matrix(rank, signed, index)`.
        For triangle-mesh presets, it must also provide `get_node_attributes(pos_name)`.
    pos_name : str
        Node attribute name storing vertex positions in R^3 for triangle-mesh presets.
    metric : MetricSpec | MetricPreset | None
        Metric/star specification. If None, Hodge stars default to identity matrices.
    star_backend : HodgeStarBackend | None
        Optional explicit backend. If provided, it overrides automatic selection.
    eps : float
        Numerical safeguard for star inverses and mass-lumping.

    Examples
    --------
    >>> ops = ExteriorCalculusOperators(sc, metric="circumcentric")
    >>> L1 = ops.dec_hodge_laplacian(1)
    """

    sc: object
    pos_name: str = "position"
    metric: MetricSpec | MetricPreset | None = field(
        default_factory=lambda: MetricSpec(preset="identity")
    )
    star_backend: HodgeStarBackend | None = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        """Post-initialize the operator object.

        Returns
        -------
        None
            This method mutates `metric` and `star_backend` in place.
        """
        self.eps = float(self.eps)
        self.metric = self._normalize_metric(self.metric)
        if self.star_backend is None:
            self.star_backend = self._select_backend()

    @staticmethod
    def _normalize_metric(
        metric: MetricSpec | MetricPreset | None,
    ) -> MetricSpec | None:
        """Normalize user-provided metric input into a `MetricSpec` or None.

        Parameters
        ----------
        metric : MetricSpec | MetricPreset | None
            User input metric specification.

        Returns
        -------
        MetricSpec | None
            Normalized metric specification.
        """
        if metric is None:
            return None
        if isinstance(metric, MetricSpec):
            return metric
        if isinstance(metric, str):
            return MetricSpec(preset=metric)  # type: ignore[arg-type]
        raise TypeError("metric must be None, a MetricSpec, or a preset string.")

    def _select_backend(self) -> HodgeStarBackend | None:
        """Select and construct the appropriate Hodge star backend.

        Returns
        -------
        HodgeStarBackend | None
            The selected backend instance, or None if metric support is disabled.

        Raises
        ------
        ValueError
            If the metric preset is not supported.
        """
        if self.metric is None:
            return None

        preset = self.metric.preset

        if preset in ("identity", "diagonal"):
            ms = MetricSpec(
                preset=preset,
                diagonal_weights=self.metric.diagonal_weights,
                primal_measures=self.metric.primal_measures,
                dual_measures=self.metric.dual_measures,
                tensors=self.metric.tensors,
                fn=self.metric.fn,
                eps=self.eps,
            )
            return DiagonalHodgeStar(metric=ms)

        if preset in ("barycentric_lumped", "circumcentric", "voronoi", "euclidean"):
            ms = MetricSpec(
                preset=preset,
                diagonal_weights=self.metric.diagonal_weights,
                primal_measures=self.metric.primal_measures,
                dual_measures=self.metric.dual_measures,
                tensors=self.metric.tensors,
                fn=self.metric.fn,
                eps=self.eps,
            )
            return TriangleMesh3DBackend(sc=self.sc, metric=ms, pos_name=self.pos_name)

        raise ValueError(f"Unsupported metric preset {preset!r}.")

    @property
    def dim(self) -> int:
        """Return the maximum simplex dimension of the complex.

        Returns
        -------
        int
            Maximum simplex dimension.
        """
        return int(self.sc.dim)

    def d_matrix(self, k: int, signed: bool = True) -> csr_matrix:
        """Return the coboundary matrix d_k : C^k -> C^{k+1}.

        Parameters
        ----------
        k : int
            Cochain degree.
        signed : bool
            If True, respect orientations. If False, take absolute values.

        Returns
        -------
        csr_matrix
            Sparse coboundary matrix.
        """
        if k < 0 or k >= self.dim + 1:
            raise ValueError(f"k must be in [0, {self.dim}], got {k}.")
        return self.sc.coincidence_matrix(k + 1, signed=signed, index=False)

    def hodge_star(self, k: int, inverse: bool = False) -> csr_matrix:
        """Return a discrete Hodge star matrix *_k.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool
            If True, return (*_k)^{-1}.

        Returns
        -------
        csr_matrix
            Sparse star matrix.
        """
        n_k = len(list(self.sc.skeleton(k)))
        if self.star_backend is None:
            return identity(n_k, format="csr")
        return self.star_backend.star(self.sc, k, inverse=inverse)

    def codifferential(self, k: int, signed: bool = True) -> csr_matrix:
        """Return the DEC codifferential delta_k : C^k -> C^{k-1}.

        Parameters
        ----------
        k : int
            Cochain degree. Must satisfy k >= 1.
        signed : bool
            Whether to use signed coboundaries.

        Returns
        -------
        csr_matrix
            Sparse codifferential matrix.
        """
        if k <= 0:
            raise ValueError("codifferential requires k >= 1.")
        star_k = self.hodge_star(k, inverse=False)
        star_km1_inv = self.hodge_star(k - 1, inverse=True)
        d_km1 = self.d_matrix(k - 1, signed=signed)
        return star_km1_inv @ (d_km1.T @ star_k)

    def dec_hodge_laplacian(self, k: int, signed: bool = True) -> csr_matrix:
        """Return the DEC Hodge Laplacian Delta_k on k-cochains.

        Parameters
        ----------
        k : int
            Cochain degree.
        signed : bool
            Whether to use signed coboundaries.

        Returns
        -------
        csr_matrix
            Sparse Laplacian matrix.
        """
        if k < 0 or k > self.dim:
            raise ValueError(f"k must be in [0, {self.dim}], got {k}.")

        term_up = 0
        if k < self.dim:
            d_k = self.d_matrix(k, signed=signed)
            delta_kp1 = self.codifferential(k + 1, signed=signed)
            term_up = delta_kp1 @ d_k

        term_down = 0
        if k > 0:
            delta_k = self.codifferential(k, signed=signed)
            d_km1 = self.d_matrix(k - 1, signed=signed)
            term_down = d_km1 @ delta_k

        return term_up + term_down

    def fem_mass_matrix_0(self) -> csr_matrix:
        """Return the consistent P1 FEM mass matrix on 0-forms.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex mass matrix.

        Raises
        ------
        RuntimeError
            If the current backend is not the triangle-mesh backend.
        """
        if not isinstance(self.star_backend, TriangleMesh3DBackend):
            raise RuntimeError(
                "fem_mass_matrix_0 requires a triangle-mesh metric preset."
            )
        return self.star_backend.fem_mass_matrix_0()

    def cotan_stiffness_0(self) -> csr_matrix:
        """Return the surface FEM stiffness matrix on 0-forms.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex stiffness matrix.

        Raises
        ------
        RuntimeError
            If the current backend is not the triangle-mesh backend.
        """
        if not isinstance(self.star_backend, TriangleMesh3DBackend):
            raise RuntimeError(
                "cotan_stiffness_0 requires a triangle-mesh metric preset."
            )
        return self.star_backend.cotan_stiffness_0()

    def riemannian_stiffness_0(self) -> csr_matrix:
        """Return anisotropic surface stiffness matrix on 0-forms.

        Returns
        -------
        csr_matrix
            Vertex-by-vertex anisotropic stiffness matrix.

        Raises
        ------
        RuntimeError
            If the current backend is not the triangle-mesh backend.
        """
        if not isinstance(self.star_backend, TriangleMesh3DBackend):
            raise RuntimeError(
                "riemannian_stiffness_0 requires a triangle-mesh metric preset."
            )
        return self.star_backend.riemannian_stiffness_0()

    def cotan_laplacian_0(self, *, lumped: bool = True) -> csr_matrix:
        """Return a cotangent Laplacian operator on 0-forms.

        Parameters
        ----------
        lumped : bool
            If True, return M_lumped^{-1} K. Otherwise return K.

        Returns
        -------
        csr_matrix
            Laplacian-like operator.

        Raises
        ------
        RuntimeError
            If the current backend is not the triangle-mesh backend.
        """
        if not isinstance(self.star_backend, TriangleMesh3DBackend):
            raise RuntimeError(
                "cotan_laplacian_0 requires a triangle-mesh metric preset."
            )
        return self.star_backend.cotan_laplacian_0(lumped=lumped)

    def riemannian_laplacian_0(self, *, lumped: bool = True) -> csr_matrix:
        """Return an anisotropic Laplacian operator on 0-forms.

        Parameters
        ----------
        lumped : bool
            If True, return M_lumped^{-1} K(G). Otherwise return K(G).

        Returns
        -------
        csr_matrix
            Laplacian-like operator.

        Raises
        ------
        RuntimeError
            If the current backend is not the triangle-mesh backend.
        """
        if not isinstance(self.star_backend, TriangleMesh3DBackend):
            raise RuntimeError(
                "riemannian_laplacian_0 requires a triangle-mesh metric preset."
            )
        return self.star_backend.riemannian_laplacian_0(lumped=lumped)
