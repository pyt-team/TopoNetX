r"""
Exterior calculus operators built on TopoNetX coincidence matrices.

This module exposes :class:`~toponetx.algorithms.exterior_calculus.ExteriorCalculusOperators`,
a user-facing wrapper around:

- topological coboundaries ``d_k``,
- diagonal discrete Hodge stars ``*_k``,
- codifferentials ``δ_k``,
- DEC Hodge Laplacians ``Δ_k``.

Notes
-----
**Mathematical background.**
Let ``K`` be a finite cell/simplicial complex of dimension ``d``. For each ``k``,
let ``C^k(K)`` denote the space of real-valued *k-cochains* (functions assigning a
number to each oriented k-cell). The *coboundary* operator is

.. math::

    d_k : C^k(K) \to C^{k+1}(K),

implemented by TopoNetX as a (co)incidence matrix between k- and (k+1)-cells (with
optional signs encoding orientation).

Discrete Exterior Calculus (DEC) introduces a (typically diagonal) discrete Hodge
star

.. math::

    *_k : C^k(K) \to C^{d-k}(K),

which depends on a choice of metric/geometry and defines an inner product on
cochains:

.. math::

    \langle \alpha, \beta \rangle_k := \alpha^\top (*_k)\, \beta.

Given ``d_{k-1}`` and the Hodge stars, the DEC *codifferential* is

.. math::

    \delta_k := (*_{k-1})^{-1}\, d_{k-1}^\top\, (*_k) : C^k(K) \to C^{k-1}(K),

and the DEC *Hodge Laplacian* is

.. math::

    \Delta_k := \delta_{k+1} d_k + d_{k-1} \delta_k : C^k(K) \to C^k(K),

with the convention that missing terms are zero when indices go out of range.

**Implementation notes.**
- ``d_k`` is obtained from :meth:`TopoNetX Complex.coincidence_matrix` as
  ``coincidence_matrix(k+1, signed=..., index=False)``.
- ``*_k`` is produced by a *backend* chosen from the user metric specification.
  If metric support is disabled (``metric=None``), then ``*_k`` defaults to the
  identity matrix.
- For triangle meshes embedded in ``R^3`` (triangle-mesh presets), additional
  surface FEM operators on 0-forms are exposed through the chosen backend:
  consistent mass matrix and stiffness matrices, plus mass-lumped Laplacians.

Examples
--------
Basic DEC Laplacian on a simplicial complex (identity metric)::

    import toponetx as tnx
    from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators

    sc = tnx.SimplicialComplex([[0, 1, 2]])
    ops = ExteriorCalculusOperators(sc, metric="identity")

    d0 = ops.d_matrix(0)  # C^0 -> C^1
    S1 = ops.hodge_star(1)  # *_1
    delta1 = ops.codifferential(1)
    L0 = ops.dec_hodge_laplacian(0)

Triangle mesh in R^3 with circumcentric stars (requires vertex positions)::

    import toponetx as tnx
    from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators

    sc = tnx.SimplicialComplex([[0, 1, 2], [0, 2, 3]])
    sc.set_simplex_attributes(
        {
            0: [0.0, 0.0, 0.0],
            1: [1.0, 0.0, 0.0],
            2: [1.0, 1.0, 0.0],
            3: [0.0, 1.0, 0.0],
        },
        name="position",
    )

    ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")
    S0 = ops.hodge_star(0)
    M0 = ops.fem_mass_matrix_0()
    K0 = ops.cotan_stiffness_0()
    Lcot = ops.cotan_laplacian_0(lumped=True)

Disable metric support (all stars become identities)::

    ops = ExteriorCalculusOperators(sc, metric=None)
    S2 = ops.hodge_star(2)  # identity on C^2
"""

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
    r"""Compute metric-aware DEC operators on a TopoNetX complex.

    This class is a thin, user-facing wrapper that constructs and exposes core
    operators of Discrete Exterior Calculus (DEC) on a TopoNetX complex.

    Notes
    -----
    **What it provides.**
    - Coboundary operators ``d_k`` obtained from TopoNetX coincidence matrices.
    - Discrete Hodge stars ``*_k`` from a backend determined by a metric preset
      (or an explicitly supplied backend).
    - Codifferentials ``δ_k`` defined via Hodge stars and transposed coboundaries:

      .. math::

          \delta_k := (*_{k-1})^{-1}\, d_{k-1}^\top\, (*_k).

    - DEC Hodge Laplacians ``Δ_k`` assembled as:

      .. math::

          \Delta_k := \delta_{k+1} d_k + d_{k-1} \delta_k.

    **Metric behavior.**
    Users may pass:
    - ``metric=None`` to disable metric support (all stars become identities),
    - a preset string (``MetricPreset``),
    - a :class:`~toponetx.algorithms.exterior_calculus.metric.MetricSpec`,
    - an explicit ``star_backend`` implementing
      :class:`~toponetx.algorithms.exterior_calculus.metric.HodgeStarBackend`,
      which overrides automatic selection.

    **Triangle mesh extras.**
    For triangle-mesh presets (e.g. ``"circumcentric"``, ``"voronoi"``,
    ``"barycentric_lumped"``, ``"euclidean"``), the backend is
    :class:`~toponetx.algorithms.exterior_calculus.metric.TriangleMesh3DBackend`
    and the complex must provide 3D positions under ``pos_name``. In that case,
    this wrapper exposes surface FEM helpers on 0-forms via the backend:
    mass matrix, stiffness matrices, and (mass-lumped) Laplacians.

    Attributes
    ----------
    sc : object
        Complex providing:

        - ``dim`` (maximum dimension),
        - ``skeleton(k)`` (iterable of k-cells / simplices),
        - ``coincidence_matrix(rank, signed, index)`` for coboundaries.

        For triangle-mesh presets, it must also provide
        ``get_node_attributes(pos_name)`` returning 3D vertex coordinates.
    pos_name : str
        Node attribute name storing vertex positions in ``R^3`` for triangle-mesh
        presets (default: ``"position"``).
    metric : MetricSpec | MetricPreset | None
        Metric/star specification.

        - If ``None``, metric support is disabled and stars default to identity.
        - If a preset string, it is normalized to ``MetricSpec(preset=...)``.
        - If ``MetricSpec``, it is used as-is (with ``eps`` set from this object).
    star_backend : HodgeStarBackend | None
        Optional explicit backend overriding automatic selection.
    eps : float
        Numerical safeguard for star inverses and mass-lumping.

    Examples
    --------
    Build operators on a simplicial complex and compute a Laplacian::

        import toponetx as tnx
        from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators

        sc = tnx.SimplicialComplex([[0, 1, 2]])
        ops = ExteriorCalculusOperators(sc, metric="identity")

        L0 = ops.dec_hodge_laplacian(0)

    Use a triangle-mesh preset (requires 3D vertex positions)::

        import toponetx as tnx
        from toponetx.algorithms.exterior_calculus import ExteriorCalculusOperators

        sc = tnx.SimplicialComplex([[0, 1, 2], [0, 2, 3]])
        sc.set_simplex_attributes(
            {
                0: [0.0, 0.0, 0.0],
                1: [1.0, 0.0, 0.0],
                2: [1.0, 1.0, 0.0],
                3: [0.0, 1.0, 0.0],
            },
            name="position",
        )

        ops = ExteriorCalculusOperators(sc, metric="circumcentric", pos_name="position")
        M0 = ops.fem_mass_matrix_0()
        Lcot = ops.cotan_laplacian_0(lumped=True)

    Disable metric support::

        ops = ExteriorCalculusOperators(sc, metric=None)
        S1 = ops.hodge_star(1)  # identity
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

        This method:

        1. Casts ``eps`` to float.
        2. Normalizes the user-supplied ``metric`` into a
           :class:`~toponetx.algorithms.exterior_calculus.metric.MetricSpec`
           (or ``None`` to disable metric support).
        3. If no explicit ``star_backend`` is provided, selects one automatically
           based on the normalized metric preset.

        Returns
        -------
        None
            This method mutates ``metric`` and ``star_backend`` in place.
        """
        self.eps = float(self.eps)
        self.metric = self._normalize_metric(self.metric)
        if self.star_backend is None:
            self.star_backend = self._select_backend()

    @staticmethod
    def _normalize_metric(
        metric: MetricSpec | MetricPreset | None,
    ) -> MetricSpec | None:
        """Normalize user-provided metric input into a ``MetricSpec`` or ``None``.

        Users may pass:

        - ``None``: disables metric support (stars become identities).
        - :class:`~toponetx.algorithms.exterior_calculus.metric.MetricSpec`: used directly.
        - a preset string: converted to ``MetricSpec(preset=<string>)``.

        Parameters
        ----------
        metric : MetricSpec | MetricPreset | None
            User input metric specification.

        Returns
        -------
        MetricSpec | None
            Normalized metric specification.

        Raises
        ------
        TypeError
            If ``metric`` is not ``None``, a ``MetricSpec``, or a preset string.
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

        The backend determines how diagonal Hodge stars ``*_k`` are computed.

        Returns
        -------
        HodgeStarBackend | None
            The selected backend instance, or ``None`` if metric support is disabled.

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
        r"""Return the coboundary matrix ``d_k : C^k -> C^{k+1}``.

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
        r"""Return a discrete Hodge star matrix ``*_k``.

        Parameters
        ----------
        k : int
            Cochain degree.
        inverse : bool
            If True, return ``(*_k)^{-1}``.

        Returns
        -------
        csr_matrix
            Sparse star matrix (diagonal in the provided backends).
        """
        n_k = len(list(self.sc.skeleton(k)))
        if self.star_backend is None:
            return identity(n_k, format="csr")
        return self.star_backend.star(self.sc, k, inverse=inverse)

    def codifferential(self, k: int, signed: bool = True) -> csr_matrix:
        r"""Return the DEC codifferential ``δ_k : C^k -> C^{k-1}``.

        Parameters
        ----------
        k : int
            Cochain degree. Must satisfy ``k >= 1``.
        signed : bool
            Whether to use signed coboundaries (orientation-aware).

        Returns
        -------
        csr_matrix
            Sparse codifferential matrix.

        Raises
        ------
        ValueError
            If ``k <= 0``.
        """
        if k <= 0:
            raise ValueError("codifferential requires k >= 1.")
        star_k = self.hodge_star(k, inverse=False)
        star_km1_inv = self.hodge_star(k - 1, inverse=True)
        d_km1 = self.d_matrix(k - 1, signed=signed)
        return star_km1_inv @ (d_km1.T @ star_k)

    def dec_hodge_laplacian(self, k: int, signed: bool = True) -> csr_matrix:
        r"""Return the DEC Hodge Laplacian ``Δ_k`` on k-cochains.

        Parameters
        ----------
        k : int
            Cochain degree.
        signed : bool
            Whether to use signed coboundaries (orientation-aware).

        Returns
        -------
        csr_matrix
            Sparse Laplacian matrix.

        Raises
        ------
        ValueError
            If ``k`` is outside ``[0, dim]``.
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
        r"""Return the consistent P1 FEM mass matrix on 0-forms.

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
        r"""Return the surface FEM stiffness matrix on 0-forms.

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
        r"""Return anisotropic surface stiffness matrix on 0-forms.

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
        r"""Return a cotangent Laplacian operator on 0-forms.

        Parameters
        ----------
        lumped : bool
            If True, return ``M_lumped^{-1} K``. Otherwise return ``K``.

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
        r"""Return an anisotropic Laplacian operator on 0-forms.

        Parameters
        ----------
        lumped : bool
            If True, return ``M_lumped^{-1} K(G)``. Otherwise return ``K(G)``.

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
