"""Path class."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

from typing_extensions import Self

from toponetx.classes.complex import Atom

__all__ = ["Path"]


class Path(Atom[tuple[Hashable]]):
    """Path class.

    A class representing an elementary p-path in a path complex, which is the building
    block for a path complex. By the definition established in the original paper [2]_,
    an elementary p-path on a non-empty set of vertices V is any sequence of vertices
    with length p + 1.

    Unlike the original paper [2]_ where elementary p-paths span the regular space of
    boundary-invariant paths, the elementary p-paths represented by this class span the
    space of simple paths with length p as proposed in [1]_.

    Parameters
    ----------
    elements : Sequence[Hashable]
        The nodes in the elementary p-path.
    construct_boundaries : bool, default=False
        If `True`, construct the entire boundary of the elementary p-path.
    reserve_sequence_order : bool, default=False
        If `True`, reserve the order of the sub-sequence of nodes in the elementary
        p-path. Else, the sub-sequence of nodes in the elementary p-path will be
        reversed if the first index is larger than the last index.
    allowed_paths : Iterable[tuple[Hashable]], optional
        A list of allowed boundaries of an elementary p-path. If None, all possible
        boundaries are constructed (similarly to simplex).
    **kwargs : keyword arguments, optional
        Additional attributes to be associated with the elementary p-path.

    Notes
    -----
    - An elementary p-path is defined as an ordered sequence of nodes (n1, ..., np) if
      we reserve the sequence order. Thus, for this case, (n1, ..., np) is different
      from (np, ..., n1). If we do not reserve the sequence order, then (n1, ..., np)
      is the same as (np, ..., n1). For this case, the first index must be smaller than
      the last index so that we can discard the other duplicate elementary p-path. As
      elementary p-paths may contain characters and numbers at the same time, we
      leverage lexicographical order to compare two indices.
    - Similarly to simplex, in order to find the boundary of an elementary p-path, we
      remove one node at a time from the elementary p-path. However, unlike simplex
      where order does not matter, the process of omitting nodes from an elementary
      p-path may result some non-existing paths. For instance, if we have an elementary
      p-path (1, 2, 3) and there is no path (1, 3), then applying boundary operation on
      the elementary p-path results in [(2,3), (1,3), (1,2)]. In order to avoid this,
      we can provide a list of allowed boundaries. If the boundary of an elementary
      p-path is not in the list of allowed boundaries, then it will not be included in
      the boundary of the elementary p-path.
    - When an elementary p-path is created and allowed paths are not specified, its
      boundary is automatically created by iteratively removing one node at a time from
      the elementary p-path, which is identical to a simplex.

    References
    ----------
    .. [1] Truong and Chin.
        Generalizing Topological Graph Neural Networks with Paths.
        https://arxiv.org/abs/2308.06838.pdf
    .. [2] Grigor'yan, Lin, Muranov, and Yau.
        Homologies of path complexes and digraphs.
        https://arxiv.org/pdf/1207.2834.pdf

    Examples
    --------
    >>> path1 = tnx.Path((1, 2, 3))
    >>> list(path1.boundary)
    []
    >>> path2 = tnx.Path((1, 2, 3), construct_boundaries=True)
    >>> list(path2.boundary)
    [(2, 3), (1, 3), (1, 2)]
    >>> path3 = tnx.Path(
    ...     (1, 2, 3), construct_boundaries=True, allowed_paths=[(1, 2), (2, 3)]
    ... )
    >>> list(path3.boundary)
    [(2, 3), (1, 2)]
    >>> path4 = tnx.Path(
    ...     (1, 2, 3), construct_boundaries=True, allowed_paths=[(1, 3), (2, 3)]
    ... )
    >>> list(path4.boundary)
    [(2, 3), (1, 3)]
    """

    def __init__(
        self,
        elements: Sequence[Hashable] | Hashable,
        construct_boundaries: bool = False,
        reserve_sequence_order: bool = False,
        allowed_paths: Iterable[tuple[Hashable]] | None = None,
        **kwargs,
    ) -> None:
        self.__check_inputs(elements, reserve_sequence_order)
        if not isinstance(elements, Sequence):
            elements = [elements]
        super().__init__(tuple(elements), **kwargs)
        if len(set(elements)) != len(self.elements):
            raise ValueError("A p-path cannot contain duplicate nodes.")

        self.construct_boundaries = construct_boundaries
        self.reserve_sequence_order = reserve_sequence_order
        if construct_boundaries:
            self._boundaries = self.construct_path_boundaries(
                elements,
                reserve_sequence_order=reserve_sequence_order,
                allowed_paths=allowed_paths,
            )
        else:
            self._boundaries = []

    @staticmethod
    def construct_path_boundaries(
        elements: Sequence[Hashable],
        reserve_sequence_order: bool = False,
        allowed_paths: Iterable[tuple[Hashable]] | None = None,
    ) -> list[tuple[Hashable, ...]]:
        """Return list of elementary p-path objects representing the boundaries.

        Parameters
        ----------
        elements : Sequence[Hashable]
            The nodes in the elementary p-path.
        reserve_sequence_order : bool, default=False
            If True, reserve the order of the sub-sequence of nodes in the elementary
            p-path. Else, the sub-sequence of nodes in the elementary p-path will be
            reversed if the first index is larger than the last index.
        allowed_paths : Iterable[tuple[Hashable]], optional
            A list of allowed boundaries. If None, all possible boundaries are
            constructed (similarly to simplex).

        Returns
        -------
        list[tuple[Hashable]]
            List of elementary p-path objects representing the boundaries.

        Notes
        -----
        - In order to construct a set of boundary (p-1)-paths of an elementary p-path
          (n1, ..., np), we iteratively omit one node at a time from the elementary
          p-path. The resulted boundary paths are elementary (p-1)-paths in the form of
          (n1, ..., ni-1, ni+1, ..., np), where i is the index of the omitted node.
          However, some of the elementary (p-1)-paths may be non-existing (or
          non-allowed in our context), so we discard them from the boundary set.

        Examples
        --------
        >>> tnx.Path.construct_path_boundaries((1, 2, 3), reserve_sequence_order=False)
        [(2, 3), (1, 3), (1, 2)]
        >>> tnx.Path.construct_path_boundaries(
        ...     (1, 2, 3), reserve_sequence_order=False, allowed_paths=[(1, 2), (2, 3)]
        ... )
        [(2, 3), (1, 2)]
        """
        boundaries = []
        for i in range(len(elements)):
            boundary = list(elements[0:i]) + list(elements[(i + 1) :])
            if (
                not reserve_sequence_order
                and len(boundary) > 1
                and str(boundary[0]) > str(boundary[-1])
            ):
                boundary.reverse()
            if allowed_paths is None or tuple(boundary) in allowed_paths:
                boundaries.append(tuple(boundary))
        return boundaries

    @property
    def boundary(self) -> list[tuple[Hashable, ...]]:
        """Return the boundary of this path.

        The boundary of this path are all elementary (p-1)-path objects representing
        the elementary (p-1)-path on the boundary of the target elementary p-path.

        Returns
        -------
        list[tuple[Hashable]]
            List of elementary p-path objects representing the boundary (p-1)-paths.
        """
        return self._boundaries

    def clone(self) -> Self:
        """Return a shallow copy of the elementary p-path.

        Returns
        -------
        Path
            A shallow copy of the elementary p-path.
        """
        return self.__class__(
            self.elements,
            construct_boundaries=self.construct_boundaries,
            **self._attributes,
        )

    def __check_inputs(self, elements: Any, reserve_sequence_order: bool) -> None:
        """Sanity check for inputs, as sequence order matters.

        Parameters
        ----------
        elements : Any
            The nodes in the elementary p-path.
        reserve_sequence_order : bool
            If True, reserve the order of the sub-sequence of nodes in the elementary
            p-path. Else, the sub-sequence of nodes in the elementary p-path will be
            reversed if the first index is larger than the last index. This variable is
            only used for sanity check on the input.
        """
        if isinstance(elements, Sequence):
            for i in elements:
                if not isinstance(i, Hashable):
                    raise ValueError(f"All nodes of a p-path must be hashable, got {i}")
            if (
                not reserve_sequence_order
                and len(elements) > 1
                and str(elements[0]) > str(elements[-1])
            ):
                raise ValueError(
                    f"An elementary p-path must have the first index smaller than the last index, got {elements}."
                )
        elif isinstance(elements, Hashable):
            if not isinstance(elements, int) or isinstance(elements, str):
                raise ValueError(
                    f"Elements of an elementary p-path must be an int or str, got {type(elements)}."
                )

    def __repr__(self) -> str:
        """Return string representation of elementary p-paths.

        Returns
        -------
        str
            The __repr__ representation of the Path object.
        """
        return f"Path({self.elements})"

    def __str__(self) -> str:
        """Return string representation of elementary p-paths.

        Returns
        -------
        str
            The __str__ representation of the Path object.
        """
        return f"Node set: {self.elements}, Boundaries: {self.boundary}, Attributes: {self._attributes}"
