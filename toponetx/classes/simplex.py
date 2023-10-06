"""Simplex Class."""

import warnings
from collections.abc import Collection, Hashable, Iterable
from itertools import combinations
from typing import Any

from toponetx.classes.complex import Atom

__all__ = ["Simplex"]


class Simplex(Atom):
    """
    A class representing a simplex in a simplicial complex.

    This class represents a simplex in a simplicial complex, which is a set of nodes with a specific dimension. The
    simplex is immutable, and the nodes in the simplex must be hashable and unique.

    Parameters
    ----------
    elements: Collection
        The nodes in the simplex.
    name : str, optional
        A name for the simplex.
    construct_tree : bool, default=True
        If True, construct the entire simplicial tree for the simplex.
    attr : keyword arguments, optional
        Additional attributes to be associated with the simplex.

    Examples
    --------
    >>> # Create a 0-dimensional simplex (point)
    >>> s = Simplex((1,))
    >>> # Create a 1-dimensional simplex (line segment)
    >>> s = Simplex((1, 2))
    >>> # Create a 2-dimensional simplex (triangle)
    >>> simplex1 = Simplex((1, 2, 3))
    >>> simplex2 = Simplex(("a", "b", "c"))
    >>> # Create a 3-dimensional simplex (tetrahedron)
    >>> simplex3 = Simplex((1, 2, 4, 5), weight=1)
    """

    def __init__(
        self, elements: Collection, name: str = "", construct_tree: bool = False, **attr
    ) -> None:
        if construct_tree is not False:
            warnings.warn(
                "The `construct_tree` argument is deprecated.",
                DeprecationWarning,
            )

        for i in elements:
            if not isinstance(i, Hashable):
                raise ValueError(f"All nodes of a simplex must be hashable, got {i}")

        super().__init__(frozenset(sorted(elements)), name, **attr)
        if len(elements) != len(self.elements):
            raise ValueError("A simplex cannot contain duplicate nodes.")

    def __contains__(self, item: Any) -> bool:
        """Return True if the given element is a subset of the nodes.

        Parameters
        ----------
        item : Any
            The element to be checked.

        Returns
        -------
        bool
            True if the given element is a subset of the nodes.

        Examples
        --------
        >>> s = Simplex((1, 2, 3))
        >>> 1 in s
        True
        >>> 4 in s
        False
        >>> (1, 2) in s
        True
        >>> (1, 4) in s
        False
        """
        if isinstance(item, Iterable):
            return frozenset(item) <= self.elements
        return super().__contains__(item)

    @staticmethod
    def construct_simplex_tree(elements: Collection) -> frozenset["Simplex"]:
        """Return set of Simplex objects representing the faces."""
        warnings.warn(
            "`Simplex.construct_simplex_tree` is deprecated.", DeprecationWarning
        )

        faceset = set()
        for r in range(len(elements), 0, -1):
            for face in combinations(elements, r):
                faceset.add(
                    Simplex(elements=sorted(face), construct_tree=False)
                )  # any face is always ordered
        return frozenset(faceset)

    @property
    def boundary(self) -> frozenset["Simplex"]:
        """Return the set of the set of all n-1 faces in of the input n-simplex.

        Returns
        -------
        frozenset[Simplex]
            A frozenset representing boundary simplices.

        Examples
        --------
        For a n-simplex [1,2,3], the boundary is all the n-1 subsets of [1,2,3] :
            (1,2), (2,3), (3,1).
        """
        """Return a set of Simplex objects representing the boundary faces."""
        warnings.warn(
            "`Simplex.boundary` is deprecated, use `SimplicialComplex.get_boundaries()` on the simplicial complex that contains this simplex instead.",
            DeprecationWarning,
        )

        return frozenset(
            Simplex(elements, construct_tree=False)
            for elements in combinations(self.elements, len(self) - 1)
        )

    def sign(self, face) -> int:
        """Calculate the sign of the simplex with respect to a given face.

        Parameters
        ----------
        face : Simplex
            A face of the simplex.
        """
        raise NotImplementedError()

    @property
    def faces(self):
        """Get the set of faces of the simplex.

        If `construct_tree` is True, return the precomputed set of faces `_faces`.
        Otherwise, construct the simplex tree and return the set of faces.

        Returns
        -------
        frozenset[Simplex]
            The set of faces of the simplex.
        """
        warnings.warn(
            "`Simplex.faces` is deprecated, use `SimplicialComplex.get_boundaries()` on the simplicial complex that contains this simplex instead.",
            DeprecationWarning,
        )

        return Simplex.construct_simplex_tree(self.elements)

    def __repr__(self) -> str:
        """Return string representation of the simplex.

        :return: A string representation of the simplex.
        :rtype: str
        """
        return f"Simplex({tuple(self.elements)})"

    def __str__(self) -> str:
        """Return string representation of the simplex.

        :return: A string representation of the simplex.
        :rtype: str
        """
        return f"Nodes set: {tuple(self.elements)}, attrs: {self._attributes}"

    def clone(self) -> "Simplex":
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex and attributes. That is, if an
        attribute is a container, that container is shared by the original and the copy. Use Python’s `copy.deepcopy`
        for new containers.

        Returns
        -------
        Simplex
        """
        return Simplex(self.elements, name=self.name, **self._attributes)
