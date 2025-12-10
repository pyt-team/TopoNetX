"""Simplex Class."""

from collections.abc import Collection, Hashable, Iterable
from functools import total_ordering
from typing import Self

from toponetx.classes.complex import Atom
from toponetx.utils.iterable import is_ordered_subset

__all__ = ["Simplex"]


@total_ordering
class Simplex[ElementType: Hashable](Atom):
    """A class representing a simplex in a simplicial complex.

    This class represents a simplex in a simplicial complex, which is a set of nodes with a specific dimension. The
    simplex is immutable, and the nodes in the simplex must be hashable and unique.

    Parameters
    ----------
    elements : Collection[Hashable]
        The nodes in the simplex.
    **kwargs : keyword arguments, optional
        Additional attributes to be associated with the simplex.

    Examples
    --------
    >>> # Create a 0-dimensional simplex (point)
    >>> s = tnx.Simplex((1,))
    >>> # Create a 1-dimensional simplex (line segment)
    >>> s = tnx.Simplex((1, 2))
    >>> # Create a 2-dimensional simplex (triangle)
    >>> simplex1 = tnx.Simplex((1, 2, 3))
    >>> simplex2 = tnx.Simplex(("a", "b", "c"))
    >>> # Create a 3-dimensional simplex (tetrahedron)
    >>> simplex3 = tnx.Simplex((1, 2, 4, 5), weight=1)
    """

    elements: frozenset[Hashable]

    def __init__(
        self,
        elements: Collection[ElementType],
        **kwargs,
    ) -> None:
        self.validate_attributes(kwargs)

        if len(elements) != len(set(elements)):
            raise ValueError("A simplex cannot contain duplicate nodes.")

        super().__init__(tuple(sorted(elements)), **kwargs)

    def __contains__(self, item: ElementType | Iterable[ElementType]) -> bool:
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
        >>> s = tnx.Simplex((1, 2, 3))
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
            item = tuple(sorted(item))
            return is_ordered_subset(item, self.elements)
        return super().__contains__(item)

    def __le__(self, other) -> bool:
        """Return True if this simplex comes before the other simplex in the lexicographic order.

        Parameters
        ----------
        other : Any
            The other simplex to compare with.

        Returns
        -------
        bool
            True if this simplex comes before the other simplex in the lexicographic order.
        """
        if not isinstance(other, Simplex):
            return NotImplemented
        return tuple(self.elements) <= tuple(other.elements)

    @staticmethod
    def validate_attributes(attributes: dict) -> None:
        """Validate the attributes of the simplex.

        Parameters
        ----------
        attributes : dict
            The attributes to be validated.

        Raises
        ------
        ValueError
            If the attributes contain the reserved keys `is_maximal` or `membership`.
        """
        if "is_maximal" in attributes or "membership" in attributes:
            raise ValueError(
                "Special attributes `is_maximal` and `membership` are reserved."
            )

    def sign(self, face: "Simplex[ElementType]") -> int:
        """Calculate the sign of the simplex with respect to a given face.

        Parameters
        ----------
        face : Simplex
            A face of the simplex.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return string representation of the simplex.

        Returns
        -------
        str
            A string representation of the simplex.
        """
        return f"Simplex({tuple(self.elements)})"

    def __str__(self) -> str:
        """Return string representation of the simplex.

        Returns
        -------
        str
            A string representation of the simplex.
        """
        return f"Nodes set: {tuple(self.elements)}, attrs: {self._attributes}"

    def clone(self) -> Self:
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex
        and attributes. That is, if an attribute is a container, that container is
        shared by the original and the copy. Use Python's `copy.deepcopy` for new
        containers.

        Returns
        -------
        Simplex
            A copy of this simplex.
        """
        return self.__class__(self.elements, **self._attributes)
