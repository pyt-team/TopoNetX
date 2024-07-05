"""HyperEdge classes."""

from collections.abc import Collection, Hashable

from toponetx.classes.complex import Atom

__all__ = ["HyperEdge"]


class HyperEdge(Atom[frozenset[Hashable]]):
    """Class for a hyperedge (or a set-type cell).

    This class represents a set-type cell in a combinatorial complex, which is a set of
    nodes with optional attributes and a rank. The nodes in a hyperedge must be
    hashable and unique, and the hyperedge itself is immutable.

    Parameters
    ----------
    elements : iterable of hashables
        The nodes in the hyperedge.
    rank : int, optional
        The rank of the hyperedge. Default is None.
    **kwargs : additional attributes
        Additional attributes of the hyperedge, as keyword arguments.

    Examples
    --------
    >>> ac1 = tnx.HyperEdge((1, 2, 3))
    >>> ac2 = tnx.HyperEdge((1, 2, 4, 5))
    >>> ac3 = tnx.HyperEdge(("a", "b", "c"))
    >>> ac3 = tnx.HyperEdge(("a", "b", "c"), rank=10)
    """

    def __init__(self, elements: Collection, rank=None, **kwargs) -> None:
        for i in elements:
            if not isinstance(i, Hashable):
                raise TypeError("Every element of HyperEdge must be hashable.")

        super().__init__(frozenset(sorted(elements)), **kwargs)
        if len(elements) != len(self.elements):
            raise ValueError("A hyperedge cannot contain duplicate nodes.")

        self._rank = rank

    def __str__(self) -> str:
        """Return a string representation of the HyperEdge.

        Returns
        -------
        str
            A string representation of the HyperEdge.
        """
        return f"Nodes set: {tuple(self.elements)}, attrs: {self._attributes}"

    def __repr__(self):
        """Return a string representation of the HyperEdge.

        Returns
        -------
        str
            A string representation of the HyperEdge.
        """
        return f"Nodes set: {tuple(self.elements)}, rank: {self._rank}"

    def __hash__(self):
        """Return a hash representation of the HyperEdge using its elements and rank.

        Returns
        -------
        int
            A hashed representation of the HyperEdge.
        """
        return hash((self.elements, self._rank))

    def __eq__(self, other):
        """Return whether all attributes of HyperEdge objects are equal.

        Parameters
        ----------
        other : HyperEdge
            The other HyperEdge that needs to be compared to the current HyperEdge.

        Returns
        -------
        bool
            A boolean value if the HyperEdge objects are equal.
        """
        if isinstance(other, HyperEdge):
            return self.__dict__ == other.__dict__  # check all attributes
        return NotImplemented

    def __ne__(self, other):
        """Return whether any attributes of HyperEdge objects are not equal.

        Parameters
        ----------
        other : HyperEdge
            The other HyperEdge that needs to be compared to the current HyperEdge.

        Returns
        -------
        bool
            A boolean value if the HyperEdge objects are not equal.
        """
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    @property
    def rank(self):
        """Rank of the HyperEdge.

        Returns
        -------
        int or None
            The rank of the HyperEdge if it is not None, None otherwise.
        """
        if self._rank is not None:
            return self._rank
        return None
