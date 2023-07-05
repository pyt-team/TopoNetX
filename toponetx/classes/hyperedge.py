"""HyperEdge classes."""


from collections.abc import Collection, Hashable

from toponetx.classes.complex import Atom

__all__ = ["HyperEdge"]


class HyperEdge(Atom):
    """Class for a hyperedge (or a set-type cell) in a combinatorial complex or a hyperedge complex.

    This class represents a set-type cell in a combinatorial complex,
    which is a set of nodes with optional attributes and a rank.
    The nodes in a hyperedge must be hashable and unique,
    and the hyperedge itself is immutable.

    Parameters
    ----------
    elements : iterable of hashables
        The nodes in the hyperedge.
    rank : int, optional
        The rank of the hyperedge. Default is None.
    name : str, optional
        The name of the hyperedge.
    **attr : additional attributes
        Additional attributes of the hyperedge, as keyword arguments.

    Examples
    --------
    >>> ac1 = HyperEdge((1, 2, 3))
    >>> ac2 = HyperEdge((1, 2, 4, 5))
    >>> ac3 = HyperEdge(("a", "b", "c"))
    >>> ac3 = HyperEdge(("a", "b", "c"), rank=10)
    """

    def __init__(self, elements: Collection, name: str = "", rank=None, **attr) -> None:
        for i in elements:
            if not isinstance(i, Hashable):
                raise TypeError("Every element of HyperEdge must be hashable.")

        super().__init__(frozenset(sorted(elements)), name, **attr)
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
        return f"Nodes set: {tuple(self.elements)}, attrs: {self._properties}"

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
        else:
            print("HyperEdge has no rank")
            return None
