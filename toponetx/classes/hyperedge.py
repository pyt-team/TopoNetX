"""HyperEdge classes."""


from collections.abc import Hashable, Iterable

__all__ = ["HyperEdge"]


class HyperEdge:
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
        The name of the hyperedge. Default is None.
    **attr : additional attributes
        Additional attributes of the hyperedge, as keyword arguments.

    Examples
    --------
    >>> ac1 = HyperEdge((1, 2, 3))
    >>> ac2 = HyperEdge((1, 2, 4, 5))
    >>> ac3 = HyperEdge(("a", "b", "c"))
    >>> ac3 = HyperEdge(("a", "b", "c"), rank=10)
    """

    def __init__(self, elements, rank=None, name=None, **attr):
        if name is None:
            self.name = ""
        else:
            self.name = name

        if isinstance(elements, Hashable) and not isinstance(elements, Iterable):
            elements = frozenset([elements])
        if elements is not None:
            for i in elements:
                if not isinstance(i, Hashable):
                    raise TypeError("Every element of HyperEdge must be hashable.")

        self.nodes = frozenset(list(elements))
        self._rank = rank

        if len(self.nodes) != len(elements):
            raise ValueError("A ranked entity cannot contain duplicate nodes.")

        self.properties = dict()
        self.properties.update(attr)

    def __getitem__(self, item):
        """Get the attribute of the hyperedge.

        Parameters
        ----------
        item : str
            The attribute name.

        Returns
        -------
        Any
            The value of the attribute.

        Raises
        ------
        KeyError
            If the attribute is not present in the hyperedge.
        """
        if item not in self.properties:
            raise KeyError(
                f"Attribute '{item}' is not present in the hyperedge '{self.name}'."
            )
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        """Set the attribute of the hyperedge.

        Parameters
        ----------
        key : str
            The attribute name.
        item : Any
            The value of the attribute.
        """
        self.properties[key] = item

    def __len__(self):
        """Compute the number of nodes in the hyperedge.

        Returns
        -------
        int
            The number of nodes in the hyperedge.
        """
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the nodes of the hyperedge.

        Returns
        -------
        Iterator
            An iterator over the nodes of the hyperedge.
        """
        return iter(self.nodes)

    def __contains__(self, e):
        """Check if e is in the nodes.

        Parameters
        ----------
        e : Any
            The element to check.

        Returns
        -------
        bool
            True if e is in the nodes, False otherwise.
        """
        return e in self.nodes

    def __str__(self):
        """Return a string representation of the HyperEdge.

        Returns
        -------
        str
            A string representation of the HyperEdge.
        """
        return f"Nodes set: {tuple(self.nodes)}, attrs: {self.properties}"

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
