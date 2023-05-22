"""HyperEdge and HyperEdgeView classes."""


try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Hashable, Iterable


__all__ = ["HyperEdge"]


class HyperEdge:
    """Class for an hyperedge (or a set-type cell) in a combinatorial complex or a hyperedge complex.

    This class represents an set-type cell in a combinatorial complex,
    which is a set of nodes with optional attributes and a rank.
    The nodes in a hyperedge must be hashable and unique,
    and the hyperedge itself is immutable.


    :param elements: The nodes in the hyperedge.
    :type elements: any iterable of hashables
    :param rank: The rank of the hyperedge, if any. Default is None.
    :type rank: int, optional
    :param name: The name of the hyperedge, if any. Default is None.
    :type name: str, optional
    :param **attr: Additional attributes of the hyperedge, as keyword arguments.

    Examples
    --------
    >>> ac1 = HyperEdge ( (1,2,3) )
    >>> ac2 = HyperEdge ( (1,2,4,5) )
    >>> ac3 = HyperEdge ( ("a","b","c") )
    >>> ac3 = HyperEdge ( ("a","b","c"), rank = 10)
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
                    raise ValueError("every element HyperEdge must be hashable.")

        self.nodes = frozenset(list(elements))
        self._rank = rank

        if len(self.nodes) != len(elements):
            raise ValueError("a ranked entity cannot contain duplicate nodes")

        self.properties = dict()
        self.properties.update(attr)

    def __getitem__(self, item):
        """Get the attribute of the hyperedge."""
        if item not in self.properties:
            raise KeyError(f"attr {item} is not an attr in the hyperedge {self.name}")
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        """Set the attribute of the hyperedge."""
        self.properties[key] = item

    def __len__(self):
        """Number of nodes in the hyperedge."""
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the nodes of the hyperedge."""
        return iter(self.nodes)

    def __contains__(self, e):
        """Check if e is in the nodes."""
        return e in self.nodes

    def __repr__(self):
        """String representation of Hyperedge.

        Returns
        -------
        str
        """
        return f"HyperEdge{tuple(self.nodes)}"

    def __str__(self):
        """String representation of a HyperEdge.

        Returns
        -------
        str
        """
        return f"Nodes set:{tuple(self.nodes)}, attrs:{self.properties}"

    @property
    def rank(self):
        """Rank of the hyperedge."""
        if self._rank is not None:
            return self._rank
        else:
            print("hyperedge has none rank")
            return None
