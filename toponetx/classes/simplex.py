"""Simplex Class."""

from collections.abc import Hashable, Iterable
from itertools import combinations

__all__ = ["Simplex"]


class Simplex:
    """
    A class representing a simplex in a simplicial complex.

    This class represents a simplex in a simplicial complex, which is a set of nodes with a specific dimension. The
    simplex is immutable, and the nodes in the simplex must be hashable and unique.

    Parameters
    ----------
    elements: Iterable
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

    def __init__(self, elements, name=None, construct_tree=True, **attr):
        if name is None:
            self.name = ""
        else:
            self.name = name
        self.construct_tree = construct_tree
        self.nodes = frozenset(elements)
        if len(self.nodes) != len(elements):
            raise ValueError("A simplex cannot contain duplicate nodes.")

        if construct_tree:
            self._faces = self.construct_simplex_tree(elements)
        else:
            self._faces = frozenset()
        self.properties = dict()
        self.properties.update(attr)

    def __contains__(self, e):
        """Return True if the given element is a subset of the nodes."""
        if len(self.nodes) == 0:
            return False
        if isinstance(e, Iterable):
            if len(e) > len(self.nodes):
                return False
            else:
                if isinstance(e, frozenset):
                    return e <= self.nodes
                else:
                    return frozenset(e) <= self.nodes
        elif isinstance(e, Hashable):
            return frozenset({e}) <= self.nodes
        else:
            return False

    @staticmethod
    def construct_simplex_tree(elements):
        """Return set of Simplex objects representing the faces."""
        faceset = set()
        numnodes = len(elements)
        for r in range(numnodes, 0, -1):
            for face in combinations(elements, r):
                faceset.add(
                    Simplex(elements=sorted(face), construct_tree=False)
                )  # any face is always ordered
        return frozenset(faceset)

    @property
    def boundary(self):
        """Return a set of Simplex objects representing the boundary faces."""
        if self.construct_tree:
            return frozenset(i for i in self._faces if len(i) == len(self) - 1)
        else:
            faces = Simplex.construct_simplex_tree(self.nodes)
            return frozenset(i for i in faces if len(i) == len(self) - 1)

    def sign(self, face):
        """Calculate the sign of the simplex with respect to a given face.

        Parameters
        ----------
        face : Simplex
            A face of the simplex.
        """
        raise NotImplementedError

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
        if self.construct_tree:
            return self._faces
        else:
            return Simplex.construct_simplex_tree(self.nodes)

    def __getitem__(self, item):
        """Get item.

        Get the value of the attribute associated with the simplex.

        :param item: The name of the attribute.
        :type item: str
        :return: The value of the attribute.
        :raises KeyError: If the attribute is not found in the simplex.
        """
        if item not in self.properties:
            raise KeyError(f"Attribute '{item}' is not found in the simplex.")
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        """Set the value of an attribute associated with the simplex.

        :param key: The name of the attribute.
        :type key: str
        :param item: The value of the attribute.
        """
        self.properties[key] = item

    def __len__(self):
        """Get the number of nodes in the simplex.

        :return: The number of nodes in the simplex.
        :rtype: int
        """
        return len(self.nodes)

    def __iter__(self):
        """Get an iterator over the nodes in the simplex.

        :return: An iterator over the nodes in the simplex.
        :rtype: iter
        """
        return iter(self.nodes)

    def __repr__(self):
        """Return string representation of the simplex.

        :return: A string representation of the simplex.
        :rtype: str
        """
        return f"Simplex{tuple(self.nodes)}"

    def __str__(self):
        """Return string representation of the simplex.

        :return: A string representation of the simplex.
        :rtype: str
        """
        return f"Nodes set: {tuple(self.nodes)}, attrs: {self.properties}"
