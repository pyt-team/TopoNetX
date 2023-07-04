"""Abstract class for complexes."""


import abc
from collections.abc import Collection, Iterator
from typing import Any

__all__ = ["Atom", "Complex"]


class Atom(abc.ABC):
    """Abstract class representing an atom in a complex.

    Parameters
    ----------
    elements : Collection
        The elements in the atom.
    name : str, optional
        A name for the atom.
    kwargs : keyword arguments, optional
        Additional attributes to be associated with the atom.
    """

    def __init__(self, elements: Collection, name: str = "", **kwargs) -> None:
        self.elements = elements
        self.name = name

        self._properties = dict()
        self._properties.update(kwargs)

    def __len__(self) -> int:
        """Return the number of elements in the atom."""
        return len(self.elements)

    def __iter__(self) -> Iterator:
        """Return an iterator over the elements in the atom.

        Returns
        -------
        Iterator
        """
        return iter(self.elements)

    def __contains__(self, item: Any) -> bool:
        """Return True if the given element is contained in this atom.

        Parameters
        ----------
        item : Any
            The item to be checked.

        Returns
        -------
        bool
        """
        return item in self.elements

    def __getitem__(self, item: Any) -> Any:
        """Return the property with the given name.

        Parameters
        ----------
        item : Any
            The name of the property.

        Returns
        -------
        Any
            The value of the property.

        Raises
        ------
        KeyError
            If the property does not exist.
        """
        return self._properties[item]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set the property with the given name to the given value.

        Parameters
        ----------
        key : Any
            The name of the property.
        value : Any
            The value of the property.
        """
        self._properties[key] = value

    def update(self, attributes: dict) -> None:
        """Update the properties of the atom.

        Parameters
        ----------
        attributes : dict
            The properties to be updated.
        """
        self._properties.update(attributes)


class Complex(abc.ABC):
    """Abstract class representing a complex.

    A complex is a space that is constructed by attaching lower-dimensional
    cells to a topological space to form a new space. The cells are attached to the space in a specific way,
    and the resulting space has a well-defined structure.

    Example of complexes:

    (1) Cell Complexes : Cell complexes can be used to represent various mathematical objects, such as graphs,
    manifolds, and discrete geometric shapes. They are useful in many areas of mathematics,
    such as algebraic topology and geometry, where they can be used to study the structure and
    properties of these objects.

    (2) Simplicial Complexes : Simplicial complexes are mathematical structures used to
    study the topological properties of shapes and spaces. They are made up of a set
    of points called vertices, and a collection of simplices (triangles, tetrahedra, etc.)
    that are connected to each other in a specific way. Each simplex is formed by a subset
    of the vertices and is considered a building block of the complex. The properties of
    the complex are determined by the combinatorial arrangement of the simplices and their
    connectivity. Simplicial complexes are used in many areas of mathematics and computer
    science, such as geometric modeling, data analysis, and machine learning.

    (3) The CombinatorialComplex class represents a combinatorial complex, which is a mathematical
    structure consisting of a set of points, a subset of the power set of points, and a ranking function
    that assigns a rank to each subset based on its size. These classes are used in many areas of mathematics
    and computer science, such as geometric modeling, data analysis, and machine learning.

    Parameters
    ----------
    name : str, optional
        Optional name for the complex.
    kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.
    """

    complex: dict[Any, Any]

    def __init__(self, name: str = "", **kwargs) -> None:
        self.name = name
        self.complex = dict()
        self.complex.update(kwargs)

    @property
    @abc.abstractmethod
    def nodes(self):
        """Return the node container."""
        pass

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Return dimension of the complex."""
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return number of cells for each rank.

        Returns
        -------
        tuple of ints
            The number of elements for each rank. If the complex is empty, an empty tuple is returned.
        """
        pass

    @abc.abstractmethod
    def skeleton(self, rank: int):
        """Return dimension of the complex."""
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        """Print basic string representation."""
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Printable representation of the complex.

        Makes an attempt to return a string that would produce an object with the same value when passed to ``eval()``,
        but may not be possible for all objects.

        Returns
        -------
        str
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return number of nodes."""
        pass

    @abc.abstractmethod
    def clone(self) -> "Complex":
        """Clone complex."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        """Return an iterator over the nodes."""
        pass

    @abc.abstractmethod
    def __contains__(self, item: Any) -> bool:
        """Check whether the complex contains an item."""
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        """Get item."""
        pass

    @abc.abstractmethod
    def remove_nodes(self, node_set) -> None:
        """Remove the given nodes from the complex.

        Any elements that become invalid due to the removal of nodes are also removed.
        """
        pass

    @abc.abstractmethod
    def add_node(self, node) -> None:
        """Add node to the complex."""
        pass

    @abc.abstractmethod
    def incidence_matrix(self):
        """Return incidence matrix of the complex."""
        pass

    @abc.abstractmethod
    def adjacency_matrix(self):
        """Return adjacency matrix of the complex."""
        pass

    @abc.abstractmethod
    def coadjacency_matrix(self):
        """Return coadjacency matrix of the complex."""
        pass
