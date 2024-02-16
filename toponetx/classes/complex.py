"""Abstract class for Complex and Atom."""


import abc
from collections.abc import Collection, Hashable, Iterator
from typing import Any

__all__ = ["Atom", "Complex"]


class Atom(abc.ABC):
    """Abstract class representing an atom in a complex.

    Parameters
    ----------
    elements : Collection[Hashable]
        The elements in the atom.
    **kwargs : keyword arguments, optional
        Additional attributes to be associated with the atom.
    """

    elements: Collection[Hashable]
    name: str

    def __init__(self, elements: Collection[Hashable], **kwargs) -> None:
        """Abstract class representing an atom in a complex.

        Parameters
        ----------
        elements : Collection[Hashable]
            The elements in the atom.
        **kwargs : keyword arguments, optional
            Additional attributes to be associated with the atom.
        """
        self.elements = elements

        self._attributes = {}
        self._attributes.update(kwargs)

    def __len__(self) -> int:
        """Return the number of elements in the atom.

        Returns
        -------
        int
            The number of elements in the atom.
        """
        return len(self.elements)

    def __iter__(self) -> Iterator:
        """Return an iterator over the elements in the atom.

        Returns
        -------
        Iterator
            Iterator over the elements in the atom.
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
            Returns `True` if the item is contained in the element
            and `False` otherwise.
        """
        return item in self.elements

    def __getitem__(self, item: Any) -> Any:
        """Return the attribute with the given name.

        Parameters
        ----------
        item : Any
            The name of the attribute.

        Returns
        -------
        Any
            The value of the attribute.

        Raises
        ------
        KeyError
            If the attribute does not exist on this atom.
        """
        return self._attributes[item]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set the attribute with the given name to the given value.

        Parameters
        ----------
        key : Any
            The name of the attribute.
        value : Any
            The value of the attribute.
        """
        self._attributes[key] = value

    def update(self, attributes: dict) -> None:
        """Update the attributes of the atom.

        Parameters
        ----------
        attributes : dict
            The attributes to be updated.
        """
        self._attributes.update(attributes)


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
    **kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.
    """

    complex: dict[Any, Any]

    def __init__(self, **kwargs) -> None:
        """Initialize a new instance of the Complex class.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Attributes to add to the complex as key=value pairs.
        """
        self.complex = {}
        self.complex.update(kwargs)

    @property
    @abc.abstractmethod
    def nodes(self):
        """Return the node container."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Return dimension of the complex.

        Returns
        -------
        int
            Returns the dimension of the complex.
        """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return number of cells for each rank.

        Returns
        -------
        tuple of ints
            The number of elements for each rank. If the complex is empty, an empty
            tuple is returned.
        """

    @abc.abstractmethod
    def skeleton(self, rank: int):
        """Return the atoms of given `rank` in this complex.

        Parameters
        ----------
        rank : int
            The rank of the skeleton.
        """

    @abc.abstractmethod
    def __str__(self) -> str:
        """Print basic string representation.

        Returns
        -------
        str
            Returns the string representation of the complex.
        """

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Printable representation of the complex.

        Makes an attempt to return a string that would produce an object with the same value when passed to ``eval()``,
        but may not be possible for all objects.

        Returns
        -------
        str
            Returns the __repr__ representation of the complex.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return number of nodes."""

    @abc.abstractmethod
    def clone(self) -> "Complex":
        """Clone complex."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        """Return an iterator over the nodes."""

    @abc.abstractmethod
    def __contains__(self, item: Any) -> bool:
        """Check whether the complex contains an item.

        Parameters
        ----------
        item : Any
            The item to be checked.

        Returns
        -------
        bool
            Returns `True` if the complex contains the item else `False`.
        """

    @abc.abstractmethod
    def __getitem__(self, key):
        """Get item.

        Parameters
        ----------
        key : hashable
            Get item based on key.

        Returns
        -------
        Hashable
            The hashable item that needs to be returned.
        """

    @abc.abstractmethod
    def remove_nodes(self, node_set: Iterator[Hashable]) -> None:
        """Remove the given nodes from the complex.

        Any elements that become invalid due to the removal of nodes are also removed.

        Parameters
        ----------
        node_set : Iterator[Hashable]
            The nodes to be removed.
        """

    @abc.abstractmethod
    def add_node(self, node: Hashable) -> None:
        """Add node to the complex.

        Parameters
        ----------
        node : Hashable
            The node to be added.
        """

    @abc.abstractmethod
    def incidence_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Return incidence matrix of the complex.

        Parameters
        ----------
        rank : int
            The rank of the atoms to consider.
        signed : bool, default=True
            If True, the incidence matrix is signed, otherwise it is unsigned.
        weight : str, optional
            The name of the attribute to use as weights for the incidence matrix.
        index : bool, default=False
            If True, the incidence matrix is indexed by the nodes of the complex.
        """

    @abc.abstractmethod
    def adjacency_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Return adjacency matrix of the complex.

        Parameters
        ----------
        rank : int
            The rank of the atoms to consider.
        signed : bool, default=True
            If True, the adjacency matrix is signed, otherwise it is unsigned.
        weight : str, optional
            The name of the attribute to use as weights for the adjacency matrix.
        index : bool, default=False
            If True, the adjacency matrix is indexed by the atoms of the complex.
        """

    @abc.abstractmethod
    def coadjacency_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Return coadjacency matrix of the complex.

        Parameters
        ----------
        rank : int
            The rank of the atoms to consider.
        signed : bool, default=True
            If True, the adjacency matrix is signed, otherwise it is unsigned.
        weight : str, optional
            The name of the attribute to use as weights for the adjacency matrix.
        index : bool, default=False
            If True, the adjacency matrix is indexed by the atoms of the complex.
        """
