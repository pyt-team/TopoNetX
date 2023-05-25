"""Abstract class for complexes."""


import abc

__all__ = ["Complex"]


class Complex:
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
    """

    def __init__(self):
        pass

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
    def shape(self) -> tuple:
        """Return number of cells for each rank."""
        pass

    @abc.abstractmethod
    def skeleton(self, rank):
        """Return dimension of the complex."""
        pass

    @abc.abstractmethod
    def __str__(self):
        """Print basic string representation."""
        pass

    @abc.abstractmethod
    def __repr__(self):
        """Print detailed string representation."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """Return number of nodes."""
        pass

    def _clear_cache(self):
        """Clear cache."""
        self.cache = {}

    @abc.abstractmethod
    def clone(self):
        """Clone complex."""

    @abc.abstractmethod
    def __iter__(self):
        """Return an iterator over the nodes."""
        pass

    @abc.abstractmethod
    def __contains__(self, item):
        """Check whether the complex contains an item."""
        pass

    @abc.abstractmethod
    def __getitem__(self, node):
        """Get item."""
        pass

    @abc.abstractmethod
    def remove_nodes(self, node_set):
        """Return dimension of the complex."""
        pass

    @abc.abstractmethod
    def add_node(self, node):
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
