# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:14:07 2022

@author: Mustafa Hajij
"""

"""
Abstract class for complexes
"""


from abc import ABC, abstractmethod

__all__ = ["Complex"]


class Complex(ABC):

    """
    An abstract class representing a complex.

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

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def skeleton(self, k):
        pass

    @abstractmethod
    def __str__(self):
        """
        String representation of CX

        Returns
        -------
        str

        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        String representation of cell complex

        Returns
        -------
        str

        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Number of nodes

        Returns
        -------
        int

        """

        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over the nodes of the cell complex

        Returns
        -------
        dict_keyiterator

        """
        pass

    @abstractmethod
    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.nodes

        Parameters
        ----------
        item : hashable or RankedEntity

        """

        return item in self.nodes

    @abstractmethod
    def __getitem__(self, node):
        """
        Returns the neighbors of node

        Parameters
        ----------
        node : Entity or hashable
            If hashable, then must be uid of node in cell complex

        Returns
        -------
        neighbors(node) : iterator

        """
        return self.neighbors(node)

    @abstractmethod
    def remove_nodes(self, node_set):
        """
        Removes nodes from cells and deletes references in cell complex nodes

        Parameters
        ----------
        node_set : an iterable of hashables or Entities
            Nodes in CC

        Returns
        -------
        cell complex : CombinatorialComplex

        """
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node, **attr):

        self._G.add_node(node, **attr)

    @abstractmethod
    def incidence_matrix(self, d, signed=True, weight=None, index=False):
        pass

    @abstractmethod
    def adjacency_matrix(self, d, signed=False, weight=None, index=False):
        pass

    @abstractmethod
    def coadjacency_matrix(self, d, signed=False, weight=None, index=False):
        pass

    @abstractmethod
    def node_adjacency_matrix(self, index=False, s=1, weight=False):
        pass
