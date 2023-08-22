"""Creation and manipulation of a combinatorial complex."""

from collections.abc import Collection, Hashable, Iterable, Iterator
from typing import Literal

import networkx as nx
import numpy as np
from networkx import Graph
from scipy.sparse import csr_matrix

from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.reportviews import HyperEdgeView, NodeView
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.exception import TopoNetXError
from toponetx.utils.structure import (
    incidence_to_adjacency,
    sparse_array_to_neighborhood_dict,
)

__all__ = ["CombinatorialComplex"]


class CombinatorialComplex(ColoredHyperGraph):
    """Class for Combinatorial Complex.

    A Combinatorial Complex (CC) is a triple CC = (S, X, rk) where:
    - S is an abstract set of entities,
    - X a subset of the power set of X, and
    - rk is the a rank function that associates for every set x in X a rank, a positive integer.

    The rank function rk must satisfy x <= y then rk(x) <= rk(y).
    We call this condition the CC condition.

    A CC is a generlization of graphs, hypergraphs, cellular and simplicial complexes.

    Parameters
    ----------
    cells : Collection, optional
    name : str, optional
        An identifiable name for the combinatorial complex.
    ranks : Collection, optional
        when cells is an iterable or dictionary, ranks cannot be None and it must be iterable/dict of the same
        size as cells.
    weight : array-like, optional
        User specified weight corresponding to setsytem of type pandas.DataFrame,
        length must equal number of rows in dataframe.
        If None, weight for all rows is assumed to be 1.
    graph_based : bool, default=False
        When true rank 1 edges must have cardinality equals to 1
    kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.

    Mathematical example
    --------------------
    Let S = {1, 2, 3, 4} be a set of abstract entities.
    Let X = {{1, 2}, {1, 2, 3}, {1, 3}, {1, 4}} be a subset of the power set of S.
    Let rk be the ranking function that assigns the
    length of a set as its rank, i.e. rk({1, 2}) = 2, rk({1, 2, 3}) = 3, etc.

    Then, (S, X, rk) is a combinatorial complex.

    Examples
    --------
    Define an empty combinatorial complex:

    >>> CC = CombinatorialComplex()

    Add cells to the combinatorial complex:

    >>> CC = CombinatorialComplex()
    >>> CC.add_cell([1, 2], rank=1)
    >>> CC.add_cell([3, 4], rank=1)
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> CC.add_cell([1, 2, 4], rank=2)
    >>> CC.add_cell([3, 4], rank=2)
    >>> CC.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
    """

    def __init__(
        self,
        cells: Collection | None = None,
        name: str = "",
        ranks: Collection | None = None,
        graph_based: bool = False,
        **kwargs,
    ) -> None:
        Complex.__init__(self, **kwargs)
        self.name = name

        self.graph_based = graph_based  # rank 1 edges have cardinality equals to 1
        self._node_membership = dict()
        self._max_complex = set()
        self._complex_set = HyperEdgeView()
        if cells is not None:
            if not isinstance(cells, Iterable):
                raise TypeError(
                    f"Input cells must be given as Iterable, got {type(cells)}."
                )

            if not isinstance(cells, Graph):
                if ranks is None:
                    for cell in cells:
                        if not isinstance(cell, HyperEdge):
                            raise ValueError(
                                f"input must be an HyperEdge {cell} object when rank is None"
                            )
                        if cell.rank is None:
                            raise ValueError(f"input HyperEdge {cell} has None rank")
                        self.add_cell(cell, rank=cell.rank)
                else:
                    if isinstance(cells, Iterable) and isinstance(ranks, Iterable):
                        if len(cells) != len(ranks):
                            raise TopoNetXError(
                                "cells and ranks must have equal number of elements"
                            )
                        else:
                            for cell, rank in zip(cells, ranks):
                                self.add_cell(cell, rank)
                if isinstance(cells, Iterable) and isinstance(ranks, int):
                    for cell in cells:
                        self.add_cell(cell, ranks)
            else:
                for node in cells.nodes:  # cells is a networkx graph
                    self.add_node(node, **cells.nodes[node])
                for edge in cells.edges:
                    u, v = edge
                    self.add_cell([u, v], 1, **cells.get_edge_data(u, v))

    def __str__(self) -> str:
        """Return detailed string representation."""
        return f"Combinatorial Complex with {len(self.nodes)} nodes and cells with ranks {self.ranks} and sizes {self.shape} "

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CombinatorialComplex(name='{self.name}')"

    def __setitem__(self, cell, attr):
        """Set the attributes of a cell in the CC."""
        return super().__setitem__(cell, attr)

    @property
    def __shortstr__(self):
        """Return the short string generic representation."""
        return "CC"

    def number_of_nodes(self, node_set=None) -> int:
        """Compute the number of nodes in node_set belonging to the CC.

        Parameters
        ----------
        node_set : an interable of Entities, optional, default: None
            If None, then return the number of nodes in the CC.

        Returns
        -------
        int
        """
        return super().number_of_nodes(node_set)

    def number_of_cells(self, cell_set=None) -> int:
        """Compute the number of cells in cell_set belonging to the CC.

        Parameters
        ----------
        cell_set : an interable of HyperEdge, optional
            If None, then return the number of cells.

        Returns
        -------
        int
        """
        return super().number_of_cells(cell_set)

    def shape(self):
        """Return shape.

        This is:
        (number of cells[i], for i in range(0,dim(CC))  )

        Returns
        -------
        tuple of ints
        """
        return super().shape()

    def order(self):
        """Compute the number of nodes in the CC.

        Returns
        -------
        order : int
        """
        return super().order()

    def _remove_node_helper(self, node) -> None:
        """Remove node from cells. Assumes node is present in the CC."""
        # Removing node in hyperedgeview
        for key in list(self.cells.hyperedge_dict.keys()):
            for key_rank in list(self.cells.hyperedge_dict[key].keys()):
                replace_key = key_rank.difference(node)
                if len(replace_key) > 0:
                    if key_rank != replace_key:
                        self._max_complex.difference_update(
                            {HyperEdge(key_rank, rank=key)}
                        )
                        del self.cells.hyperedge_dict[key][key_rank]
                else:
                    # Remove original hyperedge from the ranks
                    self._max_complex.difference_update({HyperEdge(key_rank, rank=key)})
                    del self.cells.hyperedge_dict[key][key_rank]
            if self.cells.hyperedge_dict[key] == {}:
                del self.cells.hyperedge_dict[key]

    def remove_nodes(self, node_set) -> None:
        """Remove nodes from cells.

        This also deletes references in combinatorial complex nodes.

        Parameters
        ----------
        node_set : an iterable of hashables
            Nodes in CC
        """
        return super().remove_nodes(node_set)

    def remove_node(self, node) -> None:
        """Remove node from cells.

        This also deletes any reference in the nodes of the CC.
        This also deletes cell references in higher ranks for the particular node.

        Parameters
        ----------
        node : hashable or HyperEdge

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        super()._remove_node(node)

    def set_cell_attributes(self, values, name: str | None = None) -> None:
        """Set cell attributes.

        Parameters
        ----------
        values : TYPE
            DESCRIPTION.
        name : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        Examples
        --------
        After computing some property of the cell of a combinatorial complex, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CC = CombinatorialComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2,)
        >>> CC.add_cell([3, 4], rank=2)
        >>> d = {(1, 2, 3, 4): 'red', (1, 2, 3): 'blue', (3, 4): 'green'}
        >>> CC.set_cell_attributes(d, name='color')
        >>> CC.cells[(3, 4)]['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CC = NestedCombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red','attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3}}
        >>> CC.set_cell_attributes(d)
        >>> CC.cells[(0, 1)]['color']
        'blue'
        3

        Note that if the dict contains cells that are not in `self.cells`, they are
        silently ignored.
        """
        super().set_cell_attributes(values, name)

    def get_node_attributes(self, name: str):
        """Get node attributes.

        Parameters
        ----------
        name : str
           Attribute name

        Returns
        -------
        Dictionary of attributes keyed by node.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = NestedCombinatorialComplex(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 },1: {'color': 'blue', 'attr2': 3} }
        >>> CC.set_node_attributes(d)
        >>> CC.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CC = NestedCombinatorialComplex(G)
        >>> nodes_color = CC.get_node_attributes('color')
        >>> nodes_color[1]
        'blue'
        """
        return super().get_node_attributes(name)

    def get_cell_attributes(self, name: str, rank=None):
        """Get node attributes from graph.

        Parameters
        ----------
        name : str
           Attribute name.
        rank : int
            rank of the k-cell

        Returns
        -------
        Dictionary of attributes keyed by cell or k-cells if k is not None

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = CombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red', 'attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3} }
        >>> CC.set_cell_attributes(d)
        >>> cell_color = CC.get_cell_attributes('color')
        >>> cell_color[frozenset({0, 1})]
        'blue'
        """
        return super().get_cell_attributes(name, rank)

    def _add_node(self, node, **attr) -> None:
        """Add one node as hyperedge."""
        self._add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_node(self, node, **attr) -> None:
        """Add a node."""
        self._add_node(node, **attr)

    def _add_hyperedge(self, hyperedge, rank, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge : HyperEdge, Hashable or Iterable
            a cell in a combinatorial complex
        rank : int
            the rank of a hyperedge, must be zero when the hyperedge is Hashable.
        **attr : attr associated with hyperedge

        Returns
        -------
        None.

        Notes
        -----
        The add_hyperedge is a method for adding hyperedges to the HyperEdgeView instance.
        It takes two arguments: hyperedge and rank, where hyperedge is a tuple or HyperEdge instance
        representing the hyperedge to be added, and rank is an integer representing the rank of the hyperedge.
        The add_hyperedge method then adds the hyperedge to the hyperedge_dict attribute of the HyperEdgeView
        instance, using the hyperedge's rank as the key and the hyperedge itself as the value.
        This allows the hyperedge to be accessed later using its rank.

        Note that the add_hyperedge method also appears to check whether the hyperedge being added
        is a valid hyperedge of the combinatorial complex by checking whether the hyperedge's nodes
        are contained in the _aux_complex attribute of the HyperEdgeView instance.
        If the hyperedge's nodes are not contained in _aux_complex, then the add_hyperedge method will
        not add the hyperedge to hyperedge_dict. This is done to ensure that the HyperEdgeView
        instance only contains valid hyperedges.
        """
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"rank must be a non-negative integer, got {rank}")

        if isinstance(hyperedge, str):
            if rank != 0:
                raise ValueError(f"rank must be zero for string input, got rank {rank}")
            hyperedge_set = frozenset({hyperedge})
        elif isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            if rank != 0:
                raise ValueError(f"rank must be zero for hashables, got rank {rank}")
            hyperedge_set = frozenset({hyperedge})
        elif isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            hyperedge_ = (
                hyperedge.elements
                if isinstance(hyperedge, HyperEdge)
                else frozenset(hyperedge)
            )
            if not all(isinstance(i, Hashable) for i in hyperedge_):
                raise ValueError("every element hyperedge must be hashable.")
            if rank == 0 and len(hyperedge_) > 1:
                raise ValueError(
                    "rank must be positive for higher order hyperedges, got rank = 0"
                )
            hyperedge_set = hyperedge_
        else:
            raise ValueError("Invalid hyperedge type")
        cell_add = HyperEdge(hyperedge_set, rank=rank)
        for elem in self._max_complex:
            # Check if the element is a superset
            if elem.elements.issuperset(cell_add.elements):
                # Check if the element is not equal to the cell
                if len(elem.elements) > len(cell_add.elements):
                    # Check if the rank of the element is greater than the rank
                    if elem._rank < cell_add._rank:
                        raise ValueError(
                            "a violation of the combinatorial complex condition:"
                            + f"the hyperedge {elem.elements} in the complex has rank {elem._rank} is larger than {cell_add._rank}, the rank of the input hyperedge {cell_add.elements} "
                        )

        self._max_complex.add(HyperEdge(hyperedge_set, rank=rank))
        self._add_hyperedge_helper(hyperedge_set, rank, **attr)
        # This is O(N) time complexity to add a hyper edge with O(N) space complexity
        if "weight" not in self._complex_set.hyperedge_dict[rank][hyperedge_set]:
            self._complex_set.hyperedge_dict[rank][hyperedge_set]["weight"] = 1
        if isinstance(hyperedge, HyperEdge):
            self._complex_set.hyperedge_dict[rank][hyperedge_set].update(
                hyperedge._properties
            )

    def add_cell(self, cell, rank=None, **attr):
        """Add a single cells to combinatorial complex.

        Parameters
        ----------
        cell : hashable, iterable or HyperEdge
            If hashable the cell returned will be empty.
        rank : rank of a cell

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        return super().add_cell(cell, rank, **attr)

    def add_cells_from(self, cells, ranks=None) -> None:
        """Add cells to combinatorial complex.

        Parameters
        ----------
        cells : iterable of hashables
            For hashables the cells returned will be empty.
        ranks: Iterable or int. When iterable, len(ranks) == len(cells)

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        return super().add_cells_from(cells, ranks)

    def _remove_hyperedge(self, hyperedge) -> None:
        if hyperedge not in self.cells:
            raise KeyError(f"The cell {hyperedge} is not in the complex")

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            del self._complex_set.hyperedge_dict[0][hyperedge]

        if isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.elements
        else:
            hyperedge_ = frozenset(hyperedge)
        rank = self._complex_set.get_rank(hyperedge_)
        del self._complex_set.hyperedge_dict[rank][hyperedge_]
        self._max_complex.difference_update({HyperEdge(hyperedge_, rank=rank)})

    def remove_cell(self, cell):
        """Remove a single cell from CC.

        Parameters
        ----------
        cell : hashable or RankedEntity

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex

        Notes
        -----
        Deletes reference to cell from all of its nodes.
        If any of its nodes do not belong to any other cells
        the node is dropped from self.
        """
        super().remove_cell(cell)

    def remove_cells(self, cell_set):
        """Remove cells from CC.

        Parameters
        ----------
        cell_set : iterable of hashables

        Returns
        -------
        Combinatorial Complex : NestedCombinatorialComplex
        """
        super().remove_cells(cell_set)

    def clone(self) -> "CombinatorialComplex":
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex and attributes. That is, if an
        attribute is a container, that container is shared by the original and the copy. Use Python’s `copy.deepcopy`
        for new containers.

        Returns
        -------
        CombinatorialComplex
        """
        CC = CombinatorialComplex(name=self.name, graph_based=self.graph_based)
        for cell in self.cells:
            CC.add_cell(cell, self.cells.get_rank(cell))
        return CC
