"""Creation and manipulation of a Colored Hypergraph."""

from collections.abc import Collection, Hashable, Iterable
from typing import Literal

import networkx as nx
import numpy as np
from networkx import Graph
from scipy.sparse import csr_matrix

from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.reportviews import HyperEdgeView, NodeView
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.exception import TopoNetXError
from toponetx.utils.structure import (
    compute_set_incidence,
    incidence_to_adjacency,
    sparse_array_to_neighborhood_dict,
)

__all__ = ["ColoredHyperGraph"]


class ColoredHyperGraph(Complex):
    """Class for ColoredHyperGraph Complex.

    A Colored Hypergraph (CHG) is a triple CHG = (S, X, c) where:
    - S is an abstract set of entities,
    - X a subset of the power set of X, and
    - c is the a color function that associates for every set x in X a color, a positive integer.



    A CHG is a generlization of graphs, combintorial complex, hypergraphs, cellular and simplicial complexes.

    Parameters
    ----------
    cells : Collection, optional
    name : str, optional
        An identifiable name for the Colored Hypergraph.
    ranks : Collection, optional
        Represent the color of cells.
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


    Then, (S, X, c) is a colored hypergraph.

    Examples
    --------
    Define an empty colored hypergraph:

    >>> CHG = ColoredHyperGraph()

    Add cells to the colored hypergraph:
    >>> from toponetx.classes.colored_hypergraph import ColoredHyperGraph
    >>> CHG = ColoredHyperGraph()
    >>> CHG.add_cell([1, 2], rank=1)
    >>> CHG.add_cell([3, 4], rank=1)
    >>> CHG.add_cell([1, 2, 3, 4], rank=2)
    >>> CHG.add_cell([1, 2, 4], rank=2)
    >>> CHG.add_cell([3, 4], rank=2)
    >>> CHG.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
    """

    def __init__(
        self,
        cells: Collection | None = None,
        name: str = "",
        ranks: Collection | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = name
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
                            self.add_cell(cell, rank=1)
                        else:
                            self.add_cell(cell, rank=cell.rank)
                else:
                    if isinstance(cells, Iterable) and isinstance(ranks, Iterable):
                        if len(cells) != len(ranks):
                            raise TopoNetXError(
                                "cells and ranks must have equal number of elements"
                            )
                        else:
                            for cell, color in zip(cells, ranks):
                                self.add_cell(cell, color)
                if isinstance(cells, Iterable) and isinstance(ranks, int):
                    for cell in cells:
                        self.add_cell(cell, ranks)
            else:
                for node in cells.nodes:  # cells is a networkx graph
                    self.add_node(node, **cells.nodes[node])
                for edge in cells.edges:
                    u, v = edge
                    self.add_cell([u, v], 1, **cells.get_edge_data(u, v))

    @property
    def cells(self):
        """
        Object associated with self._cells.

        Returns
        -------
        HyperEdgeView
        """
        return self._complex_set

    @property
    def nodes(self):
        """
        Object associated with self.elements.

        Returns
        -------
        NodeView

        """
        return NodeView(self._complex_set.hyperedge_dict, cell_type=HyperEdge)

    @property
    def incidence_dict(self):
        """Return dict keyed by cell uids with values the uids of nodes in each cell.

        Returns
        -------
        dict
        """
        return self._complex_set.hyperedge_dict

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape.

        This is:
        (number of cells[i], for i in range(0,dim(CHG))  )

        Returns
        -------
        tuple of ints
        """
        return self._complex_set.shape

    def skeleton(self, rank: int):
        """Return skeleton."""
        return self._complex_set.skeleton(rank)

    @property
    def ranks(self):
        """Return ranks."""
        return sorted(self._complex_set.allranks)

    @property
    def dim(self) -> int:
        """Return dimension."""
        return max(self._complex_set.allranks)

    @property
    def __shortstr__(self):
        """Return the short string generic representation."""
        return "CHG"

    def __str__(self):
        """Return detailed string representation."""
        return f"Colored Hypergraph with {len(self.nodes)} nodes and hyperedges with colors {self.ranks} and sizes {self.shape} "

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ColoredHyperGraph(name='{self.name}')"

    def __len__(self):
        """Return number of nodes."""
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self.nodes)

    def __contains__(self, item):
        """Return boolean indicating if item is in self.nodes.

        Parameters
        ----------
        item : hashable or HyperEdge
        """
        return item in self.nodes

    def get_all_incidence_structure_dict(self):
        """Get all incidence structure dictionary."""
        d = {}
        for r in range(1, self.dim + 1):
            B0r = sparse_array_to_neighborhood_dict(
                self.incidence_matrix(rank=0, to_rank=r)
            )
            d["B_0_" + str(r)] = B0r
        return d

    def __setitem__(self, cell, attr):
        """Set the attributes of a hyperedge or node in the CHG."""
        if cell in self.nodes:
            self.nodes[cell].update(attr)
            return
        # we now check if the input is a cell in the CHG
        elif cell in self.cells:
            hyperedge_ = HyperEdgeView._to_frozen_set(cell)
            rank = self.cells.get_rank(hyperedge_)
            if hyperedge_ in self._complex_set.hyperedge_dict[rank]:
                self._complex_set.hyperedge_dict[rank][hyperedge_] = attr
        else:
            raise KeyError(f"input {cell} is not in the complex")

    def __getitem__(self, node):
        """Return the attrs of a node.

        Parameters
        ----------
        node :  hashable

        Returns
        -------
        dictionary of attrs associated with node
        """
        return self.nodes[node]

    def size(self, cell):
        """Compute the number of nodes in node_set that belong to cell.

        Parameters
        ----------
        cell : hashable or HyperEdge

        Returns
        -------
        size : int
        """
        if cell not in self.cells:
            raise TopoNetXError(
                "Input cell is not in cells of the {}".format(self.__shortstr__)
            )
        return len(self._complex_set[cell])

    def number_of_nodes(self, node_set=None):
        """Compute the number of nodes in node_set belonging to the CHG.

        Parameters
        ----------
        node_set : an interable of Entities, optional, default: None
            If None, then return the number of nodes in the CHG.

        Returns
        -------
        number_of_nodes : int
        """
        if node_set:
            return len([node for node in node_set if node in self.nodes])
        return len(self.nodes)

    def number_of_cells(self, cell_set=None):
        """Compute the number of cells in cell_set belonging to the CHG.

        Parameters
        ----------
        cell_set : an interable of HyperEdge, optional
            If None, then return the number of cells.

        Returns
        -------
        number_of_cells : int
        """
        if cell_set:
            return len([cell for cell in cell_set if cell in self.cells])
        return len(self.cells)

    def order(self):
        """Compute the number of nodes in the CHG.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def degree(self, node, rank: int = 1) -> int:
        """Compute the number of cells of certain rank that contain node.

        Parameters
        ----------
        node : hashable
            Identifier for the node.
        rank : int, optional, default: 1
            Smallest size of cell to consider in degree

        Returns
        -------
        int
            Number of cells of certain rank that contain node.
        """
        if node not in self.nodes:
            raise KeyError(f"Node {node} not in {self.__shortstr__}.")
        if isinstance(rank, int):
            if rank >= 0:
                return sum(
                    [
                        1 if node in x else 0
                        for x in self._complex_set.hyperedge_dict[rank].keys()
                    ]
                )
            else:
                raise TopoNetXError("Rank must be positive")
        elif rank is None:
            rank_list = self._complex_set.hyperedge_dict.keys()
            value = 0
            for rank in rank_list:
                value += sum(
                    [
                        1 if node in x else 0
                        for x in self._complex_set.hyperedge_dict[rank].keys()
                    ]
                )
            return value

    def _remove_node(self, node) -> None:
        if isinstance(node, HyperEdge):
            pass
        elif isinstance(node, Hashable):
            node = HyperEdge([node])
        else:
            raise TypeError("node must be a HyperEdge or a hashable object")
        if node not in self.nodes:
            raise KeyError(f"node {node} not in {self.__shortstr__}")
        self._remove_node_helper(node)

    def remove_node(self, node):
        """Remove node from cells.

        This also deletes any reference in the nodes of the CHG.

        Parameters
        ----------
        node : hashable or HyperEdge

        Returns
        -------
        Colored Hypergraph : ColoredHyperGraph
        """
        self._remove_node(node)
        return self

    def remove_nodes(self, node_set) -> None:
        """Remove nodes from cells.

        This also deletes references in colored hypergraph nodes.

        Parameters
        ----------
        node_set : an iterable of hashables
            Nodes in CHG
        """
        copy_set = set()
        for node in node_set:
            if isinstance(node, Hashable):
                if isinstance(node, HyperEdge):
                    copy_set.add(list(node.elements)[0])
                else:
                    copy_set.add(node)
                if node not in self.nodes:
                    raise KeyError(f"node {node} not in {self.__shortstr__}")
            else:
                raise TypeError("node {node} must be a HyperEdge or a hashable object")
        self._remove_node_helper(copy_set)
        return self

    def _remove_node_helper(self, node) -> None:
        """Remove node from cells. Assumes node is present in the CHG."""
        # Removing node in hyperedgeview
        for key in list(self.cells.hyperedge_dict.keys()):
            for key_rank in list(self.cells.hyperedge_dict[key].keys()):
                replace_key = key_rank.difference(node)
                if len(replace_key) > 0:
                    if key_rank != replace_key:
                        del self.cells.hyperedge_dict[key][key_rank]
                else:
                    # Remove original hyperedge from the ranks
                    del self.cells.hyperedge_dict[key][key_rank]
            if self.cells.hyperedge_dict[key] == {}:
                del self.cells.hyperedge_dict[key]

    def _add_hyperedge(self, hyperedge, rank, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge : HyperEdge, Hashable or Iterable
            a cell in a colored hypergraph
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
        is a valid hyperedge of the colored hypergraph by checking whether the hyperedge's nodes
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
            if isinstance(hyperedge, HyperEdge):
                hyperedge_set = hyperedge.elements
            else:
                if not all(isinstance(i, Hashable) for i in hyperedge):
                    raise ValueError(
                        f"Input hyperedge {hyperedge} contain non-hashable elements."
                    )
                hyperedge_set = frozenset(hyperedge)

            if rank == 0 and len(hyperedge_set) > 1:
                raise ValueError(
                    "rank must be positive for hyperedges containing more than 1 element, got rank = 0"
                )
        else:
            raise ValueError("Invalid hyperedge type")
        self._add_hyperedge_helper(hyperedge_set, rank, **attr)
        if "weight" not in self._complex_set.hyperedge_dict[rank][hyperedge_set]:
            self._complex_set.hyperedge_dict[rank][hyperedge_set]["weight"] = 1
        if isinstance(hyperedge, HyperEdge):
            self._complex_set.hyperedge_dict[rank][hyperedge_set].update(
                hyperedge._properties
            )

    def _remove_hyperedge(self, hyperedge):
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

    def _add_node(self, node, **attr):
        """
        Add one node as a hyperedge.

        Parameters
        ----------
        node : hashable
            The node to add as a hyperedge.
        **attr : dict
            Additional attributes to assign to the hyperedge.

        Returns
        -------
        None
        """
        self._add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_node(self, node, **attr):
        """
        Add a node.

        Parameters
        ----------
        node : hashable
            The node to add.
        **attr : dict
            Additional attributes to assign to the node.

        Returns
        -------
        None
        """
        self._add_node(node, **attr)

    def set_node_attributes(self, values, name=None):
        """
        Set node attributes.

        Parameters
        ----------
        values : dict
            A dictionary where keys are nodes and values are the attributes to set.
        name : str or None, optional
            The name of the attribute to set for all nodes. If None, attributes will be set individually for each node.

        Returns
        -------
        None
        """
        if name is not None:
            for cell, value in values.items():
                try:
                    self.nodes[cell][name] = value
                except AttributeError:
                    pass

        else:
            for cell, d in values.items():
                try:
                    self.nodes[cell].update(d)
                except AttributeError:
                    pass
            return

    def set_cell_attributes(self, values, name=None):
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
        After computing some property of the cell of a Colored Hypergraph, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CHG = ColoredHyperGraph()
        >>> CHG.add_cell([1, 2, 3, 4], rank=2)
        >>> CHG.add_cell([1, 2, 4], rank=2,)
        >>> CHG.add_cell([3, 4], rank=2)
        >>> d = {(1, 2, 3, 4): 'red', (1, 2, 3): 'blue', (3, 4): 'green'}
        >>> CHG.set_cell_attributes(d, name='color')
        >>> CHG.cells[(3, 4)]['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CHG = ColoredHyperGraph(G)
        >>> d = {(1, 2): {'color': 'red','attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3}}
        >>> CHG.set_cell_attributes(d)
        >>> CHG.cells[(0, 1)]['color']
        'blue'
        3

        Note that if the dict contains cells that are not in `self.cells`, they are
        silently ignored.
        """
        if name is not None:
            for cell, value in values.items():
                try:
                    self.cells[cell][name] = value
                except AttributeError:
                    pass
        else:
            for cell, d in values.items():
                try:
                    self.cells[cell].update(d)
                except AttributeError:
                    pass
            return

    def get_node_attributes(self, name):
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
        >>> CHG = NestedColoredHyperGraph(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 },1: {'color': 'blue', 'attr2': 3} }
        >>> CHG.set_node_attributes(d)
        >>> CHG.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CHG = NestedColoredHyperGraph(G)
        >>> nodes_color = CHG.get_node_attributes('color')
        >>> nodes_color[1]
        'blue'
        """
        return {
            node: self.nodes.nodes[node][name]
            for node in self.nodes.nodes
            if name in self.nodes.nodes[node]
        }

    def get_cell_attributes(self, name, rank=None):
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
        >>> CHG = ColoredHyperGraph(G)
        >>> d = {(1, 2): {'color': 'red', 'attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3} }
        >>> CHG.set_cell_attributes(d)
        >>> cell_color = CHG.get_cell_attributes('color')
        >>> cell_color[frozenset({0, 1})]
        'blue'
        """
        if rank is not None:
            return {
                cell: self.skeleton(rank)[cell][name]
                for cell in self.skeleton(rank)
                if name in self.skeleton(rank)[cell]
            }
        else:
            return {
                cell: self.cells[cell].properties[name]
                for cell in self.cells
                if name in self.cells[cell].properties
            }

    def add_hyperedge_with_its_nodes(self, hyperedge_, rank, **attr):
        """Adding nodes of hyperedge helper method."""
        if rank == 0:
            if 0 not in self._complex_set.hyperedge_dict:
                self._complex_set.hyperedge_dict[0] = {}
                self._complex_set.hyperedge_dict[0][hyperedge_] = {"weight": 1}
            self._complex_set.hyperedge_dict[0][hyperedge_].update(**attr)

        else:
            self._complex_set.hyperedge_dict[rank][hyperedge_].update(**attr)
            for i in hyperedge_:
                if 0 not in self._complex_set.hyperedge_dict:
                    self._complex_set.hyperedge_dict[0] = {}

                if i not in self._complex_set.hyperedge_dict[0]:
                    self._complex_set.hyperedge_dict[0][frozenset({i})] = {"weight": 1}

    def _add_hyperedge_helper(self, hyperedge_, rank, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge_ : frozenset of hashable elements
        rank : int
        attr : arbitrary attrs

        Returns
        -------
        None.
        """
        if rank in self._complex_set.hyperedge_dict:
            if hyperedge_ in self._complex_set.hyperedge_dict[rank]:
                self.add_hyperedge_with_its_nodes(hyperedge_, rank, **attr)
            else:
                self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
                self.add_hyperedge_with_its_nodes(hyperedge_, rank, **attr)
        else:
            self._complex_set.hyperedge_dict[rank] = {}
            self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
            self.add_hyperedge_with_its_nodes(hyperedge_, rank, **attr)

    def add_cell(self, cell, rank=None, **attr):
        """Add a single cells to Colored Hypergraph.

        Parameters
        ----------
        cell : hashable, iterable or HyperEdge
            If hashable the cell returned will be empty.
        rank : rank of a cell

        Returns
        -------
        Colored Hypergraph : ColoredHyperGraph
        """
        if rank is None:
            self._add_hyperedge(cell, rank=1, **attr)
        else:
            self._add_hyperedge(cell, rank=rank, **attr)

    def add_cells_from(self, cells, ranks=None):
        """Add cells to Colored Hypergraph.

        Parameters
        ----------
        cells : iterable of hashables
            For hashables the cells returned will be empty.
        ranks: Iterable or int. When iterable, len(ranks) == len(cells)

        Returns
        -------
        Colored Hypergraph : ColoredHyperGraph
        """
        if ranks is None:
            for cell in cells:
                if not isinstance(cell, HyperEdge):
                    raise ValueError(
                        f"input must be an HyperEdge {cell} object when rank is None"
                    )
                if cell.rank is None:
                    raise ValueError(f"input HyperEdge {cell} has None rank")
                self.add_cell(cell, cell.rank)
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

    def remove_cell(self, cell):
        """Remove a single cell from CHG.

        Parameters
        ----------
        cell : hashable or RankedEntity

        Returns
        -------
        Colored Hypergraph : ColoredHyperGraph

        Notes
        -----
        Deletes reference to cell from all of its nodes.
        If any of its nodes do not belong to any other cells
        the node is dropped from self.
        """
        self._remove_hyperedge(cell)

    def get_incidence_structure_dict(self, i, j):
        """Get incidence structure dictionary."""
        return sparse_array_to_neighborhood_dict(self.incidence_matrix(i, j))

    def get_adjacency_structure_dict(self, i, j):
        """Get adjacency structure dictionary."""
        return sparse_array_to_neighborhood_dict(self.adjacency_matrix(i, j))

    def remove_cells(self, cell_set):
        """Remove cells from CHG.

        Parameters
        ----------
        cell_set : iterable of hashables

        Returns
        -------
        Colored Hypergraph : NestedColoredHyperGraph
        """
        for cell in cell_set:
            self.remove_cell(cell)

    def _incidence_matrix(
        self,
        rank,
        to_rank,
        weight=None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix.

        An incidence matrix indexed by r-ranked hyperedges k-ranked hyperedges
        r !=k, when k is None incidence_type will be considered instead

        Parameters
        ----------
        sparse : bool, default=True
        index : bool, default=False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray
        row dictionary : dict
            Dictionary identifying row with item in entityset's children
        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----
        Incidence_matrix method  is a method for generating the incidence matrix of a ColoredHyperGraph.
        An incidence matrix is a matrix that describes the relationships between the hyperedges
        of a complex. In this case, the incidence_matrix method generates a matrix where
        the rows correspond to the hyperedges of the complex and the columns correspond to the faces
        . The entries in the matrix are either 0 or 1,
        depending on whether a hyperedge contains a given face or not.
        For example, if hyperedge i contains face j, then the entry in the ith
        row and jth column of the matrix will be 1, otherwise it will be 0.

        To generate the incidence matrix, the incidence_matrix method first creates
        a dictionary where the keys are the faces of the complex and the values are
        the hyperedges that contain that face. This allows the method to quickly look up
        which hyperedges contain a given face. The method then iterates over the hyperedges in
        the HyperEdgeView instance, and for each hyperedge, it checks which faces it contains.
        For each face that the hyperedge contains, the method increments the corresponding entry
        in the matrix. Finally, the method returns the completed incidence matrix.
        """
        if rank == to_rank:
            raise ValueError("incidence must be computed for k!=r, got equal r and k.")
        children = self.skeleton(rank)
        uidset = self.skeleton(to_rank)
        return compute_set_incidence(children, uidset, sparse, index)

    def incidence_matrix(
        self,
        rank,
        to_rank,
        weight=None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CHG indexed by nodes x cells.

        Parameters
        ----------
        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.
        index : boolean, optional, default False
            If True return will include a dictionary of node uid : row number
            and cell uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray
        row dictionary : dict
            Dictionary identifying rows with nodes
        column dictionary : dict
            Dictionary identifying columns with cells
        """
        return self._incidence_matrix(rank, to_rank, sparse=sparse, index=index)

    def adjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        r,k : int, int
            Two ranks for skeletons in the input Colored Hypergraph
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.
        index: boolean, optional, default: False
            If True, will return a rowdict of row to node uid
        index : book, default=False
            indicate weather to return the indices of the adjacency matrix.

        Returns
        -------
        If index is True
            adjacency_matrix : scipy.sparse.csr.csr_matrix
            row dictionary : dict

        If index if False
            adjacency_matrix : scipy.sparse.csr.csr_matrix

        Examples
        --------
        >>> G = Graph() # networkx graph
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0,3)
        >>> G.add_edge(0,4)
        >>> G.add_edge(1, 4)
        >>> CHG = ColoredHyperGraph(cells=G)
        >>> CHG.adjacency_matrix(0, 1)
        """
        if index:
            B, row, col = self.incidence_matrix(
                rank, via_rank, sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(rank, via_rank, sparse=True, index=index)
        A = incidence_to_adjacency(B.T, s=s)
        if index:
            return A, row
        return A

    def cell_adjacency_matrix(self, index=False, s=1):
        """Compute the cell adjacency matrix.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
          all cells adjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        raise NotImplementedError()

    def node_adjacency_matrix(self, index=False, s=1):
        """Compute the node adjacency matrix."""
        raise NotImplementedError()

    def coadjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Compute the coadjacency matrix.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        r,k : two ranks for skeletons in the input Colored Hypergraph

        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        index: boolean, optional, default: False
            if True, will return a rowdict of row to node uid

        weight: bool, default=True
            If False all nonzero entries are 1.
            If True adjacency matrix will depend on weighted incidence matrix,
        index : book, default=False
            indicate weather to return the indices of the adjacency matrix.

        Returns
        -------
        If index is True
            coadjacency_matrix : scipy.sparse.csr.csr_matrix

            row dictionary : dict

        If index if False

            coadjacency_matrix : scipy.sparse.csr.csr_matrix
        """
        if index:
            B, row, col = self.incidence_matrix(
                via_rank, rank, sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(via_rank, rank, sparse=True, index=index)
        A = incidence_to_adjacency(B)
        if index:
            return A, col
        return A

    @classmethod
    def from_trimesh(cls, mesh) -> "ColoredHyperGraph":
        """Import from trimesh.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        >>> CHG = ColoredHyperGraph.from_trimesh(mesh)
        >>> CHG.nodes
        """
        raise NotImplementedError()

    def restrict_to_cells(self, cell_set, name=None):
        """Construct a Colored Hypergraph using a subset of the cells.

        Parameters
        ----------
        cell_set: iterable of hashables or RankedEntities
            A subset of elements of the Colored Hypergraph  cells
        name: str, optional

        Returns
        -------
        new Colored Hypergraph : NestedColoredHyperGraph
        """
        raise NotImplementedError()

    def restrict_to_nodes(self, node_set, name=None):
        """Restrict to a set of nodes.

        Constructs a new Colored Hypergraph  by restricting the
        cells in the Colored Hypergraph to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set: iterable of hashables
            References a subset of elements of self.nodes

        name: str, optional

        Returns
        -------
        new Colored Hypergraph : NestedColoredHyperGraph

        """
        raise NotImplementedError()

    def from_networkx_graph(self, G):
        """Construct a Colored Hypergraph from a networkx graph.

        Parameters
        ----------
        G : NetworkX graph
            A networkx graph

        Returns
        -------
        CHG such that the edges of the graph are ranked 1
        and the nodes are ranked 0.

        Examples
        --------
        >>> from networkx import Graph
        >>> G = Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0,4)
        >>> G.add_edge(0,7)
        >>> CX = ColoredHyperGraph()
        >>> CX.from_networkx_graph(G)
        >>> CX.nodes
        RankedEntitySet(:Nodes,[0, 1, 4, 7],{'weight': 1.0})
        >>> CX.cells
        RankedEntitySet(:Cells,[(0, 1), (0, 7), (0, 4)],{'weight': 1.0})
        """
        for node in G.nodes:
            self.add_node(node)
        for edge in G.edges:
            self.add_cell(edge, rank=1)

    def is_connected(self, s=1, cells=False):
        """Determine if Colored Hypergraph is :term:`s-connected <s-connected, s-node-connected>`.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        cells: boolean, optional, default: False
            If True, will determine if s-cell-connected.
            For s=1 s-cell-connected is the same as s-connected.

        Returns
        -------
        is_connected : boolean

        Notes
        -----
        A CHG is s node connected if for any two nodes v0,vn
        there exists a sequence of nodes v0,v1,v2,...,v(n-1),vn
        such that every consecutive pair of nodes v(i),v(i+1)
        share at least s cell.

        Examples
        --------
        >>> CHG = ColoredHyperGraph(cells=E)
        >>> CHG.is_connected()
        """
        raise NotImplementedError()

    def singletons(self):
        """Return a list of singleton cell.

        A singleton cell is a cell of
        size 1 with a node of degree 1.

        Returns
        -------
        singles : list
            A list of cells uids.
        """
        singletons = []
        for cell in self.cells:
            zero_elements = self.cells[cell].skeleton(0)
            if len(zero_elements) == 1:
                for n in zero_elements:
                    if self.degree(n) == 1:
                        singletons.append(cell)
        return singletons

    def remove_singletons(self, name=None):
        """Construct new CHG with singleton cells removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        new CHG : CHG
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)

    def s_connected_components(
        self, s: int = 1, cells: bool = True, return_singletons: bool = False
    ):
        """Return a generator for s-cell-connected components.

        or the :term:`s-node-connected components <s-connected component, s-node-connected component>`.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.
        cells : boolean, optional, default: True
            If True will return cell components, if False will return node components
        return_singletons : bool, optional, default : False

        Notes
        -----
        If cells=True, this method returns the s-cell-connected components as
        lists of lists of cell uids.
        An s-cell-component has the property that for any two cells e1 and e2
        there is a sequence of cells starting with e1 and ending with e2
        such that pairwise adjacent cells in the sequence intersect in at least
        s nodes. If s=1 these are the path components of the CHG.

        If cells=False this method returns s-node-connected components.
        A list of sets of uids of the nodes which are s-walk connected.
        Two nodes v1 and v2 are s-walk-connected if there is a
        sequence of nodes starting with v1 and ending with v2 such that pairwise
        adjacent nodes in the sequence share s cells. If s=1 these are the
        path components of the ColoredHyperGraph .

        Yields
        ------
        s_connected_components : iterator
            Iterator returns sets of uids of the cells (or nodes) in the s-cells(node)
            components of CHG.
        """
        raise NotImplementedError()

    def s_component_subgraphs(
        self, s: int = 1, cells: bool = True, return_singletons: bool = False
    ):
        """Return a generator for the induced subgraphs of s_connected components.

        Removes singletons unless return_singletons is set to True.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.
        cells : boolean, optional, cells=False
            Determines if cell or node components are desired. Returns
            subgraphs equal to the CHG restricted to each set of nodes(cells) in the
            s-connected components or s-cell-connected components
        return_singletons : bool, optional

        Yields
        ------
        s_component_subgraphs : iterator
            Iterator returns subgraphs generated by the cells (or nodes) in the
            s-cell(node) components.
        """
        raise NotImplementedError()

    def s_components(
        self, s: int = 1, cells: bool = True, return_singletons: bool = True
    ):
        """Compute s-connected components.

        Same as s_connected_components.

        See Also
        --------
        s_connected_components
        """
        raise NotImplementedError()

    def connected_components(self, cells: bool = False, return_singletons: bool = True):
        """Compute s-connected components.

        Same as :meth:`s_connected_components`,
        with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        raise NotImplementedError()

    def connected_component_subgraphs(self, return_singletons: bool = True):
        """Compute s-component subgraphs with s=1.

        Same as :meth:`s_component_subgraphs` with s=1.

        Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        raise NotImplementedError()

    def components(self, cells: bool = False, return_singletons: bool = True):
        """Compute s-connected components for s=1.

        Same as :meth:`s_connected_components` with s=1.

        But nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        raise NotImplementedError()

    def component_subgraphs(self, return_singletons: bool = False):
        """Compute s-component subgraphs wth s=1.

        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        raise NotImplementedError()

    def node_diameters(self, s: int = 1):
        """Return node diameters of the connected components.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        list of the diameters of the s-components and
        list of the s-component nodes
        """
        raise NotImplementedError()

    def cell_diameters(self, s: int = 1):
        """Return the cell diameters of the s_cell_connected component subgraphs.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        maximum diameter : int
        list of diameters : list
            List of cell_diameters for s-cell component subgraphs in CHG
        list of component : list
            List of the cell uids in the s-cell component subgraphs.
        """
        raise NotImplementedError()

    def diameter(self, s: int = 1) -> int:
        """Return the length of the longest shortest s-walk between nodes.

        Parameters
        ----------
        s : int, default=1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        int

        Raises
        ------
        TopoNetXError
            If CHG is not s-cell-connected

        Notes
        -----
        Two nodes are s-adjacent if they share s cells.
        Two nodes v_start and v_end are s-walk connected if there is a sequence of
        nodes v_start, v_1, v_2, ... v_n-1, v_end such that consecutive nodes
        are s-adjacent. If the graph is not connected, an error will be raised.
        """
        raise NotImplementedError()

    def cell_diameter(self, s: int = 1) -> int:
        """Return length of the longest shortest s-walk between cells.

        Parameters
        ----------
        s : int, default=1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
        int

        Raises
        ------
        TopoNetXError
            If ColoredHyperGraph is not s-cell-connected

        Notes
        -----
        Two cells are s-adjacent if they share s nodes.
        Two nodes e_start and e_end are s-walk connected if there is a sequence of
        cells e_start, e_1, e_2, ... e_n-1, e_end such that consecutive cells
        are s-adjacent. If the graph is not connected, an error will be raised.
        """
        raise NotImplementedError()

    def distance(self, source, target, s: int = 1) -> int:
        """Return shortest s-walk distance between two nodes.

        Parameters
        ----------
        source : node.uid or node
            a node in the CHG
        target : node.uid or node
            a node in the CHG
        s : int
            the number of cells

        Returns
        -------
        int

        See Also
        --------
        cell_distance

        Notes
        -----
        The s-distance is the shortest s-walk length between the nodes.
        An s-walk between nodes is a sequence of nodes that pairwise share
        at least s cells. The length of the shortest s-walk is 1 less than
        the number of nodes in the path sequence.

        Uses the networkx shortest_path_length method on the graph
        generated by the s-adjacency matrix.

        """
        raise NotImplementedError()

    def cell_distance(self, source, target, s: int = 1):
        """Return the shortest s-walk distance between two cells.

        Parameters
        ----------
        source : cell.uid or cell
            an cell in the colored hypergraph
        target : cell.uid or cell
            an cell in the colored hypergraph
        s : int, default=1
            the number of intersections between pairwise consecutive cells

        Returns
        -------
        int
            The shortest s-walk cell distance between `source` and `target`.
            A shortest s-walk is computed as a sequence of cells,
            the s-walk distance is the number of cells in the sequence
            minus 1. If no such path exists returns np.inf.

        See Also
        --------
        distance

        Notes
        -----
        The s-distance is the shortest s-walk length between the cells.
        An s-walk between cells is a sequence of cells such that consecutive pairwise
        cells intersect in at least s nodes. The length of the shortest s-walk is 1 less than
        the number of cells in the path sequence.

        Uses the networkx shortest_path_length method on the graph
        generated by the s-cell_adjacency matrix.
        """
        raise NotImplementedError()

    def clone(self) -> "ColoredHyperGraph":
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex and attributes. That is, if an
        attribute is a container, that container is shared by the original and the copy. Use Python’s `copy.deepcopy`
        for new containers.

        Returns
        -------
        ColoredHyperGraph
        """
        CHG = ColoredHyperGraph(name=self.name)
        for cell in self.cells:
            CHG.add_cell(cell, self.cells.get_rank(cell))
        return CHG
