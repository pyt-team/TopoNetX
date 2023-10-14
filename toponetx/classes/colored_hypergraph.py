"""Creation and manipulation of a Colored Hypergraph."""

from collections.abc import Collection, Hashable, Iterable
from typing import Literal, Optional

import networkx as nx
import numpy as np
import scipy.sparse
from networkx import Graph
from scipy.sparse import csr_array, csr_matrix, diags

import toponetx as tnx
from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.reportviews import ColoredHyperEdgeView, NodeView
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.utils.structure import (
    compute_set_incidence,
    incidence_to_adjacency,
    sparse_array_to_neighborhood_dict,
)

__all__ = ["ColoredHyperGraph"]


class ColoredHyperGraph(Complex):
    """Class for ColoredHyperGraph Complex.

    A Colored Hypergraph (CHG) is a triplet CHG = (S, X, c) where:
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
    >>> CHG.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
    """

    def __init__(
        self,
        cells: Collection | None = None,
        name: str = "",
        ranks: Collection | int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = name
        self._complex_set = ColoredHyperEdgeView()

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
                            raise ValueError(
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
        return NodeView(
            self._complex_set.hyperedge_dict, cell_type=HyperEdge, colored_nodes=True
        )

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
    def __shortstr__(self) -> str:
        """Return the short string generic representation."""
        return "CHG"

    def __str__(self) -> str:
        """Return detailed string representation."""
        return f"Colored Hypergraph with {len(self.nodes)} nodes and hyperedges with colors {self.ranks[1:]} and sizes {self.shape[1:]} "

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ColoredHyperGraph(name='{self.name}')"

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self.nodes)

    def __contains__(self, item) -> bool:
        """Return true/false indicating if item is in self.nodes or self.edges.

        Parameters
        ----------
        item : hashable or HyperEdge
        """
        return item in self.nodes

    def __setitem__(self, cell, **attr):
        """Set the attributes of a hyperedge or node in the CHG."""
        if cell in self.nodes:
            self.nodes[cell].update(**attr)
        # we now check if the input is a cell in the CHG
        elif cell in self.cells:
            hyperedge_ = ColoredHyperEdgeView._to_frozen_set(cell)
            rank = self.cells.get_rank(hyperedge_)
            if hyperedge_ in self._complex_set.hyperedge_dict[rank]:
                self._complex_set.hyperedge_dict[rank][hyperedge_].update(**attr)

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
        if node not in self.nodes:
            raise KeyError(f"Node {node} is not in the complex.")
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
            raise ValueError(
                "Input cell is not in cells of the {}".format(self.__shortstr__)
            )
        return len(self._complex_set[cell])

    def number_of_nodes(self, node_set=None):
        """Compute the number of nodes in node_set belonging to the CHG.

        Parameters
        ----------
        node_set : an interable of Entities, optional
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

    def degree(self, node, rank: int = 1, s: int = 0) -> int:
        """Compute the number of cells of certain rank (or all ranks) that contain node.

        Parameters
        ----------
        node : hashable
            Identifier for the node.
        rank : int, optional
               The rank at which the degree of the node is computed.
               When None, degree of the input node is computed with respect to cells of all ranks.
        s : int, optional
            Smallest size of cell to consider in degree


        Returns
        -------
        int
            Number of cells of certain rank (or all ranks) that contain node.
        """
        if node not in self.nodes:
            raise KeyError(f"Node {node} not in {self.__shortstr__}.")
        if isinstance(rank, int):
            if rank >= 0:
                if rank in self._complex_set.hyperedge_dict.keys():
                    return sum(
                        [
                            len(self._complex_set.hyperedge_dict[rank][x])
                            if node in x
                            else 0
                            for x in self._complex_set.hyperedge_dict[rank].keys()
                            if len(x) >= s
                        ]
                    )
                else:
                    raise RuntimeError(
                        f"There are no cells in the colored hypergraph with rank {rank}"
                    )

            else:
                raise ValueError("Rank must be positive")
        elif rank is None:
            rank_list = self._complex_set.hyperedge_dict.keys()
            value = 0
            for rank in rank_list:
                if rank == 0:
                    continue
                else:
                    value += sum(
                        [
                            len(self._complex_set.hyperedge_dict[rank][x])
                            if node in x
                            else 0
                            for x in self._complex_set.hyperedge_dict[rank].keys()
                            if len(x) >= s
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

    def _add_hyperedge(self, hyperedge, rank, key=None, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge : HyperEdge, Hashable or Iterable
            a cell in a colored hypergraph
        rank : int
            the rank of a hyperedge, must be zero when the hyperedge is Hashable.
        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish coloredhyperedges among nodes.
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
        self._add_hyperedge_helper(hyperedge_set, rank, key, **attr)
        if rank == 0 and hyperedge_set in self._complex_set.hyperedge_dict[0]:
            self._complex_set.hyperedge_dict[0][hyperedge_set][0].update(**attr)

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

    def _add_node(self, node, **attr) -> None:
        """Add one node as a hyperedge.

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
        if node in self.nodes:
            self._complex_set.hyperedge_dict[0][frozenset({node})][0].update(**attr)
        else:
            self._add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_node(self, node, **attr) -> None:
        """Add a node.

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

    def set_node_attributes(self, values, name: Optional[str] = None) -> None:
        """Set node attributes.

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

    def set_cell_attributes(self, values, name: Optional[str] = None) -> None:
        """Set cell attributes.

        Parameters
        ----------
        values : dict
            dictionary of attributes to set for the cell.
        name : str, optional
            name of the attribute to set for the cell.

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
        >>> d = {((1, 2, 3, 4),0): 'red', ((1, 2, 4),0): 'blue', ((3, 4),0): 'green'}
        >>> CHG.set_cell_attributes(d, name='color')
        >>> CHG.cells[((3, 4),0)]['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CHG = ColoredHyperGraph(G)
        >>> d = {((1, 2),0): {'color': 'red','attr2': 1}, ((0, 1),0): {'color': 'blue', 'attr2': 3}}
        >>> CHG.set_cell_attributes(d)
        >>> CHG.cells[((0, 1),0)]['color']
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
        >>> CHG = ColoredHyperGraph(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 },1: {'color': 'blue', 'attr2': 3} }
        >>> CHG.set_node_attributes(d)
        >>> CHG.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CHG = ColoredHyperGraph(G)
        >>> nodes_color = CHG.get_node_attributes('color')
        >>> nodes_color[1]
        'blue'
        """
        return {
            tuple(node)[0]: self.nodes[node][name]
            for node in self.nodes
            if name in self.nodes[node]
        }

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
        >>> CHG = ColoredHyperGraph(G)
        >>> d = {((1, 2),0): {'color': 'red', 'attr2': 1}, ((0, 1),0): {'color': 'blue', 'attr2': 3} }
        >>> CHG.set_cell_attributes(d)
        >>> cell_color = CHG.get_cell_attributes('color')
        >>> cell_color[(frozenset({0, 1}), 0)]
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
                cell: self.cells[cell][name]
                for cell in self.cells
                if name in self.cells[cell]
            }

    def _add_nodes_of_hyperedge(self, hyperedge_) -> None:
        """Adding nodes of a hyperedge."""
        if 0 not in self._complex_set.hyperedge_dict:
            self._complex_set.hyperedge_dict[0] = {}
        for i in hyperedge_:
            if i not in self._complex_set.hyperedge_dict[0]:
                self._complex_set.hyperedge_dict[0][frozenset({i})] = {}
                self._complex_set.hyperedge_dict[0][frozenset({i})][0] = {"weight": 1}

    def new_hyperedge_key(self, hyperedge, rank):
        """Add hyperedge new key.

        Notes
        -----
        In the standard ColoredHyperGraph class the new key is the number of existing
        hyperedges between the nodes that define that hyperedge
        (increased if necessary to ensure unused).
        The first hyperedge will have key 0, then 1, etc. If an hyperedge is removed
        further new_hyperedge_keys may not be in this order.

        Parameters
        ----------
        hyperedge :representing node elements of the hyperedge
        rank : rank (color) of the input hyperedge

        Returns
        -------
        key : int
        """
        try:
            keydict = self._complex_set.hyperedge_dict[rank][hyperedge]
        except KeyError:
            return 0
        key = len(keydict)
        while key in keydict:
            key += 1
        return key

    def _add_hyperedge_helper(self, hyperedge_, rank, key=None, **attr):
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
        if rank not in self._complex_set.hyperedge_dict:
            self._complex_set.hyperedge_dict[rank] = {}
        if key is None:
            key = self.new_hyperedge_key(hyperedge_, rank)
        if hyperedge_ not in self._complex_set.hyperedge_dict[rank]:
            self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
            self._complex_set.hyperedge_dict[rank][hyperedge_][key] = {"weight": 1}
            self._complex_set.hyperedge_dict[rank][hyperedge_][key].update(**attr)
        elif key is not self._complex_set.hyperedge_dict[rank][hyperedge_]:
            self._complex_set.hyperedge_dict[rank][hyperedge_][key] = {"weight": 1}
            self._complex_set.hyperedge_dict[rank][hyperedge_][key].update(**attr)
        else:
            self._complex_set.hyperedge_dict[rank][hyperedge_][key].update(**attr)
        self._add_nodes_of_hyperedge(hyperedge_)

    def add_cell(self, cell, rank=None, key=None, **attr):
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
            self._add_hyperedge(cell, 1, key, **attr)
        else:
            self._add_hyperedge(cell, rank, key, **attr)

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
                if isinstance(cell, HyperEdge):
                    self.add_cell(cell, cell.rank)
                else:
                    self.add_cell(cell, rank=None)
        else:
            if isinstance(cells, Iterable) and isinstance(ranks, Iterable):
                if len(cells) != len(ranks):
                    raise ValueError(
                        "cells and ranks must have equal number of elements"
                    )
                else:
                    for cell, rank in zip(cells, ranks):
                        self.add_cell(cell, rank)
        if isinstance(cells, Iterable) and isinstance(ranks, int):
            for cell in cells:
                self.add_cell(cell, ranks)

    def remove_cell(self, cell) -> None:
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

    def remove_cells(self, cell_set) -> None:
        """Remove cells from CHG.

        Parameters
        ----------
        cell_set : iterable of hashables

        Returns
        -------
        Colored Hypergraph : ColoredHyperGraph
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
        rank : int
        to_rank: int
        sparse : bool, default=True
        index : bool, default=False
            If True return will include a dictionary of children uid : row number
            and element uid : column number
        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

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
        weight: str | None = None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CHG indexed by nodes x cells.

        Parameters
        ----------
        rank : int
        to_rank: int
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default False
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

    def node_to_all_cell_incidence_matrix(
        self, weight: str | None = None, index: bool = False
    ) -> scipy.sparse.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]:
        """Nodes/all cells incidence matrix for the indexed by nodes X cells.

        Parameters
        ----------
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            If True return will include a dictionary of node uid : row number
            and cell uid : column number

        Returns
        -------
        scipy.sparse.csr.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]
            The incidence matrix, if `index` is False, otherwise
            lower (row) index dict, upper (col) index dict, incidence matrix
            where the index dictionaries map from the entity (as `Hashable` or `tuple`) to the row or col index of the matrix
        """
        return self.all_ranks_incidence_matrix(0, weight=weight, index=index)

    def all_ranks_incidence_matrix(
        self,
        rank,
        weight: str | None = None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CHG indexed by cells of rank n X all other cells.

        Parameters
        ----------
        weight : str, optional
            If not given, all nonzero entries are 1.
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
        children = self.skeleton(rank)
        all_other_ranks = []
        for i in self.ranks:
            if i is not rank:
                all_other_ranks = all_other_ranks + self.skeleton(i)
        return compute_set_incidence(children, all_other_ranks, sparse, index)

    def adjacency_matrix(self, rank, via_rank, s: int = 1, index: bool = False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        rank, via_rank : int, int
            Two ranks for skeletons in the input Colored Hypergraph
        s : int, list, optional
            Minimum number of edges shared by neighbors with node.
        index: bool, optional
            If True, will return a rowdict of row to node uid
        index : bool, optional
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
            row, col, B = self.incidence_matrix(
                rank, via_rank, sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(rank, via_rank, sparse=True, index=index)
        A = incidence_to_adjacency(B.T)
        if index:
            return row, A

        return A

    def all_cell_to_node_coadjacnecy_matrix(self, index: bool = False, s: int = None):
        """Compute the cell adjacency matrix.

        Parameters
        ----------
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
          all cells coadjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        B = self.node_to_all_cell_incidence_matrix(index=index)
        if index:
            A = incidence_to_adjacency(B[-1], s=s)
            return B[1], A
        A = incidence_to_adjacency(B, s=s)
        return A

    def node_to_all_cell_adjacnecy_matrix(self, index: bool = False, s: int = None):
        """Compute the node/all cell adjacency matrix."""
        B = self.node_to_all_cell_incidence_matrix(index=index)
        if index:
            A = incidence_to_adjacency(B[-1].T, s=s)
            return B[0], A
        A = incidence_to_adjacency(B.T, s=s)
        return A

    def coadjacency_matrix(self, rank, via_rank, s: int = 1, index: bool = False):
        """Compute the coadjacency matrix.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        rank, via_rank : two ranks for skeletons in the input Colored Hypergraph

        s : int, list, optional
            Minimum number of edges shared by neighbors with node.

        index: bool, optional
            if True, will return a rowdict of row to node uid

        weight: bool, default=True
            If False all nonzero entries are 1.
            If True adjacency matrix will depend on weighted incidence matrix,

        Returns
        -------
        If index is True
            coadjacency_matrix : scipy.sparse.csr.csr_matrix

            row dictionary : dict

        If index if False

            coadjacency_matrix : scipy.sparse.csr.csr_matrix
        """
        if index:
            row, col, B = self.incidence_matrix(
                via_rank, rank, sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(via_rank, rank, sparse=True, index=index)
        A = incidence_to_adjacency(B)
        if index:
            return col, A
        return A

    def degree_matrix(self, rank: int, index: bool = False):
        """Degree of each node in CHG.

        Parameters
        ----------
        rank : the rank (color) in the CHG to which the laplacian matrix is computed
        index: bool, default: False

        Returns
        -------
        if index is True:
            return rowdict, degree matrix
        else:
            return degree matrix
        """
        if rank < 1:
            raise ValueError(
                "rank for the degree matrix must be larger or equal to 1, got {rank}"
            )

        rowdict, _, M = self.incidence_matrix(0, rank, index=True)
        if M.shape == (0, 0):
            D = np.zeros(self.num_nodes)
        else:
            D = np.ravel(np.sum(M, axis=1))

        return (rowdict, D) if index else D

    def laplacian_matrix(self, rank, sparse=False, index=False):
        """Laplacian matrix, see [1].

        Parameters
        ----------
        rank : the rank ( or color) in the complex to which the laplacian matrix is computed
        sparse: bool, default: False
            Specifies whether the output matrix is a scipy sparse matrix or a numpy matrix.
        index: bool, default: False

        Returns
        -------
        L_d : numpy array
            Array of dim (N, N), where N is number of nodes in the CHG
        if index is True:
            return rowdict


        References
        ----------
        .. [1] Lucas, M., Cencetti, G., & Battiston, F. (2020).
            Multiorder Laplacian for synchronization in higher-order networks.
            Physical Review Research, 2(3), 033410.
        """
        if rank < 1:
            raise ValueError(
                "rank for the laplacian matrix must be larger or equal to 1, got {rank}"
            )

        row_dict, A = self.adjacency_matrix(0, rank, index=True)

        if A.shape == (0, 0):
            L = csr_array((0, 0)) if sparse else np.empty((0, 0))
            return ({}, L) if index else L

        if sparse:
            K = csr_array(diags(self.degree_matrix(rank)))
        else:
            K = np.diag(self.degree_matrix(rank))
        L = K - A

        return (row_dict, L) if index else L

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

    def restrict_to_cells(self, cell_set, name: Optional[str] = None):
        """Construct a Colored Hypergraph using a subset of the cells.

        Parameters
        ----------
        cell_set: iterable of hashables or RankedEntities
            A subset of elements of the Colored Hypergraph  cells
        name: str, optional

        Returns
        -------
        ColoredHyperGraph
        """
        from toponetx.classes.combinatorial_complex import CombinatorialComplex

        if isinstance(self, ColoredHyperGraph) and not isinstance(
            self, CombinatorialComplex
        ):
            chg = ColoredHyperGraph(name)
        elif isinstance(self, CombinatorialComplex):
            chg = CombinatorialComplex(name)
        valid_cells = []
        for c in cell_set:
            if c in self.cells:
                valid_cells.append(c)
        for c in valid_cells:
            if not isinstance(c, Iterable):
                raise ValueError(f"each element in cell_set must be Iterable, got {c}")
            chg.add_cell(c, rank=self.cells.get_rank(c))
        return chg

    def restrict_to_nodes(self, node_set, name: Optional[str] = None):
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
        ColoredHyperGraph

        """
        from toponetx.classes.combinatorial_complex import CombinatorialComplex

        if isinstance(self, ColoredHyperGraph) and not isinstance(
            self, CombinatorialComplex
        ):
            chg = ColoredHyperGraph(name)
        elif isinstance(self, CombinatorialComplex):
            chg = CombinatorialComplex(name)
        node_set = frozenset(node_set)
        for i in self.ranks:
            if i != 0:
                for cell in self.skeleton(i):
                    if isinstance(cell, frozenset):
                        c_set = cell
                    else:
                        c_set = cell[0]
                    if c_set <= node_set:
                        chg.add_cell(c_set, rank=i)
        return chg

    def from_networkx_graph(self, G) -> None:
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
        >>> CHG = ColoredHyperGraph()
        >>> CHG.from_networkx_graph(G)
        >>> CHG.nodes
        """
        for node in G.nodes:  # cells is a networkx graph
            self.add_node(node, **G.nodes[node])
        for edge in G.edges:
            u, v = edge
            self.add_cell([u, v], 1, **G.get_edge_data(u, v))

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
            zero_elements = cell[0]
            if len(zero_elements) == 1:
                for n in zero_elements:
                    if self.degree(n, None) == 1:
                        singletons.append(cell)
        return singletons

    def remove_singletons(self, name: Optional[str] = None):
        """Construct new CHG with singleton cells removed.

        Parameters
        ----------
        name: str, optional

        Returns
        -------
        ColoredHyperGraph
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)

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
        for cell, key in self.cells:
            CHG.add_cell(cell, key=key, rank=self.cells.get_rank(cell))
        return CHG
