"""Creation and manipulation of a Colored Hypergraph."""

from collections.abc import Collection, Hashable, Iterable

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
    _compute_incidence_matrix,
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
    colors : Collection, optional
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
        colors: Collection | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = name

        self._complex_set = HyperEdgeView()
        self.complex = dict()

        if cells is not None:
            if not isinstance(cells, Iterable):
                raise TypeError(
                    f"Input cells must be given as Iterable, got {type(cells)}."
                )

            if not isinstance(colors, Graph):
                if colors is None:
                    for cell in cells:
                        if not isinstance(cell, HyperEdge):
                            raise ValueError(
                                f"input must be an HyperEdge {cell} object when rank is None"
                            )
                        if cell.rank is None:
                            raise ValueError(f"input HyperEdge {cell} has None rank")
                        self.add_cell(cell, cell.rank)
                else:
                    if isinstance(cells, Iterable) and isinstance(colors, Iterable):
                        if len(cells) != len(colors):
                            raise TopoNetXError(
                                "cells and ranks must have equal number of elements"
                            )
                        else:
                            for cell, color in zip(cells, colors):
                                self.add_cell(cell, color)
                if isinstance(cells, Iterable) and isinstance(colors, int):
                    for cell in cells:
                        self.add_cell(cell, colors)
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

    def __setitem__(self, cell, attr):
        """Set the attributes of a hyperedge or node in the CHG."""
        if cell in self:
            if isinstance(cell, self.cell_type):
                if cell.elements in self.nodes:
                    self.nodes.update(attr)
            elif isinstance(cell, Iterable):
                cell = frozenset(cell)
                if cell in self.nodes:
                    self.nodes.update(attr)
                else:
                    raise KeyError(f"node {cell} is not in complex")
            elif isinstance(cell, Hashable):
                if frozenset({cell}) in self:
                    self.nodes.update(attr)
                    return
        # we now check if the input is a cell in
        elif cell in self.cells:

            hyperedge_ = HyperEdgeView._to_frozen_set(cell)
            rank = self.get_rank(hyperedge_)

            if hyperedge_ in self.hyperedge_dict[rank]:
                self.hyperedge_dict[rank][hyperedge_] = attr
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
            raise TopoNetXError("Input cell is not in cells of the CHG")
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
            return len([node for node in self.nodes if node in node_set])
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
            return len([cell for cell in self.cells if cell in cell_set])
        return len(self.cells)

    def order(self):
        """Compute the number of nodes in the CHG.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def _remove_node(self, node):
        self._remove_hyperedge(node)

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

        This also deletes references in Colored Hypergraph nodes.

        Parameters
        ----------
        node_set : an iterable of hashables or Entities
            Nodes in CHG
        """
        for node in node_set:
            self.remove_node(node)

    def _add_hyperedge_helper(self, hyperedge_, color, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge_ : frozenset of hashable elements
        color : int
        attr : arbitrary attrs

        Returns
        -------
        None.
        """
        if color not in self._complex_set.hyperedge_dict:
            self._complex_set.hyperedge_dict[color] = {}

        existing_hyperedges = self._complex_set.hyperedge_dict[color]
        new_hyperedge = hyperedge_

        counter = 1
        while new_hyperedge in existing_hyperedges:
            # Create a new key using the existing hyperedge set and a unique identifier
            new_hyperedge = frozenset(hyperedge_) | frozenset([f"{counter}"])
            counter += 1

        existing_hyperedges[new_hyperedge] = {}
        existing_hyperedges[new_hyperedge].update(attr)

        # Set weight to 1 if not present in the default dictionary of the hyperedge
        if "weight" not in existing_hyperedges[new_hyperedge]:
            existing_hyperedges[new_hyperedge]["weight"] = 1

        # Add zero-color hyperedges (hashable components)
        for i in hyperedge_:
            zero_color_hyperedge = frozenset({i})
            if 0 not in self._complex_set.hyperedge_dict:
                self._complex_set.hyperedge_dict[0] = {}
            if zero_color_hyperedge not in self._complex_set.hyperedge_dict[0]:
                self._complex_set.hyperedge_dict[0][zero_color_hyperedge] = {
                    "weight": 1
                }

    def _add_hyperedge(self, hyperedge, color=None, **attr):
        if color is None:
            color = 1

        if not isinstance(color, int):
            raise ValueError(f"Color must be an integer or None, got {color}")

        if color < 0:
            raise ValueError(
                f"Color must be a non-negative integer or None, got {color}"
            )

        if isinstance(hyperedge, str):
            if color != 0:
                raise ValueError(
                    f"Color must be zero for string input, got color {color}"
                )

            hyperedge_ = frozenset({hyperedge})
            self._add_hyperedge_helper(hyperedge_, color, **attr)
            return

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            if color != 0:
                raise ValueError(f"Color must be zero for hashables, got color {color}")

            hyperedge_ = frozenset({hyperedge})
            self._add_hyperedge_helper(hyperedge_, color, **attr)
            return

        if isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            if not isinstance(hyperedge, HyperEdge):
                hyperedge_ = frozenset(sorted(hyperedge))
                if len(hyperedge_) != len(hyperedge):
                    raise ValueError(
                        "A hyperedge cannot contain duplicate nodes, got {hyperedge_}"
                    )
            else:
                hyperedge_ = hyperedge.elements

            for i in hyperedge_:
                if not isinstance(i, Hashable):
                    raise ValueError(
                        f"Every element in a hyperedge must be hashable, got {hyperedge_}"
                    )

            self._add_hyperedge_helper(hyperedge_, color, **attr)
            if "weight" not in self._complex_set.hyperedge_dict[color][hyperedge_]:
                self._complex_set.hyperedge_dict[color][hyperedge_]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self._complex_set.hyperedge_dict[color][hyperedge_].update(
                    hyperedge._properties
                )

    def _remove_hyperedge(self, hyperedge):
        if hyperedge not in self.cells:
            raise KeyError(f"The cell {hyperedge} is not in the colored hypergraph")

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            del self._complex_set.hyperedge_dict[0][hyperedge]

        if isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.elements
        else:
            hyperedge_ = frozenset(hyperedge)
        rank = self._complex_set.get_rank(hyperedge_)
        del self._complex_set.hyperedge_dict[rank][hyperedge_]

        return

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
        self._add_hyperedge(hyperedge=node, color=0, **attr)

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
            for node, value in values.items():
                try:
                    self.nodes[node].__dict__[name] = value
                except AttributeError:
                    pass
        else:
            for node, d in values.items():
                try:
                    self.nodes[node].__dict__.update(d)
                except AttributeError:
                    pass

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
                    self.cells[cell].__dict__[name] = value
                except AttributeError:
                    pass
        else:
            for cell, d in values.items():
                try:
                    self.cells[cell].__dict__.update(d)
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
            node: self.nodes[node][name]
            for node in self.nodes
            if name in self.nodes[node]
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
        self._add_hyperedge(cell, rank, **attr)

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
        pass

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
        pass

    def get_adjacency_structure_dict(self, i, j):
        """Get adjacency structure dictionary."""
        pass

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

    def _incidence_matrix(self, rank, to_rank, weight=None, sparse=True, index=False):
        """Compute incidence matrix.

        An incidence matrix indexed by r-ranked hyperedges k-ranked hyperedges
        r !=k, when k is None incidence_type will be considered instead

        Parameters
        ----------
        incidence_type : {'up', 'down'}, default='up'
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
        Incidence_matrix method  is a method for generating the incidence matrix of a combinatorial complex.
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

        if (
            rank < to_rank
        ):  # up incidence is defined between two skeletons of different ranks
            children = self.skeleton(rank)
            uidset = self.skeleton(to_rank)

        elif (
            rank > to_rank
        ):  # up incidence is defined between two skeletons of different ranks
            children = self.skeleton(to_rank)
            uidset = self.skeleton(rank)
        return _compute_incidence_matrix(children, uidset, sparse, index)

    def incidence_matrix(
        self,
        rank,
        to_rank=None,
        weight=None,
        sparse=True,
        index=False,
    ):
        """Compute incidence matrix of the colored hypergraph."""
        return _compute_incidence_matrix(rank, to_rank, weight, sparse, index)

    def adjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        r,k : int, int
            Two ranks for skeletons in the input Colored Hypergraph, such that r<k
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
        pass

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
        pass

    def node_adjacency_matrix(self, index=False, s=1):
        """Compute the node adjacency matrix."""
        pass

    def coadjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Compute the coadjacency matrix.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        r,k : two ranks for skeletons in the input Colored Hypergraph, such that r>k

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
        pass

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

        # RNS = self.cells.restrict_to(element_subset=cell_set, name=name)
        # return NestedColoredHyperGraph(cells=RNS, name=name)

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
        pass

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

    def clone(self) -> "ColoredHyperGraph":
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex and attributes. That is, if an
        attribute is a container, that container is shared by the original and the copy. Use Python’s `copy.deepcopy`
        for new containers.

        Returns
        -------
        ColoredHyperGraph
        """
        pass
