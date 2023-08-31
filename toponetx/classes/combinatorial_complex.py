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
    compute_set_incidence,
    incidence_to_adjacency,
    sparse_array_to_neighborhood_dict,
)

__all__ = ["CombinatorialComplex"]


class CombinatorialComplex(ColoredHyperGraph):
    """Class for Combinatorial Complex.

    A Combinatorial Complex (CCC) is a triple CCC = (S, X, rk) where:
    - S is an abstract set of entities,
    - X a subset of the power set of X, and
    - rk is the a rank function that associates for every set x in X a rank, a positive integer.

    The rank function rk must satisfy x <= y then rk(x) <= rk(y).
    We call this condition the CCC condition.

    A CCC is a generlization of graphs, hypergraphs, cellular and simplicial complexes.

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

    >>> CCC = CombinatorialComplex()

    Add cells to the combinatorial complex:

    >>> CCC = CombinatorialComplex()
    >>> CCC.add_cell([1, 2], rank=1)
    >>> CCC.add_cell([3, 4], rank=1)
    >>> CCC.add_cell([1, 2, 3, 4], rank=2)
    >>> CCC.add_cell([1, 2, 4], rank=2)
    >>> CCC.add_cell([3, 4], rank=2)
    >>> CCC.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
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
        """Set the attributes of a cell in the CCC."""
        return super().__setitem__(cell, attr)

    @property
    def __shortstr__(self):
        """Return the short string generic representation."""
        return "CCC"

    def number_of_nodes(self, node_set=None) -> int:
        """Compute the number of nodes in node_set belonging to the CCC.

        Parameters
        ----------
        node_set : an interable of Entities, optional
            If None, then return the number of nodes in the CCC.

        Returns
        -------
        int
        """
        return super().number_of_nodes(node_set)

    def number_of_cells(self, cell_set=None) -> int:
        """Compute the number of cells in cell_set belonging to the CCC.

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
        (number of cells[i], for i in range(0,dim(CCC))  )

        Returns
        -------
        tuple of ints
        """
        return super().shape()

    def order(self):
        """Compute the number of nodes in the CCC.

        Returns
        -------
        order : int
        """
        return super().order()

    def _remove_node_helper(self, node) -> None:
        """Remove node from cells. Assumes node is present in the CCC."""
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
            Nodes in CCC
        """
        return super().remove_nodes(node_set)

    def remove_node(self, node) -> None:
        """Remove node from cells.

        This also deletes any reference in the nodes of the CCC.
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
        values : dict
        Dictionary of cell attributes to set keyed by cell name.
        name : str, optional

        Returns
        -------
        None.

        Examples
        --------
        After computing some property of the cell of a combinatorial complex, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CCC = CombinatorialComplex()
        >>> CCC.add_cell([1, 2, 3, 4], rank=2)
        >>> CCC.add_cell([1, 2, 4], rank=2,)
        >>> CCC.add_cell([3, 4], rank=2)
        >>> d = {(1, 2, 3, 4): 'red', (1, 2, 3): 'blue', (3, 4): 'green'}
        >>> CCC.set_cell_attributes(d, name='color')
        >>> CCC.cells[(3, 4)]['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CCC = NestedCombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red','attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3}}
        >>> CCC.set_cell_attributes(d)
        >>> CCC.cells[(0, 1)]['color']
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
        >>> CCC = NestedCombinatorialComplex(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 },1: {'color': 'blue', 'attr2': 3} }
        >>> CCC.set_node_attributes(d)
        >>> CCC.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CCC = NestedCombinatorialComplex(G)
        >>> nodes_color = CCC.get_node_attributes('color')
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
        >>> CCC = CombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red', 'attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3} }
        >>> CCC.set_cell_attributes(d)
        >>> cell_color = CCC.get_cell_attributes('color')
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
        Colored Hypergraph : ColoredHyperGraph
        """
        if self.graph_based:
            if rank == 1:
                if not isinstance(cell, Iterable):
                    TopoNetXError(
                        "Rank 1 cells in graph-based ColoredHyperGraph must be Iterable."
                    )
                if len(cell) != 2:
                    TopoNetXError(
                        f"Rank 1 cells in graph-based ColoredHyperGraph must have size equalt to 1 got {cell}."
                    )

        self._add_hyperedge(cell, rank=rank, **attr)

    def _incidence_matrix(
        self,
        rank,
        to_rank,
        incidence_type: Literal["up", "down"] = "up",
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
        to_rank: int, optional
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
        if to_rank is None:
            if incidence_type == "up":
                children = self.skeleton(rank)
                uidset = self.skeleton(rank + 1)
            elif incidence_type == "down":
                uidset = self.skeleton(rank)
                children = self.skeleton(rank - 1)
            else:
                raise ValueError(
                    "Invalid value for incidence_type. Must be 'up' or 'down'"
                )
        else:
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
        return compute_set_incidence(children, uidset, sparse, index)

    def incidence_matrix(
        self,
        rank,
        to_rank=None,
        incidence_type: str = "up",
        weight=None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CCC between rank and to_rank skeleti.

        Parameters
        ----------
        rank : int
        to_rank: int, optional
        incidence_type : {'up', 'down'}, default='up'
        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.
        sparse : bool, default=True
        index : bool, optional
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
        return self._incidence_matrix(
            rank, to_rank, incidence_type=incidence_type, sparse=sparse, index=index
        )

    def adjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        rank, via_rank : int, int
            Two ranks for skeletons in the input Colored Hypergraph
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.
        index: bool, default=False
            If True return will include a dictionary of node uid : row number
            and cell uid : column number

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
        if via_rank is not None:
            if rank > via_rank:
                raise ValueError("rank must be greater than via_rank")
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
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
          all cells adjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        B = self.incidence_matrix(
            rank=0, to_rank=None, incidence_type="up", index=index
        )
        if index:

            A = incidence_to_adjacency(B[0].transpose(), s=s)

            return A, B[2]
        A = incidence_to_adjacency(B.transpose(), s=s)
        return A

    def node_adjacency_matrix(self, index=False, s=1):
        """Compute the node adjacency matrix."""
        B = self.incidence_matrix(rank=0, to_rank=None, index=index)
        if index:
            A = incidence_to_adjacency(B[0], s=s)
            return A, B[1]
        A = incidence_to_adjacency(B, s=s)
        return A

    def coadjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Compute the coadjacency matrix.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        rank , via_rank : two ranks for skeletons in the input Colored Hypergraph, such that r>k

        s : int, list, optional
            Minimum number of edges shared by neighbors with node.

        index: bool, optional
            If True return will include a dictionary of node uid : row number
            and cell uid : column number

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
        if via_rank is not None:
            if rank < via_rank:
                raise ValueError("rank must be greater than via_rank")
        if index:
            B, row, col = self.incidence_matrix(
                via_rank, rank, incidence_type="down", sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                rank, via_rank, incidence_type="down", sparse=True, index=index
            )
        A = incidence_to_adjacency(B)
        if index:
            return A, col
        return A

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
        """Remove a single cell from CCC.

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
        """Remove cells from CCC.

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
        CCC = CombinatorialComplex(name=self.name, graph_based=self.graph_based)
        for cell in self.cells:
            CCC.add_cell(cell, self.cells.get_rank(cell))
        return CCC

    def is_connected(self, s=1, cells=False):
        """Determine if combintorial complex is :term:`s-connected <s-connected, s-node-connected>`.

        Parameters
        ----------
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.

        cells: bool, default=False
            If True, will determine if s-cell-connected.
            For s=1 s-cell-connected is the same as s-connected.

        Returns
        -------
        is_connected : bool

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
        B = self.incidence_matrix(rank=0, to_rank=None, incidence_type="up")
        if cells:
            A = incidence_to_adjacency(B, s=s)
        else:
            A = incidence_to_adjacency(B.transpose(), s=s)
        G = nx.from_scipy_sparse_matrix(A)
        return nx.is_connected(G)

    def singletons(self):
        """Return a list of singleton cell.

        A singleton cell is a cell of
        size 1 with a node of degree 1.

        Returns
        -------
        list
            A list of cells uids.
        """
        for cell in self.cells:
            zero_elements = self.cells[cell].skeleton(0)
            if len(zero_elements) == 1:
                for n in zero_elements:
                    if self.degree(n) == 1:
                        yield cell

    def remove_singletons(self, name=None):
        """Construct new CHG with singleton cells removed.

        Parameters
        ----------
        name: str, optional

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
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.
        cells : bool, default=True
            If True will return cell components, if False will return node components
        return_singletons : bool, default=False

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
        if cells:
            A, coldict = self.cell_adjacency_matrix(s=s, index=True)
            G = nx.from_scipy_sparse_matrix(A)

            for c in nx.connected_components(G):
                if not return_singletons and len(c) == 1:
                    continue
                yield {coldict[n] for n in c}
        else:
            A, rowdict = self.node_adjacency_matrix(s=s, index=True)
            G = nx.from_scipy_sparse_matrix(A)
            for c in nx.connected_components(G):
                if not return_singletons:
                    if len(c) == 1:
                        continue
                yield {rowdict[n] for n in c}

    def s_component_subgraphs(
        self, s: int = 1, cells: bool = True, return_singletons: bool = False
    ):
        """Return a generator for the induced subgraphs of s_connected components.

        Removes singletons unless return_singletons is set to True.

        Parameters
        ----------
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.
        cells : bool, default=False
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
        for idx, c in enumerate(
            self.s_components(s=s, cells=cells, return_singletons=return_singletons)
        ):
            if cells:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")
            else:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")

    def s_components(
        self, s: int = 1, cells: bool = True, return_singletons: bool = True
    ):
        """Compute s-connected components.

        Same as s_connected_components.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(
            s=s, cells=cells, return_singletons=return_singletons
        )

    def connected_components(self, cells: bool = False, return_singletons: bool = True):
        """Compute s-connected components.

        Same as :meth:`s_connected_components`,
        with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(cells=cells, return_singletons=True)

    def connected_component_subgraphs(self, return_singletons: bool = True):
        """Compute s-component subgraphs with s=1.

        Same as :meth:`s_component_subgraphs` with s=1.

        Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def components(self, cells: bool = False, return_singletons: bool = True):
        """Compute s-connected components for s=1.

        Same as :meth:`s_connected_components` with s=1.

        But nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(s=1, cells=cells)

    def component_subgraphs(self, return_singletons: bool = False):
        """Compute s-component subgraphs wth s=1.

        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def node_diameters(self, s: int = 1):
        """Return node diameters of the connected components.

        Parameters
        ----------
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        list of the diameters of the s-components and
        list of the s-component nodes
        """
        A, coldict = self.node_adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_matrix(A)
        diams = []
        comps = []
        for c in nx.connected_components(G):
            diamc = nx.diameter(G.subgraph(c))
            temp = set()
            for e in c:
                temp.add(coldict[e])
            comps.append(temp)
            diams.append(diamc)
        loc = np.argmax(diams)
        return diams[loc], diams, comps

    def cell_diameters(self, s: int = 1):
        """Return the cell diameters of the s_cell_connected component subgraphs.

        Parameters
        ----------
        s : int, list, default=1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        maximum diameter : int
        list of diameters : list
            List of cell_diameters for s-cell component subgraphs in CHG
        list of component : list
            List of the cell uids in the s-cell component subgraphs.
        """
        A, coldict = self.cell_adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_matrix(A)
        diams = []
        comps = []
        for c in nx.connected_components(G):
            diamc = nx.diameter(G.subgraph(c))
            temp = {coldict[e] for e in c}
            comps.append(temp)
            diams.append(diamc)
        loc = np.argmax(diams)
        return diams[loc], diams, comps

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
        A = self.node_adjacency_matrix(s=s)
        G = nx.from_scipy_sparse_matrix(A)
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            raise TopoNetXError(f"{self.__shortstr__} is not s-connected. s={s}")

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
        A = self.cell_adjacency_matrix(s=s)
        G = nx.from_scipy_sparse_matrix(A)
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            raise TopoNetXError(f"CHG is not s-connected. s={s}")

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
