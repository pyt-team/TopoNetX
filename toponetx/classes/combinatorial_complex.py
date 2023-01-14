"""
Class for creation and manipulation of a combinatorial complex.
The class also supports basic functions.
"""

import collections
import warnings

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from hypernetx import Hypergraph
from networkx import Graph
from scipy.sparse import csr_matrix

from toponetx.classes.abstract_cell import AbstractCell, AbstractCellView
from toponetx.classes.node import NodeView
from toponetx.classes.ranked_entity import RankedEntity
from toponetx.exception import TopoNetXError
from toponetx.utils.structure import sparse_array_to_neighborhood_dict

__all__ = ["CombinatorialComplex"]


class CombinatorialComplex:
    """Class for Combinatorial Complex.

    A Combinatorial Complex (CC) is a triple CC = (S, X, i) where:
    -  S is an abstract set of entities,
    - X a subset of the power set of X, and
    - i is the a rank function that associates for every
    set x in X a rank, a positive integer.

    The rank function i must satisfy x<=y then i(x)<=i(y).
    We call this condition the CC condition.

    A CC is a generlization of graphs, hypergraphs, cellular and simplicial complexes.

    Parameters
    ----------
    cells : (optional)iterable, default: None

    name : hashable, optional, default: None
        If None then a placeholder '_'  will be inserted as name

    ranks : (optional) an iterable, default: None.
        when cells is an iterable or dictionary, ranks cannot be None and it must be iterable/dict of the same
        size as cells.

    weight : array-like, optional, default : None
        User specified weight corresponding to setsytem of type pandas.DataFrame,
        length must equal number of rows in dataframe.
        If None, weight for all rows is assumed to be 1.

    graph_based : boolean, default is False. When true
                rank 1 edges must have cardinality equals to 1


    Mathematical example
    ---------------------
    Let S = {1, 2, 3, 4} be a set of entities.
    Let X = {{1, 2}, {1, 2, 3}, {1, 3}, {1, 4}} be a subset of the power set of S.
    Let i be the ranking function that assigns the
    length of a set as its rank, i.e. i({1, 2}) = 2, i({1, 2, 3}) = 3, etc.

    Then, (S, X, i) is a combinatorial complex.


    Examples
    ---------
        >>> # define an empty Combinatorial Complex
        >>> CC = CombinatorialComplex()
        >>> # add cells using the add_cell method
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4], rank=2)

    """

    def __init__(
        self, cells=None, name=None, ranks=None, weight=None, graph_based=False, **attr
    ):
        if not name:
            self.name = ""
        else:
            self.name = name

        self.graph_based = graph_based  # rank 1 edges have cardinality equals to 1
        # if cells is None:
        self._complex_set = AbstractCellView()
        self.complex = dict()  # dictionary for combinatorial complex attributes

        if cells is not None:

            if not isinstance(cells, Iterable):
                raise TypeError(
                    f"Input cells must be given as Iterable, got {type(cells)}."
                )

        if cells is not None:
            if not isinstance(cells, Graph):
                if ranks is None:
                    for cell in cells:
                        if not isinstance(cell, AbstractCell):
                            raise ValueError(
                                f"input must be an AbstractCell {cell} object when rank is None"
                            )
                        if cell.rank is None:
                            raise ValueError(f"input AbstractCell {cell} has None rank")
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
        RankedEntitySet
        """
        return self._complex_set

    @property
    def nodes(self):
        """
        Object associated with self._nodes.

        Returns
        -------
        RankedEntitySet

        """
        return NodeView(self._complex_set.cell_dict, cell_type=AbstractCell)

    @property
    def incidence_dict(self):
        """
        Dictionary keyed by cell uids with values the uids of nodes in each cell

        Returns
        -------
        dict

        """
        return self._complex_set.cell_dict

    @property
    def shape(self):
        """
        (number of cells[i], for i in range(0,dim(CC))  )

        Returns
        -------
        tuple

        """

        return self._complex_set.shape

    def skeleton(self, rank):
        return self._complex_set.skeleton(rank)

    @property
    def ranks(self):
        return sorted(self._complex_set.allranks)

    @property
    def dim(self):
        return max(list(self._complex_set.allranks))

    def __str__(self):
        """
        String representation of CC

        Returns
        -------
        str

        """
        return f"Combinatorial Complex with {len(self.nodes)} nodes and cells with ranks {self.ranks} and sizes {self.shape} "

    def __repr__(self):
        """
        String representation of combinatorial complex

        Returns
        -------
        str

        """
        return f"CombinatorialComplex(name={self.name})"

    def __len__(self):
        """
        Number of nodes

        Returns
        -------
        int

        """

        return len(self.nodes)

    def __iter__(self):
        """
        Iterate over the nodes of the combinatorial complex

        Returns
        -------
        dict_keyiterator

        """
        return iter(self.nodes)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.nodes

        Parameters
        ----------
        item : hashable or AbstractCell

        """

        return item in self.nodes

    def __getitem__(self, node):
        """
        Returns the attrs of of node

        Parameters
        ----------
        node :  hashable

        Returns
        -------
        dictionary of attrs assocaited with node

        """
        return self.nodes[node]

    def degree(self, node, rank=1):
        """Compute the number of cells of certain rank that contain node.

        Parameters
        ----------
        node : hashable
            identifier for the node.
        rank : positive integer, optional, default: 1
            smallest size of cell to consider in degree

        Returns
        -------
         : int

        """
        if node in self.nodes:
            memberships = set(self.nodes[node].memberships)
        else:
            raise (print(f"The input node {node} is not an element of the node set."))
        if rank >= 0:
            return len(
                set(
                    e
                    for e in memberships
                    if e in self.cells and self.cells[e].rank == rank
                )
            )
        if rank == None:
            return len(memberships)
        raise TopoNetXError("Rank must be non-negative integer")

    def size(self, cell):
        """Compute the number of nodes in node_set that belong to cell.

        Parameters
        ----------
        cell : hashable or AbstractCell

        Returns
        -------
        size : int

        """
        if cell not in self.cells:
            raise TopoNetXError("Input cell is not in cells of the CC")
        return len(self._complex_set[cell])

    def number_of_nodes(self, node_set=None):
        """Compute the number of nodes in node_set belonging to the CC

        Parameters
        ----------
        node_set : an interable of Entities, optional, default: None
            If None, then return the number of nodes in the CC.

        Returns
        -------
        number_of_nodes : int
        """
        if node_set:
            return len([node for node in self.nodes if node in node_set])
        return len(self.nodes)

    def number_of_cells(self, cell_set=None):
        """Compute the number of cells in cell_set belonging to the CC.

        Parameters
        ----------
        cell_set : an interable of AbstractCell, optional, default: None
            If None, then return the number of cells in combinatorial complex.

        Returns
        -------
        number_of_cells : int
        """
        if cell_set:
            return len([cell for cell in self.cells if cell in cell_set])
        return len(self.cells)

    def order(self):
        """Compute the number of nodes in the CC.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def remove_node(self, node):
        """Remove node from cells.

        This also deletes any reference in the nodes of the CC.

        Parameters
        ----------
        node : hashable or AbstractCell

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        self._complex_set.remove_node(node)
        return self

    def remove_nodes(self, node_set):
        """Remove nodes from cells.

        This also deletes references in combinatorial complex nodes.

        Parameters
        ----------
        node_set : an iterable of hashables or Entities
            Nodes in CC

        Returns
        -------
        Combinatorial Complex : NestedCombinatorialComplex
        """
        for node in node_set:
            self.remove_node(node)
        return self

    def add_node(self, node, **attr):
        self._complex_set.add_node(node, **attr)

    def set_node_attributes(self, values, name=None):
        if name is not None:
            for cell, value in values.items():
                try:
                    self.nodes[cell].__dict__[name] = value
                except AttributeError:
                    pass

        else:
            for cell, d in values.items():
                try:
                    self.nodes[cell].__dict__.update(d)
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

        Example
        ------

        After computing some property of the cell of a combinatorial complex, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CC = CombinatorialComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2,)
        >>> CC.add_cell([3, 4], rank=2)
        >>> d = {(1, 2, 3, 4): 'red', (1, 2, 3): 'blue', (3, 4): 'green'}
        >>> CC.set_cell_attributes(d, name='color')
        >>> CC.cells[(3, 4)].properties['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes::

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = NestedCombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red','attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3}}
        >>> CC.set_cell_attributes(d)
        >>> CC.cells[(0, 1)].properties['color']
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
        """Get node attributes from combinatorial complex

        Parameters
        ----------
        name : string
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
        return {
            node: self.nodes[node].properties[name]
            for node in self.nodes
            if name in self.nodes[node].properties
        }

    def get_cell_attributes(self, name, rank=None):
        """Get node attributes from graph

        Parameters
        ----------

        name : string
           Attribute name

        rank : integer rank of the k-cell
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

        if rank is not None:
            return {
                cell: self.skeleton(rank)[cell].properties[name]
                for cell in self.skeleton(rank)
                if name in self.skeleton(rank)[cell].properties
            }
        else:
            return {
                cell: self.cells[cell].properties[name]
                for cell in self.cells
                if name in self.cells[cell].properties
            }

    def _add_nodes_from(self, nodes):
        """Instantiate new nodes when cells are added to the CC.

        Private helper method.

        Parameters
        ----------
        nodes : iterable of hashables or RankedEntities
        """
        for node in nodes:
            self.add_node(node)

    def add_cell(self, cell, rank=None, **attr):
        """Add a single cells to combinatorial complex.

        Parameters
        ----------
        cell : hashable, iterable or AbstractCell
            If hashable the cell returned will be empty.
            rank : rank of a cell

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        if self.graph_based:
            if rank == 1:
                if not isinstance(cell, Iterable):
                    TopoNetXError(
                        f" rank 1 cells in graph-based CombinatorialComplex must be Iterable."
                    )
                if len(cell) != 2:
                    TopoNetXError(
                        f"rank 1 cells in graph-based CombinatorialComplex must have size equalt to 1 got {cell}."
                    )

        self._complex_set.add_cell(cell, rank, **attr)
        return self

    def add_cells_from(self, cells, ranks=None):
        """Add cells to combinatorial complex.

        Parameters
        ----------
        cells : iterable of hashables or RankedEntities
            For hashables the cells returned will be empty.
        ranks: Iterable or int. When iterable, len(ranks) == len(cells)

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        if ranks is None:
            for cell in cells:
                if not isinstance(cell, AbstractCell):
                    raise ValueError(
                        f"input must be an AbstractCell {cell} object when rank is None"
                    )
                if cell.rank is None:
                    raise ValueError(f"input AbstractCell {cell} has None rank")
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

        self._complex_set.remove_cell(cell)

        return self

    def get_incidence_structure_dict(self, i, j):
        return sparse_array_to_neighborhood_dict(self.incidence_matrix(i, j))

    def get_adjacency_structure_dict(self, i, j):
        return sparse_array_to_neighborhood_dict(self.adjacency_matrix(i, j))

    def get_all_incidence_structure_dict(self):
        d = {}
        for r in range(1, self.dim):
            B0r = sparse_array_to_neighborhood_dict(
                self.incidence_matrix(rank=0, to_rank=r)
            )
            d["B_0_" + {r}] = B0r
        return d

    def remove_cells(self, cell_set):
        """Remove cells from CC.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities

        Returns
        -------
        Combinatorial Complex : NestedCombinatorialComplex
        """
        for cell in cell_set:
            self.remove_cell(cell)
        return self

    def incidence_matrix(
        self, rank, to_rank, incidence_type="up", weight=None, sparse=True, index=False
    ):
        """
        An incidence matrix for the CC indexed by nodes x cells.

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
        weight = False  # not implemented at this moment

        return self._complex_set.incidence_matrix(
            rank, to_rank, incidence_type=incidence_type, sparse=sparse, index=index
        )

    @staticmethod
    def _incidence_to_adjacency(B, s=1, weight=False):
        """
        Helper method to obtain adjacency matrix from
        boolean incidence matrix for s-metrics.
        Self loops are not supported.
        The adjacency matrix will define an s-linegraph.

        Parameters
        ----------
        B : scipy.sparse.csr.csr_matrix
            incidence matrix of 0's and 1's

        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        # weight : bool, dict optional, default=True
        #     If False all nonzero entries are 1.
        #     Otherwise, weight will be as in product.

        Returns
        -------
        A : scipy.sparse.csr.csr_matrix

        """
        B = csr_matrix(B)
        weight = False  ## currently weighting is not supported

        if weight == False:
            A = B.dot(B.transpose())
            A.setdiag(0)
            A = (A >= s) * 1
        return A

    def adjacency_matrix(self, rank, to_rank, s=1, weight=False, index=False):
        """
        The sparse weighted :term:`s-adjacency matrix`

        Parameters
        ----------
        r,k : two ranks for skeletons in the input combinatorial complex, such that r<k

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
            adjacency_matrix : scipy.sparse.csr.csr_matrix

            row dictionary : dict

        If index if False

            adjacency_matrix : scipy.sparse.csr.csr_matrix


        Example
        --------
        >>> G = Graph() # networkx graph
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0,3)
        >>> G.add_edge(0,4)
        >>> G.add_edge(1, 4)
        >>> CC = CombinatorialComplex(cells=G)
        >>> CC.adjacency_matrix(0, 1)
        """

        if to_rank is not None:
            assert rank < to_rank
        if index:
            B, row, col = self.incidence_matrix(rank, to_rank, sparse=True, index=index)
        else:
            B = self.incidence_matrix(
                rank, to_rank, incidence_type="up", sparse=True, index=index
            )
        weight = False  ## currently weighting is not supported
        A = self._incidence_to_adjacency(B, s=s, weight=weight)
        if index:
            return A, row
        else:
            return A

    def cell_adjacency_matrix(self, index=False, s=1, weight=False):
        """Compute the cell adjacency matrix.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
          all cells adjacency_matrix : scipy.sparse.csr.csr_matrix

        """

        weight = False  ## Currently default weight are not supported

        B = self.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:

            A = self._incidence_to_adjacency(B[0].transpose(), s=s)

            return A, B[2]
        A = self._incidence_to_adjacency(B.transpose(), s=s)
        return A

    def node_adjacency_matrix(self, index=False, s=1, weight=False):
        """Compute the node adjacency matrix."""
        weight = False  ## Currently default weight are not supported

        B = self.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:
            A = self._incidence_to_adjacency(B[0], s=s)
            return A, B[1]
        A = self._incidence_to_adjacency(B, s=s)
        return A

    def coadjacency_matrix(self, rank, to_rank, s=1, weight=False, index=False):
        """Compute the coadjacency matrix.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        r,k : two ranks for skeletons in the input combinatorial complex, such that r>k

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
        if to_rank is not None:
            assert rank > to_rank
        if index:
            B, row, col = self.incidence_matrix(
                to_rank, rank, incidence_type="down", sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                rank, to_rank, incidence_type="down", sparse=True, index=index
            )
        weight = False  ## currently weighting is not supported
        if weight == False:
            A = B.T.dot(B)
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, col
        return A

    @staticmethod
    def from_trimesh(mesh):
        """
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                               faces=[[0, 1, 2]],
                               process=False)
        >>> CC = CombinatorialComplex.from_trimesh(mesh)
        >>> print(CC.nodes)
        """
        CC = CombinatorialComplex()
        return CC

    def restrict_to_cells(self, cell_set, name=None):
        """
        Constructs a combinatorial complex using a subset of the cells in combinatorial complex

        Parameters
        ----------
        cell_set: iterable of hashables or RankedEntities
            A subset of elements of the combinatorial complex  cells

        name: str, optional

        Returns
        -------
        new Combinatorial Complex : NestedCombinatorialComplex

        Example

        """
        raise NotImplementedError

        # RNS = self.cells.restrict_to(element_subset=cell_set, name=name)
        # return NestedCombinatorialComplex(cells=RNS, name=name)

    def restrict_to_nodes(self, node_set, name=None):
        """Restrict to a set of nodes.

        Constructs a new combinatorial complex  by restricting the
        cells in the combinatorial complex to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set: iterable of hashables
            References a subset of elements of self.nodes

        name: string, optional, default: None

        Returns
        -------
        new Combinatorial Complex : NestedCombinatorialComplex

        """
        raise NotImplementedError

    def from_networkx_graph(self, G):
        """Construct a combinatorial complex from a networkx graph.

        Parameters
        ----------
        G : NetworkX graph
            A networkx graph

        Returns
        -------
        CC such that the edges of the graph are ranked 1
        and the nodes are ranked 0.

        Example
        ------
        >>> from networkx import Graph
        >>> G = Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0,4)
        >>> G.add_edge(0,7)
        >>> CX = CombinatorialComplex.from_networkx_graph(G)
        >>> CX.nodes
        RankedEntitySet(:Nodes,[0, 1, 4, 7],{'weight': 1.0})
        >>> CX.cells
        RankedEntitySet(:Cells,[(0, 1), (0, 7), (0, 4)],{'weight': 1.0})
        """
        for node in G.nodes:
            self.add_node(node)
        for edge in G.edges:
            self.add_cell(edge, rank=1)

    def to_hypergraph(self):
        """Converts a combinatorial complex to a hypergraph.

        Example
        -------
        >>> CC = CombinatorialComplex(cells=E)
        >>> HG = CC.to_hypergraph()
        """
        raise NotImplementedError

    def is_connected(self, s=1, cells=False):
        """
        Determines if combinatorial complex is :term:`s-connected <s-connected, s-node-connected>`.

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

        A CC is s node connected if for any two nodes v0,vn
        there exists a sequence of nodes v0,v1,v2,...,v(n-1),vn
        such that every consecutive pair of nodes v(i),v(i+1)
        share at least s cell.

        Example
        -------
        >>> CC = CombinatorialComplex(cells=E)
        """
        B = self.incidence_matrix(rank=0, to_rank=None, incidence_type="up")
        if cells:
            A = self._incidence_to_adjacency(B, s=s)
        else:
            A = self._incidence_to_adjacency(B.transpose(), s=s)
        G = nx.from_scipy_sparse_matrix(A)
        return nx.is_connected(G)

    def singletons(self):
        """
        Returns a list of singleton cell. A singleton cell is a cell of
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
        """Construct new CC with singleton cells removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        new CC : CC

        Example
        -------
        >>> x1 = RankedEntity('x1',rank = 0)
        >>> x2 = RankedEntity('x2',rank = 0)
        >>> x3 = RankedEntity('x3',rank = 0)
        >>> x4 = RankedEntity('x4',rank = 0)
        >>> x5 = RankedEntity('x5',rank = 0)
        >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
        >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
        >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
        >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
        >>> y5 = RankedEntity('y5',[x4,x5], rank = 1)
        >>> y6 = RankedEntity('y6',[x4,x5], rank = 1)
        >>> w = RankedEntity('w',[x4,x5,x1],rank = 2)
        >>> # define the Ranked Entity Set
        >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,w,y6] )
        >>> CC = NestedCombinatorialComplex(cells=E)
        >>> CC_with_singletons = CC.restrict_to_nodes([x3,x2])
        >>> CC_no_singltons = CC_with_singletons.remove_singletons()
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)

    def s_connected_components(self, s=1, cells=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-cell-connected components <s-cell-connected component>`
        or the :term:`s-node-connected components <s-connected component, s-node-connected component>`
        of the combinatorial complex.

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
        s nodes. If s=1 these are the path components of the CC.

        If cells=False this method returns s-node-connected components.
        A list of sets of uids of the nodes which are s-walk connected.
        Two nodes v1 and v2 are s-walk-connected if there is a
        sequence of nodes starting with v1 and ending with v2 such that pairwise
        adjacent nodes in the sequence share s cells. If s=1 these are the
        path components of the combinatorial complex .

        Example
        -------


        Yields
        ------
        s_connected_components : iterator
            Iterator returns sets of uids of the cells (or nodes) in the s-cells(node)
            components of CC.

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

    def s_component_subgraphs(self, s=1, cells=True, return_singletons=False):
        """
        Returns a generator for the induced subgraphs of s_connected components.
        Removes singletons unless return_singletons is set to True.
        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        cells : boolean, optional, cells=False
            Determines if cell or node components are desired. Returns
            subgraphs equal to the CC restricted to each set of nodes(cells) in the
            s-connected components or s-cell-connected components
        return_singletons : bool, optional

        Yields
        ------
        s_component_subgraphs : iterator
            Iterator returns subgraphs generated by the cells (or nodes) in the
            s-cell(node) components of combinatorial complex.

        """
        for idx, c in enumerate(
            self.s_components(s=s, cells=cells, return_singletons=return_singletons)
        ):
            if cells:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")
            else:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")

    def s_components(self, s=1, cells=True, return_singletons=True):
        """
        Same as s_connected_components

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(
            s=s, cells=cells, return_singletons=return_singletons
        )

    def connected_components(self, cells=False, return_singletons=True):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(cells=cells, return_singletons=True)

    def connected_component_subgraphs(self, return_singletons=True):
        """
        Same as :meth:`s_component_subgraphs` with s=1. Returns iterator

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def components(self, cells=False, return_singletons=True):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(s=1, cells=cells)

    def component_subgraphs(self, return_singletons=False):
        """
        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def node_diameters(self, s=1):
        """
        Returns the node diameters of the connected components in combinatorial complex.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.
        Returns
        --------
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

    def cell_diameters(self, s=1):
        """
        Returns the cell diameters of the s_cell_connected component subgraphs
        in CC.

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        maximum diameter : int

        list of diameters : list
            List of cell_diameters for s-cell component subgraphs in CC

        list of component : list
            List of the cell uids in the s-cell component subgraphs.

        """

        A, coldict = self.cell_adjacency_matrix(s=s, index=True)
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

    def diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between nodes in combinatorial complex

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        diameter : int

        Raises
        ------
        TopoNetXError
            If CC is not s-cell-connected

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
            raise TopoNetXError(f"CC is not s-connected. s={s}")

    def cell_diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between cells in combinatorial complex

        Parameters
        ----------
        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Return
        ------
        cell_diameter : int

        Raises
        ------
        TopoNetXError
            If combinatorial complex is not s-cell-connected

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
            raise TopoNetXError(f"CC is not s-connected. s={s}")

    def distance(self, source, target, s=1):
        """
        Returns the shortest s-walk distance between two nodes in the combinatorial complex.

        Parameters
        ----------
        source : node.uid or node
            a node in the CC

        target : node.uid or node
            a node in the CC

        s : positive integer
            the number of cells

        Returns
        -------
        s-walk distance : int

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

        if isinstance(source, RankedEntity):
            source = source.uid
        if isinstance(target, RankedEntity):
            target = target.uid
        A, rowdict = self.node_adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_matrix(A)
        rkey = {v: k for k, v in rowdict.items()}
        try:
            path = nx.shortest_path_length(G, rkey[source], rkey[target])
            return path
        except:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf

    def cell_distance(self, source, target, s=1):
        """
        Returns the shortest s-walk distance between two cells in the combinatorial complex.

        Parameters
        ----------
        source : cell.uid or cell
            an cell in the combinatorial complex

        target : cell.uid or cell
            an cell in the combinatorial complex

        s : positive integer
            the number of intersections between pairwise consecutive cells

        Returns
        -------
        s- walk distance : the shortest s-walk cell distance
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
        if isinstance(source, RankedEntity):
            source = source.uid
        if isinstance(target, RankedEntity):
            target = target.uid
        A, coldict = self.cell_adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_matrix(A)
        ckey = {v: k for k, v in coldict.items()}
        try:
            path = nx.shortest_path_length(G, ckey[source], ckey[target])
            return path
        except:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf

    def dataframe(self, sort_rows=False, sort_columns=False, cell_weight=True):
        """Create a pandas dataframe from the combinatorial complex.

        Returns a pandas dataframe for CC indexed by the nodes and
        with column headers given by the cell names.

        Parameters
        ----------
        sort_rows : bool, optional, default=True
            sort rows based on hashable node names
        sort_columns : bool, optional, default=True
            sort columns based on hashable cell names
        cell_weight : bool, optional, default=True
        """
        mat, rdx, cdx = self.cells.incidence_matrix(
            0, None, incidence_type="up", index=True
        )
        index = [rdx[i] for i in rdx]
        columns = [cdx[j] for j in cdx]
        df = pd.DataFrame(mat.todense(), index=index, columns=columns)
        if sort_rows:
            df = df.sort_index()
        if sort_columns:
            df = df[sorted(columns)]
        return df

    @classmethod
    def from_numpy_array(
        cls,
        M,
        node_names=None,
        cell_names=None,
        node_label="nodes",
        cell_label="cells",
        name=None,
        key=None,
        static=False,
        use_nwhy=False,
    ):
        """
        Create a hypergraph from a real valued matrix represented as a 2 dimensionsl numpy array.
        The matrix is converted to a matrix of 0's and 1's so that any truthy cells are converted to 1's and
        all others to 0's.

        Parameters
        ----------
        M : real valued array-like object, 2 dimensions
            representing a real valued matrix with rows corresponding to nodes and columns to cells

        node_names : object, array-like, default=None
            List of node names must be the same length as M.shape[0].
            If None then the node names correspond to row indices with 'v' prepended.

        cell_names : object, array-like, default=None
            List of cell names must have the same length as M.shape[1].
            If None then the cell names correspond to column indices with 'e' prepended.

        name : hashable

        key : (optional) function
            boolean function to be evaluated on each cell of the array,
            must be applicable to numpy.array

        Returns
        -------
         : NestedCombinatorialComplex

        Note
        ----
        The constructor does not generate empty cells.
        All zero columns in M are removed and the names corresponding to these
        cells are discarded.


        """
        # Create names for nodes and cells
        # Validate the size of the node and cell arrays

        M = np.array(M)
        if len(M.shape) != (2):
            raise TopoNetXError("Input requires a 2 dimensional numpy array")
        # apply boolean key if available
        if key:
            M = key(M)

        if node_names is not None:
            nodenames = np.array(node_names)
            if len(nodenames) != M.shape[0]:
                raise TopoNetXError(
                    "Number of node names does not match number of rows."
                )
        else:
            nodenames = np.array([f"v{idx}" for idx in range(M.shape[0])])

        if cell_names is not None:
            cellnames = np.array(cell_names)
            if len(cellnames) != M.shape[1]:
                raise TopoNetXError(
                    "Number of cell_names does not match number of columns."
                )
        else:
            cellnames = np.array([f"e{jdx}" for jdx in range(M.shape[1])])

        # Remove empty column indices from M columns and cellnames
        colidx = np.array([jdx for jdx in range(M.shape[1]) if any(M[:, jdx])])
        colidxsum = np.sum(colidx)
        if not colidxsum:
            return Hypergraph()
        else:
            M = M[:, colidx]
            cellnames = cellnames[colidx]
            edict = dict()
            # Create an EntitySet of cells from M
            for jdx, e in enumerate(cellnames):
                edict[e] = nodenames[[idx for idx in range(M.shape[0]) if M[idx, jdx]]]
            return Hypergraph(edict, name=name)
