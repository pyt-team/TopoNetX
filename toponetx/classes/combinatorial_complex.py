"""Creation and manipulation of a combinatorial complex."""

import warnings
from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from hypernetx import Hypergraph
from networkx import Graph
from scipy.sparse import csr_matrix

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


class CombinatorialComplex(Complex):
    """Class for Combinatorial Complex.

    A Combinatorial Complex (CC) is a triple CC = (S, X, rk) where:
    -  S is an abstract set of entities,
    - X a subset of the power set of X, and
    - rk is the a rank function that associates for every
    set x in X a rank, a positive integer.

    The rank function i must satisfy x<=y then rk(x)<=rk(y).
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
    --------------------
    Let S = {1, 2, 3, 4} be a set of entities.
    Let X = {{1, 2}, {1, 2, 3}, {1, 3}, {1, 4}} be a subset of the power set of S.
    Let i be the ranking function that assigns the
    length of a set as its rank, i.e. rk({1, 2}) = 2, rk({1, 2, 3}) = 3, etc.

    Then, (S, X, rk) is a combinatorial complex.

    Examples
    --------
    >>> # define an empty Combinatorial Complex
    >>> CC = CombinatorialComplex()
    >>> # add cells using the add_cell method
    >>> CC.add_cell([1, 2], rank=1)
    >>> CC.add_cell([3, 4], rank=1)
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> CC.add_cell([1, 2, 4], rank=2)
    >>> CC.add_cell([3, 4], rank=2)
    >>> CC.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
    """

    def __init__(
        self, cells=None, name=None, ranks=None, weight=None, graph_based=False, **attr
    ):
        super().__init__()

        if not name:
            self.name = ""
        else:
            self.name = name

        self.graph_based = graph_based  # rank 1 edges have cardinality equals to 1

        # we define a combinatorial complex as the closure of a simplicial complex
        # this gives fast insertion time because the condition x<y implies rk(x)<rk(x)
        # for a new to be inserted cell x can be checked only with
        # the maximal simplex that contains x
        # the latter can be accessed in constant time inside toponetx.    \

        self._aux_complex = SimplicialComplex()

        self._complex_set = HyperEdgeView()
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
            else:

                for node in cells.nodes:  # cells is a networkx graph
                    self.add_node(node, **cells.nodes[node])
                for edge in cells.edges:
                    u, v = edge
                    self.add_cell([u, v], 1, **cells.get_edge_data(u, v))

    def _incidence_matrix_helper(self, children, uidset, sparse=True, index=False):
        """Help compute the incidence matrix."""
        from collections import OrderedDict
        from operator import itemgetter

        ndict = dict(zip(children, range(len(children))))
        edict = dict(zip(uidset, range(len(uidset))))

        ndict = OrderedDict(sorted(ndict.items(), key=itemgetter(1)))
        edict = OrderedDict(sorted(edict.items(), key=itemgetter(1)))

        r_hyperedge_dict = {j: children[j] for j in range(len(children))}
        k_hyperedge_dict = {i: uidset[i] for i in range(len(uidset))}

        r_hyperedge_dict = OrderedDict(
            sorted(r_hyperedge_dict.items(), key=itemgetter(0))
        )
        k_hyperedge_dict = OrderedDict(
            sorted(k_hyperedge_dict.items(), key=itemgetter(0))
        )

        if len(ndict) != 0:

            # if index:
            #     rowdict = {v: k for k, v in ndict.items()}
            #     coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for n in ndict:
                    for e in edict:
                        if n <= e:
                            data.append(1)
                            rows.append(ndict[n])
                            cols.append(edict[e])
                MP = csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(r_hyperedge_dict), len(k_hyperedge_dict)),
                )
            else:
                # Create an np.matrix
                MP = np.zeros((len(children), len(uidset)), dtype=int)
                for e in k_hyperedge_dict:
                    for n in r_hyperedge_dict:
                        if r_hyperedge_dict[n] <= k_hyperedge_dict[e]:
                            MP[ndict[n], edict[e]] = 1
            if index:
                return ndict, edict, MP
            else:
                return MP
        else:
            if index:
                return {}, {}, np.zeros(1)
            else:
                return np.zeros(1)

    def _incidence_matrix(
        self, rank, to_rank, incidence_type="up", weight=None, sparse=True, index=False
    ):
        """Compute incidence matrix.

        An incidence matrix indexed by r-ranked hyperedges k-ranked hyperedges
        r !=k, when k is None incidence_type will be considered instead

        Parameters
        ----------
        incidence_type : str, optional, default 'up', other options 'down'

        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
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
        if to_rank is None:
            if incidence_type == "up":
                children = self.skeleton(rank)
                uidset = self.skeleton(rank + 1, level="upper")
            elif incidence_type == "down":
                uidset = self.skeleton(rank)
                children = self.skeleton(rank - 1, level="lower")
            raise TopoNetXError("incidence_type must be 'up' or 'down' ")
        else:
            assert (
                rank != to_rank
            )  # incidence is defined between two skeletons of different ranks
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
        return self._incidence_matrix_helper(children, uidset, sparse, index)

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
        Object associated with self._nodes.

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
        (number of cells[i], for i in range(0,dim(CC))  )

        Returns
        -------
        tuple of ints
        """
        return self._complex_set.shape

    def skeleton(self, rank):
        """Return skeleton."""
        return self._complex_set.skeleton(rank)

    @property
    def ranks(self):
        """Return ranks."""
        return sorted(self._complex_set.allranks)

    @property
    def dim(self) -> int:
        """Return dimension."""
        return max(list(self._complex_set.allranks))

    def __str__(self):
        """Return detailed string representation."""
        return f"Combinatorial Complex with {len(self.nodes)} nodes and cells with ranks {self.ranks} and sizes {self.shape} "

    def __repr__(self):
        """Return string representation."""
        return f"CombinatorialComplex(name={self.name})"

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
        """Set the attributes of a hyperedge or node in the CC."""
        if cell in self:
            if isinstance(cell, self.cell_type):
                if cell.nodes in self.nodes:
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

    def degree(self, node, rank=1):
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
        if rank is None:
            return len(memberships)
        raise TopoNetXError("Rank must be non-negative integer")

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
            raise TopoNetXError("Input cell is not in cells of the CC")
        return len(self._complex_set[cell])

    def number_of_nodes(self, node_set=None):
        """Compute the number of nodes in node_set belonging to the CC.

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
        cell_set : an interable of HyperEdge, optional, default: None
            If None, then return the number of cells.

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

    def _remove_node(self, node):
        self._remove_hyperedge(node)

    def remove_node(self, node):
        """Remove node from cells.

        This also deletes any reference in the nodes of the CC.

        Parameters
        ----------
        node : hashable or HyperEdge

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex
        """
        self._remove_node(node)
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

    def _add_node(self, node, **attr):
        """Add one node as hyperedge."""
        self._add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_node(self, node, **attr):
        """Add a node."""
        self._add_node(node, **attr)

    def set_node_attributes(self, values, name=None):
        """Set node attributes."""
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
        >>> CC.cells[(3, 4)].properties['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

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
        return {
            node: self.nodes[node].properties[name]
            for node in self.nodes
            if name in self.nodes[node].properties
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
                self._complex_set.hyperedge_dict[rank][hyperedge_].update(attr)
                for i in hyperedge_:
                    if 0 not in self._complex_set.hyperedge_dict:
                        self._complex_set.hyperedge_dict[0] = {}

                    if i not in self._complex_set.hyperedge_dict[0]:
                        self._complex_set.hyperedge_dict[0][frozenset({i})] = {
                            "weight": 1
                        }
            else:
                self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
                self._complex_set.hyperedge_dict[rank][hyperedge_].update(attr)
                for i in hyperedge_:
                    if 0 not in self._complex_set.hyperedge_dict:
                        self._complex_set.hyperedge_dict[0] = {}

                    if i not in self._complex_set.hyperedge_dict[0]:
                        self._complex_set.hyperedge_dict[0][frozenset({i})] = {
                            "weight": 1
                        }
        else:
            self._complex_set.hyperedge_dict[rank] = {}
            self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
            self._complex_set.hyperedge_dict[rank][hyperedge_].update(attr)

            for i in hyperedge_:
                if 0 not in self._complex_set.hyperedge_dict:
                    self._complex_set.hyperedge_dict[0] = {}
                if i not in self._complex_set.hyperedge_dict[0]:
                    self._complex_set.hyperedge_dict[0][frozenset({i})] = {"weight": 1}

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
        if not isinstance(rank, int):
            raise ValueError(f"rank must be an integer, got {rank}")

        if rank < 0:
            raise ValueError(f"rank must be non-negative integer, got {rank}")

        if isinstance(hyperedge, str):
            if rank != 0:
                raise ValueError(f"rank must be zero for string input, got rank {rank}")
            else:
                if 0 not in self._complex_set.hyperedge_dict:
                    self._complex_set.hyperedge_dict[0] = {}
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})] = {}
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})].update(attr)
                self._aux_complex.add_simplex(Simplex(frozenset({hyperedge}), r=0))
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})][
                    "weight"
                ] = 1
                return

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            if rank != 0:
                raise ValueError(f"rank must be zero for hashables, got rank {rank}")
            else:
                if 0 not in self._complex_set.hyperedge_dict:
                    self._complex_set.hyperedge_dict[0] = {}
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})] = {}
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})].update(attr)
                self._aux_complex.add_simplex(Simplex(frozenset({hyperedge}), r=0))
                self._complex_set.hyperedge_dict[0][frozenset({hyperedge})][
                    "weight"
                ] = 1
                return
        if isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            if not isinstance(hyperedge, HyperEdge):
                hyperedge_ = frozenset(
                    sorted(hyperedge)
                )  # put the simplex in cananical order
                if len(hyperedge_) != len(hyperedge):
                    raise ValueError(
                        f"a hyperedge cannot contain duplicate nodes,got {hyperedge_}"
                    )
            else:
                hyperedge_ = hyperedge.nodes
        if isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            for i in hyperedge_:
                if not isinstance(i, Hashable):
                    raise ValueError(
                        "every element hyperedge must be hashable, input hyperedge is {hyperedge_}"
                    )
        if (
            rank == 0
            and isinstance(hyperedge, Iterable)
            and not isinstance(hyperedge, str)
        ):
            if len(hyperedge) > 1:
                raise ValueError(
                    "rank must be positive for higher order hyperedges, got rank = 0 "
                )

        self._aux_complex.add_simplex(Simplex(hyperedge_, r=rank))
        if self._aux_complex.is_maximal(hyperedge_):  # safe to insert the hyperedge
            # looking down from hyperedge to other hyperedges in the complex
            # make sure all subsets of hyperedge have lower ranks
            all_subsets = self._aux_complex.get_boundaries([hyperedge_], min_dim=1)
            for f in all_subsets:
                if frozenset(f) == frozenset(hyperedge_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] > rank:
                        rr = self._aux_complex[f]["r"]
                        self._aux_complex.remove_maximal_simplex(hyperedge_)
                        raise ValueError(
                            "a violation of the combinatorial complex condition:"
                            + f"the hyperedge {f} in the complex has rank {rr} is larger than {rank}, the rank of the input hyperedge {hyperedge_} "
                        )

            self._add_hyperedge_helper(hyperedge_, rank, **attr)
            self._complex_set.hyperedge_dict[rank][hyperedge_]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self._complex_set.hyperedge_dict[rank][hyperedge_].update(
                    hyperedge.properties
                )

        else:
            all_cofaces = self._aux_complex.get_cofaces(hyperedge_, 0)
            # looking up from hyperedge to other hyperedges in the complex
            # make sure all supersets that are in the complex of hyperedge have higher ranks

            for f in all_cofaces:
                if frozenset(f) == frozenset(hyperedge_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] < rank:
                        rr = self._aux_complex[f]["r"]
                        # all supersets in a CC must have ranks that is larger than or equal to input ranked hyperedge
                        raise ValueError(
                            "violation of the combinatorial complex condition : "
                            + f"the hyperedge {f} in the complex has rank {rr} is smaller than {rank}, the rank of the input hyperedge {hyperedge_} "
                        )
            self._aux_complex[hyperedge_]["r"] = rank
            self._add_hyperedge_helper(hyperedge_, rank, **attr)
            self._complex_set.hyperedge_dict[rank][hyperedge_]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self._complex_set.hyperedge_dict[rank][hyperedge_].update(
                    hyperedge.properties
                )

    def _add_hyperedges_from(self, hyperedges):
        if isinstance(hyperedges, Iterable):
            for s in hyperedges:
                self.hyperedges(s)
        else:
            raise ValueError("input cells must be an iterable of HyperEdge objects")

    def _remove_hyperedge(self, hyperedge):

        if hyperedge not in self.cells:
            raise KeyError(f"The cell {hyperedge} is not in the complex")

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            del self._complex_set.hyperedge_dict[0][hyperedge]

        if isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.nodes
        else:
            hyperedge_ = frozenset(hyperedge)
        rank = self._complex_set.get_rank(hyperedge_)
        del self._complex_set.hyperedge_dict[rank][hyperedge_]

        return

    def _add_nodes_from(self, nodes):
        """Instantiate new nodes when cells are added to the CC.

        Private helper method.

        Parameters
        ----------
        nodes : iterable of hashables
        """
        for node in nodes:
            self.add_node(node)

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
        if self.graph_based:
            if rank == 1:
                if not isinstance(cell, Iterable):
                    TopoNetXError(
                        "Rank 1 cells in graph-based CombinatorialComplex must be Iterable."
                    )
                if len(cell) != 2:
                    TopoNetXError(
                        f"Rank 1 cells in graph-based CombinatorialComplex must have size equalt to 1 got {cell}."
                    )

        self._add_hyperedge(cell, rank, **attr)
        return self

    def add_cells_from(self, cells, ranks=None):
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
        self._remove_hyperedge(cell)

        return self

    def get_incidence_structure_dict(self, i, j):
        """Get incidence structure dictionary."""
        return sparse_array_to_neighborhood_dict(self.incidence_matrix(i, j))

    def get_adjacency_structure_dict(self, i, j):
        """Get adjacency structure dictionary."""
        return sparse_array_to_neighborhood_dict(self.adjacency_matrix(i, j))

    def get_all_incidence_structure_dict(self):
        """Get all incidence structure dictionary."""
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
        cell_set : iterable of hashables

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
        """Compute incidence matrix for the CC indexed by nodes x cells.

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
        return self._incidence_matrix(
            rank, to_rank, incidence_type=incidence_type, sparse=sparse, index=index
        )

    def adjacency_matrix(self, rank, via_rank, s=1, index=False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        r,k : int, int
            Two ranks for skeletons in the input combinatorial complex, such that r<k
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
        >>> CC = CombinatorialComplex(cells=G)
        >>> CC.adjacency_matrix(0, 1)
        """
        if via_rank is not None:
            assert rank < via_rank
        if index:
            B, row, col = self.incidence_matrix(
                rank, via_rank, sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                rank, via_rank, incidence_type="up", sparse=True, index=index
            )
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
        B = self.incidence_matrix(
            rank=0, to_rank=None, incidence_type="up", index=index
        )
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
        if via_rank is not None:
            assert rank > via_rank
        if index:
            B, row, col = self.incidence_matrix(
                via_rank, rank, incidence_type="down", sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                rank, via_rank, incidence_type="down", sparse=True, index=index
            )
        A = incidence_to_adjacency(B.T)
        if index:
            return A, col
        return A

    @classmethod
    def from_trimesh(cls, mesh) -> "CombinatorialComplex":
        """Import from trimesh.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        >>> CC = CombinatorialComplex.from_trimesh(mesh)
        >>> CC.nodes
        """
        raise NotImplementedError()

    def restrict_to_cells(self, cell_set, name=None):
        """Construct a combinatorial complex using a subset of the cells.

        Parameters
        ----------
        cell_set: iterable of hashables or RankedEntities
            A subset of elements of the combinatorial complex  cells
        name: str, optional

        Returns
        -------
        new Combinatorial Complex : NestedCombinatorialComplex
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

        name: str, optional

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

        Examples
        --------
        >>> from networkx import Graph
        >>> G = Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0,4)
        >>> G.add_edge(0,7)
        >>> CX = CombinatorialComplex()
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

    def to_hypergraph(self):
        """Convert a combinatorial complex to a hypergraph.

        Examples
        --------
        >>> CC = CombinatorialComplex(cells=E)
        >>> HG = CC.to_hypergraph()
        """
        raise NotImplementedError

    def is_connected(self, s=1, cells=False):
        """Determine if combinatorial complex is :term:`s-connected <s-connected, s-node-connected>`.

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

        Examples
        --------
        >>> CC = CombinatorialComplex(cells=E)
        >>> CC.is_connected()
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
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)

    def s_connected_components(self, s=1, cells=True, return_singletons=False):
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
        s nodes. If s=1 these are the path components of the CC.

        If cells=False this method returns s-node-connected components.
        A list of sets of uids of the nodes which are s-walk connected.
        Two nodes v1 and v2 are s-walk-connected if there is a
        sequence of nodes starting with v1 and ending with v2 such that pairwise
        adjacent nodes in the sequence share s cells. If s=1 these are the
        path components of the combinatorial complex .

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
        """Return a generator for the induced subgraphs of s_connected components.

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
            s-cell(node) components.
        """
        for idx, c in enumerate(
            self.s_components(s=s, cells=cells, return_singletons=return_singletons)
        ):
            if cells:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")
            else:
                yield self.restrict_to_cells(c, name=f"{self.name}:{idx}")

    def s_components(self, s=1, cells=True, return_singletons=True):
        """Compute s-connected components.

        Same as s_connected_components.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(
            s=s, cells=cells, return_singletons=return_singletons
        )

    def connected_components(self, cells=False, return_singletons=True):
        """Compute s-connected components.

        Same as :meth:`s_connected_components`,
        with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(cells=cells, return_singletons=True)

    def connected_component_subgraphs(self, return_singletons=True):
        """Compute s-component subgraphs with s=1.

        Same as :meth:`s_component_subgraphs` with s=1.

        Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def components(self, cells=False, return_singletons=True):
        """Compute s-connected components for s=1.

        Same as :meth:`s_connected_components` with s=1.

        But nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(s=1, cells=cells)

    def component_subgraphs(self, return_singletons=False):
        """Compute s-component subgraphs wth s=1.

        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def node_diameters(self, s=1):
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
        """Return the cell diameters of the s_cell_connected component subgraphs.

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
        """Return the length of the longest shortest s-walk between nodes.

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
        """Return length of the longest shortest s-walk between cells.

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
        """Return shortest s-walk distance between two nodes.

        Parameters
        ----------
        source : node.uid or node
            a node in the CC
        target : node.uid or node
            a node in the CC
        s : int
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
        raise NotImplementedError

    def cell_distance(self, source, target, s=1):
        """Return the shortest s-walk distance between two cells.

        Parameters
        ----------
        source : cell.uid or cell
            an cell in the combinatorial complex
        target : cell.uid or cell
            an cell in the combinatorial complex
        s : int
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
        raise NotImplementedError
