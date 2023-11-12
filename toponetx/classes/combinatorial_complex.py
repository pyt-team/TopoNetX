"""Creation and manipulation of a combinatorial complex."""

from collections.abc import Collection, Hashable, Iterable
from typing import Any, Literal

import networkx as nx

from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.reportviews import HyperEdgeView, NodeView
from toponetx.utils.structure import compute_set_incidence

__all__ = ["CombinatorialComplex"]


class CombinatorialComplex(ColoredHyperGraph):
    r"""Class for Combinatorial Complex.

    A Combinatorial Complex (CCC) is a triple $CCC = (S, X, rk)$ where:
    - $S$ is an abstract set of entities,
    - $X$ a subset of the power set of $X$, and
    - $rk$ is the a rank function that associates for every set x in X a rank, a positive integer.

    The rank function $rk$ must satisfy $x \leq y$ then $rk(x) \leq rk(y)$.
    We call this condition the CCC condition.

    A CCC is a generlization of graphs, hypergraphs, cellular and simplicial complexes.

    Mathematical Example:

    Let $S = \{1, 2, 3, 4\}$ be a set of abstract entities.
    Let $X = \{\{1, 2\}, \{1, 2, 3\}, \{1, 3\}, \{1, 4\}\}$ be a subset of the power set of $S$.
    Let rk be the ranking function that assigns the
    length of a set as its rank, i.e. $rk(\{1, 2\}) = 2$, $rk(\{1, 2, 3\}) = 3$, etc.

    Then, $(S, X, rk)$ is a combinatorial complex.

    Parameters
    ----------
    cells : Collection, optional
        A collection of cells to add to the combinatorial complex.
    name : str, optional
        An identifiable name for the combinatorial complex.
    ranks : Collection, optional
        When cells is an iterable or dictionary, ranks cannot be None and it must be iterable/dict of the same
        size as cells.
    graph_based : bool, default=False
        When true rank 1 edges must have cardinality equals to 1.
    **kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.

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
        self._node_membership = {}
        self._complex_set = HyperEdgeView()

        if cells is not None:
            if not isinstance(cells, Iterable):
                raise TypeError(
                    f"Input cells must be given as Iterable, got {type(cells)}."
                )
            if not isinstance(cells, nx.Graph):
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
                            raise ValueError(
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
        """Set the attributes of a hyperedge or node in the CCC."""
        if cell in self.nodes:
            self.nodes[cell].update(attr)
            return
        # we now check if the input is a cell in the CCC
        elif cell in self.cells:
            hyperedge_ = HyperEdgeView._to_frozen_set(cell)
            rank = self.cells.get_rank(hyperedge_)
            if hyperedge_ in self._complex_set.hyperedge_dict[rank]:
                self._complex_set.hyperedge_dict[rank][hyperedge_] = attr
        else:
            raise KeyError(f"input {cell} is not in the complex")

    def __contains__(self, item) -> bool:
        """Return true/false indicating if item is in self.nodes.

        Parameters
        ----------
        item : hashable or HyperEdge

        Returns
        -------
        bool
        """
        return item in self.nodes

    @property
    def __shortstr__(self) -> str:
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
            The number of nodes in node_set belonging to this combinatorial complex.
        """
        return super().number_of_nodes(node_set)

    @property
    def nodes(self):
        """Object associated with self.elements.

        Returns
        -------
        NodeView
        """
        return NodeView(
            self._complex_set.hyperedge_dict, cell_type=HyperEdge, colored_nodes=False
        )

    @property
    def cells(self):
        """Object associated with self._cells.

        Returns
        -------
        HyperEdgeView
        """
        return self._complex_set

    def number_of_cells(self, cell_set=None) -> int:
        """Compute the number of cells in cell_set belonging to the CCC.

        Parameters
        ----------
        cell_set : an interable of HyperEdge, optional
            If None, then return the number of cells.

        Returns
        -------
        int
            The number of cells in cell_set belonging to this combinatorial complex.
        """
        return super().number_of_cells(cell_set)

    @property
    def shape(self):
        """Return shape.

        This is:
        (number of cells[i], for i in range(0,dim(CCC))  )

        Returns
        -------
        tuple of ints
        """
        return self._complex_set.shape

    def skeleton(
        self,
        rank: int,
        level: Literal[
            "equal",
            "upper",
            "up",
            "lower",
            "down",
            "uppereq",
            "upeq",
            "lowereq",
            "downeq",
        ] = "equal",
    ):
        """Skeleton of the CCC.

        Parameters
        ----------
        rank : int
            The rank of the skeleton.
        level : str, default="equal"
            Level of the skeleton.

        Returns
        -------
        list of HyperEdge
            The skeleton of the CCC.
        """
        return self._complex_set.skeleton(rank, level=level)

    def order(self) -> int:
        """Compute the number of nodes in the CCC.

        Returns
        -------
        int
            The number of nodes in this combinatorial complex.
        """
        return super().order()

    def _remove_node_helper(self, node) -> None:
        """Remove node from cells.

        This function assumes that the node is present in the CCC.
        """
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

    def remove_nodes(self, node_set) -> None:
        """Remove nodes from cells.

        This also deletes references in combinatorial complex nodes.

        Parameters
        ----------
        node_set : an iterable of hashables
            The nodes to remove from this combinatorial complex.
        """
        super().remove_nodes(node_set)

    def remove_node(self, node) -> None:
        """Remove node from cells.

        This also deletes any reference in the nodes of the CCC.
        This also deletes cell references in higher ranks for the particular node.

        Parameters
        ----------
        node : hashable or HyperEdge
            The node to remove from this combinatorial complex.
        """
        super()._remove_node(node)

    def set_cell_attributes(self, values, name: str | None = None) -> None:
        """Set cell attributes.

        Parameters
        ----------
        values : dict
            Dictionary of cell attributes to set keyed by cell name.
        name : str, optional
           Attribute name.

        Examples
        --------
        After computing some property of the cell of a combinatorial complex, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CCC = CombinatorialComplex()
        >>> CCC.add_cell([1, 2, 3, 4], rank=2)
        >>> CCC.add_cell([1, 2, 4], rank=2)
        >>> CCC.add_cell([3, 4], rank=2)
        >>> d = {(1, 2, 3, 4): 'red', (1, 2, 3): 'blue', (3, 4): 'green'}
        >>> CCC.set_cell_attributes(d, name='color')
        >>> CCC.cells[(3, 4)]['color']
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CCC = CombinatorialComplex(G)
        >>> d = {(1, 2): {'color': 'red','attr2': 1}, (0, 1): {'color': 'blue', 'attr2': 3}}
        >>> CCC.set_cell_attributes(d)
        >>> CCC.cells[(0, 1)]['color']
        'blue'
        3

        Note that if the dict contains cells that are not in `self.cells`, they are
        silently ignored.
        """
        super().set_cell_attributes(values, name)

    def get_node_attributes(self, name: str) -> dict[Hashable, Any]:
        """Get node attributes.

        Parameters
        ----------
        name : str
           Attribute name.

        Returns
        -------
        dict[Hashable, Any]
            Dictionary mapping each node to the value of the given attribute name.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CCC = CombinatorialComplex(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 },1: {'color': 'blue', 'attr2': 3} }
        >>> CCC.set_node_attributes(d)
        >>> CCC.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CCC = CombinatorialComplex(G)
        >>> nodes_color = CCC.get_node_attributes('color')
        >>> nodes_color[1]
        'blue'
        """
        return super().get_node_attributes(name)

    def get_cell_attributes(self, name: str, rank: int | None = None):
        """Get node attributes from graph.

        Parameters
        ----------
        name : str
           Attribute name.
        rank : int
            Restrict the returned attribute values to cells of a specific rank.

        Returns
        -------
        dict
            Dictionary of attributes keyed by cell or k-cells if k is not None.

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
        """Add one node as a hyperedge of rank 0."""
        if node in self:
            self._complex_set.hyperedge_dict[0][frozenset({node})].update(**attr)
        else:
            self._add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_node(self, node, **attr) -> None:
        """Add a node to a CCC.

        Parameters
        ----------
        node : Hashable
            The node to add to this combinatorial complex.
        **attr : keyword arguments, optional
            Attributes to add to the node as key=value pairs.
        """
        self._add_node(node, **attr)

    def _add_nodes_of_hyperedge(self, hyperedge_):
        """Adding nodes of a hyperedge.

        Parameters
        ----------
        hyperedge_ : frozenset of elements

        Returns
        -------
        None.
        """
        for i in hyperedge_:
            if 0 not in self._complex_set.hyperedge_dict:
                self._complex_set.hyperedge_dict[0] = {}
            if i not in self._complex_set.hyperedge_dict[0]:
                self._complex_set.hyperedge_dict[0][frozenset({i})] = {"weight": 1}

    def _add_hyperedge_helper(self, hyperedge_, rank, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge_ : frozenset of elements
        rank : int
        attr : arbitrary attrs

        Returns
        -------
        None.
        """
        if rank not in self._complex_set.hyperedge_dict:
            self._complex_set.hyperedge_dict[rank] = {}
        if hyperedge_ not in self._complex_set.hyperedge_dict[rank]:
            self._complex_set.hyperedge_dict[rank][hyperedge_] = {}
            self._complex_set.hyperedge_dict[rank][hyperedge_] = {"weight": 1}
            self._complex_set.hyperedge_dict[rank][hyperedge_].update(**attr)
        else:
            self._complex_set.hyperedge_dict[rank][hyperedge_].update(**attr)
        self._add_nodes_of_hyperedge(hyperedge_)

    def _CCC_condition(self, hyperedge_, rank):
        """Check if hyperedge_ satisfy the CCC condition."""
        for node in hyperedge_:
            if node in self._node_membership:
                for existing_hyperedge in self._node_membership[node]:
                    if existing_hyperedge == hyperedge_:
                        continue
                    else:
                        e_rank = self._complex_set.get_rank(existing_hyperedge)
                        if rank > e_rank:
                            if existing_hyperedge.issuperset(hyperedge_):
                                raise ValueError(
                                    "a violation of the combinatorial complex condition:"
                                    + f"the hyperedge {existing_hyperedge} in the complex has rank {e_rank} is larger than {rank}, the rank of the input hyperedge {hyperedge_} "
                                )

                        if rank < e_rank:
                            if hyperedge_.issuperset(existing_hyperedge):
                                raise ValueError(
                                    "violation of the combinatorial complex condition : "
                                    + f"the hyperedge {existing_hyperedge} in the complex has rank {e_rank} is smaller than {rank}, the rank of the input hyperedge {hyperedge_} "
                                )

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
        elif isinstance(hyperedge, (Iterable, HyperEdge)):
            if len(hyperedge) == 1:
                if rank != 0:
                    raise ValueError(
                        f"rank must be zero cells with single element, got rank {rank} with input hyperedge {hyperedge} "
                    )
            if isinstance(hyperedge, HyperEdge):
                hyperedge_ = hyperedge.elements
            else:
                if not all(isinstance(i, Hashable) for i in hyperedge):
                    raise ValueError(
                        f"Input hyperedge {hyperedge} contain non-hashable elements."
                    )
                hyperedge_ = frozenset(hyperedge)
            if rank == 0 and len(hyperedge_) > 1:
                raise ValueError(
                    "rank must be positive for higher order hyperedges, got rank = 0"
                )
            hyperedge_set = hyperedge_
        else:
            raise ValueError("Invalid hyperedge type")

        if rank == 0:
            if 0 not in self._complex_set.hyperedge_dict:
                self._complex_set.hyperedge_dict[0] = {}
            self._complex_set.hyperedge_dict[0][hyperedge_set] = {}
            self._complex_set.hyperedge_dict[0][hyperedge_set].update(attr)
            if "weight" not in self._complex_set.hyperedge_dict[0][hyperedge_set]:
                self._complex_set.hyperedge_dict[0][hyperedge_set]["weight"] = 1
        else:
            if hyperedge_set in self.cells:
                e_rank = self._complex_set.get_rank(hyperedge_set)
                if e_rank > rank:
                    self.remove_cell(hyperedge_set)
                    self._add_hyperedge_helper(hyperedge_set, rank, **attr)
                    if (
                        "weight"
                        not in self._complex_set.hyperedge_dict[rank][hyperedge_set]
                    ):
                        self._complex_set.hyperedge_dict[rank][hyperedge_set][
                            "weight"
                        ] = 1
                    return
                elif e_rank < rank:
                    self._CCC_condition(hyperedge_, rank)
                    self.remove_cell(hyperedge_set)
                    self._add_hyperedge_helper(hyperedge_set, rank, **attr)
                    if (
                        "weight"
                        not in self._complex_set.hyperedge_dict[rank][hyperedge_set]
                    ):
                        self._complex_set.hyperedge_dict[rank][hyperedge_set][
                            "weight"
                        ] = 1
                    return
                else:
                    self._add_hyperedge_helper(hyperedge_set, rank, **attr)
                    if (
                        "weight"
                        not in self._complex_set.hyperedge_dict[rank][hyperedge_set]
                    ):
                        self._complex_set.hyperedge_dict[rank][hyperedge_set][
                            "weight"
                        ] = 1
                    return
            self._CCC_condition(hyperedge_, rank)
            self._add_hyperedge_helper(hyperedge_set, rank, **attr)
            if "weight" not in self._complex_set.hyperedge_dict[rank][hyperedge_set]:
                self._complex_set.hyperedge_dict[rank][hyperedge_set]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self._complex_set.hyperedge_dict[rank][hyperedge_set].update(
                    hyperedge._attributes
                )

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
            For which rank of cells to compute the incidence matrix.
        to_rank: int, optional
        incidence_type : {'up', 'down'}, default='up'
            Whether to compute the up or down incidence matrix.
        sparse : bool, default=True
            Whether to return a sparse or dense incidence matrix.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the incidence
            matrix.

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or numpy.ndarray
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
            raise ValueError(
                "incidence matrix can be computed for k!=r, got equal r and k."
            )
        if to_rank is None:
            if incidence_type == "up":
                children = self.skeleton(rank)
                uidset = self.skeleton(rank, level="up")
            elif incidence_type == "down":
                uidset = self.skeleton(rank)
                children = self.skeleton(rank, level="down")
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

            elif rank > to_rank:
                raise ValueError("incidence matrix can be computed for r<k, got r>k.")

        return compute_set_incidence(children, uidset, sparse, index)

    def incidence_matrix(
        self,
        rank: int,
        to_rank=None,
        incidence_type: Literal["up", "down"] = "up",
        weight: str | None = None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CCC between rank and to_rank skeleti.

        Parameters
        ----------
        rank, to_rank : int
            For which rank of cells to compute the incidence matrix.
        incidence_type : {"up", "down"}, default="up"
            Whether to compute the up or down incidence matrix.
        weight : bool, default=False
            The name of the cell attribute to use as weights for the incidence matrix.
            If `None`, all cell weights are considered to be one.
        sparse : bool, default=True
            Whether to return a sparse or dense incidence matrix.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the incidence
            matrix.

        Returns
        -------
        row_indices, col_indices : dict
            Dictionary assigning each row and column of the incidence matrix to a
            cell.
        incidence_matrix : scipy.sparse.csr.csr_matrix
            The incidence matrix.
        """
        return self._incidence_matrix(
            rank, to_rank, incidence_type=incidence_type, sparse=sparse, index=index
        )

    def adjacency_matrix(self, rank, via_rank, s: int = 1, index: bool = False):
        """Sparse weighted :term:`s-adjacency matrix`.

        Parameters
        ----------
        rank, via_rank : int
            Two ranks for skeletons in the input combinatorial complex.
        s : int, default=1
            Minimum number of edges shared by neighbors with node.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the adjacency
            matrix.

        Returns
        -------
        indices : list
            List identifying the rows and columns of the adjacency matrix with the
            cells of the combinatorial complex. Only returned if `index` is True.
        adjacency_matrix : scipy.sparse.csr.csr_matrix
            The adjacency matrix of this combinatorial complex.

        Examples
        --------
        >>> CCC = CombinatorialComplex()
        >>> CCC.add_cell([1, 2], rank=1)
        >>> CCC.add_cell([3, 4], rank=1)
        >>> CCC.add_cell([1, 2, 3, 4], rank=2)
        >>> CCC.add_cell([1, 2, 4], rank=2)
        >>> CCC.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)
        >>> CCC.adjacency_matrix(0, 1)
        """
        if via_rank is not None:
            if rank > via_rank:
                raise ValueError(
                    "rank must be lesser than via_rank, must be r<k, got r>k"
                )
        return super().adjacency_matrix(rank, via_rank, s, index)

    def coadjacency_matrix(self, rank, via_rank, s: int = None, index: bool = False):
        """Compute the coadjacency matrix of self.

        The sparse weighted :term:`s-coadjacency matrix`

        Parameters
        ----------
        rank, via_rank : int
            Two ranks for skeletons in the input combinatorial complex , such that r>k.
        s : int, optional
            Minimum number of edges shared by neighbors with node.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the coadjacency
            matrix.

        Returns
        -------
        indices : list
            List identifying the rows and columns of the coadjacency matrix with the
            cells of the combinatorial complex. Only returned if `index` is True.
        coadjacency_matrix : scipy.sparse.csr.csr_matrix
            The coadjacency matrix of this combinatorial complex.
        """
        if via_rank is not None:
            if rank < via_rank:
                raise ValueError("rank must be greater than via_rank")
        return super().coadjacency_matrix(rank, via_rank, s, index)

    def dirac_operator_matrix(self, weight: str | None = None, index: bool = False):
        """Compute dirac operator matrix of self.

        Parameters
        ----------
        weight : str, optional
            The name of the cell attribute to use as weights for the dirac operator
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            If True return will include a dictionary of all cells in the complex uid

        Returns
        -------
        scipy.sparse.csr.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]
            The dirac operator matrix, if `index` is False, otherwise
        row_indices, col_indices : dict
            List identifying rows and columns of the dirac operator matrix. Only
            returned if `index` is True.
        dirac_matrix : scipy.sparse.csr.csc_matrix
            The dirac operator matrix of this combinatorial complex.

        Examples
        --------
        >>> from toponetx.classes import CombinatorialComplex
        >>> CCC = CombinatorialComplex()
        >>> CCC.add_cell([1, 2, 3, 4], rank=2)
        >>> CCC.add_cell([1, 2], rank=1)
        >>> CCC.add_cell([2, 3], rank=1)
        >>> CCC.add_cell([1, 4], rank=1)
        >>> CCC.add_cell([3, 4, 8], rank=2)
        >>> CCC.dirac_operator_matrix()
        """
        from scipy.sparse import bmat

        index_set = []
        incidence = {}
        for i in range(0, self.dim + 1):
            for j in range(i + 1, self.dim + 1):
                indexj, indexi, Bij = self.incidence_matrix(
                    i, j, weight=weight, index=True
                )
                incidence[(i, j)] = Bij
            index_set.append(indexj)
        index_set.append(indexi)
        dirac = []
        for i in range(0, self.dim + 1):
            row = []
            for j in range(0, self.dim + 1):
                if (i, j) in incidence:
                    row.append(incidence[(i, j)])
                elif (j, i) in incidence:
                    row.append(incidence[(j, i)].T)
                else:
                    row.append(None)
            dirac.append(row)
        dirac_mat = bmat(dirac)
        if index:
            d = {}
            shift = 0
            for i in index_set:
                i = {k: v + shift for k, v in i.items()}
                d.update(i)
                shift = len(d)

            return d, dirac_mat
        else:
            return dirac_mat

    def add_cells_from(self, cells, ranks: Iterable[int] | int | None = None) -> None:
        """Add cells to combinatorial complex.

        Parameters
        ----------
        cells : iterable of hashables
            For hashables the cells returned will be empty.
        ranks : iterable or int, optional
            When iterable, len(ranks) == len(cells).
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
                    raise ValueError(
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

    def add_cell(self, cell, rank=None, **attr) -> None:
        """Add a single cells to combinatorial complex.

        Parameters
        ----------
        cell : hashable, iterable or HyperEdge
            If hashable the cell returned will be empty.
        rank : int
            Rank of the cell.
        **attr : keyword arguments, optional
            Attributes to add to the cell as key=value pairs.
        """
        if self.graph_based:
            if rank == 1:
                if not isinstance(cell, Iterable):
                    raise TypeError(
                        "Rank 1 cells in graph-based CombinatorialComplex must be Iterable."
                    )
                if len(cell) != 2:
                    raise ValueError(
                        f"Rank 1 cells in graph-based CombinatorialComplex must have size equal to 1 got {cell}."
                    )

        self._add_hyperedge(cell, rank, **attr)

    def remove_cell(self, cell) -> None:
        """Remove a single cell from CCC.

        Parameters
        ----------
        cell : hashable or RankedEntity
            The cell to remove from this combinatorial complex.

        Notes
        -----
        Deletes reference to cell from all of its nodes.
        If any of its nodes do not belong to any other cells
        the node is dropped from self.
        """
        super().remove_cell(cell)

    def remove_cells(self, cell_set) -> None:
        """Remove cells from CCC.

        Parameters
        ----------
        cell_set : iterable of hashables
            The cells to remove from this combinatorial complex.
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
            A copy of this combinatorial complex.
        """
        CCC = CombinatorialComplex(name=self.name, graph_based=self.graph_based)
        for cell in self.cells:
            CCC.add_cell(cell, self.cells.get_rank(cell))
        return CCC

    def singletons(self):
        """Return a list of singleton cell.

        A singleton cell is a node of degree 0.

        Returns
        -------
        list
            A list of cells uids.

        Examples
        --------
        >>> CCC = CombinatorialComplex()
        >>> CCC.add_cell([1, 2], rank=1)
        >>> CCC.add_cell([3, 4], rank=1)
        >>> CCC.add_cell([9],rank=0)
        >>> CCC.singletons()
        """
        singletons = []
        for k in self.skeleton(0):
            if self.degree(tuple(k)[0], None) == 0:
                singletons.append(k)
        return singletons

    def remove_singletons(self):
        """Construct new CCC with singleton cells removed.

        Returns
        -------
        CombinatorialComplex
            A copy of this combinatorial complex with singleton cells removed.
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)
