"""Creation and manipulation of a Colored Hypergraph."""

import contextlib
from collections.abc import Collection, Hashable, Iterable
from itertools import chain
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse
import trimesh
from scipy.sparse import csr_array, diags
from typing_extensions import Self

from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.reportviews import ColoredHyperEdgeView, NodeView
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
    - X is a subset of the power set of S, and
    - c is the color function that associates a positive integer color or rank to each set x in X.

    A CHG is a generalization of graphs, combinatorial complexes, hypergraphs, cellular, and simplicial complexes.

    Parameters
    ----------
    cells : Collection, optional
        The initial collection of cells in the Colored Hypergraph.
    ranks : Collection, optional
        Represents the color of cells.
    **kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.

    Examples
    --------
    Define an empty colored hypergraph:

    >>> CHG = tnx.ColoredHyperGraph()

    Add cells to the colored hypergraph:

    >>> CHG = tnx.ColoredHyperGraph()
    >>> CHG.add_cell([1, 2], rank=1)
    >>> CHG.add_cell([3, 4], rank=1)
    >>> CHG.add_cell([1, 2, 3, 4], rank=2)
    >>> CHG.add_cell([1, 2, 4], rank=2)
    >>> CHG.add_cell([1, 2, 3, 4, 5, 6, 7], rank=3)

    Create a Colored Hypergraph and add groups of friends with corresponding ranks:

    >>> CHG = tnx.ColoredHyperGraph()
    >>> CHG.add_cell(
    ...     ["Alice", "Bob"], rank=1
    ... )  # Alice and Bob are in a close-knit group.
    >>> CHG.add_cell(["Charlie", "David"], rank=1)  # Another closely connected group.
    >>> CHG.add_cell(
    ...     ["Alice", "Bob", "Charlie", "David"], rank=2
    ... )  # Both groups together form a higher-ranked community.
    >>> CHG.add_cell(["Alice", "Bob", "David"], rank=2)  # Overlapping connections.
    >>> CHG.add_cell(
    ...     ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"], rank=3
    ... )  # A larger, more influential community.

    The code demonstrates how to represent social relationships using a Colored Hypergraph, where each group of friends (hyperedge) is assigned a rank based on the strength of the connection.
    """

    def __init__(
        self,
        cells: Collection | None = None,
        ranks: Collection | int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._complex_set = ColoredHyperEdgeView()

        if cells is not None:
            if not isinstance(cells, Iterable):
                raise TypeError(
                    f"Input cells must be given as Iterable, got {type(cells)}."
                )

            if not isinstance(cells, nx.Graph):
                if ranks is None:
                    for cell in cells:
                        if not isinstance(cell, HyperEdge):
                            self.add_cell(cell, rank=1)
                        else:
                            self.add_cell(cell, rank=cell.rank)
                else:
                    if isinstance(cells, Iterable) and isinstance(ranks, Iterable):
                        for cell, color in zip(cells, ranks, strict=True):
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
        """Object associated with self._cells.

        Returns
        -------
        HyperEdgeView
            HyperEdgeView of all cells and associated ranks.
        """
        return self._complex_set

    @property
    def nodes(self):
        """Object associated with self.elements.

        Returns
        -------
        NodeView
            NodeView of all nodes.
        """
        return NodeView(
            self._complex_set.hyperedge_dict.get(0, {}),
            cell_type=HyperEdge,
            colored_nodes=True,
        )

    @property
    def incidence_dict(self):
        """Return dict keyed by cell uids with values the uids of nodes in each cell.

        Returns
        -------
        dict
           Dictionary of cell uids with values.
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
            Tuple of number of cells in each rank.
        """
        return self._complex_set.shape

    def skeleton(self, rank: int):
        """Return skeleton.

        Parameters
        ----------
        rank : int
            Rank.

        Returns
        -------
        int
           Number of cells in specified rank.
        """
        return self._complex_set.skeleton(rank)

    @property
    def ranks(self):
        """Return the sorted list of ranks in the colored hypergraph.

        Returns
        -------
        list
            The sorted list of ranks.
        """
        return sorted(self._complex_set.allranks)

    @property
    def dim(self) -> int:
        """Return the dimension of the colored hypergraph.

        Returns
        -------
        int
            The dimension of the colored hypergraph.
        """
        if len(self._complex_set) == 0:
            return 0
        return max(self._complex_set.allranks)

    @property
    def __shortstr__(self) -> str:
        """Return the short string generic representation of the colored hypergraph.

        Returns
        -------
        str
            The short string generic representation.
        """
        return "CHG"

    def __str__(self) -> str:
        """Return a detailed string representation of the colored hypergraph.

        Returns
        -------
        str
            A detailed string representation.
        """
        return f"Colored Hypergraph with {len(self.nodes)} nodes and hyperedges with colors {self.ranks[1:]} and sizes {self.shape[1:]} "

    def __repr__(self) -> str:
        """Return a string representation of the colored hypergraph.

        Returns
        -------
        str
            A string representation.
        """
        return "ColoredHyperGraph()"

    def __len__(self) -> int:
        """Return the number of nodes in the colored hypergraph.

        Returns
        -------
        int
            The number of nodes.
        """
        return len(self.nodes)

    def __iter__(self):
        """Iterate over the nodes.

        Returns
        -------
        iterable
            Iterator over the nodes.
        """
        return chain(
            self.nodes,
            chain.from_iterable(
                rank_hyperedges.keys()
                for rank_hyperedges in self._complex_set.hyperedge_dict.values()
            ),
        )

    def __contains__(self, atom: Any) -> bool:
        """Check whether this colored hypergraph contains the given atom.

        Parameters
        ----------
        atom : Any
            The atom to be checked.

        Returns
        -------
        bool
            Returns `True` if this colored hypergraph contains the atom, else `False`.
        """
        return atom in self._complex_set

    def __getitem__(self, atom: Any) -> dict[Hashable, Any]:
        """Return the user-defined attributes associated with the given atom.

        Writing to the returned dictionary will update the user-defined attributes
        associated with the atom.

        Parameters
        ----------
        atom : Any
            The atom for which to return the associated user-defined attributes.

        Returns
        -------
        dict[Hashable, Any]
            The user-defined attributes associated with the given atom.

        Raises
        ------
        KeyError
            If the atom does not exist in this complex.
        """
        return self._complex_set[atom]

    def size(self, cell):
        """Compute the number of nodes in the colored hypergraph that belong to a specific cell.

        Parameters
        ----------
        cell : hashable or HyperEdge
            The cell for which to compute the size.

        Returns
        -------
        int
            The number of nodes in the colored hypergraph that belong to the specified cell.

        Raises
        ------
        ValueError
            If the input cell is not in the cells of the colored hypergraph.
        """
        if cell not in self.cells:
            raise ValueError(f"Input cell is not in cells of the {self.__shortstr__}")
        return len(self._complex_set[cell])

    def number_of_nodes(self, node_set=None):
        """Compute the number of nodes in node_set belonging to the CHG.

        Parameters
        ----------
        node_set : an interable of Entities, optional
            If None, then return the number of nodes in the CHG.

        Returns
        -------
        int
            Number of nodes in node_set belonging to the CHG.
        """
        if node_set:
            return len([node for node in node_set if node in self.nodes])
        return len(self.nodes)

    def number_of_cells(self, cell_set=None):
        """Compute the number of cells in the colored hypergraph.

        Parameters
        ----------
        cell_set : iterable of HyperEdge, optional
            If provided, computes the number of cells belonging to the specified cell_set.
            If None, returns the total number of cells in the hypergraph.

        Returns
        -------
        int
            The number of cells in the specified cell_set or the total number of cells if cell_set is None.
        """
        if cell_set:
            return len([cell for cell in cell_set if cell in self.cells])
        return len(self.cells)

    def order(self):
        """Compute the number of nodes in the CHG.

        Returns
        -------
        int
            The number of nudes in this hypergraph.
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
            Smallest size of cell to consider in degree.

        Returns
        -------
        int
            Number of cells of certain rank (or all ranks) that contain node.
        """
        if node not in self.nodes:
            raise KeyError(f"Node {node} not in {self.__shortstr__}.")
        if isinstance(rank, int):
            if rank >= 0:
                if rank in self._complex_set.hyperedge_dict:
                    return sum(
                        [
                            len(self._complex_set.hyperedge_dict[rank][x])
                            if node in x
                            else 0
                            for x in self._complex_set.hyperedge_dict[rank]
                            if len(x) >= s
                        ]
                    )

                raise RuntimeError(
                    f"There are no cells in the colored hypergraph with rank {rank}"
                )

            raise ValueError("Rank must be positive")
        if rank is None:
            rank_list = self._complex_set.hyperedge_dict.keys()
            value = 0
            for rank in rank_list:
                if rank == 0:
                    continue

                value += sum(
                    len(self._complex_set.hyperedge_dict[rank][x]) if node in x else 0
                    for x in self._complex_set.hyperedge_dict[rank]
                    if len(x) >= s
                )
            return value
        return None

    def _remove_node(self, node) -> None:
        """Remove a node from the ColoredHyperGraph.

        Parameters
        ----------
        node : HyperEdge or Hashable
            The node to be removed.

        Returns
        -------
        None
           None.

        Raises
        ------
        TypeError
            If the input node is neither a HyperEdge nor a hashable object.
        KeyError
            If the node is not present in the ColoredHyperGraph.
        """
        if isinstance(node, HyperEdge):
            pass
        elif isinstance(node, Hashable):
            node = HyperEdge([node])
        else:
            raise TypeError("node must be a HyperEdge or a hashable object")
        if node not in self.nodes:
            raise KeyError(f"node {node} not in {self.__shortstr__}")
        self._remove_node_helper(node)

    def remove_node(self, node) -> Self:
        """Remove a node from the ColoredHyperGraph.

        This method removes a node from the cells and deletes any reference in the nodes of the CHG.

        Parameters
        ----------
        node : HyperEdge or Hashable
            The node to be removed.

        Returns
        -------
        ColoredHyperGraph
            The ColoredHyperGraph instance after removing the node.
        """
        self._remove_node(node)
        return self

    def remove_nodes(self, node_set) -> None:
        """Remove nodes from cells.

        This also deletes references in colored hypergraph nodes.

        Parameters
        ----------
        node_set : an iterable of hashables
            The nodes to remove from this colored hypergraph.
        """
        copy_set = set()
        for node in node_set:
            if isinstance(node, Hashable):
                if isinstance(node, HyperEdge):
                    copy_set.add(next(iter(node.elements)))
                else:
                    copy_set.add(node)
                if node not in self.nodes:
                    raise KeyError(f"node {node} not in {self.__shortstr__}")
            else:
                raise TypeError("node {node} must be a HyperEdge or a hashable object")
        self._remove_node_helper(copy_set)

    def _remove_node_helper(self, node) -> None:
        """Remove node from cells.

        This function assumes that the node is present in the CHG.

        Parameters
        ----------
        node : hashable
            The node to be removed.

        Returns
        -------
        None
            None.
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

    def _add_hyperedge(self, hyperedge, rank, key=None, **attr):
        """Add hyperedge.

        Parameters
        ----------
        hyperedge : HyperEdge, Hashable, or Iterable
            A cell in a colored hypergraph.
        rank : int
            The rank of a hyperedge; must be zero when the hyperedge is Hashable.
        key : int, optional
            Used to distinguish colored hyperedges among nodes.
        **attr : attributes associated with hyperedge
            Attributes of the hyperedge.

        Returns
        -------
        None
            Adds a hyperedge to the ColoredHypergraph object.

        Notes
        -----
        The `_add_hyperedge` method is used for adding hyperedges to the HyperEdgeView instance.
        It takes three main arguments: hyperedge, rank, and key.
        - `hyperedge`: a tuple, HyperEdge instance, or Hashable representing the hyperedge to be added.
        - `rank`: an int representing the rank of the hyperedge.
        - `key`: an int used to distinguish colored hyperedges among nodes (optional).

        Additional attributes associated with the hyperedge can be provided through the **attr parameter.

        The method adds the hyperedge to the `hyperedge_dict` attribute of the HyperEdgeView instance,
        using the hyperedge's rank as the key and the hyperedge itself as the value.
        This allows the hyperedge to be accessed later using its rank.

        Note that the `_add_hyperedge` method checks whether the hyperedge being added is a valid hyperedge
        of the colored hypergraph by ensuring that the hyperedge's nodes are contained in the `_aux_complex`
        attribute of the HyperEdgeView instance. If the hyperedge's nodes are not in `_aux_complex`,
        the method will not add the hyperedge to `hyperedge_dict` to ensure that only valid hyperedges
        are included in the HyperEdgeView instance.
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
        elif isinstance(hyperedge, Iterable | HyperEdge):
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

    def _remove_hyperedge(self, hyperedge) -> None:
        """Remove a hyperedge from the CHG.

        Parameters
        ----------
        hyperedge : Hashable or HyperEdge
            The hyperedge to be removed from the CHG.

        Raises
        ------
        KeyError
            If the hyperedge is not present in the complex.
        """
        rank = self._complex_set.get_rank(hyperedge)
        del self._complex_set.hyperedge_dict[rank][frozenset(hyperedge)]

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
            None.
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
            None.
        """
        self._add_node(node, **attr)

    def set_node_attributes(self, values, name: str | None = None) -> None:
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
            Set the attributes of the nodes.
        """
        if name is not None:
            for cell, value in values.items():
                with contextlib.suppress(AttributeError):
                    self.nodes[cell][name] = value

        else:
            for cell, d in values.items():
                with contextlib.suppress(AttributeError):
                    self.nodes[cell].update(d)
            return

    def set_cell_attributes(self, values, name: str | None = None) -> None:
        """Set cell attributes.

        Parameters
        ----------
        values : dict
            Dictionary of attributes to set for the cell.
        name : str, optional
            Name of the attribute to set for the cell.

        Returns
        -------
        None
            Set the attributes of the cells.

        Examples
        --------
        After computing some property of the cell of a Colored Hypergraph, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CHG = tnx.ColoredHyperGraph()
        >>> CHG.add_cell([1, 2, 3, 4], rank=2)
        >>> CHG.add_cell([1, 2, 4], rank=2)
        >>> CHG.add_cell([3, 4], rank=2)
        >>> d = {((1, 2, 3, 4), 0): "red", ((1, 2, 4), 0): "blue", ((3, 4), 0): "green"}
        >>> CHG.set_cell_attributes(d, name="color")
        >>> CHG.cells[((3, 4), 0)]["color"]
        'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes:

        >>> G = nx.path_graph(3)
        >>> CHG = tnx.ColoredHyperGraph(G)
        >>> d = {
        ...     ((1, 2), 0): {"color": "red", "attr2": 1},
        ...     ((0, 1), 0): {"color": "blue", "attr2": 3},
        ... }
        >>> CHG.set_cell_attributes(d)
        >>> CHG.cells[((0, 1), 0)]["color"]
        'blue'

        Note that if the dict contains cells that are not in `self.cells`, they are
        silently ignored.
        """
        if name is not None:
            for cell, value in values.items():
                with contextlib.suppress(AttributeError):
                    self.cells[cell][name] = value
        else:
            for cell, d in values.items():
                with contextlib.suppress(AttributeError):
                    self.cells[cell].update(d)
            return

    def get_node_attributes(self, name: str):
        """Get node attributes.

        Parameters
        ----------
        name : str
           Attribute name.

        Returns
        -------
        dict
            Dictionary of attributes keyed by node.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CHG = tnx.ColoredHyperGraph(G)
        >>> d = {0: {"color": "red", "attr2": 1}, 1: {"color": "blue", "attr2": 3}}
        >>> CHG.set_node_attributes(d)
        >>> CHG.get_node_attributes("color")
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CHG = tnx.ColoredHyperGraph(G)
        >>> nodes_color = CHG.get_node_attributes("color")
        >>> nodes_color[1]
        'blue'
        """
        return {
            next(iter(node)): self.nodes[node][name]
            for node in self.nodes
            if name in self.nodes[node]
        }

    def get_cell_attributes(self, name: str, rank=None):
        """Get cell attributes from the colored hypergraph.

        Parameters
        ----------
        name : str
            Attribute name.
        rank : int, optional
            Rank of the k-cell. If specified, the function returns attributes for k-cells with the given rank.
            If not provided, the function returns attributes for all cells.

        Returns
        -------
        dict
            Dictionary of attributes keyed by cell or k-cells if rank is not None.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CHG = tnx.ColoredHyperGraph(G)
        >>> d = {
        ...     ((1, 2), 0): {"color": "red", "attr2": 1},
        ...     ((0, 1), 0): {"color": "blue", "attr2": 3},
        ... }
        >>> CHG.set_cell_attributes(d)
        >>> cell_color = CHG.get_cell_attributes("color")
        >>> cell_color[(frozenset({0, 1}), 0)]
        'blue'
        """
        if rank is not None:
            return {
                cell: self.skeleton(rank)[cell][name]
                for cell in self.skeleton(rank)
                if name in self.skeleton(rank)[cell]
            }

        return {
            cell: self.cells[cell][name]
            for cell in self.cells
            if name in self.cells[cell]
        }

    def _add_nodes_of_hyperedge(self, hyperedge_) -> None:
        """Add nodes of a hyperedge to the CHG.

        Parameters
        ----------
        hyperedge_ : Hashable or HyperEdge
            The hyperedge whose nodes are to be added.

        Notes
        -----
        This method is used to add the nodes of a hyperedge to the CHG.
        It updates the hyperedge dictionary in the complex set, assigning a default weight of 1 to each node.
        """
        if 0 not in self._complex_set.hyperedge_dict:
            self._complex_set.hyperedge_dict[0] = {}
        for i in hyperedge_:
            if i not in self._complex_set.hyperedge_dict[0]:
                self._complex_set.hyperedge_dict[0][frozenset({i})] = {}
                self._complex_set.hyperedge_dict[0][frozenset({i})][0] = {"weight": 1}

    def new_hyperedge_key(self, hyperedge, rank):
        """Add a new key for the given hyperedge.

        Notes
        -----
        In the standard ColoredHyperGraph class, the new key is determined by
        counting the number of existing hyperedges between the nodes that define
        the input hyperedge. The count is increased if necessary to ensure the key is unused.
        The first hyperedge will have key 0, then 1, and so on. If a hyperedge is removed,
        further new_hyperedge_keys may not be in this order.

        Parameters
        ----------
        hyperedge : tuple or HyperEdge
            Representing node elements of the hyperedge.
        rank : int
            Rank (color) of the input hyperedge.

        Returns
        -------
        int
            The new key assigned to the hyperedge.
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
        """Add hyperedge to the ColoredHyperGraph.

        Parameters
        ----------
        hyperedge_ : frozenset of hashable elements
            The hyperedge to be added.
        rank : int
            The rank (color) of the hyperedge.
        key : hashable identifier, optional (default=None)
            Used to distinguish hyperedges among nodes.
        **attr : arbitrary attrs
            Additional attributes associated with the hyperedge.

        Returns
        -------
        None
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
        """Add a single cell to Colored Hypergraph.

        Parameters
        ----------
        cell : hashable, iterable, or HyperEdge
            If hashable, the cell returned will be empty.
        rank : int
            The rank of a cell.
        key : int, optional
            Used to distinguish colored hyperedges among nodes.
        **attr : attributes associated with hyperedge
            Attributes of the hyperedge.

        Returns
        -------
        ColoredHyperGraph
            The modified Colored Hypergraph.

        Notes
        -----
        The add_cell method adds a cell to the Colored Hypergraph instance. The cell can be a hashable, iterable, or HyperEdge.
        If the cell is hashable, the resulting cell will be empty. The rank parameter specifies the rank of the cell.
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
            For hashables, the cells returned will be empty.
        ranks : Iterable or int
            When iterable, len(ranks) == len(cells).

        Returns
        -------
        None
            None.
        """
        if ranks is None:
            for cell in cells:
                if isinstance(cell, HyperEdge):
                    self.add_cell(cell, cell.rank)
                else:
                    self.add_cell(cell, rank=None)
        else:
            if isinstance(cells, Iterable) and isinstance(ranks, Iterable):
                for cell, rank in zip(cells, ranks, strict=True):
                    self.add_cell(cell, rank)
        if isinstance(cells, Iterable) and isinstance(ranks, int):
            for cell in cells:
                self.add_cell(cell, ranks)

    def remove_cell(self, cell) -> None:
        """Remove a single cell from the ColoredHyperGraph.

        Parameters
        ----------
        cell : Hashable or RankedEntity
            The cell to be removed.

        Returns
        -------
        None
            The cell is removed in place.

        Notes
        -----
        Deletes the reference to the cell from all of its nodes.
        If any of its nodes do not belong to any other cells,
        the node is dropped from the ColoredHyperGraph.
        """
        self._remove_hyperedge(cell)

    def get_incidence_structure_dict(self, i, j):
        """Get the incidence structure dictionary for cells of rank i and j.

        Parameters
        ----------
        i : int
            The rank (color) of the first set of cells.
        j : int
            The rank (color) of the second set of cells.

        Returns
        -------
        dict
            The incidence structure dictionary representing the relationship between cells of rank i and j.

        Notes
        -----
        The incidence structure dictionary has cells of rank i as keys and lists of cells of rank j incident to them as values.
        """
        return sparse_array_to_neighborhood_dict(self.incidence_matrix(i, j))

    def get_adjacency_structure_dict(self, i, j):
        """Get the adjacency structure dictionary for cells of rank i and j.

        Parameters
        ----------
        i : int
            The rank (color) of the first set of cells.
        j : int
            The rank (color) of the second set of cells.

        Returns
        -------
        dict
            The adjacency structure dictionary representing the relationship between cells of rank i and j.

        Notes
        -----
        The adjacency structure dictionary has cells of rank i as keys and lists of cells of rank j adjacent to them as values.
        """
        return sparse_array_to_neighborhood_dict(self.adjacency_matrix(i, j))

    def remove_cells(self, cell_set) -> None:
        """Remove cells from this colored hypergraph.

        Parameters
        ----------
        cell_set : iterable of hashables
            The cells to remove from this colored hypergraph.
        """
        for cell in cell_set:
            self.remove_cell(cell)

    def incidence_matrix(
        self,
        rank: int,
        to_rank: int,
        weight: str | None = None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute incidence matrix for the CHG indexed by nodes x cells.

        An incidence matrix indexed by r-ranked hyperedges and k-ranked hyperedges
        where r != k. When k is None, incidence_type will be considered instead.

        Parameters
        ----------
        rank : int
            The rank for computing the incidence matrix.
        to_rank : int
            The rank for computing the incidence matrix.
        weight : str, default=None
            The attribute to use as weight. If `None`, all weights are considered to be
            one.
        sparse : bool, default=True
            Whether to return a sparse or dense incidence matrix.
        index : bool, default=False
            If True, return will include a dictionary of node uid: row number
            and cell uid: column number.

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray
            The incidence matrix.
        row dictionary : dict
            Dictionary identifying rows with nodes.
        column dictionary : dict
            Dictionary identifying columns with cells.
        """
        if rank == to_rank:
            raise ValueError("incidence must be computed for k!=r, got equal r and k.")
        children = self.skeleton(rank)
        uidset = self.skeleton(to_rank)

        return compute_set_incidence(children, uidset, sparse, index)

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
            and cell uid : column number.

        Returns
        -------
        scipy.sparse.csr.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]
            The incidence matrix, if `index` is False, otherwise
            lower (row) index dict, upper (col) index dict, incidence matrix
            where the index dictionaries map from the entity (as `Hashable` or `tuple`) to the row or col index of the matrix.
        """
        if index:
            (
                row_indices,
                col_indices,
                incidence_matrix,
            ) = self.all_ranks_incidence_matrix(0, weight=weight, index=index)
            row_indices = {next(iter(k)): v for k, v in row_indices.items()}
            return row_indices, col_indices, incidence_matrix
        return self.all_ranks_incidence_matrix(0, weight=weight, index=index)

    def all_ranks_incidence_matrix(
        self,
        rank,
        weight: str | None = None,
        sparse: bool = True,
        index: bool = False,
    ):
        """Compute the incidence matrix for the Colored Hypergraph indexed by cells of rank `n` and all other cells.

        Parameters
        ----------
        rank : int
            The rank which filters the cells based on which the incidence matrix is computed.
        weight : str, optional
            If not given, all nonzero entries are 1.
        sparse : bool, optional
            The output will be sparse.
        index : bool, optional, default False
            If True, the return will include a dictionary of node uid : row number
            and cell uid : column number.

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray
            The incidence matrix.
        row dictionary : dict
            Dictionary identifying rows with nodes.
        column dictionary : dict
            Dictionary identifying columns with cells.

        Notes
        -----
        The all_ranks_incidence_matrix method computes the incidence matrix for the Colored Hypergraph,
        focusing on cells of rank `n` and all other cells. The weight parameter allows specifying the
        weight of the entries in the matrix. If index is True, dictionaries mapping node and cell identifiers
        to row and column numbers are included in the return.
        """
        children = self.skeleton(rank)
        all_other_ranks = []
        for i in self.ranks:
            if i is not rank:
                all_other_ranks = all_other_ranks + self.skeleton(i)
        return compute_set_incidence(children, all_other_ranks, sparse, index)

    def adjacency_matrix(self, rank, via_rank, s: int = 1, index: bool = False):
        """Sparse weighted s-adjacency matrix.

        Parameters
        ----------
        rank, via_rank : int, int
            Two ranks for skeletons in the input Colored Hypergraph.
        s : int, list, optional
            Minimum number of edges shared by neighbors with node.
        index : bool, optional
            If True, will return a rowdict of row to node uid.

        Returns
        -------
        row dictionary : dict
            Dictionary identifying rows with nodes. If False, this does not exist.
        adjacency_matrix : scipy.sparse.csr.csr_matrix
            The adjacency matrix.

        Examples
        --------
        >>> G = nx.Graph()  # networkx graph
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0, 3)
        >>> G.add_edge(0, 4)
        >>> G.add_edge(1, 4)
        >>> CHG = tnx.ColoredHyperGraph(cells=G)
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

    def all_cell_to_node_coadjacency_matrix(self, index: bool = False, s: int = 1):
        """Compute the cell co-adjacency matrix.

        Parameters
        ----------
        index : bool, optional, default=False
            If True, return a row dictionary of row to node uid.
        s : int, default=1
            Minimum number of edges shared by neighbors with a node.

        Returns
        -------
        row dictionary : dict
            Dictionary identifying rows with nodes. If False, this does not exist.
        all cells co-adjacency matrix : scipy.sparse.csr.csr_matrix
            Cell co-adjacency matrix.
        """
        B = self.node_to_all_cell_incidence_matrix(index=index)
        if index:
            A = incidence_to_adjacency(B[-1], s=s)
            return B[1], A
        return incidence_to_adjacency(B, s=s)

    def node_to_all_cell_adjacnecy_matrix(self, index: bool = False, s: int = 1):
        """Compute the node/all cell adjacency matrix.

        Parameters
        ----------
        index : bool, optional
            If True, will return a rowdict of row to node uid.
        s : int, default=1
            Minimum number of edges shared by neighbors with the node.

        Returns
        -------
        row dictionary : dict
            Dictionary identifying rows with nodes. If False, this does not exist.
        cell adjacency matrix : scipy.sparse.csr.csr_matrix
            The cell adjacency matrix.
        """
        B = self.node_to_all_cell_incidence_matrix(index=index)
        if index:
            A = incidence_to_adjacency(B[-1].T, s=s)
            return B[0], A
        return incidence_to_adjacency(B.T, s=s)

    def coadjacency_matrix(self, rank, via_rank, s: int = 1, index: bool = False):
        """Compute the coadjacency matrix.

        Parameters
        ----------
        rank : int
            The rank for the primary skeleton in the input Colored Hypergraph.
        via_rank : int
            The rank for the secondary skeleton in the input Colored Hypergraph.
        s : int, optional
            Minimum number of edges shared by neighbors with the node.
        index : bool, optional
            If True, return will include a dictionary of row number to node uid.

        Returns
        -------
        row dictionary : dict
            Dictionary identifying rows with nodes. If False, this does not exist.
        co-adjacency_matrix : scipy.sparse.csr.csr_matrix
            The co-adjacency matrix.
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
        """Compute the node-degree matrix.

        Parameters
        ----------
        rank : int
            The rank (color) in the Colored Hypergraph to which the degree matrix is computed.
        index : bool, default: False
            If True, return will include a dictionary of row number to node uid.

        Returns
        -------
        row dictionary : dict
            Dictionary identifying rows with nodes. If False, this does not exist.
        degree_matrix : scipy.sparse.csr.csr_matrix
            The degree matrix.

        Raises
        ------
        ValueError
            If the rank is not in the range of the Colored Hypergraph.
        """
        if not self.dim >= rank > 0:
            raise ValueError(
                f"Cannot compute degree matrix for rank {rank} in a {self.dim}-dimensional complex."
            )

        rowdict, _, M = self.incidence_matrix(0, rank, index=True)
        D = np.ravel(np.sum(M, axis=1))
        return (rowdict, D) if index else D

    def laplacian_matrix(self, rank, sparse=False, index=False):
        """Compute Laplacian matrix.

        Parameters
        ----------
        rank : int
            The rank (or color) in the complex to which the Laplacian matrix is computed.
        sparse : bool, default: False
            Specifies whether the output matrix is a scipy sparse matrix or a numpy matrix.
        index : bool, default: False
            If True, return will include a dictionary of node uid: row number.

        Returns
        -------
        numpy.ndarray, scipy.sparse.csr.csr_matrix
            Array of dimension (N, N), where N is the number of nodes in the Colored Hypergraph.
            If index is True, return a dictionary mapping node uid to row number.

        Raises
        ------
        ValueError
            If the rank is not in the range of the Colored Hypergraph.

        References
        ----------
        .. [1] Lucas, M., Cencetti, G., & Battiston, F. (2020).
            Multiorder Laplacian for synchronization in higher-order networks.
            Physical Review Research, 2(3), 033410.
        """
        if not 0 < rank <= self.dim:
            raise ValueError(
                f"Cannot compute Laplacian matrix for rank {rank} in a {self.dim}-dimensional complex."
            )

        if len(self.nodes) == 0:
            A = np.empty((0, 0))
        else:
            row_dict, A = self.adjacency_matrix(0, rank, index=True)

        if sparse:
            K = csr_array(diags(self.degree_matrix(rank)))
        else:
            K = np.diag(self.degree_matrix(rank))
        L = K - A

        return (row_dict, L) if index else L

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        """Import from a trimesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The trimesh object to import.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(
        ...     vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        ...     faces=[[0, 1, 2]],
        ...     process=False,
        ... )
        >>> CHG = tnx.ColoredHyperGraph.from_trimesh(mesh)
        >>> CHG.nodes
        """
        raise NotImplementedError

    def restrict_to_cells(self, cell_set):
        """Construct a Colored Hypergraph using a subset of the cells.

        Parameters
        ----------
        cell_set : hashable
            A subset of elements of the Colored Hypergraph cells.

        Returns
        -------
        ColoredHyperGraph
            The Colored Hypergraph constructed from the specified subset of cells.
        """
        chg = self.__class__()
        valid_cells = [c for c in cell_set if c in self.cells]
        for c in valid_cells:
            if not isinstance(c, Iterable):
                raise ValueError(f"each element in cell_set must be Iterable, got {c}")
            if isinstance(c, tuple):
                chg.add_cell(c[0], rank=self.cells.get_rank(c[0]))
            else:
                chg.add_cell(c, rank=self.cells.get_rank(c))
        return chg

    def restrict_to_nodes(self, node_set):
        """Restrict to a set of nodes.

        Constructs a new Colored Hypergraph by restricting the
        cells in the Colored Hypergraph to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set : iterable of hashables
            References a subset of elements of self.nodes.

        Returns
        -------
        ColoredHyperGraph
            A new Colored Hypergraph restricted to the specified node_set.
        """
        chg = self.__class__()
        node_set = frozenset(node_set)
        for i in self.ranks:
            if i != 0:
                for cell in self.skeleton(i):
                    c_set = cell if isinstance(cell, frozenset) else cell[0]
                    if c_set <= node_set:
                        chg.add_cell(c_set, rank=i)
        return chg

    def from_networkx_graph(self, G) -> None:
        """Construct a Colored Hypergraph from a networkx graph.

        Parameters
        ----------
        G : NetworkX graph
            A networkx graph.

        Returns
        -------
        None
            The method modifies the current Colored Hypergraph.

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0, 4)
        >>> G.add_edge(0, 7)
        >>> CHG = tnx.ColoredHyperGraph()
        >>> CHG.from_networkx_graph(G)
        >>> CHG.nodes
        NodeView([(0,), (1,), (4,), (7,)])
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
        list
            A list of cells uids.
        """
        singletons = []
        for cell in self.cells:
            zero_elements = cell[0]
            if len(zero_elements) == 1:
                singletons.extend(
                    cell for n in zero_elements if self.degree(n, None) == 1
                )
        return singletons

    def remove_singletons(self):
        """Construct new CHG with singleton cells removed.

        Returns
        -------
        ColoredHyperGraph
            Return new CHG with singleton cells removed.
        """
        cells = [cell for cell in self.cells if cell not in self.singletons()]
        return self.restrict_to_cells(cells)

    def clone(self) -> Self:
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex
        and attributes. That is, if an attribute is a container, that container is
        shared by the original and the copy. Use Python's `copy.deepcopy` for new
        containers.

        Returns
        -------
        ColoredHyperGraph
            ColoredHyperGraph.
        """
        CHG = self.__class__()
        for cell, key in self.cells:
            CHG.add_cell(cell, key=key, rank=self.cells.get_rank(cell))
        return CHG
