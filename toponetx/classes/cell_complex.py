"""Creation and manipulation of a 2d cell complex.

The class also supports attaching arbitrary attributes and data to cells.

A cell complex is abbreviated in CC.
"""

import warnings
from collections import defaultdict
from collections.abc import Collection, Hashable, Iterable, Iterator
from itertools import chain, zip_longest
from typing import Any

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
from networkx import Graph
from networkx.classes.reportviews import EdgeView, NodeView
from networkx.utils import pairwise
from typing_extensions import Self, deprecated

from toponetx.classes.cell import Cell
from toponetx.classes.combinatorial_complex import (
    ColoredHyperGraph,
    CombinatorialComplex,
)
from toponetx.classes.complex import Complex
from toponetx.classes.reportviews import CellView
from toponetx.utils import incidence_to_adjacency

__all__ = ["CellComplex"]

try:
    from hypernetx import Hypergraph
    from hypernetx.classes.entity import Entity
except ImportError:
    Hypergraph = None
    Entity = None


class CellComplex(Complex):
    r"""Class representing a cell complex.

    A cell complex is a mathematical structure that is built up from simple building blocks called cells.
    These cells can be thought of as generalized versions of familiar shapes, such as points, line segments,
    triangles, and disks. By gluing these cells together in a prescribed way, one can create complex
    geometrical objects that are of interest in topology and geometry.

    Cell complexes can be used to represent various mathematical objects, such as graphs,
    manifolds, and discrete geometric shapes. They are useful in many areas of mathematics,
    such as algebraic topology and geometry, where they can be used to study the structure and
    properties of these objects.

    In TNX the class CellComplex supports building a regular or non-regular
    2d cell complex. The class CellComplex only supports the construction
    of 2d cell complexes. If higher dimensional cell complexes are desired
    then one should utilize the class CombinatorialComplex.

    Mathematically, in TNX a cell complex it a triplet $(V, E, C)$
    where $V$ is a set of nodes, $E$ is a set of edges and $C$ is a set of 2-cells.
    In TNX each 2-cell C is consists of a finite sequence of nodes $C=(n_1,...,n_k,n_1)$ with $k\geq 2$.
    All edges between two consecutive nodes in C belong to  E.
    Regular cells have unique nodes in C whereas non-regular cells allow for duplication.

    In TNX, cell complexes are implemented to be dynamic in the sense that
    they can change by adding or subtracting objects (nodes, edges, 2-cells)
    from them.

    1. Dynamic construction of cell complexes, allowing users to add or remove objects from these
        structures after their initial creation.
    2. Compatibility with the NetworkX library, enabling users to leverage the powerful algorithms
        and data structures provided by this package.
    3. Support for attaching arbitrary attributes and data to cells in the complex, allowing users to store
        and manipulate additional information about these objects.
    4. Efficient storage and manipulation of complex data structures, using advanced data structures
        such as sparse matrices.
    5. Robust error handling and validation of input data, ensuring that the package is reliable and easy to use.

    Parameters
    ----------
    cells : iterable, optional
        A list of cells to add to the complex.
    regular : bool, default=True
        If True, then the complex is regular, otherwise it is non-regular.
    **kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.

    Examples
    --------
    Iteratively construct a cell complex:

    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> # the cell [1, 2, 3, 4] consists of the cycle (1,2), (2,3), (3,4), (4,5)
    >>> # tnx creates these edges automatically if they are not inserted in the underlying graph
    >>> CC.add_cell([2, 3, 4, 5], rank=2)
    >>> CC.add_cell([5, 6, 7, 8], rank=2)

    You can also pass a list of cells to the constructor:

    >>> c1 = tnx.Cell((1, 2, 3))  # a cell here is always assumed to be 2d
    >>> c2 = tnx.Cell((1, 2, 3, 4))
    >>> CC = tnx.CellComplex([c1, c2])

    TopoNetX is also compatible with NetworkX, allowing users to create a cell complex from a NetworkX graph:

    >>> g = nx.Graph()
    >>> g.add_edge(1, 0)
    >>> g.add_edge(2, 0)
    >>> g.add_edge(1, 2)
    >>> CC = tnx.CellComplex(g)
    >>> CC.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
    >>> CC.cells

    By default, a regular cell complex is constructed. You can change this behaviour using the
    `regular` parameter when constructing the complex.

    >>> # non-regular cell complex
    >>> # by default CellComplex constructor assumes regular cell complex
    >>> CC = tnx.CellComplex(regular=False)
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> CC.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
    >>> c1 = tnx.Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
    >>> CC.add_cell(c1)
    >>> CC.add_cell([5, 6, 7, 8], rank=2)
    >>> CC.is_regular
    """

    def __init__(self, cells=None, regular: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        self._regular = regular
        self._G = Graph()

        self._cells = CellView()
        if cells is not None:
            if isinstance(cells, Graph):
                self._G = cells
            elif isinstance(cells, Iterable) and not isinstance(cells, Graph):
                for cell in cells:
                    if isinstance(cell, Hashable) and not isinstance(
                        cell, Iterable
                    ):  # c is a node
                        self.add_node(cell)
                    elif isinstance(cell, Iterable):
                        if len(cell) == 2:
                            self.add_cell(cell, rank=1)
                        elif len(cell) == 1:
                            self.add_node(next(iter(cell)))
                        else:
                            self.add_cell(cell, rank=2)

            else:
                raise TypeError(
                    f"cells must be iterable, networkx graph or None, got {type(cells)}"
                )

    @property
    def cells(self) -> CellView:
        """Return cells.

        Returns
        -------
        CellView
            The cell view of the Complex.
        """
        return self._cells

    @property
    def edges(self) -> EdgeView:
        """Return edges.

        Returns
        -------
        EdgeView
            The edge view of the Complex.
        """
        return self._G.edges

    @property
    def nodes(self) -> NodeView:
        """Return nodes.

        Returns
        -------
        NodeView
            The node view of the Complex.
        """
        return self._G.nodes

    @property
    @deprecated(
        "`CellComplex.maxdim` is deprecated and will be removed in the future, use `CellComplex.dim` instead."
    )
    def maxdim(self) -> int:
        """Return maximum dimension.

        Returns
        -------
        int
            The maximum dimension for Cell Complex.
        """
        return self.dim

    @property
    def dim(self) -> int:
        """Return maximum dimension.

        Returns
        -------
        int
            The maximum dimension of the Cell Complex.
        """
        if len(self.nodes) == 0:
            return 0
        if len(self.edges) == 0:
            return 0
        if len(self.cells) == 0:
            return 1
        return 2

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return shape.

        Returns
        -------
        tuple[int, int, int]
            The tuple containing the number of entities corresponding to the
            nodes, edges, and cells of the Cell Complex respectively.
        """
        return len(self.nodes), len(self.edges), len(self.cells)

    def skeleton(self, rank: int):
        """Compute skeleton.

        Parameters
        ----------
        rank : {0, 1, 2}
            The rank of the skeleton.

        Returns
        -------
        Iterable
            The atoms of given `rank` in this complex.

        Raises
        ------
        ValueError
            If `rank` is not 0, 1 or 2.
        """
        if rank == 0:
            return self.nodes
        if rank == 1:
            return self.edges
        if rank == 2:
            return self.cells
        raise ValueError("Only dimensions 0,1, and 2 are supported.")

    @property
    def is_regular(self) -> bool:
        """Check the regularity condition.

        Returns
        -------
        bool
            Returns `True` if the regularity condition is satisfied.

        Examples
        --------
        >>> CC = tnx.CellComplex(regular=False)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        >>> c1 = tnx.Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        >>> CC.add_cell(c1)
        >>> CC.add_cell([5, 6, 7, 8], rank=2)
        >>> CC.is_regular
        False
        """
        return all(cell.is_regular for cell in self.cells)

    def __str__(self) -> str:
        """Return detailed string representation.

        Returns
        -------
        str
            The string representation of the Cell Complex.
        """
        return f"Cell Complex with {len(self.nodes)} nodes, {len(self.edges)} edges and {len(self.cells)} 2-cells "

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            The __repr__ representation of the Cell Complex.
        """
        return "CellComplex()"

    def __len__(self) -> int:
        """Return number of nodes.

        Returns
        -------
        int
            The number of nodes in the Cell Complex.
        """
        return len(self.nodes)

    def __iter__(self) -> Iterator:
        """Iterate over the nodes.

        Returns
        -------
        dict_keyiterator
            Iterator over nodes.
        """
        return chain(
            ((x,) for x in self.nodes),
            self.edges,
            (tuple(cell) for cell in self.cells),
        )

    def __contains__(self, item) -> bool:
        """Return boolean indicating if item is in self.nodes, self.edges or self.cells.

        Parameters
        ----------
        item : hashable
            Iterm.

        Returns
        -------
        bool
            True if item is a cell of dim 0,1,2 is in the complex, False otherwise.
        """
        if not isinstance(item, Iterable):
            return item in self._G.nodes

        if len(item) == 2 and item in self._G.edges:
            return True
        return item in self._cells

    def __getitem__(self, atom: Any) -> dict[Hashable, Any]:
        """Return the user-defined attributes associated with the given cell.

        Writing to the returned dictionary will update the user-defined attributes
        associated with the cell.

        Parameters
        ----------
        atom : Any
            The cell for which to return the associated user-defined attributes.

        Returns
        -------
        dict[Hashable, Any]
            The user-defined attributes associated with the given cell.
        """
        if isinstance(atom, Collection) and len(atom) == 1:
            atom = next(iter(atom))

        if not isinstance(atom, Iterable):
            return self._G.nodes[atom]

        atom = tuple(atom)
        if len(atom) == 2 and atom in self._G.edges:
            return self._G.edges[atom]

        return self._cells[atom]

    def _insert_cell(self, cell: tuple | list | Cell, **attr):
        """Insert cell.

        Parameters
        ----------
        cell : tuple | list | Cell
            The cell to insert.
        **attr
            The attributes of the cell to insert.
        """
        # input must be list, tuple or Cell type
        if isinstance(cell, tuple | list | Cell):
            if isinstance(cell, tuple | list):
                cell = Cell(elements=cell, **attr)
            elif isinstance(cell, Cell):
                cell.update(attr)

            if cell.elements not in self._cells._cells:
                self._cells._cells[cell.elements] = {0: cell}
            else:
                # if cell is already in the complex, insert duplications and give it differnt key
                self._cells._cells[cell.elements][
                    len(self._cells._cells[cell.elements])
                ] = cell
        else:
            raise TypeError("input must be list, tuple or Cell type")

    def _delete_cell(self, cell: tuple | list | Cell, key=None):
        """Delete cell.

        Parameters
        ----------
        cell : tuple | list | Cell
            The cell to delete.
        key : Hashable
            The key of the cell to delete.
        """
        if isinstance(cell, Cell):
            cell = cell.elements
        if cell in self._cells._cells:
            if key is None:
                del self._cells._cells[cell]
            elif key in self._cells._cells[cell]:
                del self._cells._cells[cell][key]
            else:
                raise KeyError(f"cell with key {key} is not in the complex ")
        else:
            raise KeyError(f"cell {cell} is not in the complex")

    def _cell_equivalence_class(self) -> dict[Cell, set[Cell]]:
        """Return the equivalence classes of cells in the cell complex.

        Returns
        -------
        dict[Cell, set[Cell]]
            Dict structure: `Cell` representing equivalence class -> Set of all `Cell`s in class.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell((1, 2, 3, 4), rank=2)
        >>> CC.add_cell((2, 3, 4, 1), rank=2)
        >>> CC.add_cell((1, 2, 3, 4), rank=2)
        >>> CC.add_cell((1, 2, 3, 6), rank=2)
        >>> CC.add_cell((3, 4, 1, 2), rank=2)
        >>> CC.add_cell((4, 3, 2, 1), rank=2)
        >>> CC.add_cell((1, 2, 7, 3), rank=2)
        >>> c1 = tnx.Cell((1, 2, 3, 4, 5))
        >>> CC.add_cell(c1, rank=2)
        >>> CC._cell_equivalence_class()
        """
        equiv_classes = defaultdict(set)
        all_inserted_cells = set()
        for i, c1 in enumerate(self.cells):
            for j, c2 in enumerate(self.cells):
                if i == j:
                    if j not in all_inserted_cells:
                        equiv_classes[c1].add(j)
                elif i > j or j in all_inserted_cells:
                    continue
                else:
                    if c1.is_homotopic_to(c2):
                        equiv_classes[c1].add(j)
                        all_inserted_cells.add(j)
        return equiv_classes

    def degree(self, node: Hashable, rank: int = 1) -> int:
        """Compute the number of cells of certain rank that contain node.

        Parameters
        ----------
        node : hashable
            Identifier for the node.
        rank : int, default=1
            Smallest size of cell to consider in degree.

        Returns
        -------
        int
            Number of cells of rank at least rank that contain node.
        """
        if rank > 1:
            raise NotImplementedError(
                f"Rank {rank} is currently not supported by degree."
            )
        return self._G.degree[node]

    def size(
        self,
        cell: tuple | list | Cell,
        node_set: Iterable[Hashable] | None = None,
    ) -> int:
        """Compute number of nodes in node_set that belong to cell.

        If node_set is None then returns the size of cell.

        Parameters
        ----------
        cell : hashable
            The uid of an cell in the cell complex.
        node_set : an iterable of node elements
            Node elements.

        Returns
        -------
        int
            Number of nodes in node_set that belong to cell.
        """
        if node_set:
            if isinstance(cell, Cell):
                return len(set(node_set).intersection(set(cell.elements)))
            if isinstance(cell, Iterable):
                return len(set(node_set).intersection(set(cell)))
            return None

        if cell in self.cells:
            return len(cell)
        raise KeyError(f" the key {cell} is not a key for an existing cell ")

    def number_of_nodes(self, node_set: Iterable[Hashable] | None = None):
        """Compute number of nodes in node_set belonging to cell complex.

        Parameters
        ----------
        node_set : an iterable of nodes, optional
            If None, then return the number of nodes in cell complex.

        Returns
        -------
        int
            Number of nodes in node_set belonging to cell complex.
        """
        if node_set:
            return len([node for node in self.nodes if node in node_set])
        return len(self.nodes)

    def number_of_edges(self, edge_set: Iterable[tuple] | None = None) -> int:
        """Compute number of edges in edge_set belonging to cell complex.

        Parameters
        ----------
        edge_set : an iterable of edges, optional
            If None, then return the number of edges in cell complex.

        Returns
        -------
        int
            The number of edges in edge_set belonging to cell complex.
        """
        if edge_set:
            return len(
                [
                    edge
                    for edge in self.edges
                    if edge in edge_set or edge[::-1] in edge_set
                ]
            )
        return len(self.edges)

    def number_of_cells(
        self, cell_set: Iterable[tuple | list | Cell] | None = None
    ) -> int:
        """Compute the number of cells in cell_set belonging to cell complex.

        Parameters
        ----------
        cell_set : iterable of cells, optional
            Cells can be represented as a `tuple`, `list`, or `Cell` object.
            If None, then return the number of cells in cell complex.

        Returns
        -------
        int
            The number of cells in cell_set belonging to cell complex.
        """
        if cell_set:
            return len([cell for cell in self.cells if cell in cell_set])
        return len(self.cells)

    def order(self) -> int:
        """Compute the number of nodes in the cell complex.

        Returns
        -------
        int
            Number of nodes in the cell complex.
        """
        return len(self.nodes)

    def neighbors(self, node):
        """Return nodes in cell complex which share s cell(s) with node.

        Parameters
        ----------
        node : hashable
            Node in the cell complex for which to find neighbors.

        Returns
        -------
        list
            List of neighbors.

        Raises
        ------
        KeyError
            If node is not in the cell complex.
        """
        if node not in self.nodes:
            raise KeyError(f"input {node} is not in the complex.")

        return self._G[node]

    def cell_neighbors(self, cell, s: int = 1):
        """Return the neighboring cells of the given cell.

        Parameters
        ----------
        cell : Cell or Iterable
            The cell for which to find neighbors.
        s : int, default=1
            Minimum number of nodes shared by neighboring cell.

        Returns
        -------
        list
            List of cell neighbors.
        """
        raise NotImplementedError

    def remove_node(self, node: Hashable) -> None:
        """Remove the given node from the cell complex.

        This method removes the given node from the cell complex, along with any
        cells that contain the node.

        Parameters
        ----------
        node : hashable
            The node to be removed from the cell complex.

        Raises
        ------
        RuntimeError
            If the given node does not exist in the cell complex.
        """
        if node not in self.nodes:
            raise RuntimeError("The given node does not exist in the cell complex.")
        # Remove the node from the cell complex
        self._G.remove_node(node)
        # Remove any cells that contain the node
        for cell in self.cells:
            if node in cell:
                self.remove_cell(cell)

    def remove_nodes(self, node_set: Iterable[Hashable]) -> None:
        """Remove nodes from cells.

        This also deletes references in cell complex nodes.

        Parameters
        ----------
        node_set : iterable of hashables or Entities
            Nodes in the cell complex.
        """
        for node in node_set:
            self.remove_node(node)

    def add_node(self, node: Hashable, **attr) -> None:
        """Add a single node to cell complex.

        Parameters
        ----------
        node : hashable
            The node to be added.
        **attr : keyword arguments, optional
            Attributes to add to the node as key=value pairs.
        """
        self._G.add_node(node, **attr)

    def _add_nodes_from(self, nodes: Iterable[Hashable]) -> None:
        """Instantiate new nodes when cells added to cell complex.

        Parameters
        ----------
        nodes : iterable of hashables
            Nodes to instantiate when adding cells to cell complex.
        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u_of_edge: Hashable, v_of_edge: Hashable, **attr) -> None:
        """Add edge.

        Parameters
        ----------
        u_of_edge : hashable
            First node of the edge.
        v_of_edge : hashable
            Second node of the edge.
        **attr : keyword arguments, optional
            Attributes to add to the edge as key=value pairs.
        """
        self._G.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add: Iterable[tuple], **attr) -> None:
        """Add edges.

        Parameters
        ----------
        ebunch_to_add : Iterable of edges
            Each edge must be given as a tuple (u, v).
        **attr : keyword arguments, optional
            Attributes to add to the edges as key=value pairs.
        """
        self._G.add_edges_from(ebunch_to_add, **attr)

    def add_cell(
        self,
        cell: tuple | list | Cell,
        rank: int | None = None,
        check_skeleton: bool = False,
        **attr,
    ):
        """Add a single cell to cell complex.

        Parameters
        ----------
        cell : hashable
            If hashable the cell returned will be empty.
        rank : {0, 1, 2}
            Rank of the cell to be added.
        check_skeleton : bool, default=False
            If true, this function checks the skeleton whether the given cell can be added.
        **attr : keyword arguments, optional
            Attributes to add to the cell as key=value pairs.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> c1 = tnx.Cell((2, 3, 4), color="black")
        >>> CC.add_cell(c1, weight=3)
        >>> CC.add_cell([1, 2, 3, 4], rank=2, color="red")
        >>> CC.add_cell([2, 3, 4, 5], rank=2, color="blue")
        >>> CC.add_cell([5, 6, 7, 8], rank=2, color="green")
        >>> CC.cells[(1, 2, 3, 4)]["color"]
        'red'
        """
        if isinstance(cell, Cell):  # rank check will be ignored, cells by default
            # are assumed to be of rank 2
            if self.is_insertable_cycle(
                cell, check_skeleton=check_skeleton, warnings_dis=True
            ):
                for edge in cell.boundary:
                    self._G.add_edge(edge[0], edge[1])
                if self._regular:
                    if cell.is_regular:
                        self._insert_cell(cell, **attr)
                    else:
                        raise RuntimeError(
                            "input cell violates the regularity condition."
                        )
                else:
                    self._insert_cell(cell, **attr)
            else:
                raise RuntimeError(
                    "input cell violates the regularity condition, make sure cell is regular or change complex to non-regular"
                )

        else:
            if rank == 0:
                raise RuntimeError(
                    "Use `add_node` to insert nodes or zero ranked cells."
                )
            if rank == 1:
                if len(cell) != 2:
                    raise ValueError("rank 1 cells (edges) must have exactly two nodes")
                if len(set(cell)) == 1:
                    raise ValueError(" invalid insertion : self-loops are not allowed.")
                self.add_edge(cell[0], cell[1], **attr)
            elif rank == 2:
                if isinstance(cell, Iterable):
                    if not isinstance(cell, list):
                        cell = list(cell)

                    if self.is_insertable_cycle(cell, check_skeleton):
                        edges_cell = set(zip_longest(cell, cell[1:] + [cell[0]]))
                        self._G.add_edges_from(edges_cell)
                        self._insert_cell(Cell(cell, regular=self._regular), **attr)
                    else:
                        raise ValueError(
                            f"Invalid cycle condition for cell {cell}. This input cell is not inserted, check if cell is irregular or if boundary edges of the cell are in the complex."
                        )
                else:
                    raise ValueError(f"invalid input, input {cell} must be iterable")
            else:
                raise ValueError(
                    f"Add cell only supports adding cells of dimensions 0,1 or 2-- got {rank}",
                )

    def add_cells_from(
        self,
        cell_set: Iterable[tuple | list | Cell],
        rank: int | None = None,
        check_skeleton: bool = False,
        **attr,
    ) -> None:
        """Add cells to cell complex.

        Parameters
        ----------
        cell_set : iterable of hashables or Cell
            Iterable of cells to add to the complex.
        rank : int, optional
            The rank of cells to be added. If `cell_set` consists of `Cell` objects,
            this parameter is ignored.
        check_skeleton : bool, default=False
            If true, this function checks the skeleton whether the given cell can be added.
        **attr : keyword arguments, optional
            Attributes to add to the cells as key=value pairs.
        """
        for cell in cell_set:
            self.add_cell(cell=cell, rank=rank, check_skeleton=check_skeleton, **attr)

    def remove_cell(self, cell: tuple | list | Cell):
        """Remove a single cell from Cell Complex.

        Parameters
        ----------
        cell : cell's node_set or Cell
            The cell to remove from this cell complex.

        Notes
        -----
        Deletes reference to cell, keep it boundary edges in the cell complex
        """
        if isinstance(cell, Cell):
            self._delete_cell(cell.elements)
        elif isinstance(cell, Iterable):
            if not isinstance(cell, tuple):
                cell = tuple(cell)
            self._delete_cell(cell)

    def remove_cells(self, cell_set: Iterable[tuple | list | Cell]):
        """Remove cells from a cell complex that are in cell_set.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities
            The cells to remove from this cell complex.
        """
        for cell in cell_set:
            self.remove_cell(cell)

    def clear(self):
        """Remove all cells from a cell complex."""
        for cell in self.cells:
            self.remove_cell(cell)

    def set_filtration(
        self,
        values: dict[Hashable | tuple | list | Cell, dict]
        | dict[Hashable | tuple | list | Cell, Any],
        name: str | None = None,
    ) -> None:
        """Set filtration.

        Parameters
        ----------
        values : dict
            Dictionary of filtration values to set.
            If `name` is specified, the dictionary must be of the form cell -> value.
            Otherwise, the dictionary must be of the form cell -> (attribute -> value).
        name : str, optional
            The attribute name to use to store the filtration values.

        Notes
        -----
        This is equivalent to setting a real-valued feature defined on the entire cell complex

        If the dict contains cells that are not in `self.cells`, they are
        silently ignored.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> d = {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}
        >>> CC.set_filtration(d, "f")
        """
        d_nodes, d_edges, d_cells = [{}, {}, {}]

        for k, v in values.items():
            # to do, make sure v is a number

            if not isinstance(v, int | float):
                raise ValueError(f"filtration value must be a int or float, input {v}")

            if isinstance(k, Hashable) and not isinstance(k, Iterable):  # node
                d_nodes[k] = v
            elif isinstance(k, Iterable) and len(k) == 2:  # edge
                d_edges[k] = v
            elif isinstance(k, Iterable) and len(k) != 2:  # cell
                d_cells[k] = v
        self.set_cell_attributes(d_nodes, name=name, rank=0)
        self.set_cell_attributes(d_edges, name=name, rank=1)
        self.set_cell_attributes(d_cells, name=name, rank=2)

    def get_filtration(self, name: str) -> dict[Hashable, Any]:
        """Get filtration.

        Parameters
        ----------
        name : str
            The attribute name that holds the filtration values.

        Returns
        -------
        dict
            Dictionary mapping each cell to the value of the given attribute name.

        Notes
        -----
        This is equivalent to getting a feature defined on the entire cell complex

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> d = {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}
        >>> CC.set_filtration(d, "f")
        >>> CC.get_filtration("f")
        {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}
        """
        lst = [self.get_cell_attributes(name, rank=r) for r in range(3)]
        d = {}
        for i in lst:
            if i is not None:
                d.update(i)
        return d

    def set_node_attributes(
        self,
        values: dict[Hashable, dict] | dict[Hashable, Any],
        name: str | None = None,
    ) -> None:
        """Set node attributes.

        Parameters
        ----------
        values :  dict
            Dictionary of attribute values to set.
            If `name` is specified, the dictionary must be of the form node -> value.
            Otherwise, the dictionary must be of the form node -> (attribute -> value).
        name : str, optional
            Attribute name to use for setting the attribute values.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> d = {1: {"color": "red", "attr2": 1}, 2: {"color": "blue", "attr2": 3}}
        >>> CC.set_node_attributes(d)
        >>> CC[1]["color"]
        'red'
        """
        self.set_cell_attributes(values, rank=0, name=name)

    def get_node_attributes(self, name: str) -> dict[tuple, Any]:
        """Get node attributes.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        dict
            Dictionary mapping each node to the value of the given attribute name.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> d = {1: {"color": "red", "attr2": 1}, 2: {"color": "blue", "attr2": 3}}
        >>> CC.set_node_attributes(d)
        >>> CC.get_node_attributes("color")
        {1: 'red', 2: 'blue'}
        """
        return self.get_cell_attributes(rank=0, name=name)

    def set_edge_attributes(
        self,
        values: dict[tuple, dict] | dict[tuple, Any],
        name: str | None = None,
    ) -> None:
        """Set edge attributes.

        Parameters
        ----------
        values :  dict
            Dictionary of attribute values to set.
            If `name` is specified, the dictionary must be of the form edge -> value.
            Otherwise, the dictionary must be of the form edge -> (attribute -> value).
        name : str, optional
            Attribute name to use for setting the attribute values.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> d = {
        ...     (1, 2): {"color": "red", "attr2": 1},
        ...     (2, 3): {"color": "blue", "attr2": 3},
        ... }
        >>> CC.set_edge_attributes(d)
        >>> CC.edges[(1, 2)]["color"]
        'red'
        """
        self.set_cell_attributes(values, rank=1, name=name)

    def get_edge_attributes(self, name: str) -> dict[tuple, Any]:
        """Get edge attributes.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        dict
            Dictionary mapping each edge to the value of the given attribute name.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> d = {
        ...     (1, 2): {"color": "red", "attr2": 1},
        ...     (2, 3): {"color": "blue", "attr2": 3},
        ... }
        >>> CC.set_edge_attributes(d)
        >>> CC.get_edge_attributes("color")
        {(1, 2): 'red', (2, 3): 'blue'}
        """
        return self.get_cell_attributes(rank=1, name=name)

    def set_cell_attributes(
        self,
        values: dict[Hashable | tuple | list | Cell, dict]
        | dict[Hashable | tuple | list | Cell, Any],
        rank: int,
        name: str | None = None,
    ) -> None:
        """Set cell attributes.

        Parameters
        ----------
        values :  dict
            Dictionary of attribute values to set.
            If `name` is specified, the dictionary must be of the form cell -> value.
            Otherwise, the dictionary must be of the form cell -> (attribute -> value).
        rank : {0, 1, 2}
            0 for nodes, 1 for edges, 2 for 2-cells.
            ranks > 2 are currently not supported.
        name : str, optional
            Attribute name to use for setting the attribute values.

        Notes
        -----
        If the dict contains cells that are not in `self.cells`, they are
        silently ignored.

        Examples
        --------
        After computing some property of the cell of a cell complex, you may want
        to assign a cell attribute to store the value of that property for
        each cell:

        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> d = {(1, 2, 3, 4): "red", (1, 2, 4): "blue"}
        >>> CC.set_cell_attributes(d, name="color", rank=2)
        >>> CC.cells[(1, 2, 3, 4)]["color"]
        'red'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update cell attributes::

        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> d = {
        ...     (1, 2, 3, 4): {"color": "red", "attr2": 1},
        ...     (1, 2, 4): {"color": "blue", "attr2": 3},
        ... }
        >>> CC.set_cell_attributes(d, rank=2)
        >>> CC.cells[(1, 2, 3, 4)][0]["color"]
        'red'
        """
        if rank == 0:
            nx.set_node_attributes(self._G, values, name)
        elif rank == 1:
            nx.set_edge_attributes(self._G, values, name)
        elif rank == 2:
            if name is not None:
                # if `values` is a dict using `.items()` => {cell: (key,value) } or {cell:value}

                for cell, value in values.items():
                    try:
                        if len(cell) == 2:
                            if isinstance(cell[0], Iterable) and isinstance(
                                cell[1], int
                            ):  # input cell has cell key
                                self.cells[cell][cell[0]][name] = value
                            else:
                                self.cells[cell][name] = value
                        elif isinstance(
                            self.cells[cell], list
                        ):  # all cells with same key get same attrs
                            for i in range(len(self.cells[cell])):
                                self.cells[cell][i][name] = value
                        else:
                            self.cells[cell][name] = value
                    except KeyError:
                        pass
            else:
                for cell, d in values.items():
                    try:
                        if len(cell) == 2:
                            if isinstance(cell[0], Iterable) and isinstance(
                                cell[1], int
                            ):  # input cell has cell key
                                self.cells[cell[0]][cell[1]].update(d)
                            else:  # length of cell is 2
                                self.cells[cell].update(d)
                        elif isinstance(
                            self.cells[cell], list
                        ):  # all cells with same key get same attrs
                            for i in range(len(self.cells[cell])):
                                self.cells[cell][i].update(d)
                        else:
                            self.cells[cell].update(d)
                    except KeyError:
                        pass
                return
        else:
            raise ValueError(f"Rank must be 0, 1 or 2, got {rank}")

    def get_cell_attributes(
        self, name: str, rank: int
    ) -> dict[tuple[Hashable, ...], Any]:
        """Get node attributes from graph.

        Parameters
        ----------
        name : str
           Attribute name.
        rank : int
            The rank for which to return attribute values.

        Returns
        -------
        dict
            Dictionary mapping each cell of rank `rank` to the value of the given
            attribute name.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> d = {
        ...     ((1, 2, 3, 4), 0): {"color": "red", "attr2": 1},
        ...     (1, 2, 4): {"color": "blue", "attr2": 3},
        ... }
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> CC.set_cell_attributes(d, 2)
        >>> CC.get_cell_attributes("color", 2)
        {((1, 2, 3, 4), 0): 'red', (1, 2, 4): 'blue'}
        """
        if rank == 0:
            return nx.get_node_attributes(self._G, name)
        if rank == 1:
            return nx.get_edge_attributes(self._G, name)
        if rank == 2:
            d = {}
            for n in self.cells:
                if isinstance(self.cells[n.elements], list):  # multiple cells
                    for i in range(len(self.cells[n.elements])):
                        if name in self.cells[n.elements][i]:
                            d[(n.elements, i)] = self.cells[n.elements][i][name]
                else:
                    if name in self.cells[n.elements]:
                        d[n.elements] = self.cells[n.elements][name]

            return d
        raise ValueError(f"Rank must be 0, 1 or 2, got {rank}")

    def set_cell_data(self, cell, rank, attr_name: str, attr_value):
        """Set data for a specific cell in the complex.

        Parameters
        ----------
        cell : str or tuple
            The cell to set data for.
        rank : int
            The rank of the cell.
        attr_name : str
            The name of the attribute to set.
        attr_value : object
            The value to set for the attribute.

        Raises
        ------
        KeyError
            If the specified cell is not found.

        Notes
        -----
        - For rank 0 cells (nodes), the data is stored in the 'nodes' dictionary.
        - For rank 1 cells (edges), the data is stored in the 'edges' dictionary.
        - For rank 2 cells (other cells), the data is stored in the 'cells' dictionary.
        """
        if rank == 0:
            if cell in self.nodes:
                self.nodes[cell][attr_name] = attr_value
            else:
                raise KeyError(f"{cell} is not a node in the complex.")
        elif rank == 1:
            if cell in self.edges and len(cell) == 2:
                self.edges[cell][attr_name] = attr_value
            else:
                raise KeyError(f"{cell} is not a valid edge in the complex.")
        elif rank == 2:
            if cell in self.cells:
                self.cells[cell][attr_name] = attr_value
            else:
                raise KeyError(f"{cell} is not a valid cell in the complex.")
        else:
            raise ValueError(f"Invalid rank: {rank}. Rank must be 0, 1, or 2.")

    def get_cell_data(self, cell, rank, attr_name: str | None = None):
        """Retrieve data associated with a specific cell in the complex.

        Parameters
        ----------
        cell : str or tuple
            The cell to retrieve data from.
        rank : int
            The rank of the cell.
        attr_name : str, optional
            The name of the attribute to retrieve. Default is None.

        Returns
        -------
        object
            The value associated with the specified cell and attribute.

        Raises
        ------
        KeyError
            If the specified cell or attribute is not found.

        Notes
        -----
        - For rank 0 cells (nodes), the data is retrieved from the 'nodes' dictionary.
        - For rank 1 cells (edges), the data is retrieved from the 'edges' dictionary.
        - For rank 2 cells (other cells), the data is retrieved from the 'cells' dictionary.
        """
        if rank == 0:
            container = self.nodes
        elif rank == 1:
            container = self.edges
        elif rank == 2:
            container = self.cells
        else:
            raise ValueError(f"Invalid rank: {rank}. Rank must be 0, 1, or 2.")

        if cell in container:
            if attr_name is not None:
                if attr_name in container[cell]:
                    return container[cell][attr_name]
                raise KeyError(
                    f"Node '{cell}' does not have an attribute with key {attr_name}."
                )
            return container[cell]

        if rank == 0:
            raise KeyError(f"Node '{cell}' is not present in the complex.")
        if rank == 1:
            raise KeyError(
                f"Edge '{cell}' is not present in the complex or does not have two nodes."
            )
        raise KeyError(f"Cell '{cell}' is not present in the complex.")

    def remove_equivalent_cells(self) -> None:
        """Remove equivalent cells.

        Remove cells from the cell complex which are homotopic.
        In other words, this is equivalent to identifying cells
        containing the same nodes and are equivalent up to cyclic
        permutation.

        Notes
        -----
        Remove all 2d- cells that are homotopic (equivalent to each other)

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([2, 3, 4, 1], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> print(CC.cells)
        >>> CC.remove_equivalent_cells()
        """
        # delete all cells that are added multiple times
        for cell in self._cells._cells:
            if len(self._cells._cells[cell]) > 1:
                for key in list(self._cells._cells[cell].keys())[1:]:
                    self._delete_cell(cell, key)

        equiv_classes = self._cell_equivalence_class()
        for c in list(self.cells):
            if c not in equiv_classes:
                self._delete_cell(c)

    def is_insertable_cycle(
        self,
        cell: tuple | list | Cell,
        check_skeleton: bool = True,
        warnings_dis: bool = False,
    ) -> bool:
        """Determine if a cycle is insertable to the cell complex.

        Checks regularity if this CellComplex is regular,
        existence of required edges if `check_skeleton` is True,
        and that the cell has a minimum length of 2.

        Parameters
        ----------
        cell : Cell | tuple | list
            The cell to check.
        check_skeleton : bool, default=True
            Whether to check that all edges induced by the cell are part of the underlying graph.
            If False, missing edges will be ignored.
        warnings_dis : bool, default=False
            Whether to print a warning with the reason why the cell is not insertable.

        Returns
        -------
        bool
            True if the cell can be inserted, otherwise False.
        """
        if isinstance(cell, Cell):
            cell = cell.elements
        if self._regular and len(set(cell)) != len(cell):
            if warnings_dis:
                warnings.warn(
                    "repeating nodes invalidates the 2-cell regularity condition",
                    stacklevel=2,
                )
            return False
        if check_skeleton:
            enum = zip_longest(cell, cell[1:] + cell[:1])
            for i in enum:
                if i not in self.edges:
                    if warnings_dis:
                        warnings.warn(
                            f"edge {i} is not a part of the 1-skeleton of the cell complex",
                            stacklevel=2,
                        )
                    return False
        return True

    def node_to_all_cell_incidence_matrix(
        self, weight: str | None = None, index: bool = False
    ) -> scipy.sparse.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]:
        """Nodes/all cells incidence matrix for the indexed by nodes X cells.

        Parameters
        ----------
        weight : str, optional
            The name of the cell attribute to use as weights for the incidence
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the incidence matrix.

        Returns
        -------
        row_indices : dict
            Dictionary assigning each row of the incidence matrix to a node. Only
            returned if `index` is True.
        col_indices : dict
            Dictionary assigning each column of the incidence matrix to a cell. Only
            returned if `index` is True.
        incidence_matrix : scipy.sparse.csr.csc_matrix
            The incidence matrix.
        """
        node_index = {node: i for i, node in enumerate(sorted(self._G.nodes))}
        edgelist = sorted([sorted(e) for e in self._G.edges])
        all_cell_index = {tuple(sorted(edge)): i for i, edge in enumerate(edgelist)}
        cell_index = {c.elements: i + len(edgelist) for i, c in enumerate(self.cells)}
        all_cell_index.update(cell_index)
        A = sp.sparse.lil_matrix((len(node_index), len(all_cell_index)))
        for cj, c in enumerate(all_cell_index):
            for ni, n in enumerate(node_index):
                if n in c:
                    A[ni, cj] = 1
        if index:
            return node_index, all_cell_index, A.asformat("csc")
        return A.asformat("csc")

    def node_to_all_cell_adjacnecy_matrix(
        self, s: int = 1, weight: str | None = None, index: bool = False
    ) -> scipy.sparse.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]:
        """Compute s-adjacency matrix between nodes with respect to 2-cells.

        Parameters
        ----------
        s : int, default=1
            The dimension of the cells to consider.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the incidence matrix.

        Returns
        -------
        indices : dict
            Dictionary assigning each row and column of the incidence matrix to a
            node. Only returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The adjacency matrix.

        Notes
        -----
        Two nodes are s-adjacent iff there exists a cell (1 dimensional or 2 dimensional)
        share contain them.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> CC.node_to_all_cell_adjacnecy_matrix().todense()
        matrix([[0., 2., 1., 2., 0.],
                [2., 0., 2., 1., 0.],
                [1., 2., 0., 3., 2.],
                [2., 1., 3., 0., 2.],
                [0., 0., 2., 2., 0.]])
        >>> # observe the contrast with the regular 0-adjacency matrix
        >>> CC.adjacency_matrix(0).todense()
        matrix([[0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 0.],
                [0., 1., 0., 1., 1.],
                [1., 0., 1., 0., 1.],
                [0., 0., 1., 1., 0.]])
        """
        if index:
            node_index, cell_index, M = self.node_to_all_cell_incidence_matrix(
                weight, index
            )

            return node_index, incidence_to_adjacency(M.T, s)

        return incidence_to_adjacency(
            self.node_to_all_cell_incidence_matrix(weight, index).T, s
        )

    def all_cell_to_node_coadjacency_matrix(
        self, s: int = 1, weight: str | None = None, index: bool = False
    ) -> scipy.sparse.csc_matrix | tuple[dict, dict, scipy.sparse.csc_matrix]:
        """Return coadjecency matrix of cells with respect to nodes.

        Parameters
        ----------
        s : int, default=1
            Minimum number of nodes that cells must share to be considered adjacent.
        weight : str, optional
            The name of the cell attribute to use as weights for the coadjacency
            matrix. If `None`, the matrix is binary.
        index : bool, optional
            Whether to return the indices of the rows and columns of the adjacency
            matrix.

        Returns
        -------
        indices : dict
            Dictionary assigning each row and column of the coadjacency matrix to a
            cell. Only returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The coadjacency matrix.

        Notes
        -----
        Two cells (1 dimensional or 2 dimensional) are s-coadjacent iff
        they share `s` nodes.

        Examples
        --------
        >>> CX = tnx.CellComplex([[1, 2, 3, 4], [2, 3, 6]])
        >>> index, m = CX.all_cell_to_node_coadjacency_matrix(s=1, index=True)
        >>> # m_ij iff cell i is coadjacency to cell j. Dimension of cells i,j are arbirary
        >>> print(m.todense(), index)
        """
        if index:
            node_index, cell_index, M = self.node_to_all_cell_incidence_matrix(
                weight, index
            )

            return cell_index, incidence_to_adjacency(M, s)

        return incidence_to_adjacency(
            self.node_to_all_cell_incidence_matrix(weight, index), s
        )

    def incidence_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> scipy.sparse.csr_matrix | tuple[dict, dict, scipy.sparse.csr_matrix]:
        """Incidence matrix for the cell complex indexed by nodes x cells.

        Parameters
        ----------
        rank : int
            The rank for which an incidence matrix should be computed.
        signed : bool, default=True
            Whether the returned incidence matrix should be signed (i.e., respect orientations) or unsigned.
        weight : str, optional
            The name of the simplex attribute to use as weights for the incidence
            matrix. If `None`, the matrix is binary.
        index : bool, optional
            Whether to return the indices of the rows and columns of the incidende matrix.

        Returns
        -------
        row_indices, col_indices : dict
            Dictionary assigning each row and column of the incidence matrix to a
            cell. Only returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The incidence matrix.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> B0 = CC.incidence_matrix(0)
        >>> B1 = CC.incidence_matrix(1)
        >>> B2 = CC.incidence_matrix(2)
        >>> B1.dot(B2).todense()
        >>> B0.dot(B1).todense()

        Note that in this example, the first three cells are equivalent and hence they have similar incidence to lower
        edges they are incident to.

        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([4, 3, 2, 1], rank=2)
        >>> CC.add_cell([2, 3, 4, 1], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> B1 = CC.incidence_matrix(1)
        >>> B2 = CC.incidence_matrix(2)
        >>> B1.dot(B2).todense()

        Non-regular cell complex example:

        >>> CC = tnx.CellComplex(regular=False)
        >>> CC.add_cell([1, 2, 3, 2], rank=2)
        >>> CC.add_cell([3, 4, 5, 3, 4, 5], rank=2)
        >>> B1 = CC.incidence_matrix(1)
        >>> B2 = CC.incidence_matrix(2)
        >>> print(B2.todense())  # observe the non-unit entries
        >>> B1.dot(B2).todense()

        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> row, column, B1 = CC.incidence_matrix(1, index=True)
        >>> print(row)
        >>> print(column)
        >>> print(B1.todense())
        """
        if rank == 0:
            A = sp.sparse.lil_matrix((0, len(self._G.nodes)))
            if index:
                node_index = {node: i for i, node in enumerate(sorted(self._G.nodes))}
                if signed:
                    return {}, node_index, A.asformat("csr")
                return {}, node_index, abs(A.asformat("csr"))

            if signed:
                return A.asformat("csr")
            return abs(A.asformat("csr"))

        if rank == 1:
            nodelist = sorted(
                self._G.nodes
            )  # always output boundary matrix in dictionary order
            edgelist = sorted([sorted(e) for e in self._G.edges])
            A = sp.sparse.lil_matrix((len(nodelist), len(edgelist)))
            node_index = {node: i for i, node in enumerate(nodelist)}
            for ei, e in enumerate(edgelist):
                (u, v) = e[:2]
                ui = node_index[u]
                vi = node_index[v]
                A[ui, ei] = -1
                A[vi, ei] = 1
            if index:
                edge_index = {tuple(sorted(edge)): i for i, edge in enumerate(edgelist)}
                if signed:
                    return node_index, edge_index, A.asformat("csr")
                return node_index, edge_index, abs(A.asformat("csr"))

            if signed:
                return A.asformat("csr")
            return abs(A.asformat("csr"))
        if rank == 2:
            edgelist = sorted([sorted(e) for e in self._G.edges])

            A = sp.sparse.lil_matrix((len(edgelist), len(self.cells)))

            edge_index = {
                tuple(sorted(edge)): i for i, edge in enumerate(edgelist)
            }  # orient edges
            for celli, cell in enumerate(self.cells):
                edge_visiting_dic = {}  # this dictionary is cell dependent
                # mainly used to handle the cell complex non-regular case
                for edge in cell.boundary:
                    ei = edge_index[tuple(sorted(edge))]
                    if ei not in edge_visiting_dic:
                        if edge in edge_index:
                            edge_visiting_dic[ei] = 1
                        else:
                            edge_visiting_dic[ei] = -1
                    else:
                        if edge in edge_index:
                            edge_visiting_dic[ei] = edge_visiting_dic[ei] + 1
                        else:
                            edge_visiting_dic[ei] = edge_visiting_dic[ei] - 1

                    A[ei, celli] = edge_visiting_dic[
                        ei
                    ]  # this will update everytime we visit this edge for non-regular cell complexes
                    # the regular case can be handled more efficiently :
                    # if edge in edge_index:
                    #    A[ei, celli] = 1
                    # else:
                    #    A[ei, celli] = -1
            if index:
                cell_index = {c.elements: i for i, c in enumerate(self.cells)}
                if signed:
                    return edge_index, cell_index, A.asformat("csr")
                return edge_index, cell_index, abs(A.asformat("csr"))

            if signed:
                return A.asformat("csr")
            return abs(A.asformat("csr"))
        raise ValueError(f"Only dimensions 0, 1 and 2 are supported, got {rank}.")

    def hodge_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> scipy.sparse.csr_matrix | tuple[dict, scipy.sparse.csr_matrix]:
        """Compute the Hodge Laplacian matrix for this cell complex.

        Parameters
        ----------
        rank : {0, 1, 2}
            Dimension of the Laplacian matrix.
        signed : bool
            Whether to return the signed or unsigned hodge laplacian.
        weight : str, optional
            The name of the cell attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Indicates whether to return the indices that define the hodge laplacian
            matrix.

        Returns
        -------
        indices : dict
            Dictionary assigning each row and column of the incidence matrix to a cell.
            Only returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The Hodge Laplacian matrix of rank `rank`.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> CC.hodge_laplacian_matrix(1)
        """
        if rank == 0:  # return L0, the unit graph laplacian
            if index:
                nodelist, _, B_next = self.incidence_matrix(
                    rank + 1, weight=weight, index=True
                )
                L_hodge = B_next @ B_next.transpose()
                if signed:
                    return nodelist, L_hodge
                return nodelist, abs(L_hodge)

            B_next = self.incidence_matrix(rank + 1, weight=weight)
            L_hodge = B_next @ B_next.transpose()
            if signed:
                return L_hodge
            return abs(L_hodge)
        if rank < 2:  # rank == 1, return L1
            if self.dim == 2:
                edge_list, cell_list, B_next = self.incidence_matrix(
                    rank + 1, weight=weight, index=True
                )
                B = self.incidence_matrix(rank, weight=weight, index=False)
                L_hodge = B_next @ B_next.transpose() + B.transpose() @ B
            else:
                B = self.incidence_matrix(rank, weight=weight)
                L_hodge = B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return edge_list, L_hodge
            return L_hodge

        if rank == 2 and self.dim == 2:
            edge_list, cell_list, B = self.incidence_matrix(
                rank, weight=weight, index=True
            )
            L_hodge = B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)

            if index:
                return cell_list, L_hodge
            return L_hodge
        if rank == 2 and self.dim != 2:
            raise ValueError(
                "The input complex does not have cells of dim 2. "
                f"The maximal cell dimension is {self.dim}, got {rank}"
            )

        raise ValueError(
            f"Rank should be larger than 0 and <= {self.dim} (maximal dimension cells), got {rank}."
        )

    def up_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute up laplacian.

        Parameters
        ----------
        rank : {0, 1}
            dimension of the up Laplacian matrix.
        signed : bool
            Whether the returned laplacian matrix should be signed (i.e., respect
            orientations) or unsigned.
        weight : str, optional
            The name of the cell attribute to use as weights for the up laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the laplacian
            matrix.

        Returns
        -------
        row_indices, col_indices : list
            List identifying the rows and columns of the laplacian matrix with the
            cells of this complex. Only returned if `index` is True.
        up_laplacian : scipy.sparse.csr.csr_matrix
            The upper laplacian matrix of this cell complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> L1_up = CC.up_laplacian_matrix(1)

        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3], rank=2)
        >>> CC.add_cell([3, 4, 5], rank=2)
        >>> index, L1_up = CC.up_laplacian_matrix(1, index=True)
        """
        if weight is not None:
            raise ValueError("Weighted Laplacian is not supported in this version.")

        if rank == 0 or rank < self.dim:
            row, col, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            L_up = B_next @ B_next.transpose()
        else:
            raise ValueError(
                f"Rank should larger than 0 and <= {self.dim - 1} (maximal dimension cells-1), got {rank}."
            )
        if not signed:
            L_up = abs(L_up)

        if index:
            return row, L_up
        return L_up

    def down_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute down laplacian.

        Parameters
        ----------
        rank : {1, 2}
            Rank of the down Laplacian matrix.
        signed : bool
            Whether to return the signed or unsigned down laplacian.
        weight : str, optional
            The name of the cell attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the laplacian.

        Returns
        -------
        indices : dict
            Dictionary assigning each row and column of the laplacian matrix to a cell.
            Only returned if `index` is True.
        down_laplacian : scipy.sparse.csr.csr_matrix
            The down laplacian matrix of this cell complex.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([2, 3, 4, 1], rank=2)
        >>> CC.add_cell([1, 2, 4], rank=2)
        >>> CC.add_cell([3, 4, 8], rank=2)
        >>> CC.down_laplacian_matrix(2)
        """
        if weight:
            raise ValueError("Weighted Laplacian is not supported in this version.")

        if 0 < rank <= self.dim:
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {self.dim} (maximal dimension cells), got {rank}."
            )
        if not signed:
            L_down = abs(L_down)
        if index:
            return row, L_down
        return L_down

    def adjacency_matrix(
        self,
        rank: int,
        signed: bool = False,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute adjacency matrix for a given rank.

        Parameters
        ----------
        rank : int
            The rank for which an adjacency matrix should be computed.
        signed : bool, default=False
            Whether the returned adjacency matrix should be signed (i.e., respect orientations) or unsigned.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the adjacency
            matrix.

        Returns
        -------
        indices : dict
            Dictionary mapping each row and column of the coadjacency matrix to a cell.
            Only returned if `index` is True.
        adjacency_matrix : scipy.sparse.csr.csr_matrix
            The adjacency matrix of rank `rank` of this cell complex.
        """
        if index:
            ind, _, incidence = self.incidence_matrix(
                rank + 1, signed=signed, index=True
            )
        else:
            incidence = self.incidence_matrix(rank + 1, signed=signed)

        incidence = incidence.T

        if index:
            return ind, incidence_to_adjacency(incidence)
        return incidence_to_adjacency(incidence)

    def coadjacency_matrix(
        self,
        rank: int,
        signed: bool = False,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute coadjacency matrix for a given rank.

        Parameters
        ----------
        rank : int
            Rank of the coadjacency matrix.
        signed : bool
            Whether to return the signed or unsigned coadjacency matrix.
        weight : str, optional
            The name of the cell attribute to use as weights for the coadjacency
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the coadjacency
            matrix.

        Returns
        -------
        indices : dict
            Dictionary mapping each row and column of the coadjacency matrix to a cell.
            Only returned if `index` is True.
        coadjacency_matrix : scipy.sparse.csr.csr_matrix
            The coadjacency matrix of this cell complex.
        """
        if index:
            _, ind, incidence = self.incidence_matrix(rank, signed=signed, index=True)
            return ind, incidence_to_adjacency(incidence)

        incidence = self.incidence_matrix(rank, signed=signed)
        return incidence_to_adjacency(incidence)

    def dirac_operator_matrix(
        self,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute dirac operator matrix matrix.

        Parameters
        ----------
        signed : bool, default=False
            Whether the returned dirac matrix should be signed (i.e., respect
            rientations) or unsigned.
        weight : str, optional
            The name of the cell attribute to use as weights for the dirac operator
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the dirac operator matrix.

        Returns
        -------
        row_indices, col_indices : dict
            List identifying rows and columns of the dirac operator matrix. Only
            returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The dirac operator matrix of this cell complex.

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> CC = tnx.CellComplex(G)
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.dirac_operator_matrix()
        """
        from scipy.sparse import bmat

        _, index0, B0 = self.incidence_matrix(0, weight=weight, index=True)
        _, index1, B1 = self.incidence_matrix(1, weight=weight, index=True)
        index1 = {k: v + len(index0) for k, v in index1.items()}
        _, index2, B2 = self.incidence_matrix(2, weight=weight, index=True)
        index2 = {k: v + len(index0) + len(index1) for k, v in index2.items()}

        dirac = bmat([[None, B1, None], [B1.T, None, B2], [None, B2.T, None]])

        if index:
            d = {}
            d.update(index0)
            d.update(index1)
            d.update(index2)
            if signed:
                return d, dirac
            return d, abs(dirac)
        if signed:
            return dirac
        return abs(dirac)

    def restrict_to_cells(
        self,
        cell_set: Iterable[Cell | tuple],
        keep_edges: bool = False,
    ):
        """Construct cell complex using a subset of the cells in cell complex.

        Parameters
        ----------
        cell_set : Iterable[Cell | tuple]
            A subset of elements of the cell complex's cells (self.cells) and edges (self.edges).
            Cells can be represented as Cell objects or tuples with length > 2.
        keep_edges : bool, default=False
            If False, discards edges not required by or included in `cell_set`
            If True, all previous edges are kept.

        Returns
        -------
        CellComplex
            A new cell complex restricted to the cells in `cell_set`.

        Examples
        --------
        >>> c1 = tnx.Cell((1, 2, 3))
        >>> c2 = tnx.Cell((1, 2, 4))
        >>> c3 = tnx.Cell((1, 2, 5))
        >>> CC = tnx.CellComplex([c1, c2, c3])
        >>> CC.add_edge(1, 0)
        >>> cx1 = CC.restrict_to_cells([c1, (0, 1)])
        >>> cx1.cells
        CellView([Cell((1, 2, 3))])
        """
        CC = CellComplex(cells=self._G.copy())

        edges = set()

        for cell in cell_set:
            if cell in self.cells:
                raw_cell = self.cells.raw(cell)
                if isinstance(raw_cell, list):
                    for c in raw_cell:
                        edges.update(
                            {
                                tuple(sorted(edge))
                                for edge in pairwise(c.elements + c.elements[:1])
                            }
                        )
                        CC.add_cell(c, rank=2)
                else:
                    edges.update(
                        {
                            tuple(sorted(edge))
                            for edge in pairwise(
                                raw_cell.elements + raw_cell.elements[:1]
                            )
                        }
                    )
                    CC.add_cell(raw_cell, rank=2)
            elif len(cell) == 2 and cell in self.edges:
                edges.add(tuple(sorted((cell[0], cell[1]))))

        if not keep_edges:
            # remove all edges that are not included (directly or through 2-cells)
            for edge in self._G.edges:
                edge_tuple = tuple(sorted((edge[0], edge[1])))
                if edge_tuple not in edges:
                    CC._G.remove_edge(edge_tuple[0], edge_tuple[1])

        return CC

    def restrict_to_nodes(self, node_set: Iterable[Hashable]):
        """Restrict cell complex to nodes.

        This constructs a new cell complex by restricting the cells in the cell complex to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set : iterable of hashables
            The nodes to restrict the cell complex to.

        Returns
        -------
        CellComplex
            A new cell complex restricted to the nodes in `node_set`.

        Examples
        --------
        >>> c1 = tnx.Cell((1, 2, 3))
        >>> c2 = tnx.Cell((1, 2, 4))
        >>> c3 = tnx.Cell((1, 2, 5))
        >>> CC = tnx.CellComplex([c1, c2, c3])
        >>> CC.add_edge(1, 0)
        >>> CC.restrict_to_nodes([1, 2, 3, 0])
        """
        _G = Graph(self._G.subgraph(node_set))
        CC = CellComplex(_G)
        cells = [cell for cell in self.cells if CC.is_insertable_cycle(cell, True)]
        CC = CellComplex(_G)

        for cell in cells:
            CC.add_cell(cell, **self.get_cell_data(cell, 2))
        return CC

    def to_combinatorial_complex(self):
        """Convert this cell complex to a combinatorial complex.

        A cell complex is a type of combinatorial complex.
        The rank of an element in a cell complex is its dimension, so vertices have rank 0,
        edges have rank 1, and faces have rank 2.

        Returns
        -------
        CombinatorialComplex
            The combinatorial complex corresponding to this cell complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2, weight=1)
        >>> CC.add_cell([2, 3, 4, 5], rank=2, weight=4)
        >>> CC.add_cell([5, 6, 7, 8], rank=2, weight=0)
        >>> CC.add_node(0, color="red")
        >>> CCC = CC.to_combinatorial_complex()
        >>> CCC.cells
        """
        CCC = CombinatorialComplex()
        for c in self.cells:
            CCC.add_cell(c, rank=2, **self.get_cell_data(c, 2))
        for c in self.edges:
            CCC.add_cell(c, rank=1, **self.get_cell_data(c, 1))
        for c in self.nodes:
            CCC.add_node(c, **self.get_cell_data(c, 0))

        return CCC

    def to_colored_hypergraph(self):
        """Convert to colored hypergraph.

        A cell complex is a type of combinatorial complex.
        The rank of an element in a cell complex is its dimension, so vertices have rank 0,
        edges have rank 1, and faces have rank 2.

        Returns
        -------
        ColoredHyperGraph
            The colored hypergraph corresponding to this cell complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2, weight=1)
        >>> CC.add_cell([2, 3, 4, 5], rank=2, weight=4)
        >>> CC.add_cell([5, 6, 7, 8], rank=2, weight=0)
        >>> CC.add_node(0, color="red")
        >>> CCC = CC.to_colored_hypergraph()
        >>> CCC.cells
        """
        CHG = ColoredHyperGraph()
        for c in self.cells:
            CHG.add_cell(c, rank=2, **self.get_cell_data(c, 2))
        for c in self.edges:
            CHG.add_cell(c, rank=1, **self.get_cell_data(c, 1))
        for c in self.nodes:
            CHG.add_node(c, **self.get_cell_data(c, 0))

        return CHG

    def to_hypergraph(self):
        """Convert this cell complex to a hypergraph.

        Returns
        -------
        Hypergraph
            The hyergraph corresponding to this cell complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2, color="red")
        >>> CC.add_cell([2, 3, 4, 5], rank=2)
        >>> CC.add_cell([5, 6, 7, 8], rank=2)
        >>> CC.to_hypergraph()
        """
        from hypernetx.classes.entity import EntitySet

        cells = []
        cells.extend(
            Entity(
                str(list(cell.elements)),
                elements=cell.elements,
                **self.get_cell_data(cell, 2),
            )
            for cell in self.cells
        )
        cells.extend(
            Entity(str(list(cell)), elements=cell, **self.get_cell_data(cell, 1))
            for cell in self.edges
        )
        E = EntitySet("CX_to_HG", elements=cells)
        HG = Hypergraph(E)
        nodes = [
            Entity(cell, elements=[], **self.get_cell_data(cell, 0))
            for cell in self.nodes
        ]
        HG._add_nodes_from(nodes)
        return HG

    def is_connected(self):
        """Determine if cell complex is s-connected.

        Returns
        -------
        bool
            True if the cell complex is connected, False otherwise.
        """
        return nx.is_connected(self._G)

    def singletons(self):
        """Return list of singleton cell.

        A singleton cell is a node of degree 0.

        Returns
        -------
        list of Hashable
            List of singleton nodes in this cell complex.

        Examples
        --------
        >>> CC = CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.add_cell([2, 3, 4, 5], rank=2)
        >>> CC.add_cell([5, 6, 7, 8], rank=2)
        >>> CC.add_node(0)
        >>> CC.add_node(10)
        >>> CC.singletons()
        [0, 10]
        """
        return [node for node in self.nodes if self.degree(node) == 0]

    def clone(self) -> Self:
        """Create a clone of the CellComplex.

        Returns
        -------
        CellComplex
            A copy of this CellComplex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2, weight=5)
        >>> CC.add_cell([2, 3, 4, 5], rank=2)
        >>> CC.add_cell([5, 6, 7, 8], rank=2)
        >>> CC.add_node(0)
        >>> CC.clone()
        """
        _G = self._G.copy()
        CC = self.__class__(_G)
        for cell in self.cells:
            CC.add_cell(cell.clone())
        return CC

    def euler_characterisitics(self) -> int:
        """Return the euler characteristics of the cell complex.

        Returns
        -------
        int
            The Euler characteristics of the cell complex.
        """
        return len(self.nodes) - len(self.edges) + len(self.cells)

    def remove_singletons(self) -> None:
        """Remove singleton nodes (see `CellComplex.singletons()`)."""
        for node in self.singletons():
            self._G.remove_node(node)

    def get_linegraph(self, s: int = 1, cells: bool = True) -> nx.Graph:
        """Create line graph of self.

        If cells=True (default), the cells will be the vertices of the line graph.
        Two vertices are connected by an s-line-graph edge if the corresponding cell
        complex edges intersect in at least s cell complex nodes.

        If cells=False, the cell complex nodes will be the vertices of the line graph.
        Two vertices are connected if the nodes they correspond to share at least s
        incident cell complex edges.

        Parameters
        ----------
        s : int
            The width of the connections.
        cells : bool, optional
            Determines if cells or nodes will be the vertices in the line graph.

        Returns
        -------
        nx.Graph
            A NetworkX graph representing the s-linegraph of the Cell Complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([0, 1, 2, 3, 4], rank=2)
        >>> CC.get_linegraph()
        """
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"'s' must be a positive integer, got {s}.")

        if not isinstance(cells, bool):
            raise TypeError(f"'cells' must be a boolean, got {cells}.")

        if cells:
            M = self.all_cell_to_node_coadjacency_matrix(s=s)
        else:
            M = self.node_to_all_cell_adjacnecy_matrix(s=s)

        return nx.from_scipy_sparse_array(M)

    def to_hasse_graph(self) -> nx.DiGraph:
        """Create Hasse graph of self.

        Returns
        -------
        nx.DiGraph
            A NetworkX Digraph representing the Hasse graph of the Cell Complex.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cell([1, 2, 3, 4], rank=2)
        >>> CC.to_hasse_graph()
        """
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node((n,))
        for e in self.edges:
            (u, v) = sorted(e)

            G.add_node((u, v))
            G.add_edge((u,), (u, v))
            G.add_edge((v,), (u, v))
        for c in self.cells:
            G.add_node(c.elements)
            for e in list(c.boundary):
                (u, v) = sorted(e)
                G.add_edge((u, v), c.elements)
                G.add_edge((u, v), c.elements)
        return G

    def from_networkx_graph(self, G: nx.Graph) -> None:
        """Add edges and nodes from a graph G to self.

        Parameters
        ----------
        G : nx.Graph
            A NetworkX graph.

        Examples
        --------
        >>> CC = tnx.CellComplex()
        >>> CC.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
        >>> CC.from_networkx_graph(G)
        >>> CC.edges
        """
        for edge in G.edges:
            self.add_cell(edge, rank=1)
        for node in G.nodes:
            self.add_node(node)

    @classmethod
    def from_trimesh(cls, mesh) -> Self:
        """Convert from trimesh object.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            A trimesh object.

        Returns
        -------
        CellComplex
            The cell complex generated from the trimesh provided.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(
        ...     vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        ...     faces=[[0, 1, 2]],
        ...     process=False,
        ... )
        >>> CC = tnx.CellComplex.from_trimesh(mesh)
        >>> CC[0]["position"]
        """
        CC = cls(mesh.faces)

        first_ind = np.min(mesh.faces)

        CC.set_cell_attributes(
            dict(
                zip(
                    range(first_ind, len(mesh.vertices) + first_ind),
                    mesh.vertices,
                    strict=False,
                )
            ),
            name="position",
            rank=0,
        )

        return CC

    @classmethod
    def load_mesh(cls, file_path, process: bool = False) -> Self:
        """Load a mesh.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The file path of the data to be loaded.
        process : bool, default=False
            Whether trimesh should process the mesh before loading it.

        Returns
        -------
        CellComplex
            The cell complex corresponding to the mesh.

        Notes
        -----
        file supported : obj, off, glb

        Examples
        --------
        >>> CC = tnx.CellComplex.load_mesh("bunny.obj")
        """
        import trimesh

        mesh = trimesh.load_mesh(file_path, process=process, force=None)
        return cls.from_trimesh(mesh)
