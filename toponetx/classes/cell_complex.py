"""
Class for creation and manipulation of a 2d cell complex.
The class also supports attaching arbitrary attributes and data to cells.
"""


import warnings

try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Iterable, Hashable

from itertools import zip_longest

import networkx as nx
import numpy as np
from hypernetx import Hypergraph
from hypernetx.classes.entity import Entity
from networkx import Graph
from scipy.sparse import csr_matrix

from toponetx.classes.cell import Cell, CellView
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.ranked_entity import DynamicCell, Node, RankedEntitySet
from toponetx.exception import TopoNetXError

__all__ = ["CellComplex"]


class CellComplex:
    """Class representing a cell complex.

    In mathematics, a cell complex is a space that is constructed by attaching lower-dimensional
    cells to a topological space to form a new space. The cells are attached to the space in a specific way,
    and the resulting space has a well-defined structure.

    For example, a cell complex can be used to represent a simplicial complex,
    which is a collection of points, line segments, triangles, and higher-dimensional
    simplices that are connected in a specific way. Each simplex in the simplicial complex
    is called a cell, and the cell complex consists of all of these cells and the way they are connected.

    A cell complex is a mathematical structure consisting of a
    set of points (called vertices or 0-cells), a set of line segments
    (called edges or 1-cells), and a set of polygons (called faces or 2-cells),
    such that the vertices, edges, and faces are connected in a consistent way.

    Cell complexes can be used to represent various mathematical objects, such as graphs,
    manifolds, and discrete geometric shapes. They are useful in many areas of mathematics,
    such as algebraic topology and geometry, where they can be used to study the structure and
    properties of these objects.


    This class represents a cell complex, which is a space constructed by
    attaching cells of different dimensions to a topological space.


    In TNX the class CellComplex supports building a regular or non-regular
    2d cell complex. The class CellComplex only supports the construction
    of 2d cell complexes. If higher order constructions are desired
    then one should utilize the class CombinatorialComplex.


    In TNX cell complexes are implementes to be dynamic in the sense that
    they can change by adding or subtracting objects (nodes, edges, cells)
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

        #Example 0
            >>> # Cell Complex can be empty
            >>> cx = CellComplex()
        #Example 1
            >>> cx = CellComplex()
            >>> cx.add_cell([1, 2, 3, 4], rank=2)
            >>> cx.add_cell([2, 3, 4, 5], rank=2)
            >>> cx.add_cell([5, 6, 7, 8], rank=2)
        #Example 2
            >>> c1 = Cell((1, 2, 3)) # a cell here is always assumed to be 2d
            >>> c2 = Cell((1, 2, 3, 4))
            >>> cx = CellComplex([c1, c2])
        #Example 3
            >>> g = Graph()
            >>> g.add_edge(1, 0)
            >>> g.add_edge(2, 0)
            >>> g.add_edge(1, 2)
            >>> cx = CellComplex(g)
            >>> cx.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
            >>> cx.cells
        #Example 4
            >>> # non-regular cell complex
            >>> cx = CellComplex(regular=False)
            >>> cx.add_cell([1, 2, 3, 4], rank=2)
            >>> cx.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
            >>> c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
            >>> cx.add_cell(c1)
            >>> cx.add_cell([5, 6, 7, 8],rank=2)
            >>> cx.is_regular
        #Example 5
            >>> cx = CellComplex()
            >>> cx.add_cell([1, 2, 3, 4], rank=2, weight=5)
            >>> cx.add_cell([2, 3, 4, 5], rank=2, weight=10)
            >>> cx.add_cell([5, 6, 7, 8], rank=2, weight=13)

    """

    def __init__(self, cells=None, name=None, regular=True, **attr):
        if not name:
            self.name = ""
        else:
            self.name = name

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
                            self.add_node(tuple(cell)[0])
                        else:
                            self.add_cell(cell, rank=2)

            else:
                raise ValueError(
                    f"cells must be iterable, networkx graph or None, got {type(cells)}"
                )
        self.complex = dict()  # dictionary for cell complex attributes
        self.complex.update(attr)

    @property
    def cells(self):
        """
        Object associated with self._cells.

        Returns
        -------

        """
        return self._cells

    @property
    def edges(self):
        """
        Object associated with self._edges.

        Returns
        -------

        """
        return self._G.edges

    @property
    def nodes(self):
        """
        Object associated with self._nodes.

        Returns
        -------
        RankedEntitySet

        """
        return self._G.nodes

    @property
    def maxdim(self):
        if len(self.nodes) == 0:
            return 0
        if len(self.edges) == 0:
            return 0
        if len(self.cells) == 0:
            return 1
        return 2

    @property
    def dim(self):
        return self.maxdim

    @property
    def shape(self):
        """
        (number of cells[i], for i in range(0,dim(cc))  )

        Returns
        -------
        tuple

        """
        return len(self.nodes), len(self.edges), len(self.cells)

    def skeleton(self, rank):
        if rank == 0:
            return self.nodes
        if rank == 1:
            return self.edges
        if rank == 2:
            return self.cells
        raise TopoNetXError("Only dimensions 0,1, and 2 are supported.")

    @property
    def is_regular(self):
        """Check the regularity condition of the cell complex.

        Returns
        -------
        bool

        Example
        -------
            >>> cx = CellComplex(regular=False)
            >>> cx.add_cell([1, 2, 3, 4], rank=2)
            >>> cx.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
            >>> c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5),regular=False)
            >>> cx.add_cell(c1)
            >>> cx.add_cell([5, 6, 7, 8], rank=2)
            >>> cx.is_regular
        """
        for cell in self.cells:
            if not cell.is_regular:
                return False
        return True

    def __str__(self):
        """
        String representation of cx

        Returns
        -------
        str

        """
        return f"Cell Complex with {len(self.nodes)} nodes, {len(self.edges)} edges  and {len(self.cells)} 2-cells "

    def __repr__(self):
        """
        String representation of cell complex

        Returns
        -------
        str

        """
        return f"CellComplex(name={self.name})"

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
        Iterate over the nodes of the cell complex

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
        item : hashable or RankedEntity

        """

        return item in self.nodes

    def __getitem__(self, node):
        """
        Returns the neighbors of node

        Parameters
        ----------
        node : Entity or hashable
            If hashable, then must be uid of node in cell complex

        Returns
        -------
        neighbors(node) : iterator

        """
        return self.neighbors(node)

    def degree(self, node, rank=1):
        """
        The number of cells of certain rank that contain node.

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
        return self._G.degree[node]

    def size(self, cell, node_set=None):
        """
        The number of nodes in node_set that belong to cell.
        If node_set is None then returns the size of cell

        Parameters
        ----------
        cell : hashable
            The uid of an cell in the cell complex

        Returns
        -------
        size : int

        """
        if node_set:
            return len(set(node_set).intersection(set(self.cells[cell])))
        else:
            if cell in self.cells:

                return len(self.cells[cell])
            else:
                raise KeyError(f" the key {cell} is not a key for an existing cell ")

    def number_of_nodes(self, node_set=None):
        """
        The number of nodes in node_set belonging to cell complex.

        Parameters
        ----------
        node_set : an interable of nodes, optional, default: None
            If None, then return the number of nodes in cell complex.

        Returns
        -------
        number_of_nodes : int

        """
        if node_set:
            return len([node for node in self.nodes if node in node_set])
        else:
            return len(self.nodes)

    def number_of_edges(self, edge_set=None):
        """
        The number of cells in cell_set belonging to cell complex.

        Parameters
        ----------
        cell_set : an interable of RankedEntities, optional, default: None
            If None, then return the number of cells in cell complex.

        Returns
        -------
        number_of_cells : int
        """
        if edge_set:
            return len([edge for edge in self.cells if edge in edge_set])
        else:
            return len(self.edges)

    def number_of_cells(self, cell_set=None):
        """
        The number of cells in cell_set belonging to cell complex.

        Parameters
        ----------
        cell_set : an interable of RankedEntities, optional, default: None
            If None, then return the number of cells in cell complex.

        Returns
        -------
        number_of_cells : int
        """
        if cell_set:
            return len([cell for cell in self.cells if cell in cell_set])
        else:
            return len(self.cells)

    def order(self):
        """
        The number of nodes in cc.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def neighbors(self, node):
        """
        The nodes in cell complex which share s cell(s) with node.

        Parameters
        ----------
        node : hashable or Entity
            uid for a node in cell complex or the node Entity

        s : int, list, optional, default : 1
            Minimum rank of cells shared by neighbors with node.

        Returns
        -------
         : list
            List of neighbors

        Example
        -------


        """
        if node not in self.nodes:
            print(f"Node is not in cell complex {self.name}.")
            return

        return self._G[node]

    def cell_neighbors(self, cell, s=1):
        """
        The cells in cell Complex which share s nodes(s) with cells.

        Parameters
        ----------
        cell : hashable or RankedEntity
            uid for a cell in cell complex or the cell RankedEntity

        s : int, list, optional, default : 1
            Minimum number of nodes shared by neighbors cell node.

        Returns
        -------
         : list
            List of cell neighbors

        """
        """
        if not cell in self.cells:
            print(f"cell is not in cc {self.name}.")


        node = self.cells[cell].uid
        return self.dual().neighbors(node, s=s)
        """
        raise NotImplementedError

    def remove_node(self, node):
        """Remove the given node from the cell complex.

        This method removes the given node from the cell complex, along with any
        cells that contain the node.

        Parameters
        ----------
        node : hashable
            The node to be removed from the cell complex.

        Raises
        ------
        TopoNetXError
            If the given node does not exist in the cell complex.

        """

        if node not in self.nodes:
            raise TopoNetXError("The given node does not exist in the cell complex.")
        # Remove the node from the cell complex
        self._G.remove_node(node)
        # Remove any cells that contain the node
        for cell in self.cells:
            if node in cell:
                self.remove_cell(cell)

    def remove_nodes(self, node_set):
        """Remove nodes from cells.

        This also deletes references in cell complex nodes.

        Parameters
        ----------
        node_set : an iterable of hashables or Entities
            Nodes in cc

        Returns
        -------
        cell complex : Cell Complex
        """
        for node in node_set:
            self.remove_node(node)

    def add_node(self, node, **attr):
        """Add a single node to cell complex."""
        self._G.add_node(node, **attr)

    def _add_nodes_from(self, nodes):
        """
        Private helper method instantiates new nodes when cells added to cell complex.

        Parameters
        ----------
        nodes : iterable of hashables or RankedEntities

        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self._G.add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self._G.add_edge_from(ebunch_to_add, **attr)

    def add_cell(self, cell, rank=None, check_skeleton=False, **attr):
        """Add a single cell to cell complex.

        Parameters
        ----------
        cell : hashable or RankedEntity
            If hashable the cell returned will be empty.
        uid : unique identifier that identifies the cell
        rank : rank of a cell, supported ranks is 1 or 2

        Returns
        -------
        Cell Complex : CellComplex

        Example
        -------
        >>> cx = CellComplex()
        >>> c1 = Cell((2, 3, 4), color='black')
        >>> cx.add_cell(c1, weight=3)
        >>> cx.add_cell([1, 2, 3, 4], rank=2, color='red')
        >>> cx.add_cell([2, 3, 4, 5], rank=2, color='blue')
        >>> cx.add_cell([5, 6, 7, 8], rank=2, color='green')
        >>> cx.cells[(1, 2, 3, 4)]['color']
        'red'


        Notes
        -----
        - Rank must be 0,1,2
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
                        self._cells.insert_cell(cell, **attr)
                    else:
                        raise TopoNetXError(
                            "input cell violates the regularity condition."
                        )
                else:
                    self._cells.insert_cell(cell, **attr)
            else:
                print(
                    "Invalid cycle condition, the input cell cannot be inserted to the cell complex"
                )
        else:
            if rank == 0:
                raise TopoNetXError(
                    "Use add_node to insert nodes or zero ranked cells."
                )
            elif rank == 1:
                if len(cell) != 2:
                    raise ValueError("rank 1 cells (edges) must have exactly two nodes")
                if len(set(cell)) == 1:
                    raise ValueError(" invalid insertion : self-loops are not allowed.")
                else:
                    self.add_edge(cell[0], cell[1], **attr)

            elif rank == 2:
                if isinstance(cell, Iterable):
                    if not isinstance(cell, list):
                        cell = list(cell)

                    if self.is_insertable_cycle(
                        cell, check_skeleton=check_skeleton, warnings_dis=True
                    ):

                        edges_cell = set(zip_longest(cell, cell[1:] + [cell[0]]))
                        for edge in edges_cell:
                            self._G.add_edges_from(edges_cell)
                        self._cells.insert_cell(
                            Cell(cell, regular=self._regular), **attr
                        )
                    else:
                        print(
                            "Invalid cycle condition, check if edges of the input cells are in the 1-skeleton."
                        )
                        print(" To ignore this check, set check_skeleton = False.")
                else:
                    raise ValueError("invalid input")
            else:
                raise ValueError(
                    f"Add cell only supports adding cells of dimensions 0,1 or 2-- got {rank}",
                )

        return self

    def add_cells_from(self, cell_set, rank=None, check_skeleton=False, **attr):
        """
        Add cells to cell complex .

        Parameters
        ----------
        cell_set : iterable of hashables or Cell
            For hashables the cells returned will be empty.

        rank : integer (optional), default is None
               when each element in cell_set is an iterable then
               rank must be a number that indicates the rank
               of the added cells.

        Returns
        -------
        Cell Complex : CellComplex

        """

        for cell in cell_set:
            self.add_cell(cell=cell, rank=rank, check_skeleton=check_skeleton, **attr)
        return self

    def remove_cell(self, cell):
        """
        Removes a single cell from Cell Complex.

        Parameters
        ----------
        cell : cell's node_set or Cell

        Returns
        -------
        Cell Complex : CellComplex

        Notes
        -----

        Deletes reference to cell, keep it boundary edges in the cell complex

        """
        if isinstance(cell, Cell):
            self._cells.delete_cell(cell.elements)
        elif isinstance(cell, Iterable):
            if not isinstance(cell, tuple):
                cell = tuple(cell)
            self._cells.delete_cell(cell)
        return self

    def remove_cells(self, cell_set):
        """
        Removes cells from a cell complex that are in cell_set.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities

        Returns
        -------
        cell complex : CellComplex

        """
        for cell in cell_set:
            self.remove_cell(cell)
        return self

    def clear(self):
        """
        Removes all cells from a cell complex.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities

        Returns
        -------
        cell complex : CellComplex

        """
        for cell in self.cells:
            self.remove_cell(cell)
        return self

    def set_filtration(self, values, name=None):
        """
        Parameters
        ----------
        values : dic
        name : str

        Returns
        -------
        None

        Note : this is equivalent to setting a real-valued feature defined on the entire cell complex


        Note : If the dict contains cells that are not in `self.cells`, they are
        silently ignored.


        Example
        -------
        >>> G = nx.path_graph(3)
        >>> cc = CellComplex(G)
        >>> d = {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}

        >>> cc.set_filtration(d, "f")

        """
        import numbers

        d_nodes, d_edges, d_cells = [{}, {}, {}]

        for k, v in values.items():

            # to do, make sure v is a number

            if not isinstance(v, (int, float)):
                raise ValueError(f"filtration value must be a int or float, input {v}")

            if isinstance(k, Hashable) and not isinstance(k, Iterable):  # node
                d_nodes[k] = v
            elif isinstance(k, Iterable) and len(k) == 2:  # edge
                d_edges[k] = v
            elif isinstance(k, Iterable) and len(k) != 2:  # cell
                d_cells[k] = v
        self.set_node_attributes(d_nodes, name)
        self.set_edge_attributes(d_edges, name)
        self.set_cell_attributes(d_cells, name)

    def get_filtration(self, name):

        """

        Note : this is equivalent to getting a feature defined on the entire cell complex

        >>> G = nx.path_graph(3)
        >>> cc = CellComplex(G)
        >>> d = {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}
        >>> cc.set_filtration(d, "f")
        >>> cc.get_filtration("f")

        {0: 1, 1: 0, 2: 2, (0, 1): 1, (1, 2): 3}

        """

        lst = [
            self.get_node_attributes(name),
            self.get_edge_attributes(name),
            self.get_cell_attributes(name),
        ]
        d = {}
        for i in lst:
            if i is not None:
                d.update(i)
        return d

    def set_node_attributes(self, values, name=None):
        """
        Parameters
        ----------
        values : dic
        name : str

        Returns
        -------
        None

        Example
        -------
        >>> G = nx.path_graph(3)
        >>> cc = CellComplex(G)
        >>> d = {0: {'color': 'red', 'attr2': 1 }, 1: {'color': 'blue', 'attr2': 3}}
        >>> cc.set_node_attributes(d)

        """

        if name is not None:
            # if `values` is a dict using `.items()` => {cell: value}

            for cell, value in values.items():
                try:
                    self.nodes[cell][name] = value
                except KeyError:
                    pass
        else:

            for cell, d in values.items():
                try:
                    self.nodes[cell].update(d)
                except KeyError:
                    pass
            return

    def set_edge_attributes(self, values, name=None):
        """
        Parameters
        ----------
        values : dic
        name : str

        Returns
        -------
        None


        Example
        -------

        >>> G = nx.path_graph(3)
        >>> cc = CellComplex(G)
        >>> d={ (0,1) : {'color':'red','attr2':1 },(1,2): {'color':'blue','attr2':3 } }
        >>> cc.set_edge_attributes(d)

        """

        if name is not None:
            # if `values` is a dict using `.items()` => {edge: value}

            for cell, value in values.items():
                try:
                    self.edges[cell][name] = value
                except KeyError:
                    pass
        else:
            for cell, d in values.items():
                try:
                    self.edges[cell].update(d)
                except KeyError:
                    pass
            return

    def get_edge_attributes(self, name):
        """Get edge attributes from cell complex

        Parameters
        ----------

        name : string
           Attribute name

        Returns
        -------
        Dictionary of attributes keyed by edge.

        """
        return {
            edge: self.edges[edge][name]
            for edge in self.edges
            if name in self.edges[edge]
        }

    def set_cell_attributes(self, values, name=None):
        """


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

            After computing some property of the cell of a cell complex, you may want
            to assign a cell attribute to store the value of that property for
            each cell:

            >>> cc = CellComplex()
            >>> cc.add_cell([1,2,3,4], rank=2)
            >>> cc.add_cell([1,2,4], rank=2,)
            >>> cc.add_cell([3,4,8], rank=2)
            >>> d={(1,2,3,4):'red',(1,2,4):'blue'}
            >>> cc.set_cell_attributes(rank,name='color')
            >>> cc.cells[(1,2,3,4)]['color']
            'red'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes::

            Examples
            --------
            >>> G = nx.path_graph(3)
            >>> cc = CellComplex(G)
            >>> cc.add_cell([1,2,3,4], rank=2)
            >>> cc.add_cell([1,2,3,4], rank=2)
            >>> cc.add_cell([1,2,4], rank=2,)
            >>> cc.add_cell([3,4,8], rank=2)
            >>> d={ (1,2,3,4): { 'color':'red','attr2':1 },(1,2,4): {'color':'blue','attr2':3 } }
            >>> cc.set_cell_attributes(d)
            >>> cc.cells[(1,2,3,4)][0]['color']
            'red'

        Note : If the dict contains cells that are not in `self.cells`, they are
        silently ignored.

        """

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

    def get_node_attributes(self, name):
        """Get node attributes from cell complex

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
        >>> cc = CellComplex(G)
        >>> d={0: {'color':'red','attr2':1 },1: {'color':'blue','attr2':3 } }
        >>> cc.set_node_attributes(d)
        >>> cc.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> cc = CellComplex(G)
        >>> nodes_color = cc.get_node_attributes('color')
        >>> nodes_color[1]
        'blue'

        """
        return {
            node: self.nodes[node][name]
            for node in self.nodes
            if name in self.nodes[node]
        }

    def get_cell_attributes(self, name, rank=None):
        """Get node attributes from graph

        Parameters
        ----------

        name : string
           Attribute name

        k : integer rank of the k-cell
        Returns
        -------
        Dictionary of attributes keyed by cell or k-cells if k is not None

        Examples
        --------
        >>> import networkx as nx
        >>> G = nx.path_graph(3)

        >>> d={ ((1,2,3,4),0): { 'color':'red','attr2':1 },(1,2,4): {'color':'blue','attr2':3 } }
        >>> cc = CellComplex(G)
        >>> cc.add_cell([1,2,3,4], rank=2)
        >>> cc.add_cell([1,2,3,4], rank=2)
        >>> cc.add_cell([1,2,4], rank=2,)
        >>> cc.add_cell([3,4,8], rank=2)
        >>> cc.set_cell_attributes(d)
        >>> cell_color=cc.get_cell_attributes('color',2)
        >>> cell_color
        '{((1, 2, 3, 4), 0): 'red', (1, 2, 4): 'blue'}
        """
        if rank is not None:
            if rank == 0:
                return self.get_cell_attributes(name)
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
            raise TopoNetXError(f"Rank must be 0, 1 or 2, got {rank}")

    def remove_equivalent_cells(self):
        """
        Remove cells from the cell complex which are homotopic.
        In other words, this is equivalent to identifying cells
        containing the same nodes and are equivalent up to cyclic
        permutation.

         Note
         ------
         Remove all 2d- cells that are homotpic (equivalent to each other)

         Returns
         -------
         None.

         Example
         -------
            >>> import networkx as nx
            >>> G = nx.path_graph(3)
            >>> cc = CellComplex(G)
            >>> cc.add_cell([1,2,3,4], rank=2)
            >>> cc.add_cell([1,2,3,4], rank=2)
            >>> cc.add_cell([2,3,4,1], rank=2)
            >>> cc.add_cell([1,2,4], rank=2,)
            >>> cc.add_cell([3,4,8], rank=2)
            >>> cc.remove_equivalent_cells()

        """
        self.cells.remove_equivalent_cells()

    def is_insertable_cycle(self, cell, check_skeleton=True, warnings_dis=False):

        if isinstance(cell, Cell):
            cell = cell.elements
        if len(cell) <= 1:
            if warnings_dis:
                warnings.warn(f"a cell must contain at least 2 edges, got {len(cell)}")
            return False
        if self._regular:
            if len(set(cell)) != len(cell):
                if warnings_dis:
                    warnings.warn(
                        "repeating nodes invalidates the 2-cell regular condition"
                    )
                return False
        if check_skeleton:
            enum = zip_longest(cell, cell[1:] + [cell[0]])
            for i in enum:
                if i not in self.edges:
                    if warnings_dis:
                        warnings.warn(
                            f"edge {i} is not a part of the 1 skeleton of the cell complex",
                            stacklevel=2,
                        )
                    return False
        return True

    def incidence_matrix(self, rank, signed=True, weight=None, index=False):
        """
        An incidence matrix for the cc indexed by nodes x cells.

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
        incidence_matrix : scipy.sparse.csr.csr_matrix

        row list : list
            list of cells in the complex with the same
            order of the row of the matrix

        column list : list
            list of cells in the complex with the same
            order of the column of the matrix
        Example1
        -------
            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3,4],rank=2)
            >>> cx.add_cell([3,4,5],rank=2)
            >>> B0 = cx.incidence_matrix(0)
            >>> B1 = cx.incidence_matrix(1)
            >>> B2 = cx.incidence_matrix(2)
            >>> B1.dot(B2).todense()
            >>> B0.dot(B1).todense()

        Example2
        --------
            ## note that in this example, the first three cells are
            ## equivalent and hence they have similar incidence to lower edges
            ## they are incident to
            >>> import networkx as nx
            >>> G = nx.path_graph(3)
            >>> cx = CellComplex(G)
            >>> cx.add_cell([1,2,3,4], rank=2)
            >>> cx.add_cell([4,3,2,1], rank=2)
            >>> cx.add_cell([2,3,4,1], rank=2)
            >>> cx.add_cell([1,2,4], rank=2,)
            >>> cx.add_cell([3,4,8], rank=2)
            >>> B1 = cx.incidence_matrix(1)
            >>> B2 = cx.incidence_matrix(2)
            >>> B1.dot(B2).todense()

        Example3
        -------
            # non-regular complex example
            >>> cx = CellComplex(regular=False)
            >>> cx.add_cell([1,2,3,2],rank=2)
            >>> cx.add_cell([3,4,5,3,4,5],rank=2)
            >>> B1 = cx.incidence_matrix(1)
            >>> B2 = cx.incidence_matrix(2)
            >>> print(B2.todense()) # observe the non-unit entries
            >>> B1.dot(B2).todense()
        Example4
        -------
            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3,4],rank=2)
            >>> cx.add_cell([3,4,5],rank=2)
            >>> row,column,B1 = cx.incidence_matrix(1,index=True)
            >>> print(row)
            >>> print(column)
            >>> print(B1.todense())

        """
        import scipy as sp
        import scipy.sparse

        if rank == 0:
            A = sp.sparse.lil_matrix((1, len(self._G.nodes)))
            for i in range(0, len(self._G.nodes)):
                A[0, i] = 1
            if index:
                node_index = {node: i for i, node in enumerate(sorted(self._G.nodes))}
                if signed:
                    return node_index, [], A.asformat("csc")
                else:
                    return node_index, [], abs(A.asformat("csc"))
            else:
                if signed:
                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))

        elif rank == 1:
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
                    return node_index, edge_index, A.asformat("csc")
                else:
                    return node_index, edge_index, abs(A.asformat("csc"))
            else:
                if signed:
                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))
        elif rank == 2:
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
                    ]  # this will update everytime we visit this edge for non-regular cc
                    # the regular case can be handled more efficiently :
                    # if edge in edge_index:
                    #    A[ei, celli] = 1
                    # else:
                    #    A[ei, celli] = -1
            if index:
                cell_index = {cell.elements: i for i, c in enumerate(self.cells)}
                if signed:
                    return edge_index, cell_index, A.asformat("csc")
                else:
                    return edge_index, cell_index, abs(A.asformat("csc"))
            else:
                if signed:
                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))
        else:
            raise ValueError(f"Only dimension 0,1 and 2 are supported, got {d}")

    @staticmethod
    def _incidence_to_adjacency(inc, weight=False):
        """
        Helper method to obtain adjacency matrix from
        boolean incidence matrix for s-metrics.
        Self loops are not supported.
        The adjacency matrix will define an s-linegraph.

        Parameters
        ----------
        inc : scipy.sparse.csr.csr_matrix
            incidence matrix of 0's and 1's


        # weight : bool, dict optional, default=True
        #     If False all nonzero entries are 1.
        #     Otherwise, weight will be as in product.

        Returns
        -------
        a matrix : scipy.sparse.csr.csr_matrix

        >>> cx = CellComplex()
        >>> cx.add_cell([1,2,3,5,6],rank=2)
        >>> cx.add_cell([1,2,4,5,3,0],rank=2)
        >>> cx.add_cell([1,2,4,9,3,0],rank=2)
        >>> B1 = cx.incidence_matrix(1,signed = False)
        >>> A1 = cx._incidence_to_adjacency(B1)

        """

        inc = csr_matrix(inc)
        weight = False  # currently weighting is not supported
        inc = abs(inc)  # make sure the incidence matrix has only positive entries
        if weight is False:
            adj = inc.dot(inc.transpose())
            adj.setdiag(0)
        return adj

    def hodge_laplacian_matrix(self, rank, signed=True, weight=None, index=False):
        """Compute the hodge-laplacian matrix for the cc.

        Parameters
        ----------
        rank : int, dimension of the Laplacian matrix.
            Supported dimension are 0,1 and 2

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

        index : boolean, optional, default False
                indicates wheather to return the indices that define the incidence matrix


        Returns
        -------
        Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension



        Example1
        -------
            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3,4],rank=2)
            >>> cx.add_cell([3,4,5],rank=2)
            >>> L1 = cx.hodge_laplacian_matrix(1)


        Example2
        --------
            ## note that in this example, the first three cells are
            ## equivalent and hence they have similar incidence to lower edges
            ## they are incident to
            >>> import networkx as nx
            >>> G = nx.path_graph(3)
            >>> cx = CellComplex(G)
            >>> cx.add_cell([1,2,3,4], rank=2)
            >>> cx.add_cell([4,3,2,1], rank=2)
            >>> cx.add_cell([2,3,4,1], rank=2)
            >>> cx.add_cell([1,2,4], rank=2,)
            >>> cx.add_cell([3,4,8], rank=2)
            >>> B1 = cx.incidence_matrix(1)
            >>> B2 = cx.incidence_matrix(2)
            >>> B1.dot(B2).todense()

        Example3
        -------
            # non-regular complex example
            >>> cx = CellComplex(regular=False)
            >>> cx.add_cell([1,2,3,2],rank=2)
            >>> cx.add_cell([3,4,5,3,4,5],rank=2)
            >>> B1 = cx.incidence_matrix(1)
            >>> B2 = cx.incidence_matrix(2)
            >>> B1.dot(B2).todense()

        """

        if rank == 0:  # return L0, the unit graph laplacian
            if index:
                nodelist, _, inc_next = self.incidence_matrix(
                    rank + 1, weight=weight, index=True
                )
                lap = inc_next @ inc_next.transpose()
                if signed:
                    return nodelist, lap
                else:
                    return nodelist, abs(lap)
            else:
                inc_next = self.incidence_matrix(rank + 1, weight=weight)
                lap = inc_next @ inc_next.transpose()
                if signed:
                    return lap
                else:
                    return abs(lap)
        elif rank < 2:  # rank == 1, return L1

            if self.maxdim == 2:
                edge_list, cell_list, inc_next = self.incidence_matrix(
                    rank + 1, weight=weight, index=True
                )
                inc = self.incidence_matrix(rank, weight=weight, index=False)
                lap = inc_next @ inc_next.transpose() + inc.transpose() @ inc
            else:
                lap = inc.transpose() @ inc
            if not signed:
                lap = abs(lap)
            if index:
                return edge_list, lap
            else:
                return lap

        elif rank == 2 and self.maxdim == 2:

            edge_list, cell_list, inc = self.incidence_matrix(
                rank, weight=weight, index=True
            )
            lap = inc.transpose() @ inc
            if not signed:
                lap = abs(lap)

            if index:
                return cell_list, lap
            else:
                return lap
        elif rank == 2 and self.maxdim != 2:
            raise ValueError(
                f"the input complex does not have cells of dim 2, max cell dim is {self.maxdim} (maximal dimension cells), got {d}"
            )
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.maxdim} (maximal dimension cells), got {d}"
            )

    def up_laplacian_matrix(self, rank, signed=True, weight=None, index=False):
        """

        Parameters
        ----------
        d : int, dimension of the up Laplacian matrix.
            Supported dimension are 0,1

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

        index : boolean, optional, default False
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension
        Returns
        -------
        up Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension


        Example1
        -------
            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3,4],rank=2)
            >>> cx.add_cell([3,4,5],rank=2)
            >>> L1_up = cx.up_laplacian_matrix(1)

        Example2
        -------
            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3],rank=2)
            >>> cx.add_cell([3,4,5],rank=2)
            >>> index , L1_up = cx.up_laplacian_matrix(1,index=True)
            >>> print(index)
            >>> print(L1_up)


        """

        weight = None  # this feature is not supported in this version

        if rank == 0:
            row, col, inc_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            lap_up = inc_next @ inc_next.transpose()
        elif rank < self.maxdim:
            row, col, inc_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            lap_up = inc_next @ inc_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension cells-1), got {d}"
            )
        if not signed:
            lap_up = abs(lap_up)

        if index:
            return row, lap_up
        else:
            return lap_up

    def down_laplacian_matrix(self, rank, signed=True, weight=None, index=False):
        """

        Parameters
        ----------
        d : int, dimension of the down Laplacian matrix.
            Supported dimension are 0,1

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

        index : boolean, optional, default False
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension
        Returns
        -------
        down Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension

        Example
        -------
          >>> import networkx as nx
          >>> G = nx.path_graph(3)
          >>> cc = CellComplex(G)
          >>> cc.add_cell([1,2,3,4], rank=2)
          >>> cc.add_cell([1,2,3,4], rank=2)
          >>> cc.add_cell([2,3,4,1], rank=2)
          >>> cc.add_cell([1,2,4], rank=2,)
          >>> cc.add_cell([3,4,8], rank=2)
          >>> cc.down_laplacian_matrix(2)


        """
        weight = None  # this feature is not supported in this version

        if rank <= self.maxdim and rank > 0:
            row, column, inc = self.incidence_matrix(rank, weight=weight, index=True)
            lap_down = inc.transpose() @ inc
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {self.maxdim} (maximal dimension cells), got {rank}."
            )
        if not signed:
            lap_down = abs(lap_down)
        if index:
            return row, lap_down
        else:
            return lap_down

    def adjacency_matrix(self, rank, signed=False, weight=None, index=False):
        """Compute adjacency matrix for a given rank."""
        weight = None  # this feature is not supported in this version

        ind, lap_up = self.up_laplacian_matrix(
            rank, signed=signed, weight=weight, index=True
        )
        lap_up.setdiag(0)

        if not signed:
            lap_up = abs(lap_up)
        if index:
            return ind, lap_up
        else:
            return lap_up

    def coadjacency_matrix(self, rank, signed=False, weight=None, index=False):
        """Compute coadjacency matrix for a given rank."""
        weight = None  # this feature is not supported in this version

        ind, lap_down = self.down_laplacian_matrix(
            rank, signed=signed, weight=weight, index=True
        )
        lap_down.setdiag(0)
        if not signed:
            lap_down = abs(lap_down)
        if index:
            return ind, lap_down
        else:
            return lap_down

    def k_hop_incidence_matrix(self, rank, k):
        """Compute k-hop incidence matrix for a given rank."""
        inc = self.incidence_matrix(rank, signed=True)
        if rank < self.maxdim and rank >= 0:
            adj = self.adjacency_matrix(rank, signed=True)
        if rank <= self.maxdim and rank > 0:
            coadj = self.coadjacency_matrix(rank, signed=True)
        if rank == self.maxdim:
            return inc @ np.power(coadj, k)
        elif rank == 0:
            return inc @ np.power(adj, k)
        else:
            return inc @ np.power(adj, k) + inc @ np.power(coadj, k)

    def k_hop_coincidence_matrix(self, rank, k):
        coinc = self.coincidence_matrix(rank, signed=True)
        if rank < self.maxdim and rank >= 0:
            adj = self.adjacency_matrix(rank, signed=True)
        if rank <= self.maxdim and rank > 0:
            coadj = self.coadjacency_matrix(rank, signed=True)
        if rank == self.maxdim:
            return np.power(coadj, k) @ coinc
        elif rank == 0:
            return np.power(adj, k) @ coinc
        else:
            return np.power(adj, k) @ coinc + np.power(coadj, k) @ coinc

    def cell_adjacency_matrix(self, signed=True, weight=None, index=False):
        """Compute adjacency matrix.

        Example
        -------
        >>> cx = CellComplex()
        >>> cx.add_cell([1,2,3],rank=2)
        >>> cx.add_cell([1,4],rank=1)
        >>> A = cx.cell_adjacency_matrix()
        """
        cc = self.to_combinatorial_complex()

        inc = cc.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:

            adj = cc._incidence_to_adjacency(inc[0].transpose())

            return adj, inc[2]
        else:
            adj = cc._incidence_to_adjacency(inc.transpose())
            return adj

    def node_adjacency_matrix(self, index=False, s=1, weight=False):

        cc = self.to_combinatorial_complex()

        inc = cc.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:

            adj = cc._incidence_to_adjacency(inc[0], s=s)

            return adj, inc[1]
        else:
            adj = cc._incidence_to_adjacency(inc, s=s)
            return adj

    def restrict_to_cells(self, cell_set, name=None):
        """
        Constructs a cell complex using a subset of the cells in cell complex

        Parameters
        ----------
        cell_set: iterable of hashables or Cell
            A subset of elements of the cell complex's cells (self.cells)

        name: str, optional

        Returns
        -------
        new cell complex : CellComplex

        Example

        >>> cx = CellComplex()
        >>> c1 = Cell((1, 2, 3))
        >>> c2 = Cell((1, 2, 4))
        >>> c3 = Cell((1, 2, 5))
        >>> cx = CellComplex([c1, c2, c3])
        >>> cx.add_edge(1, 0)
        >>> cx1 = cx.restrict_to_cells([c1, (0, 1)])
        >>> cx1.cells
        CellView([Cell(1, 2, 3)])

        """
        rns = []
        edges = []
        for cell in cell_set:
            if cell in self.cells:
                rns.append(cell)
            elif edge in self.edges:
                edges.append(edge)

        cx = CellComplex(cells=rns, name=name)
        for edge in edges:
            cx.add_edge(edge[0], edge[1])
        return cx

    def restrict_to_nodes(self, node_set, name=None):
        """Restrict cell complex to nodes.

        This constructs a new cell complex  by restricting the cells in the cell complex to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set: iterable of hashables
            References a subset of elements of self.nodes

        name: string, optional, default: None

        Returns
        -------
        new Cell Complex : Cellcomplex

        Example
        >>> cx = CellComplex()
        >>> c1 = Cell((1, 2, 3))
        >>> c2 = Cell((1, 2, 4))
        >>> c3 = Cell((1, 2, 5))
        >>> cx = CellComplex([c1, c2, c3])
        >>> cx.add_edge(1, 0)
        >>> cx.restrict_to_nodes([1, 2, 3, 0])
        """

        _G = Graph(self._G.subgraph(node_set))
        cx = CellComplex(_G)
        cells = []
        for cell in self.cells:
            if cx.is_insertable_cycle(cell, True):
                cells.append(cell)
        cx = CellComplex(_G)

        for cell in cells:
            cx.add_cell(cell)
        return cx

    def to_combinatorial_complex(self):
        """Convert to combinatorial complex.

        A cell complex is a type of combinatorial complex.
        The rank of an element in a cell complex is its dimension, so vertices have rank 0,
        edges have rank 1, and faces have rank 2.

        Example
        -------
        >>> cx = CellComplex()
        >>> cx.add_cell([1,2,3,4],rank=2)
        >>> cx.add_cell([2,3,4,5],rank=2)
        >>> cx.add_cell([5,6,7,8],rank=2)
        >>> cc= cx.to_combinatorial_complex()
        >>> cc.cells
        """
        all_cells = []

        for node in self.nodes:
            all_cells.append(Node(elements=node, **self.nodes[node]))

        for edge in self.edges:
            all_cells.append(DynamicCell(elements=edge, rank=1, **self.edges[edge]))
        for cell in self.cells:
            all_cells.append(
                DynamicCell(elements=cell.elements, rank=2, **self.cells[cell])
            )
        return CombinatorialComplex(
            RankedEntitySet("", all_cells, safe_insert=False), name="_"
        )

    def to_hypergraph(self):
        """Convert to hypergraph.

        Example
        -------
        >>> cx = CellComplex()
        >>> cx.add_cell([1,2,3,4],rank=2)
        >>> cx.add_cell([2,3,4,5],rank=2)
        >>> cx.add_cell([5,6,7,8],rank=2)
        >>> HG = cx.to_hypergraph()
        >>> HG

        """
        from hypernetx.classes.entity import EntitySet

        cells = []
        for cell in self.cells:
            cells.append(Entity(str(list(cell.elements)), elements=cell.elements))
        for cell in self.edges:
            cells.append(Entity(str(list(cell)), elements=cell))
        E = EntitySet("CX_to_HG", elements=cells)
        HG = Hypergraph(E)
        nodes = []
        for cell in self.nodes:
            nodes.append(Entity(cell, elements=[]))
        HG._add_nodes_from(nodes)
        return HG

    def is_connected(self, s=1, cells=False):
        """
        Determines if cell complex is :term:`s-connected <s-connected, s-node-connected>`.

        Parameters
        ----------
        s: int, optional, default: 1

        cells: boolean, optional, default: False
            If True, will determine if s-cell-connected.
            For s=1 s-cell-connected is the same as s-connected.

        Returns
        -------
        is_connected : boolean

        Notes
        -----

        A cx is s node connected if for any two nodes v0,vn
        there exists a sequence of nodes v0,v1,v2,...,v(n-1),vn
        such that every consecutive pair of nodes v(i),v(i+1)
        share at least s cell.


        """
        import networkx as nx

        return nx.is_connected(self._G)

    def singletons(self):
        """
        Returns a list of singleton cell. A singleton cell is a node of degree 0.

        Returns
        -------
        singles : list
            A list of cells uids.

            >>> cx = CellComplex()
            >>> cx.add_cell([1,2,3,4],rank=2)
            >>> cx.add_cell([2,3,4,5],rank=2)
            >>> cx.add_cell([5,6,7,8],rank=2)
            >>> cx.add_node(0)
            >>> cx.add_node(10)
            >>> cx.singletons()

        """

        return [node for node in self.nodes if self.degree(node) == 0]

    def remove_singletons(self, name=None):
        """Remove singletons.

        This constructs clone of cell complex with singleton cells removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        Cell Complex : CellComplex
        """

        for node in self.singletons():
            self._G.remove_node(node)

    def s_connected_components(self, s=1, cells=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-cell-connected components <s-cell-connected component>`
        or the :term:`s-node-connected components <s-connected component, s-node-connected component>`
        of the cell complex.

        Parameters
        ----------
        s : int, optional, default: 1

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
        s nodes. If s=1 these are the path components of the cc.

        If cells=False this method returns s-node-connected components.
        A list of sets of uids of the nodes which are s-walk connected.
        Two nodes v1 and v2 are s-walk-connected if there is a
        sequence of nodes starting with v1 and ending with v2 such that pairwise
        adjacent nodes in the sequence share s cells. If s=1 these are the
        path components of the cell complex .

        Example
        -------


        Yields
        ------
        s_connected_components : iterator
            Iterator returns sets of uids of the cells (or nodes) in the s-cells(node)
            components of cc.

        """

        if cells:
            adj, coldict = self.cell_adjacency_matrix(s=s, index=True)
            graph = nx.from_scipy_sparse_matrix(adj)

            for c in nx.connected_components(graph):
                if not return_singletons and len(c) == 1:
                    continue
                yield {coldict[n] for n in c}
        else:
            adj, rowdict = self.node_adjacency_matrix(s=s, index=True)
            graph = nx.from_scipy_sparse_matrix(adj)
            for c in nx.connected_components(graph):
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
        s : int, optional, default: 1

        cells : boolean, optional, cells=False
            Determines if cell or node components are desired. Returns
            subgraphs equal to the cc restricted to each set of nodes(cells) in the
            s-connected components or s-cell-connected components
        return_singletons : bool, optional

        Yields
        ------
        s_component_subgraphs : iterator
            Iterator returns subgraphs generated by the cells (or nodes) in the
            s-cell(node) components of cell complex.

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
        Returns the node diameters of the connected components in cell complex.

        Parameters
        ----------
        list of the diameters of the s-components and
        list of the s-component nodes
        """

        adj, coldict = self.node_adjacency_matrix(s=s, index=True)
        graph = nx.from_scipy_sparse_matrix(adj)
        diams = []
        comps = []
        for c in nx.connected_components(graph):
            diamc = nx.diameter(graph.subgraph(c))
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
        in cc.

        Parameters
        ----------
        s : int, optional, default: 1

        Returns
        -------
        maximum diameter : int

        list of diameters : list
            List of cell_diameters for s-cell component subgraphs in cc

        list of component : list
            List of the cell uids in the s-cell component subgraphs.

        """

        adj, coldict = self.cell_adjacency_matrix(s=s, index=True)
        graph = nx.from_scipy_sparse_matrix(adj)
        diams = []
        comps = []
        for c in nx.connected_components(graph):
            diamc = nx.diameter(graph.subgraph(c))
            temp = set()
            for e in c:
                temp.add(coldict[e])
            comps.append(temp)
            diams.append(diamc)
        loc = np.argmax(diams)
        return diams[loc], diams, comps

    def diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between nodes in cell complex

        Parameters
        ----------
        s : int, optional, default: 1

        Returns
        -------
        diameter : int

        Raises
        ------
        TopoNetXError
            If cc is not s-cell-connected

        Notes
        -----
        Two nodes are s-adjacent if they share s cells.
        Two nodes v_start and v_end are s-walk connected if there is a sequence of
        nodes v_start, v_1, v_2, ... v_n-1, v_end such that consecutive nodes
        are s-adjacent. If the graph is not connected, an error will be raised.

        """
        adj = self.node_adjacency_matrix(s=s)
        graph = nx.from_scipy_sparse_matrix(adj)
        if nx.is_connected(graph):
            return nx.diameter(graph)
        raise TopoNetXError(f"cc is not s-connected. s={s}")

    def cell_diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between cells in cell complex

        Parameters
        ----------
        s : int, optional, default: 1

        Return
        ------
        cell_diameter : int

        Raises
        ------
        TopoNetXError
            If cell complex is not s-cell-connected

        Notes
        -----
        Two cells are s-adjacent if they share s nodes.
        Two nodes e_start and e_end are s-walk connected if there is a sequence of
        cells e_start, e_1, e_2, ... e_n-1, e_end such that consecutive cells
        are s-adjacent. If the graph is not connected, an error will be raised.

        """

        adj = self.cell_adjacency_matrix(s=s)
        graph = nx.from_scipy_sparse_matrix(adj)
        if nx.is_connected(graph):
            return nx.diameter(graph)
        raise TopoNetXError(f"cell complex is not s-connected. s={s}")

    def distance(self, source, target, s=1):
        """
        Returns the shortest s-walk distance between two nodes in the cell complex.

        Parameters
        ----------
        source : node.uid or node
            a node in the cc

        target : node.uid or node
            a node in the cc

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

        if isinstance(source, Cell):
            source = source.uid
        if isinstance(target, Cell):
            target = target.uid
        adj, rowdict = self.node_adjacency_matrix(s=s, index=True)
        graph = nx.from_scipy_sparse_matrix(adj)
        rkey = {v: k for k, v in rowdict.items()}
        try:
            path = nx.shortest_path_length(graph, rkey[source], rkey[target])
            return path
        except Exception:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf

    def cell_distance(self, source, target, s=1):
        """
        Returns the shortest s-walk distance between two cells in the cell complex.

        Parameters
        ----------
        source : cell.uid or cell
            an cell in the cell complex

        target : cell.uid or cell
            an cell in the cell complex

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

        if isinstance(source, Cell):
            source = source.uid
        if isinstance(target, Cell):
            target = target.uid
        adj, coldict = self.cell_adjacency_matrix(s=s, index=True)
        graph = nx.from_scipy_sparse_matrix(adj)
        ckey = {v: k for k, v in coldict.items()}
        try:
            path = nx.shortest_path_length(graph, ckey[source], ckey[target])
            return path
        except Exception:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf

    def from_networkx_graph(self, G):
        """
        add edges and nodes from a a graph G to self

        >>> cx = CellComplex()
        >>> cx.add_cells_from([[1,2,4],[1,2,7] ],rank=2)

        >>> G= Graph()
        >>> G.add_edge(1,0)
        >>> G.add_edge(2,0)
        >>> G.add_edge(1,2)
        >>> cx.from_networkx_graph(G)
        >>> cx.edges
        """

        for edge in G.edges:
            self.add_cell(edge, rank=1)
        for node in G.nodes:
            self.add_node(node)

    @staticmethod
    def from_trimesh(mesh):
        """
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                               faces=[[0, 1, 2]],
                               process=False)
        >>> cx = CellComplex.from_trimesh(mesh)
        >>> print(cx.nodes)
        >>> print(cx.cells)
        >>> cx.nodes[0]['position']

        """
        # try to see the index of the first vertex

        cx = CellComplex(mesh.faces)

        first_ind = np.min(mesh.faces)

        if first_ind == 0:

            cx.set_node_attributes(
                dict(zip(range(len(mesh.vertices)), mesh.vertices)), name="position"
            )
        else:  # first index starts at 1.

            cx.set_node_attributes(
                dict(
                    zip(range(first_ind, len(mesh.vertices) + first_ind), mesh.vertices)
                ),
                name="position",
            )

        return cx

    @staticmethod
    def load_mesh(file_path, process=False, force=None):
        """
        file_path: str, the file path of the data to be loadeded

        process : bool, trimesh will try to process the mesh before loading it.

        force: (str or None)
            options: 'mesh' loader will "force" the result into a mesh through concatenation
                     None : will not force the above.
        Note:
            file supported : obj, off, glb
        >>> cx = CellComplex.load_mesh("bunny.obj")

        >>> cx.nodes

        """
        import trimesh

        mesh = trimesh.load_mesh(file_path, process=process, force=None)
        return CellComplex.from_trimesh(mesh)
