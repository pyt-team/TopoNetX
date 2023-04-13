"""
# --------------------------------------------------------
# Class supporting basic functions and constructions methods over Combinatorial Complex
# --------------------------------------------------------
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
from hypernetx.classes.entity import Entity, EntitySet
from networkx import Graph
from scipy.sparse import csr_matrix

from toponetx.classes.complex import Complex
from toponetx.classes.dynamic_cell import DynamicCell
from toponetx.classes.node import Node
from toponetx.classes.ranked_entity import RankedEntity, RankedEntitySet
from toponetx.exception import TopoNetXError

__all__ = ["DynamicCombinatorialComplex"]


class DynamicCombinatorialComplex(Complex):
    """Class for (Dynamic) Combinatorial Complex.

    A Combinatorial Complex (CC) is a triple CC=(S, X, i) where S is an abstract set of entities,
    X a subset of the power set of X and i is the a rank function that associates for every
    set x in X a rank.

    A CC is a generlization of graphs, hyppergraphs, cellular and simplicial complexes.

    In TopoNetX, DynamicCombinatorialComplex implements CCs dynamically and the user is allowed to
    add or subtract objects to the CC after the initial construction.
    The dynamic structure of CCs require the user to keep track of its objects,
    by using a unique names for each cell.

    The DynamicCombinatorialComplex class is a mathematical structure used to study the
    combinatorial properties of sets and collections of objects. It is made up of a set
    of entities called vertices, a subset of the power set of vertices called cells, and
    a ranking function that assigns a rank to each cell based on its size. The class supports
    basic functions and construction methods over combinatorial complexes, and allows the user
    to dynamically add or subtract objects from the complex after its initial construction.
    It is used in many areas of mathematics, such as graph theory, algebraic topology,
    and discrete geometry.

    In a CC each cell is instantiated as an RankedEntity (see the ranked_entity.py)
    and given an identifier or uid. Ranked Entities
    keep track of inclusion and ranking relationships and can be Dynamic.

    Mathematical example
    ---------------------
    Let S = {1, 2, 3, 4} be a set of entities.
    Let X = {{1, 2}, {1, 2, 3}, {1, 3}, {1, 4}} be a subset of the power set of S.
    Let i be the ranking function that assigns the
    length of a set as its rank, i.e. i({1, 2}) = 2, i({1, 2, 3}) = 3, etc.

    Then, (S, X, i) is a combinatorial complex.

    Example 0
        >>> # DynamicCombinatorialComplex can be empty
        >>> CC = DynamicCombinatorialComplex()
    Example 1
        >>> # from the constructor pass a list of cells and a list of correponding ranks
        >>> CC = DynamicCombinatorialComplex(cells=[[1, 2, 3], [2, 3], [0]], ranks=[2,1,0])
    Example 2
        >>> # create a collection of Node objects and a collection of DynamicCells
        >>> # and pass them to RankedEntitySet which is then passed to DynamicCombinatorialComplex
        >>> x1 = Node(1)
        >>> x2 = Node(2)
        >>> x3 = Node(3)
        >>> x4 = Node(4)
        >>> x5 = Node(5)
        >>> y1 = DynamicCell(elements=[x1, x2], rank=1)
        >>> y2 = DynamicCell(elements=[x2, x3], rank=1)
        >>> y3 = DynamicCell(elements=[x3, x4], rank=1)
        >>> y4 = DynamicCell(elements=[x4, x1], rank=1)
        >>> y5 = DynamicCell(elements=[x4, x5], rank=1)
        >>> w = DynamicCell(elements=[x4, x5, x1], rank=2)
        >>> # define the Ranked Entity Set
        >>> E = RankedEntitySet('E', [y1, y2, y3, y4, y5, w])
        >>> # pass the ranked entity set to the DynamicCombinatorialComplex constructor
        >>> CC = DynamicCombinatorialComplex(cells=E)

    Example 3
        >>> # create a collection of Node objects and a collection of DynamicCells
        >>> # and pass them to DynamicCombinatorialComplex
        >>> x1 = Node(1)
        >>> x2 = Node(2)
        >>> x3 = Node(3)
        >>> x4 = Node(4)
        >>> x5 = Node(5)
        >>> y1 = DynamicCell(elements=[x1, x2], rank=1)
        >>> y2 = DynamicCell(elements=[x2, x3], rank=1)
        >>> y3 = DynamicCell(elements=[x3, x4], rank=1)
        >>> y4 = DynamicCell(elements=[x4, x1], rank=1)
        >>> y5 = DynamicCell(elements=[x4, x5], rank=1)
        >>> w = DynamicCell(elements=[x4, x5, x1], rank=2)
        # define the DynamicCombinatorialComplex from a list of cells
        >>> CC = DynamicCombinatorialComplex(cells=[y1, y2, y3, y4, y5, w])

    Example 4
        >>> # create dictionary that defines the CC
        >>> # and pass it to DynamicCombinatorialComplex
        >>> d = {}
        >>> d["x1"] = Node(1)
        >>> d["x2"] = Node(2)
        >>> d["x3"] = Node(3)
        >>> d["x4"] = Node(4)
        >>> d["x5"] = Node(5)
        >>> d["y1"] = DynamicCell([d['x1'],d['x2']], rank=1)
        >>> d["y2"] = DynamicCell([d['x2'],d['x3']], rank=1)
        >>> d["y3"] = DynamicCell([d['x3'],d['x4']], rank=1)
        >>> d["y4"]= DynamicCell([d['x4'],d['x1']], rank=1)
        >>> d["y5"] = DynamicCell([d['x4'],d['x5']], rank=1)
        >>> d["w"] = DynamicCell([d['x4'],d['x5'],d['x1']], rank=2)
        >>> # define the DynamicCombinatorialComplex from a dictionary of cells
        >>> CC = DynamicCombinatorialComplex(cells=d)
    Example 5
        >>> # create networkx graph
        >>> # and pass it to DynamicCombinatorialComplex
        >>> G = Graph() # networkx graph
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0, 3)
        >>> G.add_edge(0, 4)
        >>> G.add_edge(1, 4)
        >>> CC = DynamicCombinatorialComplex(cells=G)

    Parameters
    ----------
    cells : (optional) RankedEntitySet, dict, iterable, default: None

    name : hashable, optional, default: None
        If None then a placeholder '_'  will be inserted as name

    ranks : (optional) an iterable, default: None.
        when cells is an iterable or dictionary, ranks cannot be None and it must be iterable/dict of the same
        size as cells.

    weight : array-like, optional, default : None
        User specified weight corresponding to setsytem of type pandas.DataFrame,
        length must equal number of rows in dataframe.
        If None, weight for all rows is assumed to be 1.

    safe_insertion : boolean, default is False. When True the CC condition is enforced.

    """

    def __init__(
        self, cells=None, name=None, ranks=None, weight=None, safe_insertion=False
    ):
        if not name:
            self.name = ""
        else:
            self.name = name

        if cells is None:
            setsystem = RankedEntitySet("_", elements=[])
        elif isinstance(cells, RankedEntitySet):
            setsystem = cells
        elif isinstance(cells, RankedEntity) and not isinstance(cells, RankedEntitySet):
            RankedEntitySet(
                "_",
            )
            setsystem = RankedEntitySet("_", cells.incidence_dict)
            setsystem.add(cells)
        elif isinstance(cells, dict):
            # Must be a dictionary with values equal to iterables of Entities and hashables.
            # Keys will be uids for new cells and values of the dictionary will generate the nodes.
            setsystem = RankedEntitySet("_", cells)
        elif isinstance(cells, list) and ranks is None:
            for i in cells:
                if not isinstance(i, RankedEntity):
                    raise TopoNetXError(
                        "input cells elements must be of type EankedEntity, DynamicCells or Node when ranks is None"
                    )
            setsystem = RankedEntitySet("_", cells)
        elif isinstance(cells, tuple) and ranks is None:
            for i in cells:
                if not isinstance(i, RankedEntity):
                    raise TopoNetXError(
                        "input cells elements must be ranked entities when ranks is None"
                    )
            setsystem = RankedEntitySet("_", cells)
        elif isinstance(cells, Graph):
            # cells is an instance of networkX (undirected) graph
            _cells = []
            for e in cells.edges:
                _cells.append(
                    DynamicCell(elements=e, rank=1, **cells.edges[e[0], e[1]])
                )
            for n in cells:
                _cells.append(Node(elements=n, **cells.nodes[n]))

            setsystem = RankedEntitySet("_", _cells)
        elif not isinstance(cells, RankedEntitySet):
            assert ranks is not None  # ranks cannot be None
            assert len(ranks) == len(
                cells
            )  # number of cells must be the same number of entities
            # If no ids are given, return default ids indexed by position in iterator
            # This should be an iterable of sets

            entities = []
            for element, rank in zip(cells, ranks):
                assert isinstance(rank, int)
                if isinstance(element, collections.Hashable):
                    if rank != 0:
                        raise TopoNetXError(f"Node must have zero rank, got {rank}")
                    entities.append(Node(element))
                elif isinstance(element, Iterable):
                    if len(element) == 1:  # e.g element=[0]
                        if isinstance(element[0], collections.Hashable):
                            if isinstance(element, collections.Hashable):
                                if rank != 0:
                                    raise TopoNetXError(
                                        f"Node must have zero rank, got {rank}"
                                    )
                                entities.append(Node(element[0]))

                    else:
                        entities.append(DynamicCell(elements=element, rank=rank))

            setsystem = RankedEntitySet("_", elements=entities)

        self._nodes = setsystem.skeleton(0, f"{self.name}:Nodes")

        self._cells = setsystem.skeleton(1, f"{self.name}:Cells", level="upper")

        self._ranks = sorted(tuple(self._cells.all_ranks))

    @property
    def cells(self):
        """
        Object associated with self._cells.

        Returns
        -------
        RankedEntitySet
        """
        return self._cells

    @property
    def nodes(self):
        """
        Object associated with self._nodes.

        Returns
        -------
        RankedEntitySet

        """
        return self._nodes

    @property
    def incidence_dict(self):
        """
        Dictionary keyed by cell uids with values the uids of nodes in each cell

        Returns
        -------
        dict

        """
        return [self.skeleton(i).incidence_dict for i in self.shape]

    @property
    def shape(self):
        """
        (number of cells[i], for i in range(0,dim(CC))  )

        Returns
        -------
        tuple

        """
        if len(self.cells) == 2:
            return 0
        else:
            return sorted(tuple(self._cells.all_ranks))

    def skeleton(self, rank):
        if rank == 0:
            return self.nodes
        return self.cells.skeleton(rank)

    @property
    def dim(self):
        if len(self.cells) == 0:
            return 0
        return max(self._cells.all_ranks)

    def __str__(self):
        """
        String representation of CC

        Returns
        -------
        str

        """
        return f"Combinatorial Complex with {len(self.nodes)} nodes and {len(self.cells)} cells "

    def __repr__(self):
        """
        String representation of combinatorial complex

        Returns
        -------
        str

        """
        return f"DynamicCombinatorialComplex(name={self.name})"

    def __len__(self):
        """
        Number of nodes

        Returns
        -------
        int

        """

        return len(self._nodes)

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
        item : hashable or RankedEntity

        """
        if isinstance(item, Node):
            return item.uid in self.nodes.elements
        return item in self.nodes.elements

    def __getitem__(self, node):
        """
        Returns the neighbors of node

        Parameters
        ----------
        node : Entity or hashable
            If hashable, then must be uid of node in combinatorial complex

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
        elif rank is None:
            return len(memberships)
        raise TopoNetXError("Rank must be non-negative integer")

    def size(self, cell, node_set=None):
        """
        The number of nodes in node_set that belong to cell.
        If node_set is None then returns the size of cell

        Parameters
        ----------
        cell : hashable
            The uid of an cell in the CC

        Returns
        -------
        size : int

        """

        assert isinstance(cell, RankedEntity)

        if cell not in self.cells:
            raise TopoNetXError("Input cell in not in cells of the CC")

        if node_set:
            return len(set(node_set).intersection(set(self.cells[cell])))
        return len(self.cells[cell])

    def number_of_nodes(self, node_set=None):
        """
        The number of nodes in node_set belonging to combinatorial complex.

        Parameters
        ----------
        node_set : an interable of Entities, optional, default: None
            If None, then return the number of nodes in combinatorial complex.

        Returns
        -------
        number_of_nodes : int

        """
        if node_set:
            return len([node for node in self.nodes if node in node_set])
        return len(self.nodes)

    def number_of_cells(self, cell_set=None):
        """
        The number of cells in cell_set belonging to combinatorial complex.

        Parameters
        ----------
        cell_set : an interable of RankedEntities, optional, default: None
            If None, then return the number of cells in combinatorial complex.

        Returns
        -------
        number_of_cells : int
        """
        if cell_set:
            return len([cell for cell in self.cells if cell in cell_set])
        return len(self.cells)

    def order(self):
        """
        The number of nodes in CC.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def neighbors(self, node, s=1):
        """
        The nodes in combinatorial complex which share s cell(s) with node.

        Parameters
        ----------
        node : hashable or Entity
            uid for a node in combinatorial complex or the node Entity

        s : int, list, optional, default : 1
            Minimum rank of cells shared by neighbors with node.

        Returns
        -------
         : list
            List of neighbors

        Example
        -------
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,z,w] )
            >>> CC = DynamicCombinatorialComplex(cells=E)

            >>> CC.neighbors('x1')
                ['x4', 'x5', 'x2']
            >>> CC.neighbors('x1',2)
                ['x4', 'x5']

        """
        if node not in self.nodes.elements:
            print(f"Node is not in combintorial complex {self.name}.")
            return
        node = self.nodes[
            node
        ].uid  # this allows node to be an Entity instead of a string
        memberships = set(self.nodes[node].memberships).intersection(self.cells.uidset)
        cell_set = {e for e in memberships if self.cells[e].rank >= s}

        neighborlist = set()
        for e in cell_set:
            neighborlist.update(self.cells[e].uidset)
        neighborlist.discard(node)
        return list(neighborlist)

    def cell_neighbors(self, cell, s=1):
        """
        The cells in Combinatorial Complex which share s nodes(s) with cells.

        Parameters
        ----------
        cell : hashable or RankedEntity
            uid for a cell in combintorial complex or the cell RankedEntity

        s : int, list, optional, default : 1
            Minimum number of nodes shared by neighbors cell node.

        Returns
        -------
         : list
            List of cell neighbors

        """
        """
        if not cell in self.cells:
            print(f"cell is not in CC {self.name}.")


        node = self.cells[cell].uid
        return self.dual().neighbors(node, s=s)
        """
        raise NotImplementedError

    def remove_node(self, node):
        """Remove node from CC.

        This deletes reference in combinatorial complex nodes.

        Parameters
        ----------
        node : hashable or RankedEntity
            a node in combinatorial complex

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

        Example:
        --------
        >>> x1 = RankedEntity('x1',rank = 0)
        >>> x2 = RankedEntity('x2',rank = 0)
        >>> x3 = RankedEntity('x3',rank = 0)
        >>> x4 = RankedEntity('x4',rank = 0)
        >>> x5 = RankedEntity('x5',rank = 0)
        >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
        >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
        >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
        >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
        >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
        >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
        >>> # define the Ranked Entity Set
        >>> E = RankedEntitySet('E', [y1, y2, y3, y4, y5, w])
        >>> CC = DynamicCombinatorialComplex(cells=E)

        CC.remove_node(x1)

        """
        if node not in self._nodes:
            return self

        if not isinstance(node, RankedEntity):
            node = self._nodes[node]
        for (
            cell
        ) in node.memberships:  # supposed to be the cells that contain that node.
            if cell in self._cells:
                self._cells[cell].remove(node)
                if len(self._cells[cell]) == 0:  # if we delete all elements
                    # of the cell, then its not a valid cell
                    self.remove_cell(cell)
        self._nodes.remove(node)
        return self

    def remove_nodes(self, node_set):
        """
        Removes nodes from cells and deletes references in combinatorial complex nodes

        Parameters
        ----------
        node_set : an iterable of hashables or Entities
            Nodes in CC

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

        """
        for node in node_set:
            self.remove_node(node)
        return self

    def add_node(self, node, **attr):

        if node in self._cells:
            warnings.warn(
                "Cannot add a node. Node uid already is a cell in the combinatorial complex"
            )
        elif node in self._nodes and isinstance(node, Node):
            self._nodes[node].__dict__.update(node.properties)
        elif node not in self._nodes:
            if isinstance(node, Node):
                self._nodes.add(node)
            else:  # must be hashable
                self._nodes.add(Node(elements=node, **attr))

    def set_node_attributes(self, values, name=None):

        if name is not None:
            # if `values` is a dict using `.items()` => {cell: value}

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

            After computing some property of the cell of a combinatorial complex, you may want
            to assign a cell attribute to store the value of that property for
            each cell:

            >>> CC = DynamicCombinatorialComplex()
            >>> CC.add_cell([1, 2, 3, 4], rank=2)
            >>> CC.add_cell([1, 2,4], rank=2,)
            >>> CC.add_cell([3,4], rank=2)
            >>> d={(1, 2, 3, 4):'red',(1, 2, 3):'blue',(3,4):'green'}
            >>> CC.set_cell_attributes(d,name='color')
            >>> CC.cells[(3,4)].properties['color']
            'green'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update edge attributes::

            Examples
            --------
            >>> G = nx.path_graph(3)
            >>> CC = DynamicCombinatorialComplex(G)
            >>> d={(1, 2): {'color':'red','attr2':1 },(0, 1): {'color':'blue','attr2':3 } }
            >>> CC.set_cell_attributes(d)
            >>> CC.cells[(0, 1)].properties['color']
            'blue'
            3

        Note that if the dict contains cells that are not in `self.cells`, they are
        silently ignored::

        """

        if name is not None:
            # if `values` is a dict using `.items()` => {cell: value}

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
        """Get node attributes from combintorial complex

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
        >>> CC = DynamicCombinatorialComplex(G)
        >>> d={0: {'color':'red','attr2':1 },1: {'color':'blue','attr2':3 } }
        >>> CC.set_node_attributes(d)
        >>> CC.get_node_attributes('color')
        {0: 'red', 1: 'blue'}

        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3], color="blue")
        >>> CC = DynamicCombinatorialComplex(G)
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
        >>> CC = DynamicCombinatorialComplex(G)
        >>> d={(1, 2): {'color':'red','attr2':1 },(0, 1): {'color':'blue','attr2':3 } }
        >>> CC.set_cell_attributes(d)
        >>> cell_color=CC.get_cell_attributes('color')
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
        """
        Private helper method instantiates new nodes when cells added to combinatorial complex.

        Parameters
        ----------
        nodes : iterable of hashables or RankedEntities

        """
        for node in nodes:
            self.add_node(node)

    def add_cell(self, cell, rank=None, **attr):
        """

        Adds a single cells to combinatorial complex.

        Parameters
        ----------
        cell : hashable or RankedEntity
            If hashable the cell returned will be empty.
        uid : unique identifier that identifies the cell
        rank : rank of a cell

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

        Notes
        -----
        -Rank must be None when input cell is a RankedEntity
        -Rank must be positive integer when cell is hashable
        -Rank must be larger than or equal to one for all cells

        When adding an cell to a combinatorial complex children must be removed
        so that nodes do not have elements.
        Each node (element of cell) must be instantiated as a node,
        making sure its uid isn't already present in the self.
        If an added cell contains nodes that cannot be added to combinatorial complex
        then an error will be raised.

        """
        if isinstance(cell, RankedEntity) or isinstance(cell, DynamicCell):
            if cell in self._cells:
                warnings.warn("Cannot add cell. Cell uid already in CC")
            elif cell in self._nodes:
                warnings.warn("Cannot add cell. Cell uid is already a Node")

            if rank is not None:
                raise TopoNetXError(
                    "rank is already passed in the cell, set rank to None and insert the cell "
                )

            if len(cell) > 0:  # insert zero cells
                zeo_cells = cell.skeleton(0)
                for c in zeo_cells:
                    self.add_node(c)
                self._cells.add(cell)
                for n in cell.elements:
                    self._nodes[n].memberships[cell.uid] = self._cells[cell.uid]
            else:
                # len of cell is zero
                warnings.warn("Cell is empty, input cell will not be inserted.")
        else:
            if not isinstance(rank, int):
                raise TopoNetXError(
                    "rank's type must be integer when cell is hashable,"
                    + "got {type(rank)}"
                )
            #    print("")
            if rank == 0:
                raise TopoNetXError("Use add_node to add rank zero cells")
            if rank < 0:
                raise TopoNetXError(f"rank must be positive integer got, {rank}")
            else:
                c = DynamicCell(elements=cell, rank=rank, **attr)
            for i in c:  # update table of nodes
                self._nodes.add(Node(i))
            for n in c.elements:  # update membership
                self._nodes[n].memberships[c.uid] = self._cells[c.uid]
            self._cells.add(c)  # add cell
        return self

    def add_cells_from(self, cell_set):
        """Add cells to combinatorial complex .

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities
            For hashables the cells returned will be empty.

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

        """
        for cell in cell_set:
            self.add_cell(cell)
        return self

    def add_node_to_cell(self, node, cell):
        """Add node to a cell in combinatorial complex cells.

        Parameters
        ----------
        node: hashable or RankedEntity
            If Entity, only uid and properties will be used.
            If uid is already in nodes then the known node will
            be used

        cell: uid of cell or cell, must belong to self.cells

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex
        """
        if cell in self._cells:
            if not isinstance(cell, RankedEntity) or not isinstance(cell, DynamicCell):
                cell = self._cells[cell]
            if node in self._nodes:
                self._cells[cell].add(self._nodes[node])
            else:
                if not isinstance(node, RankedEntity) or not isinstance(
                    cell, DynamicCell
                ):
                    node = RankedEntity(node, rank=0)
                else:
                    node = RankedEntity(node.uid, rank=0, **node.properties)
                self._cells[cell].add(node)
                self._nodes.add(node)

        return self

    def remove_cell(self, cell):
        """Remove a single cell from CC.

        Parameters
        ----------
        cell : hashable or RankedEntity

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

        Notes
        -----
        Deletes reference to cell from all of its nodes.
        If any of its nodes do not belong to any other cells
        the node is dropped from self.
        """
        if cell in self._cells:
            if not isinstance(cell, RankedEntity) or not isinstance(cell, DynamicCell):
                cell = self._cells[cell]
            for node in cell.uidset:
                cell.remove(node)
                if len(self._nodes[node]._memberships) == 1:
                    self._nodes.remove(node)
            self._cells.remove(cell)
        return self

    def remove_cells(self, cell_set):
        """Remove cells from CC.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities

        Returns
        -------
        Combinatorial Complex : DynamicCombinatorialComplex

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
        if rank > 0:
            return self.cells.incidence_matrix(
                rank, to_rank, incidence_type=incidence_type, sparse=sparse, index=index
            )

        cells = self.cells
        for node in self.nodes:
            cells.add_element(node, safe_insert=False)
        return cells.incidence_matrix(
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
        a matrix : scipy.sparse.csr.csr_matrix

        """
        B = csr_matrix(B)
        weight = False  # Currently weighting is not supported

        if weight is False:
            A = B.dot(B.transpose())
            A.setdiag(0)
            A = (A >= s) * 1
        return A

    def adjacency_matrix(self, rank, via_rank, s=1, weight=False, index=False):
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
        >>> G.add_edge(0, 3)
        >>> G.add_edge(0, 4)
        >>> G.add_edge(1, 4)
        >>> CC = DynamicCombinatorialComplex(cells=G)
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
        weight = False  # Currently weighting is not supported
        A = self._incidence_to_adjacency(B, s=s, weight=weight)
        if index:
            return A, row
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
        B = self.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:
            A = self._incidence_to_adjacency(B[0].transpose(), s=s)
            return A, B[2]
        A = self._incidence_to_adjacency(B.transpose(), s=s)
        return A

    def node_adjacency_matrix(self, index=False, s=1, weight=False):
        """Compute the node adjacency matrix."""
        B = self.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:
            A = self._incidence_to_adjacency(B[0], s=s)
            return A, B[1]

        A = self._incidence_to_adjacency(B, s=s)
        return A

    def coadjacency_matrix(self, rank, via_rank, s=1, weight=False, index=False):
        """
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
        weight = False  # Currently weighting is not supported
        if weight is False:
            A = B.T.dot(B)
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, col
        return A

    def collapse_cells(
        self,
        name=None,
        return_equivalence_classes=False,
    ):
        """
        Constructs a new Combinatorial Complex gotten by identifying cells containing the same nodes

        Parameters
        ----------
        name : hashable, optional, default: None

        return_equivalence_classes: boolean, optional, default: False
            Returns a dictionary of cell equivalence classes keyed by frozen sets of nodes

        Returns
        -------
        new Combinatorial Complex : DynamicCombinatorialComplex
            Equivalent cells are collapsed to a single cell named by a representative of the equivalent
            cells followed by a colon and the number of cells it represents.

        equivalence_classes : dict
            A dictionary keyed by representative cell names with values equal to the cells in
            its equivalence class

        Notes
        -----
        Two cells are identified if their respective elements are the same.
        Using this as an equivalence relation, the uids of the cells are partitioned into
        equivalence classes.

        A single cell from the collapsed cells followed by a colon and the number of elements
        in its equivalence class as uid for the new cell
        Example
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> y6 = RankedEntity('y6',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1, y2, y3, y4, y5, w,y6] )
            >>> CC = DynamicCombinatorialComplex(cells=E)


        """
        temp = self.cells.collapse_identical_elements(
            "_", return_equivalence_classes=return_equivalence_classes
        )
        if return_equivalence_classes:
            return DynamicCombinatorialComplex(cells=temp[0], name=name), temp[1]
        return DynamicCombinatorialComplex(cells=temp, name=name)

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
        new Combinatorial Complex : DynamicCombinatorialComplex

        Example
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> y6 = RankedEntity('y6',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1, y2, y3, y4, y5, w,y6] )
            >>> CC = DynamicCombinatorialComplex(cells=E)
        """

        rns = self.cells.restrict_to(element_subset=cell_set, name=name)
        return DynamicCombinatorialComplex(cells=rns, name=name)

    def restrict_to_nodes(self, node_set, name=None):
        """
        Constructs a new combinatorial complex  by restricting the cells in the combintorial complex to
        the nodes referenced by node_set.

        Parameters
        ----------
        node_set: iterable of hashables
            References a subset of elements of self.nodes

        name: string, optional, default: None

        Returns
        -------
        new Combinatorial Complex : DynamicCombinatorialComplex

        Example
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> y6 = RankedEntity('y6',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1, y2, y3, y4, y5, w,y6] )
            >>> CC = DynamicCombinatorialComplex(cells=E)
            >>> CC.restrict_to_nodes([x3,x2])
        """
        memberships = set()
        innernodes = set()
        for node in node_set:
            innernodes.add(node.uid)
            if node in self.nodes:
                memberships.update(set(self.nodes[node].memberships))
        newcellset = []
        tempp = []
        for e in memberships:
            if e in self.cells:
                temp = self.cells[e].skeleton(0).uidset.intersection(innernodes)
                tempp.append([e, temp])

                if temp:
                    newcellset.append(
                        RankedEntity(
                            e,
                            elements=list(temp),
                            rank=self.cells[e].rank,
                            **self.cells[e].properties,
                        )
                    )

        return DynamicCombinatorialComplex(RankedEntitySet("", newcellset), name=name)

    @staticmethod
    def from_networkx_graph(G):
        """Construct a combinatorial complex from a networkx graph.

        Parameters
        ----------
        G : NetworkX graph
            A networkx graph

        Returns
        -------
        CombintorialComplex such that the edges of the graph are ranked 1
        and the nodes are ranked 0.

        Example
        ------
        >>> from networkx import Graph
        >>> G = Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(0, 4)
        >>> G.add_edge(0,7)
        >>> CX = DynamicCombinatorialComplex.from_networkx_graph(G)
        >>> CX.nodes
        RankedEntitySet(:Nodes,[0, 1, 4, 7],{'weight': 1.0})
        >>> CX.cells
        RankedEntitySet(:Cells,[(0, 1), (0, 7), (0, 4)],{'weight': 1.0})
        """
        from networkx import Graph

        if isinstance(G, Graph):
            # cells is an instance of networkX (undirected) graph
            return DynamicCombinatorialComplex(cells=G)
        else:
            raise TopoNetXError(
                f"type of input G must be a networkx graph, got {type(G)}"
            )

    def to_hypergraph(self):
        """Construct a hypergraph from a combinatorial complex.

        Example
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1, y2, y3, y4, y5, w])
            >>> CC = DynamicCombinatorialComplex(cells=E)
            >>> HG = CC.to_hypergraph()
        """
        CC = self.collapse_cells()
        cells = []
        for cell in self.cells:
            cells.append(
                Entity(
                    cell, elements=CC.cells[cell].elements, **CC.cells[cell].properties
                )
            )

        E = EntitySet("CC_to_HG", elements=cells)
        HG = Hypergraph(E)
        nodes = []
        for node in self.nodes:
            nodes.append(
                Entity(
                    node, elements=CC.nodes[node].elements, **CC.nodes[node].properties
                )
            )
        HG._add_nodes_from(nodes)
        return HG

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

            # example
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E', [y1, y2, y3, y4, y5, w])
            >>> CC = DynamicCombinatorialComplex(cells=E)

        """
        B = self.incidence_matrix(0, None, incidence_type="up")
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
        """
        Constructs clone of CC with singleton cells removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        new CC : CC

        Example :
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1, x2], rank=1)
            >>> y2 = RankedEntity('y2',[x2, x3], rank=1)
            >>> y3 = RankedEntity('y3',[x3, x4], rank=1)
            >>> y4 = RankedEntity('y4',[x4, x1], rank=1)
            >>> y5 = RankedEntity('y5',[x4, x5], rank=1)
            >>> y6 = RankedEntity('y6',[x4, x5], rank=1)
            >>> w = RankedEntity('w',[x4, x5, x1], rank=2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1, y2, y3, y4, y5, w,y6] )
            >>> CC = DynamicCombinatorialComplex(cells=E)
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
        except Exception:
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
        except Exception:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf

    def dataframe(self, sort_rows=False, sort_columns=False, cell_weight=True):
        """
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
         : DynamicCombinatorialComplex

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

    @classmethod
    def from_dataframe(
        cls,
        df,
        columns=None,
        rows=None,
        name=None,
        fillna=0,
        transpose=False,
        transforms=[],
        key=None,
        node_label="nodes",
        cell_label="cells",
        static=False,
        use_nwhy=False,
    ):
        """
        Create a hypergraph from a Pandas Dataframe object using index to label vertices
        and Columns to label cells. The values of the dataframe are transformed into an
        incidence matrix.
        Note this is different than passing a dataframe directly
        into the Hypergraph constructor. The latter automatically generates a static hypergraph
        with cell and node labels given by the cell values.

        Parameters
        ----------
        df : Pandas.Dataframe
            a real valued dataframe with a single index

        columns : (optional) list, default = None
            restricts df to the columns with headers in this list.

        rows : (optional) list, default = None
            restricts df to the rows indexed by the elements in this list.

        name : (optional) string, default = None

        fillna : float, default = 0
            a real value to place in empty cell, all-zero columns will not generate
            an cell.

        transpose : (optional) bool, default = False
            option to transpose the dataframe, in this case df.Index will label the cells
            and df.columns will label the nodes, transpose is applied before transforms and
            key

        transforms : (optional) list, default = []
            optional list of transformations to apply to each column,
            of the dataframe using pd.DataFrame.apply().
            Transformations are applied in the order they are
            given (ex. abs). To apply transforms to rows or for additional
            functionality, consider transforming df using pandas.DataFrame methods
            prior to generating the hypergraph.

        key : (optional) function, default = None
            boolean function to be applied to dataframe. Must be defined on numpy
            arrays.

        See also
        --------
        from_numpy_array())


        Returns
        -------
        : Hypergraph

        Notes
        -----
        The `from_dataframe` constructor does not generate empty cells.
        All-zero columns in df are removed and the names corresponding to these
        cells are discarded.
        Restrictions and data processing will occur in this order:

            1. column and row restrictions
            2. fillna replace NaNs in dataframe
            3. transpose the dataframe
            4. transforms in the order listed
            5. boolean key

        This method offers the above options for wrangling a dataframe into an incidence
        matrix for a hypergraph. For more flexibility we recommend you use the Pandas
        library to format the values of your dataframe before submitting it to this
        constructor.

        """

        if type(df) != pd.core.frame.DataFrame:
            raise TopoNetXError("Error: Input object must be a pandas dataframe.")

        if columns:
            df = df[columns]
        if rows:
            df = df.loc[rows]

        df = df.fillna(fillna)
        if transpose:
            df = df.transpose()

        # node_names = np.array(df.index)
        # cell_names = np.array(df.columns)

        for t in transforms:
            df = df.apply(t)
        if key:
            mat = key(df.values) * 1
        else:
            mat = df.values * 1

        params = {
            "node_names": np.array(df.index),
            "cell_names": np.array(df.columns),
            "name": name,
            "node_label": node_label,
            "cell_label": cell_label,
            "static": static,
            "use_nwhy": use_nwhy,
        }
        return cls.from_numpy_array(mat, **params)
