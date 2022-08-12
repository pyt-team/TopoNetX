"""
Regular 2d Cell Complex
"""


import warnings
from collections.abc import Iterable
from itertools import zip_longest

import networkx as nx
import numpy as np
from hypernetx import Hypergraph
from hypernetx.classes.entity import Entity
from networkx import Graph
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix

from toponetx.classes.cell import Cell, CellView
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.ranked_entity import RankedEntity, RankedEntitySet
from toponetx.exception import TopoNetXError

__all__ = ["CellComplex"]


class CellComplex:

    """

    In TNX cell complexes are implementes to be dynamic in the sense that
    they can change by adding or subtracting objects
    from them.
        Example 0
            >> # Cellomplex can be empty
            >>> CC = CellComplex( )
        Example 1
            >>> CX = CellComplex()
            >>> CX.add_cell([1,2,3,4],rank=2)
            >>> CX.add_cell([2,3,4,5],rank=2)
            >>> CX.add_cell([5,6,7,8],rank=2)
        Example 2
            >>> c1= Cell( (1,2,3))
            >>> c2= Cell( (1,2,3,4) )
            >>> CX = CellComplex( [c1,c2] )
        Example 3
            >>> G= Graph()
            >>> G.add_edge(1,0)
            >>> G.add_edge(2,0)
            >>> G.add_edge(1,2)
            >>> CX = CellComplex(G)
            >>> CX.add_cells_from([[1,2,4],[1,2,7] ],rank=2)
            >>> CX.cells
    """

    def __init__(self, cells=None, name=None):
        if not name:
            self.name = ""
        else:
            self.name = name

        self._G = Graph()

        self._cells = CellView()
        if cells is not None:
            if isinstance(cells, Graph):
                self._G = cells

            elif isinstance(cells, Iterable) and not isinstance(cells, Graph):
                for c in cells:
                    self.add_cell(c)

            else:
                raise ValueError(
                    f"cells must be iterable, networkx graph or None, got {type(cells)}"
                )

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
        elif len(self.edges) == 0:
            return 0
        elif len(self.cells) == 0:
            return 1
        else:
            return 2

    @property
    def dim(self):
        return self.maxdim

    @property
    def shape(self):
        """
        (number of cells[i], for i in range(0,dim(CC))  )

        Returns
        -------
        tuple

        """
        return len(self.nodes), len(self.edges), len(self.cells)

    def skeleton(self, k):
        if k == 0:
            return self.nodes
        if k == 1:
            return self.edges
        if k == 2:
            return self.cells

    def __str__(self):
        """
        String representation of CC

        Returns
        -------
        str

        """
        return f"Cell Complex with {len(self.nodes)} nodes, {len(self.edges)} edges  and {len(self.cells)} 2-cells "

    def __repr__(self):
        """
        String representation of combinatorial complex

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

        return item in self.nodes

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
        return self._G.degree[node]

    def size(self, cell, nodeset=None):
        """
        The number of nodes in nodeset that belong to cell.
        If nodeset is None then returns the size of cell

        Parameters
        ----------
        cell : hashable
            The uid of an cell in the CC

        Returns
        -------
        size : int

        """
        if nodeset:
            return len(set(nodeset).intersection(set(self.cells[cell])))
        else:
            if cell in self.cells:

                return len(self.cells[cell])
            else:
                raise KeyError(f" the key {cell} is not a key for an existing cell ")

    def number_of_nodes(self, nodeset=None):
        """
        The number of nodes in nodeset belonging to combinatorial complex.

        Parameters
        ----------
        nodeset : an interable of Entities, optional, default: None
            If None, then return the number of nodes in cell complex.

        Returns
        -------
        number_of_nodes : int

        """
        if nodeset:
            return len([n for n in self.nodes if n in nodeset])
        else:
            return len(self.nodes)

    def number_of_edges(self, edgeset=None):
        """
        The number of cells in cellset belonging to cell complex.

        Parameters
        ----------
        cellset : an interable of RankedEntities, optional, default: None
            If None, then return the number of cells in cell complex.

        Returns
        -------
        number_of_cells : int
        """
        if edgeset:
            return len([e for e in self.cells if e in edgeset])
        else:
            return len(self.edges)

    def number_of_cells(self, cellset=None):
        """
        The number of cells in cellset belonging to combinatorial complex.

        Parameters
        ----------
        cellset : an interable of RankedEntities, optional, default: None
            If None, then return the number of cells in combinatorial complex.

        Returns
        -------
        number_of_cells : int
        """
        if cellset:
            return len([e for e in self.cells if e in cellset])
        else:
            return len(self.cells)

    def order(self):
        """
        The number of nodes in CC.

        Returns
        -------
        order : int
        """
        return len(self.nodes)

    def neighbors(self, node):
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


        """
        if not node in self.nodes:
            print(f"Node is not in cell complex {self.name}.")
            return

        return self._G.neighbors

    def cell_neighbors(self, cell, s=1):
        """
        The cells in cell Complex which share s nodes(s) with cells.

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
        """
        Removes node from cells and deletes reference in combinatorial complex nodes

        Parameters
        ----------
        node : hashable or RankedEntity
            a node in combinatorial complex

        Returns
        -------
        Cell Complex : CellComplex
        Example:


        """
        self._G.remove_node(node)
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
        Combinatorial Complex : CombinatorialComplex

        """
        for node in node_set:
            self.remove_node(node)
        return self

    def add_node(self, node):

        self._G.add_node(node)

    def _add_nodes_from(self, nodes):
        """
        Private helper method instantiates new nodes when cells added to combinatorial complex.

        Parameters
        ----------
        nodes : iterable of hashables or RankedEntities

        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u, v):
        self._G.add_edge(u, v)

    def add_cell(self, cell, uid=None, rank=None, check_skeleton=False):
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
        Cell Complex : CellComplex

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
        if isinstance(cell, Cell):
            for e in cell:
                self._G.add_edge(e[0], e[1])
            self._cells.insert_cell(cell)

        else:
            if rank == 0:
                self._G.add_node(cell, name=uid)
            elif rank == 1:
                if len(cell) != 2:
                    raise ValueError("rank 2 cell must have exactly two nodes")
                elif len(set(cell)) == 1:
                    raise ValueError(" invalid insertion : self-loops are not allowed.")
                else:
                    self.add_edge(cell[0], cell[1])

            elif rank == 2:
                if isinstance(cell, Cell):
                    self._cells.insert(Cell)
                elif isinstance(cell, tuple) or isinstance(cell, list):

                    if self.is_insertable_cycle(cell, check_skeleton=check_skeleton):
                        edges_cell = set(zip_longest(cell, cell[1:] + [cell[0]]))
                        for e in edges_cell:
                            self._G.add_edges_from(edges_cell)
                        self._cells.insert_cell(Cell(cell))
                    else:
                        print(
                            "Invalid cycle condition, the input cell can be inserted to the cell complex"
                        )
                        print(
                            "to force the cell complex condition, set check_skeleton = False."
                        )
                else:
                    raise ValueError("invalid input")
            else:
                raise ValueError(
                    f"Add cell only supports adding cells of dimensions 0,1 or 2--got {rank}",
                )

        return self

    def add_cells_from(self, cell_set, rank=None):
        """
        Add cells to cell complex .

        Parameters
        ----------
        cell_set : iterable of hashables or Cell
            For hashables the cells returned will be empty.

        Returns
        -------
        Cell Complex : CellComplex

        """

        for cell in cell_set:
            self.add_cell(cell=cell, rank=rank)
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

        Deletes reference to cell, keep it boundary edges in the cell comple

        """
        if isinstance(cell, Cell) or isinstance(cell, int):
            self._cells.delete_cell(cell)
        if isinstance(cell, tuple) or isinstance(cell, cell):
            self._cells.delete_cell_by_set(cell)
        return self

    def remove_cells(self, cell_set):
        """
        Removes cells from CC.

        Parameters
        ----------
        cell_set : iterable of hashables or RankedEntities

        Returns
        -------
        Combinatorial Complex : CombinatorialComplex

        """
        for cell in cell_set:
            self.remove_cell(cell)
        return self

    def is_insertable_cycle(self, cell, check_skeleton=True, warnings_dis=False):

        if isinstance(cell, Cell):
            cell = cell.elements
        if len(cell) <= 2:
            return False
        if len(set(cell)) != len(cell):
            if warnings_dis:
                warnings.warn(f"repeating nodes invalidates the 2-cell condition")
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

    def incidence_matrix(self, d, sign=True, weights=None, index=False):
        """
        An incidence matrix for the CC indexed by nodes x cells.

        Parameters
        ----------
        weights : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weights dictionary values.

        index : boolean, optional, default False
            If True return will include a dictionary of node uid : row number
            and cell uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix

        row dictionary : dict
            Dictionary identifying rows with nodes

        column dictionary : dict
            Dictionary identifying columns with cells

        >>> CX = CellComplex()
        >>> CX.add_cell([1,2,3,5,6],rank=2)
        >>> CX.add_cell([1,2,4,5,3,0],rank=2)
        >>> CX.add_cell([1,2,4,9,3,0],rank=2)
        >>> B1 = CX.incidence_matrix(1)
        >>> B2 = CX.incidence_matrix(2)
        >>> B1.dot(B2).todense()

        """
        weights = None  # not supported at this version
        import scipy as sp
        import scipy.sparse

        if d == 0:

            A = sp.sparse.lil_matrix((1, len(self._G.nodes)))
            for i in range(0, len(self._G.nodes)):
                A[0, i] = 1
            if index:
                if sign:
                    return self._G.nodes, [], A.asformat("csc")
                else:
                    return self._G.nodes, [], abs(A.asformat("csc"))
            else:
                if sign:

                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))

        elif d == 1:
            nodelist = sorted(
                self._G.nodes
            )  # always output boundary matrix in dictionary order
            edgelist = sorted(self._G.edges)
            A = sp.sparse.lil_matrix((len(nodelist), len(edgelist)))
            node_index = {node: i for i, node in enumerate(nodelist)}
            for ei, e in enumerate(edgelist):
                (u, v) = sorted(e[:2])
                ui = node_index[u]
                vi = node_index[v]
                A[ui, ei] = -1
                A[vi, ei] = 1
            if index:
                if sign:
                    return nodelist, edgelist, A.asformat("csc")
                else:
                    return nodelist, edgelist, abs(A.asformat("csc"))
            else:
                if sign:
                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))
        elif d == 2:
            edgelist = sorted(self._G.edges)

            A = sp.sparse.lil_matrix((len(edgelist), len(self.cells)))

            edge_index = {
                tuple(sorted(edge)): i for i, edge in enumerate(edgelist)
            }  # orient edges
            for celli, cell in enumerate(self.cells):
                for edge in cell.boundary:
                    ei = edge_index[tuple(sorted(edge))]
                    if edge in edge_index:
                        A[ei, celli] = 1
                    else:
                        A[ei, celli] = -1
            if index:
                cell_index = {cell: i for i, cell in enumerate(self.cells)}
                if sign:
                    return edge_index, cell_index, A.asformat("csc")
                else:
                    return edge_index, cell_index, abs(A.asformat("csc"))
            else:
                if sign:
                    return A.asformat("csc")
                else:
                    return abs(A.asformat("csc"))
        else:
            raise ValueError(f"only dimension 0,1 and 2 are supported, got {d}")

    @staticmethod
    def _incidence_to_adjacency(M, weights=False):
        """
        Helper method to obtain adjacency matrix from
        boolean incidence matrix for s-metrics.
        Self loops are not supported.
        The adjacency matrix will define an s-linegraph.

        Parameters
        ----------
        M : scipy.sparse.csr.csr_matrix
            incidence matrix of 0's and 1's

        s : int, optional, default: 1

        # weights : bool, dict optional, default=True
        #     If False all nonzero entries are 1.
        #     Otherwise, weights will be as in product.

        Returns
        -------
        a matrix : scipy.sparse.csr.csr_matrix

        >>> CX = CellComplex()
        >>> CX.add_cell([1,2,3,5,6],rank=2)
        >>> CX.add_cell([1,2,4,5,3,0],rank=2)
        >>> CX.add_cell([1,2,4,9,3,0],rank=2)
        >>> B1 = CX.incidence_matrix(1)
        >>> B2 = CX.incidence_matrix(2)

        """

        M = csr_matrix(M)
        weights = False  ## currently weighting is not supported

        if weights == False:
            A = M.dot(M.transpose())
            A.setdiag(0)
        return A

    def hodge_laplacian_matrix(self, d, signed=True, index=False):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L = B_next @ B_next.transpose()
        elif d < 2:
            B_next = self.incidence_matrix(d + 1)
            B = self.incidence_matrix(d)
            L = B_next @ B_next.transpose() + B.transpose() @ B
        elif d == self.maxdim:
            B = self.incidence_matrix(d)
            L = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.maxdim} (maximal dimension cells), got {d}"
            )
        if signed:
            return L
        else:
            return abs(L)

    def up_laplacian_matrix(self, d, signed=True):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension cells-1), got {d}"
            )
        if signed:
            return L_up
        else:
            return abs(L_up)

    def down_laplacian_matrix(self, d, signed=True):
        if d <= self.maxdim and d > 0:
            B = self.incidence_matrix(d)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.maxdim} (maximal dimension cells), got {d}."
            )
        if signed:
            return L_down
        else:
            return abs(L_down)

    def adjacency_matrix(self, d, signed=False):

        L_up = self.up_laplacian_matrix(d, signed)
        L_up.setdiag(0)

        if signed:
            return L_up
        else:
            return abs(L_up)

    def coadjacency_matrix(self, d, signed=False):

        L_down = self.down_laplacian_matrix(d, signed)
        L_down.setdiag(0)
        if signed:
            return L_down
        else:
            return abs(L_down)

    def k_hop_incidence_matrix(self, d, k):
        Bd = self.incidence_matrix(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.maxdim:
            return Bd @ np.power(coAd, k)
        elif d == 0:
            return Bd @ np.power(Ad, k)
        else:
            return Bd @ np.power(Ad, k) + Bd @ np.power(coAd, k)

    def k_hop_coincidence_matrix(self, d, k):
        BTd = self.coincidence_matrix(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.maxdim:
            return np.power(coAd, k) @ BTd
        elif d == 0:
            return np.power(Ad, k) @ BTd
        else:
            return np.power(Ad, k) @ BTd + np.power(coAd, k) @ BTd

    def adjacency_matrix(self, d, weights=False, index=False):  ## , weights=False):
        """
        The sparse weighted :term:`s-adjacency matrix`

        Parameters
        ----------
        r,k : two ranks for skeletons in the input combinatorial complex, such that r<k

        s : int, optional, default: 1

        index: boolean, optional, default: False
            if True, will return a rowdict of row to node uid

        weights: bool, default=True
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
        >>> CX = CellComplex()
        >>> CX.add_cell([1,2,3],rank=2)
        >>> CX.add_cell([1,4],rank=1)
        >>> A0 = CX.adjacency_matrix(0)

        """

        if index:
            MP, row, col = self.incidence_matrix(
                d, sign=False, weights=weights, index=index
            )
        else:
            MP = self.incidence_matrix(d + 1, sign=False, weights=weights, index=index)
        weights = False  ## currently weighting is not supported
        A = self._incidence_to_adjacency(MP, weights=weights)
        if index:
            return A, row
        else:
            return A

    def cell_adjacency_matrix(self, index=False, weights=False):
        """
        >>> CX = CellComplex()
        >>> CX.add_cell([1,2,3],rank=2)
        >>> CX.add_cell([1,4],rank=1)
        >>> A = CX.cell_adjacency_matrix()


        """

        CC = self.to_combinatorial_complex()
        weights = False  ## Currently default weights are not supported

        M = CC.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:

            A = CC._incidence_to_adjacency(M[0].transpose())

            return A, M[2]
        else:
            A = CC._incidence_to_adjacency(M.transpose())
            return A

    def node_adjacency_matrix(self, index=False, s=1, weights=False):

        CC = self.to_combinatorial_complex()
        weights = False  ## Currently default weights are not supported

        M = CC.incidence_matrix(0, None, incidence_type="up", index=index)
        if index:

            A = CC._incidence_to_adjacency(M[0], s=s)

            return A, M[1]
        else:
            A = CC._incidence_to_adjacency(M, s=s)
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
        new Combinatorial Complex : CombinatorialComplex
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



        """
        """
        temp = self.cells.collapse_identical_elements(
                "_", return_equivalence_classes=return_equivalence_classes
            )
        if return_equivalence_classes:
            return CombinatorialComplex(cells = temp[0], name = name), temp[1]
        else:
            return CombinatorialComplex(cells = temp, name = name)
        """
        raise NotImplementedError

    def restrict_to_cells(self, cellset, name=None):
        """
        Constructs a combinatorial complex using a subset of the cells in combinatorial complex

        Parameters
        ----------
        cellset: iterable of hashables or RankedEntities
            A subset of elements of the combinatorial complex  cells

        name: str, optional

        Returns
        -------
        new Combinatorial Complex : CombinatorialComplex

        Example

        >>> CX = CellComplex()
        >>> c1= Cell((1,2,3))
        >>> c2= Cell((1,2,4))
        >>> c3= Cell((1,2,5))
        >>> CX = CellComplex([c1,c2,c3])
        >>> CX.add_edge(1,0)
        >>> CX.restrict_to_cells([c1,[0,1]])


        """
        RNS = []
        edges = []
        for i in cellset:
            if i in self.cells:
                RNS.append(i)
            elif i in self.edges:
                edges.append(i)

        CX = CellComplex(cells=RNS, name=name)
        for i in edges:
            CX.add_edge(i[0], i[1])
        return CX

    def restrict_to_nodes(self, nodeset, name=None):
        """
        Constructs a new combinatorial complex  by restricting the cells in the combintorial complex to
        the nodes referenced by nodeset.

        Parameters
        ----------
        nodeset: iterable of hashables
            References a subset of elements of self.nodes

        name: string, optional, default: None

        Returns
        -------
        new Combinatorial Complex : CombinatorialComplex

        Example
        >>> CX = CellComplex()
        >>> c1= Cell((1,2,3))
        >>> c2= Cell((1,2,4))
        >>> c3= Cell((1,2,5))
        >>> CX = CellComplex([c1,c2,c3])
        >>> CX.add_edge(1,0)
        >>> CX.restrict_to_nodes([1,2,3,0])

        """

        _G = Graph(self._G.subgraph(nodeset))
        CX = CellComplex(_G)
        cells = []
        for cell in self.cells:
            if CX.is_insertable_cycle(cell, True):
                cells.append(cell)
        CX = CellComplex(_G)

        for e in cells:
            CX.add_cell(e)
        return CX

    def to_combinatorial_complex(self):
        """
        >>> CX = CellComplex()
        >>> CX.add_cell([1,2,3,4],rank=2)
        >>> CX.add_cell([2,3,4,5],rank=2)
        >>> CX.add_cell([5,6,7,8],rank=2)
        """
        all_cells = []

        for e in self.edges:
            all_cells.append(RankedEntity(uid=str(e), elements=e, rank=1))
        for cell in self.cells:
            all_cells.append(
                RankedEntity(uid=str(cell), elements=cell.elements, rank=2)
            )
        return CombinatorialComplex(RankedEntitySet("", all_cells), name="_")

    def to_hypergraph(self):

        """
        Example

        """
        from hypernetx.classes.entity import EntitySet

        cells = []
        for n in self.cells:
            cells.append(Entity(n, elements=n.elements))

        for n in self.edges:
            cells.append(Entity(n, elements=n))
        E = EntitySet("CC_to_HG", elements=cells)
        HG = Hypergraph(E)
        nodes = []
        for n in self.nodes:
            nodes.append(Entity(n, elements=[]))
        HG._add_nodes_from(nodes)
        return HG

    def is_connected(self, s=1, cells=False):
        """
        Determines if combinatorial complex is :term:`s-connected <s-connected, s-node-connected>`.

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

        A CX is s node connected if for any two nodes v0,vn
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

            >>> CX = CellComplex()
            >>> CX.add_cell([1,2,3,4],rank=2)
            >>> CX.add_cell([2,3,4,5],rank=2)
            >>> CX.add_cell([5,6,7,8],rank=2)
            >>> CX.add_node(0)
            >>> CX.add_node(10)

        """

        return [i for i in self.nodes if self.degree(i) == 0]

    def remove_singletons(self, name=None):
        """
        Constructs clone of cell complex with singleton cells removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        Cell Complex : CellComplex


        """

        for n in self.singletons():
            self._G.remove_node(n)

    def s_connected_components(self, s=1, cells=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-cell-connected components <s-cell-connected component>`
        or the :term:`s-node-connected components <s-connected component, s-node-connected component>`
        of the combinatorial complex.

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
        s : int, optional, default: 1

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
        s : int, optional, default: 1

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
        s : int, optional, default: 1

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
        s : int, optional, default: 1

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
            raise TopoNetXError(f"cell complex is not s-connected. s={s}")

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

        if isinstance(source, Cell):
            source = source.uid
        if isinstance(target, Cell):
            target = target.uid
        A, rowdict = self.node_adjacency_matrix(s=s, index=True)
        g = nx.from_scipy_sparse_matrix(A)
        rkey = {v: k for k, v in rowdict.items()}
        try:
            path = nx.shortest_path_length(g, rkey[source], rkey[target])
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

        if isinstance(source, Cell):
            source = source.uid
        if isinstance(target, Cell):
            target = target.uid
        A, coldict = self.cell_adjacency_matrix(s=s, index=True)
        g = nx.from_scipy_sparse_matrix(A)
        ckey = {v: k for k, v in coldict.items()}
        try:
            path = nx.shortest_path_length(g, ckey[source], ckey[target])
            return path
        except:
            warnings.warn(f"No {s}-path between {source} and {target}")
            return np.inf
