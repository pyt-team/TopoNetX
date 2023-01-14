"""Classes representing cells."""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from collections import Counter, defaultdict, deque
from itertools import zip_longest

import numpy as np

__all__ = ["Cell", "CellView"]


class Cell:
    """Class representing a 2d cell.

    A 2d cell is the elementry building block used
    to build a 2d cell complex (regular or non-regular).

    Parameters
    ----------
    elements : any iterable of hashables.
        Elements order is important and defines the 2
        cell up to cyclic permutation.
    name : str, optional, default: None
    is_regular : bool
        checks for regularity conditions of the cell. Raises ValueError when
        the True and when the input elements do not satisfy the regullarity condition
    attr : keyword arguments, optional, default: {}
        properties belonging to cell added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----
    - a cell is defined as an ordered sequence of nodes (n1,...,nk)
      each two consequitive nodes (ni,n_{i+1}) define an edge in the boundary of the cell
      note that the last edge (n_k,n1) is also included in the boundary of the cell
      and it is used to close the cell
      so a Cell that is defined as c = Cell((1, 2, 3))
      will have a c.boundary = [(1, 2), (2, 3), (3, 1)] which consists of three edges.

    - The regularity condition of a 2d cell :
            a 2d cell is regular if and only if there is no
            repeatition in the boundary edges that define the cell
            by default Cell is assumed to be regular unless otherwise specified.
            self loops are not allowed in the boundary of the edge
    Examples
        >>> cell1 = Cell((1, 2, 3))
        >>> cell2 = Cell((1, 2, 4, 5), weight=1)
        >>> cell3 = Cell(("a", "b", "c"))
    """

    def __init__(self, elements, name=None, regular=True, **attr):
        if name is None:
            self.name = "_"
        else:
            self.name = name
        self._regular = regular
        elements = list(elements)
        self._boundary = list(
            zip_longest(elements, elements[1:] + [elements[0]])
        )  # list of edges define the boundary of the 2d cell
        if len(elements) <= 1:
            raise ValueError(
                f"cell must contain at least 2 edges, got {len(elements)+1}"
            )

        if regular:
            _adjdict = {}
            for e in self._boundary:
                if e[0] in _adjdict:
                    raise ValueError(
                        f" node {e[0]} is repeated multiple times in the input cell."
                        + " input cell violates the cell complex regularity condition."
                    )
                _adjdict[e[0]] = e[1]
        else:

            for e in self._boundary:
                if e[0] == e[1]:
                    raise ValueError(
                        f"self loops are not permitted, got {(e[0],e[1])} as an edge in the cell's boundary"
                    )

        self.nodes = tuple(elements)
        self.properties = dict()
        self.properties.update(attr)

    def __getitem__(self, item):
        if item not in self.properties:
            raise KeyError(f"attr {item} is not an attr in the cell {self.name}")
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        self.properties[key] = item

    @property
    def is_regular(self):
        """
        Returns true is the Cell is a regular cell, and False otherwise
        """

        if self._regular:  # condition enforced
            return True
        else:
            _adjdict = {}
            for e in self._boundary:
                if e[0] in _adjdict:
                    return False
                _adjdict[e[0]] = e[1]

        return True

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def sign(self, edge):
        if isinstance(edge, Iterable):
            if len(edge) == 2:
                if tuple(edge) in self.boundary:
                    return 1
                elif tuple(edge)[::-1] in self.boundary:
                    return -1
                else:
                    raise KeyError(
                        f"the input {edge} is not in the boundary of the cell"
                    )

            raise ValueError(f"the input {edge} is not a valud edge")
        raise ValueError(f"the input {edge} must be iterable")

    def __contains__(self, e):
        return e in self.nodes

    def __repr__(self):
        """
        String representation of regular cell
        Returns
        -------
        str
        """
        return f"Cell{self.elements}"

    @property
    def boundary(self):
        """

        a 2d cell is characterized by its boundary edges


        Return
        ------
        iterator of tuple representing boundary edges given in cyclic order
        """
        return iter(self._boundary)

    @property
    def elements(self):
        return self.nodes

    def reverse(self):
        """Reverse the sequence of nodes that defines the cell.

        This returns a new cell with the new reversed elements.

        Return : Cell
        ------
        """
        c = Cell(self.nodes[::-1], name=self.name, regular=self._regular)
        c.properties = self.properties
        return c

    def is_homotopic_to(self, cell):
        """
        Parameters
        ----------
        cell : tuple, list or Cell

        Return
        ------
        _ : bool
            Return True is self is homotopic to input cell and False otherwise.
        """

        return Cell._are_homotopic(self, cell) or Cell._are_homotopic(
            self.reverse(), cell
        )

    @staticmethod
    def _are_homotopic(cell1, cell):
        """
        Parameters:
        ---------
        cell1 : Cell
        cell : tuple, list or Cell

        Return : bool
        ------
            return True is self is homotopic to input cell and False otherwise.

        Note :
        -------
            in a 2d-cell complex, two 2d-cells are homotopic iff one
                of them can be obtaine from the other by a cylic rotation
                of the boundary verties defining the 2d cells.
        """

        if isinstance(cell, tuple) or isinstance(cell, list):
            seq = cell
        elif isinstance(cell, Cell):
            seq = cell.elements
        else:
            raise ValueError(
                "input type must be a tuple/list of nodes defining a cell or Cell"
            )

        if len(cell1) != len(cell):
            return False

        mset1 = Counter(seq)
        mset2 = Counter(cell1.elements)
        if mset1 != mset2:
            return False

        size = len(seq)
        deq1 = deque(cell1.elements)
        deq2 = deque(seq)
        for _ in range(size):
            deq2.rotate()
            if deq1 == deq2:
                return True
        return False

    def __str__(self):
        """
        String representation of regular cell
        Returns
        -------
        str
        """
        return f"Nodes set:{self.nodes}, boundary edges:{self.boundary}, attrs:{self.properties}"


class CellView:
    """A CellView class for cells of a CellComplex

    Parameters
    ----------
    name : str

    Examples
    --------
         >>> CV = CellView()
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (2,3,4,1) )
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (1,2,3,6) )
         >>> c1=Cell((1,2,3,4,5))
         >>> CV.insert_cell(c1)
         >>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        self._cells = dict()

    def __getitem__(self, cell):
        """
        Parameters
        ----------
        cell : tuple list or cell
            DESCRIPTION.
        Returns
        -------
        TYPE : dict or ilst or dicts
            return dict of properties associated with that cells
        """

        if isinstance(cell, Cell):

            if cell.elements not in self._cells:
                raise KeyError(f"cell {cell} is not in the cell dictionary")

            elif len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k].properties
            else:
                return [
                    self._cells[cell.elements][c].properties
                    for c in self._cells[cell.elements]
                ]
        elif isinstance(cell, tuple) or isinstance(cell, list):

            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    k = next(iter(self._cells[cell].keys()))
                    return self._cells[cell][k].properties
                else:
                    return [self._cells[cell][c].properties for c in self._cells[cell]]
            else:
                raise KeyError(f"cell {cell} is not in the cell dictionary")

        else:
            raise KeyError(f"input must be a tuple, list or a cell")

    # Set methods
    def __len__(self):
        if len(self._cells) == 0:
            return 0
        else:
            return np.sum([len(self._cells[cell]) for cell in self._cells])

    def __iter__(self):
        return iter(
            [
                self._cells[cell][key]
                for cell in self._cells
                for key in self._cells[cell]
            ]
        )

    def __contains__(self, e):
        if isinstance(e, tuple) or isinstance(e, list):
            return e in self._cells

        elif isinstance(e, Cell):
            return e.elements in self._cells
        else:
            return False

    def __repr__(self):
        """C
        String representation of regular cell
        Returns
        -------
        str
        """
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"

    def __str__(self):
        """
        String representation of regular cell

        Returns
        -------
        str
        """

        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"

    def insert_cell(self, cell, **attr):
        if isinstance(cell, tuple) or isinstance(cell, list):
            cell = Cell(elements=cell, name=str(len(self._cells)), **attr)
            if cell.elements not in self._cells:
                self._cells[cell.elements] = {0: cell}
            else:
                self._cells[cell.elements][len(self._cells[cell.elements])] = cell

        elif isinstance(cell, Cell):
            cell.properties.update(attr)
            if cell.elements not in self._cells:
                self._cells[cell.elements] = {0: cell}
            else:
                self._cells[cell.elements][len(self._cells[cell.elements])] = cell

        else:
            raise ValueError("input must be list, tuple or Cell type")

    def delete_cell(self, cell, key=None):

        if isinstance(cell, Cell):
            cell = cell.elements

        if cell in self._cells:
            if key is None:
                del self._cells[cell]
            elif key in self._cells[cell]:
                del self._cells[cell][key]
            else:
                raise KeyError(f"cell with key {key} is not in the complex ")
        else:
            raise KeyError(f"cell {cell} is not in the complex")

    def _cell_equivelance_class(self):
        """


        Returns
        -------
        equiv : TYPE
            DESCRIPTION.


        Example
        ------
         >>> CV = CellView()
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (2,3,4,1) )
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (1,2,3,6) )
         >>> CV.insert_cell ( (3,4,1,2) )
         >>> CV.insert_cell ( (4,3,2,1) )
         >>> CV.insert_cell ( (1,2,7,3) )
         >>> c1=Cell((1,2,3,4,5))
         >>> CV.insert_cell(c1)
         >>> d = CV._cell_equivelance_class()
         >>> d

        """
        equiv_classes = defaultdict(set)
        all_inserted_cells = set()
        for i, c1 in enumerate(self):
            for j, c2 in enumerate(self):
                if i == j:
                    if j not in all_inserted_cells:
                        equiv_classes[c1].add(j)
                elif i > j:
                    continue
                elif j in all_inserted_cells:
                    continue
                else:
                    if c1.is_homotopic_to(c2):
                        equiv_classes[c1].add(j)
                        all_inserted_cells.add(j)
        return equiv_classes

    def remove_equivalent_cells(self):
        """
        Example:
        ---------
         >>> CV = CellView()
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (2,3,4,1) )
         >>> CV.insert_cell ( (1,2,3,4) )
         >>> CV.insert_cell ( (1,2,3,6) )
         >>> CV.insert_cell ( (3,4,1,2) )
         >>> CV.insert_cell ( (4,3,2,1) )
         >>> CV.insert_cell ( (1,2,7,3) )
         >>> c1=Cell((1,2,3,4,5))
         >>> CV.insert_cell(c1)
         >>> CV.remove_equivalent_cells()
         >>> CV


        """
        equiv_classes = self._cell_equivelance_class()
        for c in list(self):
            if c not in equiv_classes:
                d = self._cells[c.elements]
                if len(d) == 1:
                    self.delete_cell(c)
                else:
                    for k, v in d.items():
                        if len(d) == 1:
                            break
                    else:
                        self.delete_cell(c, k)
