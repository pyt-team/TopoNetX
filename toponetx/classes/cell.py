"""


"""
from collections import Counter, defaultdict, deque
from itertools import zip_longest

__all__ = ["Cell", "CellView"]


class Cell:
    """A Regular 2d cell class.
    Parameters
    ==========

    elements: any iterable of hashables. Elements order is important and defines the 2
    cell up to cyclic permutation.
    name : str

    Examples
        >>> cell1 = Cell ( (1,2,3) )
        >>> cell2 = Cell ( (1,2,4,5) )
        >>> cell3 = Cell ( ("a","b","c") )
    """

    def __init__(self, elements, name=None, **attr):

        if name is None:
            self.name = "_"
        else:
            self.name = name
        elements = list(elements)
        self._boundary = frozenset(
            zip_longest(
                elements, elements[1:] + [elements[0]]
            )  # set of edges define the boundary of the 2d cell
        )
        if len(self._boundary) < 2:
            raise ValueError(
                f" cell must contain at least 2 edges, got {len(self._boundary)}"
            )
        _adjdict = {}
        for e in self._boundary:
            if e[0] in _adjdict:
                raise ValueError(
                    f" node {e[0]} is repeated multiple times in the input cell."
                    + " input cell violates the cell complex regularity condition."
                )
            _adjdict[e[0]] = e[1]
        self.nodes = tuple(elements)
        self.properties = dict()
        self.properties.update(attr)

    def __getitem__(self, item):
        if item not in self.properties:
            raise KeyError(f"attr {item} is not an attr in the cell {self.name}")
        else:
            return self.properties[item]

    # Set methods
    def __len__(self):
        return self.nodes

    def __iter__(self):
        return iter(self.nodes)

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
        return self._boundary

    @property
    def elements(self):
        return self.nodes

    def is_homotopic_to(self, cell):
        """
        Paramters:
        ---------
        cell : tuple, list or Cell

        Return : bool
        ------
            return True is self is homotopic to input cell and False otherwise.
        """

        if isinstance(cell, tuple) or isinstance(cell, list):
            seq = cell
        elif isinstance(cell, Cell):
            seq = cell.elements
        else:
            raise ValueError(
                "input cell must be a tuple/list of nodes defining a cell or Cell"
            )

        if len(self) != len(cell):
            return False

        mset1 = Counter(seq)
        mset2 = Counter(self.elements)
        if mset1 != mset2:
            return False

        size = len(seq)
        deq1 = deque(self.elements)
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

        if isinstance(cell, Cell):
            if len(self._cells[cell.elements]) == 1:
                return self._cells[cell.elements][0].properties
            else:
                return [
                    self._cells[cell.elements][c].properties
                    for c in self._cells[cell.elements]
                ]
        elif isinstance(cell, tuple) or isinstance(cell, list):
            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    return self._cells[cell][0].properties
                else:
                    return [self._cells[cell][c].properties for c in self._cells[cell]]
        else:
            raise KeyError(f"cell {cell} is not in the cell dictionary")

    # Set methods
    def __len__(self):
        return len(self._cells)

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
            if cell.elements not in self._cells:
                self._cells[cell.elements] = {0: cell}
            else:
                self._cells[cell.elements][len(self._cells[cell.elements])] = cell

        else:
            raise ValueError("input must be list, tuple or Cell type")

    def delete_cell(self, cell, key=None):
        if cell in self._cells:
            if key is None:
                del self._cells[cell]
            elif key in self._cells[cell]:
                del self._cells[cell][key]
            else:
                raise KeyError(f"cell with key {key} is not in the complex ")
        else:
            raise KeyError(f"cell {cell} is not in the complex")

    @staticmethod
    def _cyclic_permutation_equiv(seq1, seq2):
        mset1 = Counter(seq1)
        mset2 = Counter(seq2)
        if mset1 != mset2:
            return False

        size = len(seq1)
        deq1 = deque(seq1)
        deq2 = deque(seq2)
        for _ in range(size):
            deq2.rotate()
            if deq1 == deq2:
                return True
        return False

    def collapse_identical_elements(self, return_equivalence_classes=False):

        return defaultdict(set)
