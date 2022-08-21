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
        self.nodes = elements
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
        return f"{self.elements}"

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

    def __str__(self):
        """
        String representation of regular cell
        Returns
        -------
        str
        """
        return f"Nodes set: : {self.nodes}, boundary edges : {self.boundary}, attrs: {self.properties}"


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
    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        self._cells = dict()
        self._cell_index = dict()
        self._num_cells = len(self._cells)

    def __getitem__(self, key):
        if key in self._cells:
            return self._cells[key]
        elif key in self._cell_index:
            return self._cell_index[key]
        else:
            raise KeyError(f"key {key} is not in the cell dictionary")

    # Set methods
    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells.values())

    def __contains__(self, e):
        if isinstance(e, tuple) or isinstance(e, list):
            for c in self._cells:
                if tuple(e) == tuple(self._cells[c].elements):
                    return True
            return False
        elif isinstance(e, Cell):
            return e in self._cell_index

    def __repr__(self):
        """C
        String representation of regular cell
        Returns
        -------
        str
        """
        return f"CellView({[cell for cell in self._cells.values()]})"

    def __str__(self):
        """
        String representation of regular cell
        Returns
        -------
        str
        329025
        """

        return f"{[cell for cell in self._cells.values()]} "

    def insert_cell(self, cell, **attr):
        if isinstance(cell, tuple) or isinstance(cell, list):
            cell = Cell(elements=cell, name=str(len(self._cells)), **attr)
            l = len(self._cells)
            self._cells[l] = cell
            self._cell_index[cell] = l
        elif isinstance(cell, Cell):
            cell.properties.update(attr)
            cell.name = str(len(self._cells))
            l = len(self._cells)
            self._cells[l] = cell
            self._cell_index[cell] = l
        else:
            raise ValueError("input must be list, tuple or Cell type")

    def delete_cell(self, key):
        if key in self._cells:
            cell = self._cells[key]
            del self._cells[key]
            del self._cell_index[cell]
        elif key in self._cell_index:
            index = self._cell_index[key]
            del self._cells[index]
            del self._cell_index[key]
        else:
            raise KeyError(f"The cell with key {key} is not in the complex")

    def delete_cell_by_set(self, elements):
        elements = tuple(elements)
        key_found = False
        to_be_deleted = []
        for e in self._cell_index:
            lst = e.elements

            if tuple(lst) == elements:
                index = self._cell_index[e]
                to_be_deleted.append([index, e])
                key_found = True
        for keys in to_be_deleted:
            del self._cells[keys[0]]
            del self._cell_index[keys[1]]

        if key_found == False:
            raise KeyError(f"The cell {elements} is not in the complex")

    @staticmethod
    def _is_cyc_perm(seq1, seq2):
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
