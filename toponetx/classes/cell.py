"""


"""
from collections import Counter, defaultdict, deque
from itertools import zip_longest

__all__ = ["Cell", "CellView"]


class Cell:
    """A Regular 2d cell class.
    Parameters
    ==========

    elements: any iterable of hashables.
        Elements order is important and defines the 2
        cell up to cyclic permutation.

    name : str

    is_regular : bool
        checks for regularity conditions of the cell. Raises ValueError when
        the True and when the input elements do not satisfy the regullarity condition

    Note
    ------
    The regularity condition of a 2d cell :
            a 2d cell is regular if and only if there is no
            repeatition in the edges

    Examples
        >>> cell1 = Cell ( (1,2,3) )
        >>> cell2 = Cell ( (1,2,4,5),weight = 1 )
        >>> cell3 = Cell ( ("a","b","c") )
    """

    def __init__(self, elements, name=None, is_regular=True, **attr):

        if name is None:
            self.name = "_"
        else:
            self.name = name
        self._is_regular = is_regular
        elements = list(elements)
        self._boundary = list(
            zip_longest(elements, elements[1:] + [elements[0]])
        )  # list of edges define the boundary of the 2d cell

        if is_regular:
            if len(elements) == 2:
                raise ValueError(f" cell must contain at least 2 edges, got 2")
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
                        f"loops are not permitted, got f{e[0]} and f{e[1]} as an edge in the cell"
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

    def __len__(self):
        return len(self.nodes)

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
        return iter(self._boundary)

    @property
    def elements(self):
        return self.nodes

    def reverse(self):
        """

        reverse the sequnce of nodes that define the cell and return a new cell with the new reversed elements


        Paramters:
        ---------
         None

        Return : Cell
        ------

        """
        c = Cell(self.nodes[::-1], name=self.name, is_regular=self._is_regular)
        c.properties = self.properties
        return c

    def is_homotopic_to(self, cell):
        """
        Paramters:
        ---------
        cell : tuple, list or Cell

        Return : bool
        ------
            return True is self is homotopic to input cell and False otherwise.
        """

        return Cell._are_homotopic(self, cell) or Cell._are_homotopic(
            self.reverse(), cell
        )

    @staticmethod
    def _are_homotopic(cell1, cell):
        """
        Paramters:
        ---------
        cell1 : Cell
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

        else:
            raise KeyError(f"input must be a tuple, list or a cell")

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
            cell.properties.update(attr)
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
