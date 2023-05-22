"""Cell and CellView classes."""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from collections import Counter, deque
from itertools import zip_longest

import numpy as np

__all__ = ["Cell"]


class Cell:
    r"""Class representing a 2D cell.

     A 2D cell is an elementary building block used to build a 2D cell complex, whether regular or non-regular.

     Parameters
     ----------
     elements : iterable of hashable objects
         An iterable that contains hashable objects representing the nodes of the cell. The order of the elements is important
         and defines the cell up to cyclic permutation.
     name : str, optional
         A string representing the name of the cell. The default value is None.
     regular : bool, optional
         A boolean indicating whether the cell satisfies the regularity condition. The default value is True.
         A 2D cell is regular if and only if there is no repetition in the boundary edges that define the cell.
         By default, the cell is assumed to be regular unless otherwise specified. Self-loops are not allowed in the boundary
         of the cell. If a cell violates the cell complex regularity condition, a ValueError is raised.
     **attr : keyword arguments, optional
         Properties belonging to the cell can be added as key-value pairs. Both the key and value must be hashable.

     Notes
     -----
     - A cell is defined as an ordered sequence of nodes (n1, ..., nk), where each two consecutive nodes (ni, ni+1)
       define an edge in the boundary of the cell. Note that the last edge (nk, n1) is also included in the boundary
       of the cell and is used to close the cell. For instance, if a Cell is defined as `c = Cell((1, 2, 3))`,
       then `c.boundary` will return `[(1, 2), (2, 3), (3, 1)]`, which consists of three edges.
    - When cell is created, its boundary is automatically created as a
       set of edges that encircle the cell.
     Examples
     --------
     >>> cell1 = Cell((1, 2, 3))
     >>> cell2 = Cell((1, 2, 4, 5), weight=1)
     >>> cell3 = Cell(("a", "b", "c"))
     >>> # create geometric cell:
     >>> v0 = (0, 0)
     >>> v1 = (1, 0)
     >>> v2 = (1, 1)
     >>> v3 = (0, 1)
     # create the cell with the vertices and edges
     >>> cell = Cell([v0, v1, v2, v3],type="square")
     >>> cell["type"]
     >>> print(list(cell.boundary))
     [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)),
      ((0, 1), (0, 0))]
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
                        f" Node {e[0]} is repeated multiple times in the input cell."
                        + " Input cell violates the cell complex regularity condition."
                    )
                _adjdict[e[0]] = e[1]
        else:

            for e in self._boundary:
                if e[0] == e[1]:
                    raise ValueError(
                        f"self loops are not permitted, got {(e[0],e[1])} as an edge in the cell's boundary"
                    )

        self._elements = tuple(elements)
        self.properties = dict()
        self.properties.update(attr)

    def __getitem__(self, item):
        r"""Retrieve the value associated with the given key in the properties dictionary.

        Parameters
        ----------
        item : hashable
            The key to retrieve from the properties dictionary.

        Returns
        -------
        The value associated with the given key in the properties dictionary.

        Raises
        ------
        KeyError:
            If the given key is not in the properties dictionary.
        """
        if item not in self.properties:
            raise KeyError(f"attr {item} is not an attr in the cell {self.name}")
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        r"""Set the value associated with the given key in the properties dictionary.

        Parameters
        ----------
        key : hashable
            The key to set in the properties dictionary.
        item : hashable
            The value to associate with the given key in the properties dictionary.

        Returns
        -------
        None
        """
        self.properties[key] = item

    @property
    def is_regular(self):
        r"""
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
        r"""Get the number of elements in the cell.

        Returns
        -------
        int
            The number of elements in the cell.
        """
        return len(self._elements)

    def __iter__(self):
        r"""Iterate over the elements in the cell.

        Returns
        -------
        iterator
            An iterator over the elements in the cell.
        """
        return iter(self._elements)

    def sign(self, edge):
        r"""The sign method of the Cell class takes an edge as input and returns the sign of the edge with respect to the cell. If the edge is in the boundary of the cell, then the sign is 1 if the edge is in the counterclockwise direction around the cell and -1 if it is in the clockwise direction. If the edge is not in the boundary of the cell, a KeyError is raised.

        Args:

        edge: an iterable representing the edge whose sign with respect to the cell is to be computed.
        Returns:

        1: if the edge is in the boundary of the cell and is in the counterclockwise direction around the cell.
        -1: if the edge is in the boundary of the cell and is in the clockwise direction around the cell.
        Raises:

        KeyError: if the input edge is not in the boundary of the cell.
        ValueError: if the input edge is not iterable or if its length is not 2.

        """

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
        return e in self._elements

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
        r"""
        a 2d cell is characterized by its boundary edges
        Return
        ------
        iterator of tuple representing boundary edges given in cyclic order
        """
        return iter(self._boundary)

    @property
    def elements(self):
        """Elements of the cell."""
        return self._elements

    def reverse(self):
        """Reverse the sequence of nodes that defines the cell.

        This returns a new cell with the new reversed elements.

        Return : Cell
        ------
        """
        c = Cell(self._elements[::-1], name=self.name, regular=self._regular)
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
        return f"Nodes set:{self._elements}, boundary edges:{self.boundary}, attrs:{self.properties}"
