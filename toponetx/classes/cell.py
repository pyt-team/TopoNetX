"""Cell class."""

from collections import Counter, deque
from collections.abc import Collection, Hashable, Iterable, Sequence
from itertools import zip_longest
from typing import Literal

from typing_extensions import Self

from toponetx.classes.complex import Atom

__all__ = ["Cell"]


class Cell(Atom[tuple[Hashable]]):
    """Class representing a 2D cell.

    A 2D cell is an elementary building block used to build a 2D cell complex, whether regular or non-regular.

    Parameters
    ----------
    elements : iterable of hashable objects
        An iterable that contains hashable objects representing the nodes of the cell. The order of the elements is important
        and defines the cell up to cyclic permutation.
    regular : bool, optional
        A boolean indicating whether the cell satisfies the regularity condition. The default value is True.
        A 2D cell is regular if and only if there is no repetition in the boundary edges that define the cell.
        By default, the cell is assumed to be regular unless otherwise specified. Self-loops are not allowed in the boundary
        of the cell. If a cell violates the cell complex regularity condition, a ValueError is raised.
    **kwargs : keyword arguments, optional
        Attributes belonging to the cell can be added as key-value pairs. Both the key and value must be hashable.

    Notes
    -----
    - A cell is defined as an ordered sequence of nodes (n_1, ..., n_k), where each two
      consecutive nodes (n_i, n_{i+1}) define an edge in the boundary of the cell. Note
      that the last edge (n_k, n_1) is also included in the boundary of the cell. For
      instance, if a Cell is defined as `c = Cell((1, 2, 3))`, then `c.boundary` will
      return `[(1, 2), (2, 3), (3, 1)]`, which consists of three edges.
    - When a cell is created, its boundary is automatically created as a set of edges
      that encircle the cell.

    Examples
    --------
    >>> cell1 = tnx.Cell((1, 2, 3))
    >>> cell2 = tnx.Cell((1, 2, 4, 5), weight=1)
    >>> cell3 = tnx.Cell(("a", "b", "c"))
    >>> # create geometric cell:
    >>> v0 = (0, 0)
    >>> v1 = (1, 0)
    >>> v2 = (1, 1)
    >>> v3 = (0, 1)
    # create the cell with the vertices and edges
    >>> cell = tnx.Cell([v0, v1, v2, v3], type="square")
    >>> cell["type"]
    >>> list(cell.boundary)
    [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)),
    ((0, 1), (0, 0))]
    """

    def __init__(self, elements: Collection, regular: bool = True, **kwargs) -> None:
        super().__init__(tuple(elements), **kwargs)

        self._regular = regular
        elements = list(elements)
        self._boundary = list(
            zip_longest(elements, elements[1:] + [elements[0]])
        )  # list of edges defines the boundary of the 2d cell
        if len(elements) <= 1:
            raise ValueError(
                f"cell must contain at least 2 edges, got {len(elements)+1}"
            )

        if regular:
            _adjdict = {}
            for e in self._boundary:
                if e[0] in _adjdict:
                    raise ValueError(
                        f"Node {e[0]} is repeated multiple times in the input cell. "
                        "Input cell violates the cell complex regularity condition."
                    )
                _adjdict[e[0]] = e[1]
        else:
            for e in self._boundary:
                if e[0] == e[1]:
                    raise ValueError(
                        f"self loops are not permitted, got {(e[0],e[1])} as an edge in the cell's boundary"
                    )

    def clone(self) -> Self:
        """Clone the Cell with all attributes.

        The clone method by default returns an independent shallow copy of the cell and
        attributes. That is, if an attribute is a container, that container is shared
        by the original and the copy. Use Python's `copy.deepcopy` for new containers.

        Returns
        -------
        Cell
            A copy of this cell.
        """
        return self.__class__(self.elements, self._regular, **self._attributes)

    @staticmethod
    def is_valid_cell(elements: Sequence, regular: bool = False) -> bool:
        """Check if a 2D cell defined by a list of elements is valid.

        Parameters
        ----------
        elements : Sequence
            List of elements defining the cell.
        regular : bool, default=False
            Indicates if the cell is regular.

        Returns
        -------
        bool
            True if the cell is valid, False otherwise.
        """
        _boundary = list(
            zip_longest(elements, elements[1:] + [elements[0]])
        )  # list of edges define the boundary of the 2d cell
        if len(elements) <= 1:
            return False

        if regular:
            _adjdict = {}
            for e in _boundary:
                if e[0] in _adjdict:
                    return False

                _adjdict[e[0]] = e[1]
        else:
            for e in _boundary:
                if e[0] == e[1]:
                    return False
        return True

    @property
    def is_regular(self) -> bool:
        """Check if a cell is regular.

        Returns
        -------
        bool
            True if the Cell is regular, and False otherwise.
        """
        if self._regular:  # condition enforced
            return True

        _adjdict = {}
        for e in self._boundary:
            if e[0] in _adjdict:
                return False
            _adjdict[e[0]] = e[1]

        return True

    def sign(self, edge) -> Literal[-1, 1]:
        """Compute the sign of the given edge with respect to this cell.

        This takes an edge as input and computes the sign of the edge with respect to the cell.

        If the edge is in the boundary of the cell, then the sign is 1 if the edge is in the
        counterclockwise direction around the cell and -1 if it is in the clockwise direction.

        If the edge is not in the boundary of the cell, a KeyError is raised.

        Parameters
        ----------
        edge : Iterable
            The edge whose sign should be computed.

        Returns
        -------
        {1, -1}
            1: if the edge is in the counterclockwise direction around the cell.
            -1: if the edge is in the clockwise direction around the cell.

        Raises
        ------
        KeyError
            If the input edge is not in the boundary of the cell.
        ValueError
            If the input edge is not valid.
        TypeError
            If the input edge is not iterable.
        """
        if not isinstance(edge, Iterable):
            raise TypeError(f"The input {edge} must be iterable.")

        edge = tuple(edge)
        if len(edge) == 2:
            if tuple(edge) in self.boundary:
                return 1
            if tuple(edge)[::-1] in self.boundary:
                return -1
            raise KeyError(f"Ihe input {edge} is not in the boundary of the cell.")

        raise ValueError(f"The input {edge} is not a valid edge.")

    def __repr__(self) -> str:
        """Return string representation of regular cell.

        Returns
        -------
        str
            The string representation of the regular cell.
        """
        return f"Cell({self.elements})"

    @property
    def boundary(self):
        """Boundary.

        A 2d cell is characterized by its boundary edges.

        Returns
        -------
        Iterable[tuple[int, int]]
            Iterator of tuple representing boundary edges given in cyclic order.

        Examples
        --------
        For a cell [1,2,3,4,5], the boundary is the edges that define that cell :
            (1,2), (2,3), (3,4), (4,5) and (5,1).
        """
        return iter(self._boundary)

    def reverse(self):
        """Reverse the sequence of nodes that defines the cell.

        Returns
        -------
        Cell
            New cell with the new reversed elements.
        """
        return Cell(
            self.elements[::-1],
            regular=self._regular,
            **self._attributes,
        )

    def is_homotopic_to(self, cell) -> bool:
        """Check if self is homotopic to input cell.

        Parameters
        ----------
        cell : tuple, list or Cell
            The cell to check for homotopy.

        Returns
        -------
        bool
            Return True is self is homotopic to input cell and False otherwise.
        """
        return Cell._are_homotopic(self, cell) or Cell._are_homotopic(
            self.reverse(), cell
        )

    @staticmethod
    def _are_homotopic(cell1, cell) -> bool:
        """Check if cell1 is homotopic to input cell.

        Parameters
        ----------
        cell1 : Cell
            First cell to check for homotopy.
        cell : tuple | list | Cell
            Second cell to check for homotopy.

        Returns
        -------
        bool
            Returns True is self is homotopic to input cell and False otherwise.

        Notes
        -----
        in a 2d-cell complex, two 2d-cells are homotopic iff one
            of them can be obtaine from the other by a cylic rotation
            of the boundary verties defining the 2d cells.
        """
        if isinstance(cell, tuple | list):
            seq = cell
        elif isinstance(cell, Cell):
            seq = cell.elements
        else:
            raise TypeError(
                "fInput {cell} must be a tuple/list of nodes defining a cell or Cell"
            )

        if not isinstance(cell1, Cell):
            raise TypeError("first argument cell must be a cell")

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

    def __str__(self) -> str:
        """Return string representation of regular cell.

        Returns
        -------
        str
            Returns string representation of regular cell.
        """
        return f"Nodes set:{self.elements}, boundary edges:{self.boundary}, attrs:{self._attributes}"
