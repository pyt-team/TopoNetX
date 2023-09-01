"""Module with views.

Such as:
HyperEdgeView, CellView, SimplexView, NodeView.
"""
from collections.abc import Collection, Hashable, Iterable, Iterator
from itertools import chain
from typing import Any

import numpy as np

from toponetx.classes.cell import Cell
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.simplex import Simplex
from toponetx.exception import TopoNetXError

__all__ = [
    "HyperEdgeView",
    "ColoredHyperEdgeView",
    "CellView",
    "SimplexView",
    "NodeView",
]


class CellView:
    """A CellView class for cells of a CellComplex.

    Parameters
    ----------
    name : str, optional
        The name of the cell view.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

        # Initialize a dictionary to hold cells, with keys being the tuple
        # that defines the cell, and values being dictionaries of cell objects
        # with different properties
        self._cells = dict()

    def __getitem__(self, cell):
        """Return the properties of a given cell.

        Parameters
        ----------
        cell : tuple list or cell
            The cell of interest.

        Returns
        -------
        TYPE : dict or list of dicts
            The properties associated with the cell.

        Raises
        ------
        KeyError
            If the cell is not in the cell dictionary.
        """
        if isinstance(cell, Cell):

            if cell.elements not in self._cells:
                raise KeyError(
                    f"cell {cell.__repr__()} is not in the cell dictionary",
                )

            # If there is only one cell with these elements, return its properties
            elif len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k]._properties

            # If there are multiple cells with these elements, return the properties of all cells
            else:
                return [
                    self._cells[cell.elements][c]._properties
                    for c in self._cells[cell.elements]
                ]

        # If a tuple or list is passed in, assume it represents a cell
        elif isinstance(cell, tuple) or isinstance(cell, list):

            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    k = next(iter(self._cells[cell].keys()))
                    return self._cells[cell][k]._properties
                else:
                    return [self._cells[cell][c]._properties for c in self._cells[cell]]
            else:
                raise KeyError(f"cell {cell} is not in the cell dictionary")

        else:
            raise TypeError("Input must be a tuple, list or a cell.")

    def raw(self, cell: tuple | list | Cell) -> Cell | list[Cell]:
        """Indexes the raw cell objects analogous to the overall index of CellView.

        Parameters
        ----------
        cell : tuple, list, or cell
            The cell of interest.

        Returns
        -------
        TYPE : Cell or list of Cells
            The raw Cell objects.
            If more than one cell with the same boundary exists, returns a list;
            otherwise a single cell.

        Raises
        ------
        KeyError
            If the cell is not in the cell dictionary.
        """
        if isinstance(cell, Cell):

            if cell.elements not in self._cells:
                raise KeyError(f"cell {cell.__repr__()} is not in the cell dictionary")

            # If there is only one cell with these elements, return its properties
            elif len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k]

            # If there are multiple cells with these elements, return the properties of all cells
            else:
                return [
                    self._cells[cell.elements][c] for c in self._cells[cell.elements]
                ]

        # If a tuple or list is passed in, assume it represents a cell
        elif isinstance(cell, tuple) or isinstance(cell, list):

            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    k = next(iter(self._cells[cell].keys()))
                    return self._cells[cell][k]
                else:
                    return [self._cells[cell][c] for c in self._cells[cell]]
            else:
                raise KeyError(f"cell {cell} is not in the cell dictionary")

        else:
            raise TypeError("Input must be a tuple, list or a cell.")

    def __len__(self) -> int:
        """Return the number of cells in the cell view."""
        return sum(len(self._cells[cell]) for cell in self._cells)

    def __iter__(self) -> Iterator:
        """Iterate over all cells in the cell view."""
        return iter(
            [
                self._cells[cell][key]
                for cell in self._cells
                for key in self._cells[cell]
            ]
        )

    def __contains__(self, e: tuple | list | Cell) -> bool:
        """Check if a given element is in the cell view.

        Parameters
        ----------
        e : tuple, list, or cell
            The element to check.

        Returns
        -------
        bool
            Whether or not the element is in the cell view.
        """
        while True:
            if isinstance(e, Cell):
                break
            elif isinstance(e, tuple):
                break
            elif isinstance(e, list):
                break
            else:
                raise TypeError("Input must be of type: tuple, list or a cell.")

        e = Cell(e)
        e_homotopic_to = [e.is_homotopic_to(x) for x in self._cells]
        return any(e_homotopic_to)

    def __repr__(self) -> str:
        """Return a string representation of the cell view."""
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"

    def __str__(self) -> str:
        """Return a string representation of the cell view."""
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"


class ColoredHyperEdgeView:
    """A class for viewing the cells/hyperedges of a colored hypergraph.

    Provides methods for accessing, and retrieving
    information about the cells/hyperedges of a complex.

    Parameters
    ----------
    name : str, optional
        The name of the view.

    Examples
    --------
    >>> hev = ColoredHyperEdgeView()
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.hyperedge_dict = {}

    def __getitem__(self, hyperedge):
        """Get item.

        Parameters
        ----------
        hyperedge : Hashable or ColoredHyperEdge
            DESCRIPTION.

        Returns
        -------
        TYPE : dict or list or dicts
            return dict of properties associated with that hyperedges
        """
        if isinstance(hyperedge, Iterable):
            if len(hyperedge) == 2:
                if isinstance(hyperedge[0], Iterable) and isinstance(hyperedge[1], int):
                    hyperedge_elements, key = hyperedge
                else:
                    raise KeyError(
                        "Input hyperedge must of the form (Iterable representing elements of hyperedge, key)"
                    )
        hyperedge = HyperEdgeView()._to_frozen_set(hyperedge_elements)
        hyperedge_elements = hyperedge
        rank = self.get_rank(hyperedge_elements)
        return self.hyperedge_dict[rank][hyperedge_elements][key]

    @property
    def shape(self) -> tuple[int, ...]:
        """Compute shape."""
        shape = []
        for i in self.allranks:
            sm = sum([len(self.hyperedge_dict[i][k]) for k in self.hyperedge_dict[i]])
            shape.append(sm)
        return tuple(shape)

    def __len__(self) -> int:
        """Compute the number of nodes."""
        return sum(self.shape[1:])

    def __iter__(self) -> Iterator:
        """Iterate over the hyperedges."""
        lst = []
        for r in self.hyperedge_dict:
            if r == 0:
                continue
            else:
                for he in self.hyperedge_dict[r]:
                    for k in self.hyperedge_dict[r][he]:
                        lst.append((he, k))
        return iter(lst)

    def __contains__(self, hyperedge: Collection) -> bool:
        """Check if hyperedge is in the hyperedges.

        Note
        ----
        Assumption of input here hyperedge = ( elements of hyperedge, key of hyperedge)
        """
        if len(self.hyperedge_dict) == 0:
            return False
        if isinstance(hyperedge, Iterable):
            if len(hyperedge) == 0:
                return False
            if len(hyperedge) == 2:
                if isinstance(hyperedge, HyperEdge):
                    hyperedge_elements = hyperedge.elements
                    key = 0
                elif isinstance(hyperedge[0], Iterable) and isinstance(
                    hyperedge[1], int
                ):
                    hyperedge_elements_ = hyperedge[0]
                    if not isinstance(hyperedge_elements_, HyperEdge):
                        hyperedge_elements, key = hyperedge
                    else:
                        _, key = hyperedge
                        hyperedge_elements = hyperedge_elements_.elements
                else:
                    hyperedge_elements = hyperedge
                    key = 0
            else:
                hyperedge_elements = hyperedge
                key = 0
            all_ranks = self.allranks
        else:
            return False
        if isinstance(hyperedge_elements, Iterable) and not isinstance(
            hyperedge_elements, HyperEdge
        ):
            if len(hyperedge_elements) == 0:
                return False
            else:
                for i in all_ranks:
                    if frozenset(hyperedge_elements) in self.hyperedge_dict[i]:
                        if key in self.hyperedge_dict[i][frozenset(hyperedge_elements)]:
                            return True
                        else:
                            return False
                return False
        elif isinstance(hyperedge_elements, HyperEdge):
            if len(hyperedge_elements) == 0:
                return False
            else:
                for i in all_ranks:
                    if frozenset(hyperedge_elements.elements) in self.hyperedge_dict[i]:
                        return True
                return False

    def __repr__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
        """
        return f"ColoredHyperEdgeView({[(tuple(x[0]),x[1]) for x in self]})"

    def __str__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
        """
        return f"ColoredHyperEdgeView({[(tuple(x[0]),x[1]) for x in self]})"

    def skeleton(self, rank, store_hyperedge_key=True):
        """Skeleton of the complex."""
        if rank not in self.hyperedge_dict:
            return []
        if store_hyperedge_key:
            return sorted(
                [
                    (he, k)
                    for he in self.hyperedge_dict[rank]
                    for k in self.hyperedge_dict[rank][he]
                ]
            )
        else:
            return sorted(
                [
                    he
                    for he in self.hyperedge_dict[rank]
                    for k in self.hyperedge_dict[rank][he]
                ]
            )

    def get_rank(self, e):
        """Get rank.

        Parameters
        ----------
        e : Iterable, Hashable or ColoredHyperEdge

        Returns
        -------
        int, the rank of the colored hyperedge e
        """
        if isinstance(e, HyperEdge):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e.elements) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e.elements} is not in the complex")
        elif isinstance(e, str):
            if frozenset({e}) in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {frozenset({e})} is not in the complex")
        elif isinstance(e, Iterable):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e} is not in the complex")
        elif isinstance(e, Hashable) and not isinstance(e, Iterable):
            if frozenset({e}) in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {frozenset({e})} is not in the complex")

    @property
    def allranks(self):
        """All ranks."""
        return sorted(list(self.hyperedge_dict.keys()))


class HyperEdgeView:
    """A class for viewing the cells/hyperedges of a combinatorial complex.

    Provides methods for accessing, and retrieving
    information about the cells/hyperedges of a complex.

    Parameters
    ----------
    name : str, optional
        The name of the view.

    Examples
    --------
    >>> hev = HyperEdgeView()
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.hyperedge_dict = {}

    @staticmethod
    def _to_frozen_set(hyperedge):
        if isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset(hyperedge)
        elif isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.elements
        elif isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset([hyperedge])
        return hyperedge_

    def __getitem__(self, hyperedge):
        """Get item.

        Parameters
        ----------
        hyperedge : Hashable or HyperEdge
            DESCRIPTION.

        Returns
        -------
        TYPE : dict or list or dicts
            return dict of properties associated with that hyperedges
        """
        hyperedge_ = HyperEdgeView._to_frozen_set(hyperedge)
        rank = self.get_rank(hyperedge_)
        return self.hyperedge_dict[rank][hyperedge_]

    @property
    def shape(self) -> tuple[int, ...]:
        """Compute shape."""
        return tuple(len(self.hyperedge_dict[i]) for i in self.allranks)

    def __len__(self) -> int:
        """Compute the number of nodes."""
        return sum(self.shape)

    def __iter__(self) -> Iterator:
        """Iterate over the hyperedges."""
        return chain.from_iterable(self.hyperedge_dict.values())

    def __contains__(self, e: Collection) -> bool:
        """Check if e is in the hyperedges."""
        if len(self.hyperedge_dict) == 0:
            return False
        all_ranks = self.allranks
        if isinstance(e, Iterable):
            if len(e) == 0:
                return False
            else:
                for i in all_ranks:
                    if frozenset(e) in self.hyperedge_dict[i]:
                        return True
                return False
        elif isinstance(e, HyperEdge):
            if len(e) == 0:
                return False
            else:
                for i in all_ranks:
                    if frozenset(e.elements) in self.hyperedge_dict[i]:
                        return True
                return False
        elif isinstance(e, Hashable):
            return frozenset({e}) in self.hyperedge_dict[0]

    def __repr__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
        """
        return f"HyperEdgeView({[tuple(x) for x in self]})"

    def __str__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
        """
        return f"HyperEdgeView({[tuple(x) for x in self]})"

    def skeleton(self, rank, level=None):
        """Skeleton of the complex."""
        if level is None or level == "equal":
            elements = []
            if rank in self.allranks:
                return sorted(list(self.hyperedge_dict[rank].keys()))
            else:
                return []

        elif level == "upper" or level == "up":
            elements = []
            for rank_i in self.allranks:
                if rank_i >= rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)

        elif level == "lower" or level == "down":
            elements = []
            for rank_i in self.allranks:
                if rank_i <= rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)
        else:
            raise TopoNetXError(
                "level must be None, equal, 'upper', 'lower', 'up', or 'down' "
            )

    def get_rank(self, e):
        """Get rank.

        Parameters
        ----------
        e : Iterable, Hashable or HyperEdge

        Returns
        -------
        int, the rank of the hyperedge e
        """
        if isinstance(e, HyperEdge):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e.elements) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e.elements} is not in the complex")
        elif isinstance(e, str):
            if frozenset({e}) in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {frozenset({e})} is not in the complex")
        elif isinstance(e, Iterable):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e} is not in the complex")
        elif isinstance(e, Hashable) and not isinstance(e, Iterable):
            if frozenset({e}) in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {frozenset({e})} is not in the complex")

    @property
    def allranks(self):
        """All ranks."""
        return sorted(list(self.hyperedge_dict.keys()))

    def _get_lower_rank(self, rank):
        if len(self.allranks) == 0:
            return -1

        ranks = sorted(self.allranks)
        if rank <= min(ranks) or rank >= max(ranks):
            return -1
        return ranks[ranks.index(rank) - 1]

    def _get_higher_rank(self, rank):
        if len(self.allranks) == 0:
            return -1
        ranks = sorted(self.allranks)
        if rank <= min(ranks) or rank >= max(ranks):
            return -1
        return ranks[ranks.index(rank) + 1]


class SimplexView:
    """Simplex View class.

    The SimplexView class is used to provide a view/read only information
    into a subset of the nodes in a simplex.
    These classes are used in conjunction with the SimplicialComplex class
    for view/read only purposes for simplices in simplicial complexes.

    Parameters
    ----------
    name : str, optional
        Name of the SimplexView instance.

    Attributes
    ----------
    max_dim : int
        Maximum dimension of the simplices in the SimplexView instance.
    faces_dict : list of dict
        A list containing dictionaries of faces for each dimension.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

        self.max_dim = -1
        self.faces_dict = []

    def __getitem__(self, simplex):
        """Get the dictionary of properties associated with the given simplex.

        Parameters
        ----------
        simplex : tuple, list or Simplex
            A tuple or list of nodes representing a simplex.

        Returns
        -------
        dict or list or dict
            A dictionary of properties associated with the given simplex.
        """
        if isinstance(simplex, Simplex):
            if simplex.elements in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex.elements]
        elif isinstance(simplex, Iterable):
            simplex = frozenset(simplex)
            if simplex in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex]
            else:
                raise KeyError(f"input {simplex} is not in the simplex dictionary")

        elif isinstance(simplex, Hashable):
            if frozenset({simplex}) in self:
                return self.faces_dict[0][frozenset({simplex})]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the number of simplices in each dimension.

        Returns
        -------
        tuple of ints
            A tuple of integers representing the number of simplices in each dimension.
        """
        return tuple(len(self.faces_dict[i]) for i in range(len(self.faces_dict)))

    def __len__(self) -> int:
        """Return the number of simplices in the SimplexView instance."""
        return sum(self.shape)

    def __iter__(self) -> Iterator:
        """Return an iterator over all simplices in the simplex view."""
        return chain.from_iterable(self.faces_dict)

    def __contains__(self, item) -> bool:
        """Check if a simplex is in the simplex view.

        Parameters
        ----------
        item : Any
            The simplex to be checked for membership in the simplex view

        Returns
        -------
        bool
            True if the simplex is in the simplex view, False otherwise

        Examples
        --------
        Check if a node is in the simplex view:

        >>> view = SimplexView()
        >>> view.faces_dict.append({frozenset({1}): {'weight': 1}})
        >>> view.max_dim = 0
        >>> 1 in view
        True
        >>> 2 in view
        False

        Check if a simplex is in the simplex view:

        >>> view.faces_dict.append({frozenset({1, 2}): {'weight': 1}})
        >>> view.max_dim = 1
        >>> {1, 2} in view
        True
        >>> {1, 3} in view
        False
        >>> {1, 2, 3} in view
        False
        """
        if isinstance(item, Iterable):
            item = frozenset(item)
            if not 0 < len(item) <= self.max_dim + 1:
                return False
            return item in self.faces_dict[len(item) - 1]
        elif isinstance(item, Hashable):
            return frozenset({item}) in self.faces_dict[0]
        return False

    def __repr__(self) -> str:
        """Return string representation that can be used to recreate it."""
        all_simplices: list[tuple[int, ...]] = []
        for i in range(len(self.faces_dict)):
            all_simplices += [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"

    def __str__(self) -> str:
        """Return detailed string representation of the simplex view."""
        all_simplices: list[tuple[int, ...]] = []
        for i in range(len(self.faces_dict)):
            all_simplices += [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"


class NodeView:
    """Node view class."""

    def __init__(
        self, objectdict, cell_type, colored_nodes=False, name: str = ""
    ) -> None:
        self.name = name
        if len(objectdict) != 0:
            self.nodes = objectdict[0]
        else:
            self.nodes = {}

        if cell_type is None:
            raise ValueError("cell_type cannot be None")

        self.cell_type = cell_type
        self.colored_nodes = colored_nodes

    def __repr__(self) -> str:
        """Return string representation of nodes.

        Returns
        -------
        str
        """
        all_nodes = [tuple(j) for j in self.nodes.keys()]

        return f"NodeView({all_nodes})"

    def __iter__(self) -> Iterator:
        """Return an iterator over all nodes in the node view."""
        return iter(self.nodes)

    def __getitem__(self, cell):
        """Get item.

        Parameters
        ----------
        cell : tuple list or AbstractCell or Simplex
            A cell.

        Returns
        -------
        dict or list
            Dict of properties associated with that cells.
        """
        if isinstance(cell, self.cell_type):
            if cell.elements in self.nodes:
                return self.nodes[cell.elements]
        elif isinstance(cell, Iterable):
            cell = frozenset(cell)
            if cell in self.nodes:
                if self.colored_nodes:
                    return self.nodes[cell][0]
                else:
                    return self.nodes[cell]
            else:
                raise KeyError(f"input {cell} is not in the node set of the complex")

        elif isinstance(cell, Hashable):
            if cell in self:
                if self.colored_nodes:
                    return self.nodes[frozenset({cell})][0]
                else:
                    return self.nodes[frozenset({cell})]

    def __len__(self) -> int:
        """Compute the number of nodes."""
        return len(self.nodes)

    def __contains__(self, e) -> bool:
        """Check if e is in the nodes."""
        if isinstance(e, Hashable) and not isinstance(e, self.cell_type):
            return frozenset({e}) in self.nodes

        elif isinstance(e, self.cell_type):
            return e.elements in self.nodes

        elif isinstance(e, Iterable):
            if len(e) == 1:
                return frozenset(e) in self.nodes
        else:
            return False
