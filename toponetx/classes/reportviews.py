"""Module with views.

Such as:
HyperEdgeView, CellView, SimplexView, NodeView.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterable, Iterator, Sequence
from itertools import chain
from typing import Any, Generic, Literal, TypeVar

from toponetx.classes.cell import Cell
from toponetx.classes.complex import Atom
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.path import Path
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplex_trie import SimplexTrie

__all__ = [
    "AtomView",
    "CellView",
    "ColoredHyperEdgeView",
    "HyperEdgeView",
    "NodeView",
    "PathView",
    "SimplexView",
]

T_Atom = TypeVar("T_Atom", bound=Atom)


class AtomView(ABC, Generic[T_Atom]):
    """Abstract class representing a read-only view on a collection of atoms."""

    @abstractmethod
    def __contains__(self, atom: Any) -> bool:
        """Check if a given element is in the view.

        Parameters
        ----------
        atom : Any
            The element to check.

        Returns
        -------
        bool
            Whether the element is in the view.
        """

    @abstractmethod
    def __getitem__(self, atom: Any) -> dict:
        """Get the attributes of a given element.

        Parameters
        ----------
        atom : Any
            The element of interest.

        Returns
        -------
        dict
            The attributes associated with the element.

        Raises
        ------
        KeyError
            If the element is not in the view.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[T_Atom]:
        """Iterate over all elements in the view.

        Returns
        -------
        Iterator
            Iterator to iterate over all elements in the view.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of elements in the view.

        Returns
        -------
        int
            The number of elements in the view.
        """


class CellView(AtomView[Cell]):
    """A CellView class for cells of a CellComplex."""

    # Dictionary to hold cells, with keys being the tuple that defines the cell, and
    # values being dictionaries of cell objects with different attributes
    _cells: dict[tuple[Hashable, ...], dict[int, Cell]]

    def __init__(self) -> None:
        self._cells = {}

    def __getitem__(self, cell: Any) -> dict[Hashable, Any]:
        """Return the attributes of a given cell.

        Parameters
        ----------
        cell : Any
            The cell of interest.

        Returns
        -------
        dict[Hashable, Any]
            The attributes associated with the cell.

        Raises
        ------
        KeyError
            If the cell is not in the cell dictionary.
        """
        if isinstance(cell, Cell):
            if cell.elements not in self._cells:
                raise KeyError(
                    f"cell {cell!r} is not in the cell dictionary",
                )

            # If there is only one cell with these elements, return its attributes
            if len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k]._attributes

            # If there are multiple cells with these elements, return the attributes of all cells
            return [
                self._cells[cell.elements][c]._attributes
                for c in self._cells[cell.elements]
            ]

        # If a tuple or list is passed in, assume it represents a cell
        if isinstance(cell, Iterable):
            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    k = next(iter(self._cells[cell].keys()))
                    return self._cells[cell][k]._attributes
                return [self._cells[cell][c]._attributes for c in self._cells[cell]]

            raise KeyError(f"cell {cell} is not in the cell dictionary")

        raise TypeError("Input must be a tuple, list or a cell.")

    def raw(self, cell: tuple | list | Cell) -> Cell | list[Cell]:
        """Index the raw cell objects analogous to the overall index of CellView.

        Parameters
        ----------
        cell : tuple, list, or cell
            The cell of interest.

        Returns
        -------
        Cell or list of Cells
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
                raise KeyError(f"cell {cell!r} is not in the cell dictionary")

            # If there is only one cell with these elements, return its attributes
            if len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k]

            # If there are multiple cells with these elements, return the attributes of all cells
            return [self._cells[cell.elements][c] for c in self._cells[cell.elements]]

        # If a tuple or list is passed in, assume it represents a cell
        if isinstance(cell, tuple | list):
            cell = tuple(cell)
            if cell in self._cells:
                if len(self._cells[cell]) == 1:
                    k = next(iter(self._cells[cell].keys()))
                    return self._cells[cell][k]
                return [self._cells[cell][c] for c in self._cells[cell]]
            raise KeyError(f"cell {cell} is not in the cell dictionary")

        raise TypeError("Input must be a tuple, list or a cell.")

    def __len__(self) -> int:
        """Return the number of cells in the cell view.

        Returns
        -------
        int
            The number of cells in the cell view.
        """
        return sum(len(self._cells[cell]) for cell in self._cells)

    def __iter__(self) -> Iterator[Cell]:
        """Iterate over all cells in the cell view.

        Returns
        -------
        Iterator
            Iterator to iterate over all cells in the cell view.
        """
        return iter(
            [
                self._cells[cell][key]
                for cell in self._cells
                for key in self._cells[cell]
            ]
        )

    def __contains__(self, atom: Any) -> bool:
        """Check if a given element is in the cell view.

        Parameters
        ----------
        atom : Any
            The element to check.

        Returns
        -------
        bool
            Whether the element is in the cell view.
        """
        if not isinstance(atom, Cell | tuple | list):
            return False

        atom = Cell(atom)
        return any(atom.is_homotopic_to(x) for x in self._cells)

    def __repr__(self) -> str:
        """Return a string representation of the cell view.

        Returns
        -------
        str
            The __repr__ representation of the cell view.
        """
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"

    def __str__(self) -> str:
        """Return a string representation of the cell view.

        Returns
        -------
        str
            The __str__ representation of the cell view.
        """
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in self._cells[cell]]})"


class ColoredHyperEdgeView(AtomView):
    """A class for viewing the cells/hyperedges of a colored hypergraph.

    Provides methods for accessing, and retrieving
    information about the cells/hyperedges of a complex.

    Examples
    --------
    >>> hev = tnx.ColoredHyperEdgeView()
    """

    def __init__(self) -> None:
        self.hyperedge_dict = {}

    def __getitem__(self, atom: Any) -> dict[Hashable, Any]:
        """Return the user-defined attributes associated with the given hyperedge.

        Parameters
        ----------
        atom : Any
            The hyperedge for which to return the associated user-defined attributes.

        Returns
        -------
        dict[Hashable, Any]
            The user-defined attributes associated with the given atom.

        Raises
        ------
        KeyError
            If the hyperedge does not exist.
        """
        if isinstance(atom, Hashable) and not isinstance(atom, Collection):
            atom = (atom,)

        if len(atom) == 0:
            raise KeyError(f"Hyperedge {atom} is not in the complex.")
        if len(atom) == 2:
            if isinstance(atom, HyperEdge):
                hyperedge_elements = atom.elements
                key = 0
            elif isinstance(atom[0], Iterable) and isinstance(atom[1], int):
                hyperedge_elements_ = atom[0]
                if not isinstance(hyperedge_elements_, HyperEdge):
                    hyperedge_elements, key = atom
                else:
                    _, key = atom
                    hyperedge_elements = hyperedge_elements_.elements
            else:
                hyperedge_elements = atom
                key = 0
        else:
            hyperedge_elements = atom
            key = 0

        if not isinstance(hyperedge_elements, Iterable) or len(hyperedge_elements) == 0:
            raise KeyError(f"Hyperedge {atom} is not in the complex.")

        for i in self.allranks:
            if frozenset(hyperedge_elements) in self.hyperedge_dict[i]:
                return self.hyperedge_dict[i][frozenset(hyperedge_elements)][key]
        raise KeyError(f"Hyperedge {atom} is not in the complex.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Compute shape.

        Returns
        -------
        tuple[int, ...]
            The shape of the ColoredHyperEdge.
        """
        shape = []
        for i in self.allranks:
            sm = sum(len(self.hyperedge_dict[i][k]) for k in self.hyperedge_dict[i])
            shape.append(sm)
        return tuple(shape)

    def __len__(self) -> int:
        """Compute the number of nodes.

        Returns
        -------
        int
            The number of nodes in the ColoredHyperEdge.
        """
        return sum(self.shape[1:])

    def __iter__(self) -> Iterator:
        """Iterate over the hyperedges.

        Returns
        -------
        Iterator
            The iterator to iterate over the hyperedges.
        """
        lst = []
        for r in self.hyperedge_dict:
            if r == 0:
                continue

            lst.extend(
                (he, k)
                for he in self.hyperedge_dict[r]
                for k in self.hyperedge_dict[r][he]
            )
        return iter(lst)

    def __contains__(self, atom: Any) -> bool:
        """Check if hyperedge is in the hyperedges.

        Parameters
        ----------
        atom : Any
            The hyperedge to check.

        Returns
        -------
        bool
            Return `True` if the hyperedge is contained within the hyperedges.

        Notes
        -----
        Assumption of input here hyperedge = ( elements of hyperedge, key of hyperedge)
        """
        if isinstance(atom, Hashable) and not isinstance(atom, Collection):
            atom = (atom,)

        if len(atom) == 0:
            return False
        if len(atom) == 2:
            if isinstance(atom, HyperEdge):
                hyperedge_elements = atom.elements
                key = 0
            elif isinstance(atom[0], Iterable) and isinstance(atom[1], int):
                hyperedge_elements_ = atom[0]
                if not isinstance(hyperedge_elements_, HyperEdge):
                    hyperedge_elements, key = atom
                else:
                    _, key = atom
                    hyperedge_elements = hyperedge_elements_.elements
            else:
                hyperedge_elements = atom
                key = 0
        else:
            hyperedge_elements = atom
            key = 0

        if not isinstance(hyperedge_elements, Iterable) or len(hyperedge_elements) == 0:
            return False

        for i in self.allranks:
            if frozenset(hyperedge_elements) in self.hyperedge_dict[i]:
                return key in self.hyperedge_dict[i][frozenset(hyperedge_elements)]
        return False

    def __repr__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
            The __repr__ string representation of the hyperedges.
        """
        return f"ColoredHyperEdgeView({[(tuple(x[0]),x[1]) for x in self]})"

    def __str__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
            The __str__ string representation of the hyperedges.
        """
        return f"ColoredHyperEdgeView({[(tuple(x[0]),x[1]) for x in self]})"

    def skeleton(self, rank: int, store_hyperedge_key: bool = True):
        """Skeleton of the complex.

        Parameters
        ----------
        rank : int
            Rank of the skeleton.
        store_hyperedge_key : bool, default=True
            Whether to return the hyperedge key or not.

        Returns
        -------
        list of frozensets
            The skeleton of rank `rank`.
        """
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

        return sorted(
            [
                he
                for he in self.hyperedge_dict[rank]
                for k in self.hyperedge_dict[rank][he]
            ]
        )

    def get_rank(self, edge):
        """Get the rank of a given hyperedge.

        Parameters
        ----------
        edge : Iterable, Hashable or ColoredHyperEdge
            The edge for which to get the rank.

        Returns
        -------
        int
            The rank of the given colored hyperedge.
        """
        if isinstance(edge, HyperEdge):
            if len(edge) == 0:
                return 0

            for i in list(self.allranks):
                if frozenset(edge.elements) in self.hyperedge_dict[i]:
                    return i
            raise KeyError(f"hyperedge {edge.elements} is not in the complex")
        if isinstance(edge, str):
            if frozenset({edge}) in self.hyperedge_dict[0]:
                return 0
            raise KeyError(f"hyperedge {frozenset({edge})} is not in the complex")
        if isinstance(edge, Iterable):
            if len(edge) == 0:
                return 0

            for i in list(self.allranks):
                if frozenset(edge) in self.hyperedge_dict[i]:
                    return i
            raise KeyError(f"hyperedge {edge} is not in the complex")
        if isinstance(edge, Hashable) and not isinstance(edge, Iterable):
            if frozenset({edge}) in self.hyperedge_dict[0]:
                return 0
            raise KeyError(f"hyperedge {frozenset({edge})} is not in the complex")
        return None

    @property
    def allranks(self) -> list[int]:
        """All ranks.

        Returns
        -------
        list[int]
            The sorted list of all ranks.
        """
        return sorted(self.hyperedge_dict.keys())


class HyperEdgeView(AtomView):
    """A class for viewing the cells/hyperedges of a combinatorial complex.

    Provides methods for accessing, and retrieving
    information about the cells/hyperedges of a complex.

    Examples
    --------
    >>> hev = tnx.HyperEdgeView()
    """

    def __init__(self) -> None:
        self.hyperedge_dict = {}

    @staticmethod
    def _to_frozen_set(hyperedge):
        """Convert a hyperedge into a frozen set.

        Parameters
        ----------
        hyperedge : HyperEdge | Iterable | Hashable
            The hyperedge that is to be converted to a frozen set.

        Returns
        -------
        frozenset
            Returns a frozenset of the elements contained in the hyperedge.
        """
        if isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.elements
        elif isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset(hyperedge)
        elif isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset([hyperedge])
        return hyperedge_

    def __getitem__(self, hyperedge: Any) -> dict:
        """Get item.

        Parameters
        ----------
        hyperedge : Hashable or HyperEdge
            DESCRIPTION.

        Returns
        -------
        dict or list or dicts
            Return dict of attributes associated with that hyperedges.
        """
        hyperedge_ = HyperEdgeView._to_frozen_set(hyperedge)
        rank = self.get_rank(hyperedge_)
        return self.hyperedge_dict[rank][hyperedge_]

    @property
    def shape(self) -> tuple[int, ...]:
        """Compute shape.

        Returns
        -------
        tuple[int, ...]
            A tuple representing the shape of the hyperedge.
        """
        return tuple(len(self.hyperedge_dict[i]) for i in self.allranks)

    def __len__(self) -> int:
        """Compute the number of nodes.

        Returns
        -------
        int
            The number of nodes present in the HyperEdgeView.
        """
        return sum(self.shape)

    def __iter__(self) -> Iterator[HyperEdge]:
        """Iterate over the hyperedges.

        Returns
        -------
        Iterator
            Iterator object over the hyperedges.
        """
        return chain.from_iterable(self.hyperedge_dict.values())

    def __contains__(self, atom: Collection) -> bool:
        """Check if e is in the hyperedges.

        Parameters
        ----------
        atom : Collection
            The hyperedge that needs to be checked for containership
            in the HyperEdgeView.

        Returns
        -------
        bool
            Returns `True` if the hyperedge e is contained within the HyperEdgeView,
            else return `False`.
        """
        if len(self.hyperedge_dict) == 0:
            return False
        all_ranks = self.allranks
        if isinstance(atom, HyperEdge):
            if len(atom) == 0:
                return False

            for i in all_ranks:
                if frozenset(atom.elements) in self.hyperedge_dict[i]:
                    return True
            return False
        if isinstance(atom, Iterable):
            if len(atom) == 0:
                return False
            return any(frozenset(atom) in self.hyperedge_dict[i] for i in all_ranks)
        if isinstance(atom, Hashable):
            return frozenset({atom}) in self.hyperedge_dict[0]
        return None

    def __repr__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
            The __repr__ string representation of HyperEdgeView.
        """
        return f"HyperEdgeView({[tuple(x) for x in self]})"

    def __str__(self) -> str:
        """Return string representation of hyperedges.

        Returns
        -------
        str
            The __str__ string representation of HyperEdgeView.
        """
        return f"HyperEdgeView({[tuple(x) for x in self]})"

    def skeleton(
        self,
        rank: int,
        level: Literal[
            "equal",
            "upper",
            "up",
            "lower",
            "down",
            "uppereq",
            "upeq",
            "lowereq",
            "downeq",
        ] = "equal",
    ):
        """Skeleton of the complex.

        Parameters
        ----------
        rank : int
            Rank of the skeleton.
        level : str, default="equal"
            Level of the skeleton.

        Returns
        -------
        list of frozensets
            The skeleton of rank `rank`.
        """
        if level == "equal":
            if rank in self.allranks:
                return sorted(self.hyperedge_dict[rank].keys())
            return []
        if level in {"upper", "up"}:
            elements = []
            for rank_i in self.allranks:
                if rank_i > rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)
        if level in {"lower", "down"}:
            elements = []
            for rank_i in self.allranks:
                if rank_i < rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)
        if level in {"uppereq", "upeq"}:
            elements = []
            for rank_i in self.allranks:
                if rank_i >= rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)
        if level in {"lowereq", "downeq"}:
            elements = []
            for rank_i in self.allranks:
                if rank_i <= rank:
                    elements = elements + list(self.hyperedge_dict[rank_i].keys())
            return sorted(elements)
        raise ValueError(
            "level must be 'equal', 'uppereq', 'lowereq', 'upeq', 'downeq', 'uppereq', 'lower', 'up', or 'down'"
        )

    def get_rank(self, edge):
        """Get the rank of a hyperedge.

        Parameters
        ----------
        edge : Iterable, Hashable or HyperEdge
            The edge for which to get the rank.

        Returns
        -------
        int
            The rank of the given hyperedge.
        """
        if isinstance(edge, HyperEdge):
            if len(edge) == 0:
                return 0

            for i in list(self.allranks):
                if frozenset(edge.elements) in self.hyperedge_dict[i]:
                    return i
            raise KeyError(f"hyperedge {edge.elements} is not in the complex")
        if isinstance(edge, str):
            if frozenset({edge}) in self.hyperedge_dict[0]:
                return 0
            raise KeyError(f"hyperedge {frozenset({edge})} is not in the complex")
        if isinstance(edge, Iterable):
            if len(edge) == 0:
                return 0

            for i in list(self.allranks):
                if frozenset(edge) in self.hyperedge_dict[i]:
                    return i
            raise KeyError(f"hyperedge {edge} is not in the complex")
        if isinstance(edge, Hashable) and not isinstance(edge, Iterable):
            if frozenset({edge}) in self.hyperedge_dict[0]:
                return 0
            raise KeyError(f"hyperedge {frozenset({edge})} is not in the complex")
        return None

    @property
    def allranks(self):
        """All ranks.

        Returns
        -------
        list[hashable]
            The sorted list of all ranks.
        """
        return sorted(self.hyperedge_dict.keys())

    def _get_lower_rank(self, rank):
        """Get a lower rank compared to given rank.

        Parameters
        ----------
        rank : int
            The rank to be used to get a lower rank.

        Returns
        -------
        int
            A rank below the current rank available in the HyperEdgeView.
        """
        if len(self.allranks) == 0:
            return -1

        ranks = sorted(self.allranks)
        if rank <= min(ranks) or rank >= max(ranks):
            return -1
        return ranks[ranks.index(rank) - 1]

    def _get_higher_rank(self, rank):
        """Get a higher rank compared to given rank.

        Parameters
        ----------
        rank : int
            The rank to be used to get a higher rank.

        Returns
        -------
        int
            A rank above the current rank available in the HyperEdgeView.
        """
        if len(self.allranks) == 0:
            return -1
        ranks = sorted(self.allranks)
        if rank <= min(ranks) or rank >= max(ranks):
            return -1
        return ranks[ranks.index(rank) + 1]


class SimplexView(AtomView[Simplex]):
    """Simplex View class.

    The SimplexView class is used to provide a view/read only information
    into a subset of the nodes in a simplex.
    These classes are used in conjunction with the SimplicialComplex class
    for view/read only purposes for simplices in simplicial complexes.

    Parameters
    ----------
    simplex_trie : SimplexTrie
        A SimplexTrie instance containing the simplices in the simplex view.
    """

    def __init__(self, simplex_trie: SimplexTrie) -> None:
        self._simplex_trie = simplex_trie

    def __getitem__(self, simplex: Any) -> dict[Hashable, Any]:
        """Get the dictionary of attributes associated with the given simplex.

        Parameters
        ----------
        simplex : tuple, list or Simplex
            A tuple or list of nodes representing a simplex.

        Returns
        -------
        dict
            A dictionary of attributes associated with the given simplex.

        Raises
        ------
        KeyError
            If the simplex is not in the simplex view.
        """
        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex = (simplex,)
        else:
            simplex = tuple(sorted(simplex))

        node = self._simplex_trie.find(simplex)
        if node is None:
            raise KeyError(f"Simplex {simplex} is not in the simplex view.")
        return node.attributes

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the number of simplices in each dimension.

        Returns
        -------
        tuple of ints
            A tuple of integers representing the number of simplices in each dimension.
        """
        return tuple(self._simplex_trie.shape)

    def __len__(self) -> int:
        """Return the number of simplices in the SimplexView instance.

        Returns
        -------
        int
            Returns the number of simplices in the SimplexView instance.
        """
        return len(self._simplex_trie)

    def __iter__(self) -> Iterator:
        """Return an iterator over all simplices in the simplex view.

        Returns
        -------
        Iterator
            Returns an iterator over all simplices in the simplex view.
        """
        return iter(node.simplex for node in self._simplex_trie)

    def __contains__(self, atom: Any) -> bool:
        """Check if a simplex is in the simplex view.

        Parameters
        ----------
        atom : Any
            The simplex to be checked for membership in the simplex view.

        Returns
        -------
        bool
            True if the simplex is in the simplex view, False otherwise.

        Examples
        --------
        Check if a node is in the simplex view:

        >>> view = tnx.SimplexView()
        >>> view.faces_dict.append({frozenset({1}): {"weight": 1}})
        >>> view.max_dim = 0
        >>> 1 in view
        True
        >>> 2 in view
        False

        Check if a simplex is in the simplex view:

        >>> view.faces_dict.append({frozenset({1, 2}): {"weight": 1}})
        >>> view.max_dim = 1
        >>> {1, 2} in view
        True
        >>> {1, 3} in view
        False
        >>> {1, 2, 3} in view
        False
        """
        if isinstance(atom, Hashable) and not isinstance(atom, Iterable):
            atom = (atom,)
        else:
            atom = tuple(sorted(atom))
        return atom in self._simplex_trie

    def __repr__(self) -> str:
        """Return string representation that can be used to recreate it.

        Returns
        -------
        str
            Returns the __repr__ representation of the object.
        """
        return f"SimplexView({[tuple(simplex.elements) for simplex in self._simplex_trie]})"

    def __str__(self) -> str:
        """Return detailed string representation of the simplex view.

        Returns
        -------
        str
            Returns the __str__ representation of the object.
        """
        return f"SimplexView({[tuple(simplex.elements) for simplex in self._simplex_trie]})"


class NodeView:
    """Node view class.

    Parameters
    ----------
    nodes : dict[Hashable, Any]
        A dictionary of nodes with their attributes.
    cell_type : type
        The type of the cell.
    colored_nodes : bool, optional
        Whether or not the nodes are colored.
    """

    def __init__(
        self, nodes: dict[Hashable, Any], cell_type, colored_nodes: bool = False
    ) -> None:
        self.nodes = nodes

        if cell_type is None:
            raise ValueError("cell_type cannot be None")

        self.cell_type = cell_type
        self.colored_nodes = colored_nodes

    def __repr__(self) -> str:
        """Return string representation of nodes.

        Returns
        -------
        str
            Returns the __repr__ representation of the object.
        """
        return f"NodeView({list(map(tuple, self.nodes.keys()))})"

    def __iter__(self) -> Iterator:
        """Return an iterator over all nodes in the node view.

        Returns
        -------
        Iterator
            Returns an iterator over all nodes in the node view.
        """
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
            Dict of attributes associated with that cells.
        """
        if isinstance(cell, Iterable):
            cell = frozenset(cell)
            if cell in self.nodes:
                if self.colored_nodes:
                    return self.nodes[cell][0]
                return self.nodes[cell]
        elif isinstance(cell, Hashable):
            if cell in self:
                if self.colored_nodes:
                    return self.nodes[frozenset({cell})][0]
                return self.nodes[frozenset({cell})]

        raise KeyError(f"input {cell} is not in the node set of the complex")

    def __len__(self) -> int:
        """Compute the number of nodes.

        Returns
        -------
        int
            Returns the number of nodes.
        """
        return len(self.nodes)

    def __contains__(self, e) -> bool:
        """Check if e is in the nodes.

        Parameters
        ----------
        e : Hashable | Iterable
            The node to check for.

        Returns
        -------
        bool
            Return `True` if e is contained in NodeView, else Return `False`.
        """
        if isinstance(e, Hashable) and not isinstance(e, self.cell_type):
            return frozenset({e}) in self.nodes
        if isinstance(e, self.cell_type):
            return e.elements in self.nodes
        if isinstance(e, Iterable):
            if len(e) == 1:
                return frozenset(e) in self.nodes
            return None
        return False


class PathView(AtomView[Path]):
    """Path view class."""

    def __init__(self) -> None:
        self.max_dim = -1
        self.faces_dict = []

    def __getitem__(self, path: Any) -> dict:
        """Get the dictionary of attributes associated with the given path.

        Parameters
        ----------
        path : Any
            A tuple or list of nodes representing a path.
            It can also be a Path object.
            It can also be a single node represented by int or str.

        Returns
        -------
        dict or list or dict
            A dictionary of attributes associated with the given path.

        Raises
        ------
        KeyError
            If the path is not in this view.
        """
        if isinstance(path, Path):
            path = path.elements
        if isinstance(path, Hashable) and not isinstance(path, Iterable):
            path = (path,)

        path = tuple(path)
        if path in self.faces_dict[len(path) - 1]:
            return self.faces_dict[len(path) - 1][path]

        raise KeyError(f"input {path} is not in the path dictionary")

    def __iter__(self) -> Iterator:
        """Return an iterator over all paths in the path view.

        Returns
        -------
        Iterator
            Iterator over all paths in the paths view.
        """
        return chain.from_iterable(self.faces_dict)

    def __contains__(self, atom: Any) -> bool:
        """Check if a path is in the path view.

        Parameters
        ----------
        atom : Any
            The path to be checked for membership in the path view.

        Returns
        -------
        bool
            True if the path is in the path view, False otherwise.
        """
        if isinstance(atom, Sequence):
            atom = tuple(atom)
            if not 0 < len(atom) <= self.max_dim + 1:
                return False
            return atom in self.faces_dict[len(atom) - 1]
        if isinstance(atom, Path):
            atom = atom.elements
            if not 0 < len(atom) <= self.max_dim + 1:
                return False
            return atom in self.faces_dict[len(atom) - 1]
        if isinstance(atom, Hashable):
            return (atom,) in self.faces_dict[0]
        return False

    def __len__(self) -> int:
        """Return the number of simplices in the SimplexView instance.

        Returns
        -------
        int
            Returns the number of simplices in the SimplexView instance.
        """
        return sum(self.shape)

    def __repr__(self) -> str:
        """Return string representation that can be used to recreate it.

        Returns
        -------
        str
            Returns the __repr__ representation of the object.
        """
        all_paths: list[tuple[int | str, ...]] = []
        for i in range(len(self.faces_dict)):
            all_paths += [tuple(j) for j in self.faces_dict[i]]

        return f"PathView({all_paths})"

    def __str__(self) -> str:
        """Return detailed string representation of the path view.

        Returns
        -------
        str
            Returns the __str__ representation of the object.
        """
        all_paths: list[tuple[int | str, ...]] = []
        for i in range(len(self.faces_dict)):
            all_paths += [tuple(j) for j in self.faces_dict[i]]

        return f"PathView({all_paths})"

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the number of paths in each dimension.

        Returns
        -------
        tuple of ints
            A tuple of integers representing the number of paths in each dimension.
        """
        return tuple(len(self.faces_dict[i]) for i in range(len(self.faces_dict)))
