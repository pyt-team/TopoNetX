try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Iterable, Hashable

from itertools import combinations

import numpy as np
from simplicial_complex import SimplicialComplex

from toponetx import TopoNetXError
from toponetx.classes.simplex import NodeView, Simplex, SimplexView

__all__ = ["AbstractCell", "AbstractCellView"]


class AbstractCell:
    """A AbstractCell class.
    Parameters
    ==========

    elements: any iterable of hashables.

    name : str

    Examples
        >>> ac1 = AbstractCell ( (1,2,3) )
        >>> ac2 = AbstractCell ( (1,2,4,5) )
        >>> ac3 = AbstractCell ( ("a","b","c") )
    """

    def __init__(self, elements, rank=None, name=None, **attr):

        if name is None:
            self.name = ""
        else:
            self.name = name

        if isinstance(elements, Hashable) and not isinstance(elements, Iterable):
            elements = frozenset([elements])
        self.nodes = frozenset(list(elements))
        self._rank = rank

        if len(self.nodes) != len(elements):
            raise ValueError("a ranked entity cannot contain duplicate nodes")

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
        String representation of AbstractCell
        Returns
        -------
        str
        """
        return f"AbstractCell{tuple(self.nodes)}"

    def __str__(self):
        """
        String representation of a AbstractCell
        Returns
        -------
        str
        """
        return f"Nodes set:{tuple(self.nodes)}, attrs:{self.properties}"

    @property
    def rank(self):
        if self._rank is not None:
            return self._rank
        else:
            print("cell has none rank")
            return None


class AbstractCellView:
    """A AbstractCellView class for cells of a combintorial_complex
    Parameters
    ----------
    name : str

    Examples
    --------
         >>> SC = AbstractCellView()
         >>> SC.add_cell ( (1,2,3,4), rank =4 )
         >>> SC.add_cell ( (2,3,4,1) , rank =4 )
         >>> SC.add_cell ( (1,2,3,6), rank =4 )
         >>> SC.add_cell((1,2,3,4,5) , rank =5)
         >>> SC.add_cell((1,2,3,4,5,6) , rank =5)
         #>>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        self.all_ranks = set()

        self.cell_dict = {}
        self._aux_complex = SimplicialComplex()  # used to track inserted elements

    def __getitem__(self, cell):
        """
        Parameters
        ----------
        cell : tuple list or Simplex
            DESCRIPTION.
        Returns
        -------
        TYPE : dict or ilst or dicts
            return dict of properties associated with that cells

        """
        pass

    def __setitem__(self, cell, rank, **attr):
        pass

    # Set methods
    @property
    def shape(self):
        if len(self.cell_dict) == 0:
            print("Complex is empty.")
        else:
            return [len(self.cell_dict[i]) for i in self.allranks]

    def __len__(self):
        if len(self.cell_dict) == 0:
            return 0
        else:
            return np.sum(self.shape)

    def __iter__(self):
        all_cells = []
        for i in list(self.allranks):
            all_cells = all_cells + list(self.cell_dict[i].keys())
        return iter(all_cells)

    def __contains__(self, e):

        if isinstance(e, Iterable):

            if len(e) == 0:
                return False
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.cell_dict[i]:
                        return True
                return False

        elif isinstance(e, AbstractCell):

            if len(e) == 0:
                return False
            else:
                for i in list(self.allranks):

                    if frozenset(e) in self.cell_dict[i]:
                        return True
                return False

        elif isinstance(e, Hashable):

            return e in self.cell_dict[0]

        else:
            return False

    def skeleton(self, k, name=None, level=None):

        if name is None and level is None:
            name = "X" + str(k)
        elif name is None and level == "equal":
            name = "X" + str(k)
        elif name is None and level == "upper":
            name = "X>=" + str(k)
        elif name is None and level == "up":
            name = "X>=" + str(k)
        elif name is None and level == "lower":
            name = "X<=" + str(k)
        elif name is None and level == "down":
            name = "X<=" + str(k)
        else:
            assert isinstance(name, str)
        if level is None or level == "equal":
            elements = []
            if k in self.allranks:

                return list(self.cell_dict[k].keys())
            else:
                return []

        elif level == "upper" or level == "up":
            elements = []
            for l in self.allranks:
                if l >= k:

                    elements = elements + list(self.cell_dict[l].keys())
            return elements

        elif level == "lower" or level == "down":
            elements = []
            for l in self.allranks:
                if l <= k:
                    elements = elements + list(self.cell_dict[l].keys())
            return elements
        else:
            raise TopoNetXError(
                "level must be None, equal, 'upper', 'lower', 'up', or 'down' "
            )

    def __repr__(self):
        """C
        String representation of cells
        Returns
        -------
        str
        """
        pass

        # return f"AbstractCellView({all_simplices})"

    def __str__(self):
        """
        String representation of cells

        Returns
        -------
        str
        """
        pass

    def get_rank(self, e):

        if isinstance(e, Iterable):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.cell_dict[i]:
                        return i
                raise KeyError("cell is not in the complex")

        elif isinstance(e, AbstractCell):

            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):

                    if frozenset(e.nodes) in self.cell_dict[i]:
                        return i
                raise KeyError("cell is not in the complex")

        elif isinstance(e, Hashable):

            if e in self.cell_dict[0]:
                return 0
            else:
                raise KeyError("cell is not in the complex")

    @property
    def allranks(self):
        return list(self.cell_dict.keys())

    def add_cells_from(self, cells):
        if isinstance(cells, Iterable):
            for s in cells:
                self.cells(s)
        else:
            raise ValueError("input cells must be an iterable of AbstractCell objects")

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

    def _add_cell(self, cell_, rank, **attr):
        if rank in self.cell_dict:
            if cell_ in self.cell_dict[rank]:
                self.cell_dict[rank][cell_].update(attr)
                for i in cell_:
                    if 0 not in self.cell_dict:
                        self.cell_dict[0] = {}

                    if i not in self.cell_dict[0]:
                        self.cell_dict[0][i] = {"weight": 1}

                return
            else:
                self.cell_dict[rank][cell_] = {}
                self.cell_dict[rank][cell_].update(attr)
        else:
            self.cell_dict[rank] = {}
            self.cell_dict[rank][cell_] = {}
            self.cell_dict[rank][cell_].update(attr)

            for i in cell_:
                if 0 not in self.cell_dict:
                    self.cell_dict[0] = {}

                if i not in self.cell_dict[0]:
                    self.cell_dict[0][i] = {"weight": 1}

    def add_cell(self, cell, rank, **attr):
        if isinstance(cell, Hashable) and not isinstance(cell, Iterable):
            if rank != 0:
                ValueError("rank must be zero for hashables")
            else:
                self.cell_dict[0][frozenset({cell})] = {}
                self.cell_dict[0][frozenset({cell})].update(attr)

        if isinstance(cell, Iterable) or isinstance(cell, AbstractCell):
            if not isinstance(cell, AbstractCell):
                cell_ = frozenset(sorted(cell))  # put the simplex in cananical order
                if len(cell_) != len(cell):
                    raise ValueError("a cell cannot contain duplicate nodes")
            else:

                cell_ = cell.nodes

        self._aux_complex.add_simplex(Simplex(cell_, r=rank))
        if self._aux_complex.is_maximal(cell_):  # safe to insert the cell
            # print(cell_)
            all_cofaces = self._aux_complex.get_boundaries([cell_], min_dim=1)
            for f in all_cofaces:
                if frozenset(f) == frozenset(cell_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] > rank:
                        self._aux_complex.remove_maximal_simplex(cell_)
                        raise ValueError(
                            "violation of the combintorial complex condition"
                        )
            self._add_cell(cell_, rank, **attr)
            self.cell_dict[rank][cell_]["weight"] = 1

        else:
            all_cofaces = self._aux_complex.get_cofaces(cell_, 0)
            # print(all_cofaces)
            for f in all_cofaces:
                if frozenset(f) == frozenset(cell_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] < rank:
                        raise ValueError(
                            "violation of the combintorial complex condition"
                        )
            self._aux_complex[cell_]["r"] = rank
            self._add_cell(cell_, rank, **attr)
            self.cell_dict[rank][cell_]["weight"] = 1

    def remove_cell(self, cell):

        if isinstance(cell, Hashable) and not isinstance(cell, Iterable):
            del self.cell_dict[0][cell]

        if isinstance(cell, AbstractCell):
            cell_ = cell.nodes
        else:
            cell_ = frozenset(cell)
        rank = self.get_rank(cell_)
        del self.cell_dict[rank][cell_]

        return
