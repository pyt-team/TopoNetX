import numpy as np

try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Iterable, Hashable

from toponetx import TopoNetXError
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = ["AbstractCell", "AbstractCellView"]


class AbstractCell:
    """A AbstractCell class.
    Parameters
    ==========

    elements: any iterable of hashables.
    rank: int, rank of a cell, default is None.
    name: string, optional default is None.
    name : str

    Examples
        >>> ac1 = AbstractCell ( (1,2,3) )
        >>> ac2 = AbstractCell ( (1,2,4,5) )
        >>> ac3 = AbstractCell ( ("a","b","c") )
        >>> ac3 = AbstractCell ( ("a","b","c"), rank = 10)
    """

    def __init__(self, elements, rank=None, name=None, **attr):

        if name is None:
            self.name = ""
        else:
            self.name = name

        if isinstance(elements, Hashable) and not isinstance(elements, Iterable):
            elements = frozenset([elements])
        if elements is not None:
            for i in elements:
                if not isinstance(i, Hashable):
                    raise ValueError("every element AbstractCell must be hashable.")

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
         >>> CC = AbstractCellView()
         >>> CC.add_cell ( (1,2,3,4), rank =4 )
         >>> CC.add_cell ( (2,3,4,1) , rank =4 )
         >>> CC.add_cell ( (1,2,3,6), rank =4 )
         >>> CC.add_cell((1,2,3,4,5) , rank =5)
         >>> CC.add_cell((1,2,3,4,5,6) , rank =5)
         >>> CC.add_cell( AbstractCell( (0,1,2,3,4))  , rank =5)
         #>>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = ""
        else:
            self.name = name

        self.cell_dict = {}
        self._aux_complex = SimplicialComplex()  # used to track inserted elements

    @staticmethod
    def _to_frozen_set(cell):
        if isinstance(cell, Iterable):
            cell_ = frozenset(cell)

        elif isinstance(cell, AbstractCell):
            cell_ = cell.nodes
        elif isinstance(cell, Hashable) and not isinstance(cell, Iterable):
            cell_ = frozenset([cell])
        else:
            cell_ = frozenset(cell)
        return cell_

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
        cell_ = AbstractCellView._to_frozen_set(cell)
        rank = self.get_rank(cell_)
        return self.cell_dict[rank][cell_]

    def __setitem__(self, cell, item):

        cell_ = AbstractCellView._to_frozen_set(cell)
        rank = self.get_rank(cell_)
        self.cell_dict[rank][cell_] = item

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
        for i in sorted(list(self.allranks)):
            all_cells = all_cells + [tuple(k) for k in self.cell_dict[i].keys()]
        return iter(all_cells)

    def __contains__(self, e):

        if len(self.cell_dict) == 0:
            return False

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

                    if frozenset(e.nodes) in self.cell_dict[i]:
                        return True
                return False

        elif isinstance(e, Hashable):

            return frozenset({e}) in self.cell_dict[0]

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
        all_cells = []
        for i in sorted(list(self.allranks)):
            all_cells = all_cells + [tuple(k) for k in self.cell_dict[i].keys()]
        return f"AbstractCellView({all_cells}) "

    def __str__(self):
        """
        String representation of cells

        Returns
        -------
        str
        """
        all_cells = []

        for i in sorted(list(self.allranks)):
            all_cells = all_cells + [tuple(k) for k in self.cell_dict[i].keys()]

        return f"AbstractCellView({all_cells}) "

    def get_rank(self, e):
        """
        Parameters
        ----------
        e : Iterable, Hashable or AbstractCell

        Returns
        -------
        int, the rank of the cell e


        """

        if isinstance(e, Iterable):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.cell_dict[i]:
                        return i
                raise KeyError(f"cell {e} is not in the complex")

        elif isinstance(e, AbstractCell):

            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):

                    if frozenset(e.nodes) in self.cell_dict[i]:
                        return i
                raise KeyError(f"cell {e} is not in the complex")

        elif isinstance(e, Hashable):

            if e in self.cell_dict[0]:
                return 0
            else:
                raise KeyError(f"cell {e} is not in the complex")

    @property
    def allranks(self):
        return sorted(list(self.cell_dict.keys()))

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
        """
        Parameters
        ----------
        cell_ : frozenset of hashable elements
        rank : int
        attr : arbitrary attrs
        Returns
        -------
        None.

        """
        if rank in self.cell_dict:
            if cell_ in self.cell_dict[rank]:
                self.cell_dict[rank][cell_].update(attr)
                for i in cell_:
                    if 0 not in self.cell_dict:
                        self.cell_dict[0] = {}

                    if i not in self.cell_dict[0]:
                        self.cell_dict[0][frozenset({i})] = {"weight": 1}
            else:
                self.cell_dict[rank][cell_] = {}
                self.cell_dict[rank][cell_].update(attr)
                for i in cell_:
                    if 0 not in self.cell_dict:
                        self.cell_dict[0] = {}

                    if i not in self.cell_dict[0]:
                        self.cell_dict[0][frozenset({i})] = {"weight": 1}
        else:
            self.cell_dict[rank] = {}
            self.cell_dict[rank][cell_] = {}
            self.cell_dict[rank][cell_].update(attr)

            for i in cell_:
                if 0 not in self.cell_dict:
                    self.cell_dict[0] = {}
                if i not in self.cell_dict[0]:
                    self.cell_dict[0][frozenset({i})] = {"weight": 1}

    def add_node(self, node, **attr):
        self.add_cell(node, 0, **attr)

    def add_cell(self, cell, rank, **attr):

        if not isinstance(rank, int):
            raise ValueError(f"rank must be an integer, got {type(rank)}")

        if rank < 0:
            raise ValueError(f"rank must be non-negative integer, got {rank}")

        if isinstance(cell, Hashable) and not isinstance(cell, Iterable):
            if rank != 0:
                raise ValueError(f"rank must be zero for hashables, got rank {rank}")
            else:
                self.cell_dict[0] = {}
                self.cell_dict[0][frozenset({cell})] = {}
                self.cell_dict[0][frozenset({cell})].update(attr)
                self._aux_complex.add_simplex(Simplex(frozenset({cell}), r=0))
                self.cell_dict[0][frozenset({cell})]["weight"] = 1
                return
        if isinstance(cell, Iterable) or isinstance(cell, AbstractCell):
            if not isinstance(cell, AbstractCell):
                cell_ = frozenset(sorted(cell))  # put the simplex in cananical order
                if len(cell_) != len(cell):
                    raise ValueError(
                        f"a cell cannot contain duplicate nodes,got {cell_}"
                    )
            else:
                cell_ = cell.nodes
        if isinstance(cell, Iterable) or isinstance(cell, AbstractCell):
            for i in cell_:
                if not isinstance(i, Hashable):
                    raise ValueError(
                        "every element cell must be hashable, input cell is {cell_}"
                    )
        if rank == 0:
            raise ValueError(
                "rank must be positive for higher order cells, got {rank} "
            )

        self._aux_complex.add_simplex(Simplex(cell_, r=rank))
        if self._aux_complex.is_maximal(cell_):  # safe to insert the cell
            # looking down from cell to other cells in the complex
            # make sure all subsets of cell have lower ranks
            all_subsets = self._aux_complex.get_boundaries([cell_], min_dim=1)
            for f in all_subsets:
                if frozenset(f) == frozenset(cell_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] > rank:
                        rr = self._aux_complex[f]["r"]
                        self._aux_complex.remove_maximal_simplex(cell_)
                        raise ValueError(
                            "a violation of the combinatorial complex condition:"
                            + f"the cell {f} in the complex has rank {rr} is larger than {rank}, the rank of the input cell {cell_} "
                        )

            self._add_cell(cell_, rank, **attr)
            self.cell_dict[rank][cell_]["weight"] = 1
            if isinstance(cell, AbstractCell):
                self.cell_dict[rank][cell_].update(cell.properties)

        else:
            all_cofaces = self._aux_complex.get_cofaces(cell_, 0)
            # looking up from cell to other cells in the complex
            # make sure all supersets that are in the complex of cell have higher ranks

            for f in all_cofaces:
                if frozenset(f) == frozenset(cell_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] < rank:
                        rr = self._aux_complex[f]["r"]
                        # all supersets in a CC must have ranks that is larger than or equal to input ranked cell
                        raise ValueError(
                            "violation of the combintorial complex condition : "
                            + f"the cell {f} in the complex has rank {rr} is smaller than {rank}, the rank of the input cell {cell_} "
                        )
            self._aux_complex[cell_]["r"] = rank
            self._add_cell(cell_, rank, **attr)
            self.cell_dict[rank][cell_]["weight"] = 1
            if isinstance(cell, AbstractCell):
                self.cell_dict[rank][cell_].update(cell.properties)

    def remove_cell(self, cell):

        if cell not in self:
            raise KeyError(f"The cell {cell} is not in the complex")

        if isinstance(cell, Hashable) and not isinstance(cell, Iterable):
            del self.cell_dict[0][cell]

        if isinstance(cell, AbstractCell):
            cell_ = cell.nodes
        else:
            cell_ = frozenset(cell)
        rank = self.get_rank(cell_)
        del self.cell_dict[rank][cell_]

        return

    def remove_node(self, node):
        self.remove_cell(node)

    def _incidence_matrix_helper(self, children, uidset, sparse=True, index=False):

        """

        Parameters
        ----------

        Returns
        -------

        Notes
        -----


        """

        if sparse:
            from scipy.sparse import csr_matrix

        ndict = dict(zip(children, range(len(children))))
        edict = dict(zip(uidset, range(len(uidset))))

        r_cell_dict = {j: children[j] for j in range(len(children))}
        k_cell_dict = {i: uidset[i] for i in range(len(uidset))}

        if len(ndict) != 0:

            if index:
                rowdict = {v: k for k, v in ndict.items()}
                coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for n in ndict:
                    for e in edict:
                        if n <= e:
                            data.append(1)
                            rows.append(ndict[n])
                            cols.append(edict[e])
                MP = csr_matrix(
                    (data, (rows, cols)), shape=(len(r_cell_dict), len(k_cell_dict))
                )
            else:
                # Create an np.matrix
                MP = np.zeros((len(children), len(uidset)), dtype=int)
                for e in k_cell_dict:
                    for n in r_cell_dict:
                        if r_cell_dict[n] <= k_cell_dict[e]:
                            MP[ndict[n], edict[e]] = 1
            if index:
                return MP, rowdict, coldict
            else:
                return MP
        else:
            if index:
                return np.zeros(1), {}, {}
            else:
                return np.zeros(1)

    def incidence_matrix(
        self, r, k, incidence_type="up", weight=None, sparse=True, index=False
    ):
        """
        An incidence matrix indexed by r-ranked cells k-ranked cells
        r !=k, when k is None incidence_type will be considered instead

        Parameters
        ----------

        incidence_type : str, optional, default 'up', other options 'down'

        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying row with item in entityset's children

        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----


        """
        weight = None  # weight is not supported in this version
        assert r != k  # r and k must be different
        if k is None:
            if incidence_type == "up":
                children = self.skeleton(r)
                uidset = self.skeleton(r + 1, level="upper")
            elif incidence_type == "down":
                uidset = self.skeleton(r)
                children = self.skeleton(r - 1, level="lower")
            else:
                raise TopoNetXError("incidence_type must be 'up' or 'down' ")
        else:
            assert (
                r != k
            )  # incidence is defined between two skeletons of different ranks
            if (
                r < k
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(r)
                uidset = self.skeleton(k)

            elif (
                r > k
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(k)
                uidset = self.skeleton(r)
        return self._incidence_matrix_helper(children, uidset, sparse, index)

    def adjacency_matrix(self, r, k, s=1, weights=False, index=False):
        """
        A adjacency matrix for the RankedEntitySet of the r-ranked considering thier adjacency with respect to k-ranked entities
        r < k

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of uid : row number

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying row with item in entityset's children

        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----



        """

        if k is not None:
            assert r < k  # rank k must be smaller than rank r

        if index:

            MP, row, col = self.incidence_matrix(
                r, k, incidence_type="up", sparse=True, index=index
            )
        else:
            MP = self.incidence_matrix(
                r, k, incidence_type="up", sparse=True, index=index
            )

        weights = False  ## currently weighting is not supported
        if weights == False:
            A = MP.dot(MP.transpose())
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, row
        else:
            return A

    def coadjacency_matrix(self, r, k, s=1, weights=False, index=False):
        """
        A coadjacency matrix for the RankedEntitySet of the r-ranked considering thier adjacency with respect to k-ranked entities
        r > k

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        Returns
        -------
        coadjacency_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        column dictionary of the associted incidence matrix : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----




        """
        # TODO : None case
        assert r < k  # rank k must be larger than rank r
        if index:

            MP, row, col = self.incidence_matrix(
                r, k, incidence_type="down", sparse=True, index=index
            )
        else:
            MP = self.incidence_matrix(
                k, r, incidence_type="down", sparse=True, index=index
            )
        weights = False  ## currently weighting is not supported
        if weights == False:
            A = MP.T.dot(MP)
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, col
        else:
            return A

    def _restrict_to(self, element_subset, name=None):
        """

        Parameters
        ----------
        element_subset : iterable


        name: hashable, optional
        Returns
        -------

        """

        pass
