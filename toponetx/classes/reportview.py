import numpy as np

try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Iterable, Hashable

from itertools import combinations

from toponetx import TopoNetXError
from toponetx.classes.cell import Cell
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.simplex import Simplex

__all__ = ["HyperEdgeView", "CellView", "SimplexView"]


class CellView:
    r"""A CellView class for cells of a CellComplex
    Parameters
    ----------
    name : str
        The name of the cell view.
    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        # Initialize a dictionary to hold cells, with keys being the tuple
        # that defines the cell, and values being dictionaries of cell objects
        # with different properties
        self._cells = dict()

    def __getitem__(self, cell):
        r"""Return the properties of a given cell.

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
                raise KeyError(f"cell {cell} is not in the cell dictionary")

            # If there is only one cell with these elements, return its properties
            elif len(self._cells[cell.elements]) == 1:
                k = next(iter(self._cells[cell.elements].keys()))
                return self._cells[cell.elements][k].properties

            # If there are multiple cells with these elements, return the properties of all cells
            else:
                return [
                    self._cells[cell.elements][c].properties
                    for c in self._cells[cell.elements]
                ]

        # If a tuple or list is passed in, assume it represents a cell
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
            raise KeyError("input must be a tuple, list or a cell")

    def __len__(self):
        r"""Return the number of cells in the cell view."""
        if len(self._cells) == 0:
            return 0
        else:
            return np.sum([len(self._cells[cell]) for cell in self._cells])

    def __iter__(self):
        r"""Iterate over all cells in the cell view."""
        return iter(
            [
                self._cells[cell][key]
                for cell in self._cells
                for key in self._cells[cell]
            ]
        )

    def __contains__(self, e):
        r"""Check if a given element is in the cell view.

        Parameters
        ----------
        e : tuple or Cell
            The element to check.

        Returns
        -------
        bool
            Whether or not the element is in the cell view.

        """
        if isinstance(e, tuple) or isinstance(e, list):
            return e in self._cells

        elif isinstance(e, Cell):
            return e.elements in self._cells
        else:
            return False

    def __repr__(self):
        r"""Return a string representation of the cell view."""
        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"

    def __str__(self):
        r"""Return a string representation of the cell view."""

        return f"CellView({[self._cells[cell][key] for cell in self._cells for key in  self._cells[cell]] })"


class HyperEdgeView:
    """A class for viewing the hyperedges of a combinatorial complex.

    Provides methods for accessing, manipulating, and retrieving
    information about the hyperedges of a complex.

    Parameters
    ----------
    name : str
        The name of the view.

    Examples
    --------
         >>> CC = HyperEdgeView()
         >>> CC.add_hyperedge ( (1,2,3,4), rank =4 )
         >>> CC.add_hyperedge ( (2,3,4,1) , rank =4 )
         >>> CC.add_hyperedge ( (1,2,3,6), rank =4 )
         >>> CC.add_hyperedge((1,2,3,4,5) , rank =5)
         >>> CC.add_hyperedge((1,2,3,4,5,6) , rank =5)
         >>> CC.add_hyperedge( HyperEdge( (0,1,2,3,4))  , rank =5)
         #>>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = ""
        else:
            self.name = name

        self.hyperedge_dict = {}

        from toponetx.classes.simplicial_complex import SimplicialComplex

        self._aux_complex = SimplicialComplex()  # used to track inserted elements

    @staticmethod
    def _to_frozen_set(hyperedge):
        if isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset(hyperedge)

        elif isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.nodes
        elif isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            hyperedge_ = frozenset([hyperedge])
        else:
            hyperedge_ = frozenset(hyperedge)
        return hyperedge_

    def __getitem__(self, hyperedge):
        """
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

    def __setitem__(self, hyperedge, item):

        hyperedge_ = HyperEdgeView._to_frozen_set(hyperedge)
        rank = self.get_rank(hyperedge_)
        self.hyperedge_dict[rank][hyperedge_] = item

    # Set methods
    @property
    def shape(self):
        if len(self.hyperedge_dict) == 0:
            print("Complex is empty.")
        else:
            return [len(self.hyperedge_dict[i]) for i in self.allranks]

    def __len__(self):
        if len(self.hyperedge_dict) == 0:
            return 0
        else:
            return np.sum(self.shape)

    def __iter__(self):
        all_hyperedges = []
        for i in sorted(list(self.allranks)):
            all_hyperedges = all_hyperedges + [
                tuple(k) for k in self.hyperedge_dict[i].keys()
            ]
        return iter(all_hyperedges)

    def __contains__(self, e):

        if len(self.hyperedge_dict) == 0:
            return False

        if isinstance(e, Iterable):
            if len(e) == 0:
                return False
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.hyperedge_dict[i]:
                        return True
                return False

        elif isinstance(e, HyperEdge):

            if len(e) == 0:
                return False
            else:
                for i in list(self.allranks):

                    if frozenset(e.nodes) in self.hyperedge_dict[i]:
                        return True
                return False

        elif isinstance(e, Hashable):

            return frozenset({e}) in self.hyperedge_dict[0]

        else:

            return False

    def __repr__(self):
        """C
        String representation of hyperedges
        Returns
        -------
        str
        """
        all_hyperedges = []
        for i in sorted(list(self.allranks)):
            all_hyperedges = all_hyperedges + [
                tuple(k) for k in self.hyperedge_dict[i].keys()
            ]
        return f"CellView({all_hyperedges}) "

    def __str__(self):
        """
        String representation of hyperedges

        Returns
        -------
        str
        """
        all_hyperedges = []

        for i in sorted(list(self.allranks)):
            all_hyperedges = all_hyperedges + [
                tuple(k) for k in self.hyperedge_dict[i].keys()
            ]

        return f"HyperEdgeView({all_hyperedges}) "

    def skeleton(self, rank, name=None, level=None):

        if name is None and level is None:
            name = "X" + str(rank)
        elif name is None and level == "equal":
            name = "X" + str(rank)
        elif name is None and level == "upper":
            name = "X>=" + str(rank)
        elif name is None and level == "up":
            name = "X>=" + str(rank)
        elif name is None and level == "lower":
            name = "X<=" + str(rank)
        elif name is None and level == "down":
            name = "X<=" + str(rank)
        else:
            assert isinstance(name, str)
        if level is None or level == "equal":
            elements = []
            if rank in self.allranks:

                return sorted(list(self.hyperedge_dict[rank].keys()))
            else:
                return []

        elif level == "upper" or level == "up":
            elements = []
            for rank in self.allranks:
                if rank >= rank:

                    elements = elements + list(self.hyperedge_dict[rank].keys())
            return sorted(elements)

        elif level == "lower" or level == "down":
            elements = []
            for rank in self.allranks:
                if rank <= rank:
                    elements = elements + list(self.hyperedge_dict[rank].keys())
            return sorted(elements)
        else:
            raise TopoNetXError(
                "level must be None, equal, 'upper', 'lower', 'up', or 'down' "
            )

    def get_rank(self, e):
        """
        Parameters
        ----------
        e : Iterable, Hashable or HyperEdge

        Returns
        -------
        int, the rank of the hyperedge e


        """

        if isinstance(e, Iterable):
            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):
                    if frozenset(e) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e} is not in the complex")

        elif isinstance(e, HyperEdge):

            if len(e) == 0:
                return 0
            else:
                for i in list(self.allranks):

                    if frozenset(e.nodes) in self.hyperedge_dict[i]:
                        return i
                raise KeyError(f"hyperedge {e} is not in the complex")

        elif isinstance(e, Hashable) and not isinstance(e, Iterable):

            if e in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {e} is not in the complex")

        elif isinstance(e, str):

            if e in self.hyperedge_dict[0]:
                return 0
            else:
                raise KeyError(f"hyperedge {e} is not in the complex")

    @property
    def allranks(self):
        return sorted(list(self.hyperedge_dict.keys()))

    def add_hyperedges_from(self, hyperedges):
        if isinstance(hyperedges, Iterable):
            for s in hyperedges:
                self.hyperedges(s)
        else:
            raise ValueError(
                "input hyperedges must be an iterable of HyperEdge objects"
            )

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

    def _add_hyperedge(self, hyperedge_, rank, **attr):
        """
        Parameters
        ----------
        hyperedge_ : frozenset of hashable elements
        rank : int
        attr : arbitrary attrs
        Returns
        -------
        None.

        """
        if rank in self.hyperedge_dict:
            if hyperedge_ in self.hyperedge_dict[rank]:
                self.hyperedge_dict[rank][hyperedge_].update(attr)
                for i in hyperedge_:
                    if 0 not in self.hyperedge_dict:
                        self.hyperedge_dict[0] = {}

                    if i not in self.hyperedge_dict[0]:
                        self.hyperedge_dict[0][frozenset({i})] = {"weight": 1}
            else:
                self.hyperedge_dict[rank][hyperedge_] = {}
                self.hyperedge_dict[rank][hyperedge_].update(attr)
                for i in hyperedge_:
                    if 0 not in self.hyperedge_dict:
                        self.hyperedge_dict[0] = {}

                    if i not in self.hyperedge_dict[0]:
                        self.hyperedge_dict[0][frozenset({i})] = {"weight": 1}
        else:
            self.hyperedge_dict[rank] = {}
            self.hyperedge_dict[rank][hyperedge_] = {}
            self.hyperedge_dict[rank][hyperedge_].update(attr)

            for i in hyperedge_:
                if 0 not in self.hyperedge_dict:
                    self.hyperedge_dict[0] = {}
                if i not in self.hyperedge_dict[0]:
                    self.hyperedge_dict[0][frozenset({i})] = {"weight": 1}

    def add_node(self, node, **attr):
        self.add_hyperedge(hyperedge=node, rank=0, **attr)

    def add_hyperedge(self, hyperedge, rank, **attr):
        """
        Parameters
        ----------
        hyperedge : HyperEdge, Hashable or Iterable
            a hyperedge in a combinatorial complex
        rank : int
            the rank of a hyperedge, must be zero when the hyperedge is Hashable.
        **attr : attr associated with hyperedge


        Returns
        -------
        None.

        Note
        -----
         The add_hyperedge is a method for adding hyperedges to the HyperEdgeView instance.
         It takes two arguments: hyperedge and rank, where hyperedge is a tuple or HyperEdge instance
         representing the hyperedge to be added, and rank is an integer representing the rank of the hyperedge.
         The add_hyperedge method then adds the hyperedge to the hyperedge_dict attribute of the HyperEdgeView
         instance, using the hyperedge's rank as the key and the hyperedge itself as the value.
         This allows the hyperedge to be accessed later using its rank.

        Note that the add_hyperedge method also appears to check whether the hyperedge being added
        is a valid hyperedge of the combinatorial complex by checking whether the hyperedge's nodes
        are contained in the _aux_complex attribute of the HyperEdgeView instance.
        If the hyperedge's nodes are not contained in _aux_complex, then the add_hyperedge method will
        not add the hyperedge to hyperedge_dict. This is done to ensure that the HyperEdgeView
        instance only contains valid hyperedges of the complex.

        """

        if not isinstance(rank, int):
            raise ValueError(f"rank must be an integer, got {rank}")

        if rank < 0:
            raise ValueError(f"rank must be non-negative integer, got {rank}")

        if isinstance(hyperedge, str):
            if rank != 0:
                raise ValueError(f"rank must be zero for string input, got rank {rank}")
            else:
                if 0 not in self.hyperedge_dict:
                    self.hyperedge_dict[0] = {}
                self.hyperedge_dict[0][frozenset({hyperedge})] = {}
                self.hyperedge_dict[0][frozenset({hyperedge})].update(attr)
                self._aux_complex.add_simplex(Simplex(frozenset({hyperedge}), r=0))
                self.hyperedge_dict[0][frozenset({hyperedge})]["weight"] = 1
                return

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            if rank != 0:
                raise ValueError(f"rank must be zero for hashables, got rank {rank}")
            else:
                if 0 not in self.hyperedge_dict:
                    self.hyperedge_dict[0] = {}
                self.hyperedge_dict[0][frozenset({hyperedge})] = {}
                self.hyperedge_dict[0][frozenset({hyperedge})].update(attr)
                self._aux_complex.add_simplex(Simplex(frozenset({hyperedge}), r=0))
                self.hyperedge_dict[0][frozenset({hyperedge})]["weight"] = 1
                return
        if isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            if not isinstance(hyperedge, HyperEdge):
                hyperedge_ = frozenset(
                    sorted(hyperedge)
                )  # put the simplex in cananical order
                if len(hyperedge_) != len(hyperedge):
                    raise ValueError(
                        f"a hyperedge cannot contain duplicate nodes,got {hyperedge_}"
                    )
            else:
                hyperedge_ = hyperedge.nodes
        if isinstance(hyperedge, Iterable) or isinstance(hyperedge, HyperEdge):
            for i in hyperedge_:
                if not isinstance(i, Hashable):
                    raise ValueError(
                        "every element hyperedge must be hashable, input hyperedge is {hyperedge_}"
                    )
        if (
            rank == 0
            and isinstance(hyperedge, Iterable)
            and not isinstance(hyperedge, str)
        ):
            if len(hyperedge) > 1:
                raise ValueError(
                    "rank must be positive for higher order hyperedges, got rank = 0 "
                )

        self._aux_complex.add_simplex(Simplex(hyperedge_, r=rank))
        if self._aux_complex.is_maximal(hyperedge_):  # safe to insert the hyperedge
            # looking down from hyperedge to other hyperedges in the complex
            # make sure all subsets of hyperedge have lower ranks
            all_subsets = self._aux_complex.get_boundaries([hyperedge_], min_dim=1)
            for f in all_subsets:
                if frozenset(f) == frozenset(hyperedge_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] > rank:
                        rr = self._aux_complex[f]["r"]
                        self._aux_complex.remove_maximal_simplex(hyperedge_)
                        raise ValueError(
                            "a violation of the combinatorial complex condition:"
                            + f"the hyperedge {f} in the complex has rank {rr} is larger than {rank}, the rank of the input hyperedge {hyperedge_} "
                        )

            self._add_hyperedge(hyperedge_, rank, **attr)
            self.hyperedge_dict[rank][hyperedge_]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self.hyperedge_dict[rank][hyperedge_].update(hyperedge.properties)

        else:
            all_cofaces = self._aux_complex.get_cofaces(hyperedge_, 0)
            # looking up from hyperedge to other hyperedges in the complex
            # make sure all supersets that are in the complex of hyperedge have higher ranks

            for f in all_cofaces:
                if frozenset(f) == frozenset(hyperedge_):
                    continue
                if "r" in self._aux_complex[f]:  # f is part of the CC
                    if self._aux_complex[f]["r"] < rank:
                        rr = self._aux_complex[f]["r"]
                        # all supersets in a CC must have ranks that is larger than or equal to input ranked hyperedge
                        raise ValueError(
                            "violation of the combintorial complex condition : "
                            + f"the hyperedge {f} in the complex has rank {rr} is smaller than {rank}, the rank of the input hyperedge {hyperedge_} "
                        )
            self._aux_complex[hyperedge_]["r"] = rank
            self._add_hyperedge(hyperedge_, rank, **attr)
            self.hyperedge_dict[rank][hyperedge_]["weight"] = 1
            if isinstance(hyperedge, HyperEdge):
                self.hyperedge_dict[rank][hyperedge_].update(hyperedge.properties)

    def remove_hyperedge(self, hyperedge):

        if hyperedge not in self:
            raise KeyError(f"The hyperedge {hyperedge} is not in the complex")

        if isinstance(hyperedge, Hashable) and not isinstance(hyperedge, Iterable):
            del self.hyperedge_dict[0][hyperedge]

        if isinstance(hyperedge, HyperEdge):
            hyperedge_ = hyperedge.nodes
        else:
            hyperedge_ = frozenset(hyperedge)
        rank = self.get_rank(hyperedge_)
        del self.hyperedge_dict[rank][hyperedge_]

        return

    def remove_node(self, node):
        self.remove_hyperedge(node)

    def _incidence_matrix_helper(self, children, uidset, sparse=True, index=False):
        """
        Parameters
        ----------
        Returns
        -------
        Notes
        -----
        """
        from collections import OrderedDict
        from operator import itemgetter

        if sparse:
            from scipy.sparse import csr_matrix

        ndict = dict(zip(children, range(len(children))))
        edict = dict(zip(uidset, range(len(uidset))))

        ndict = OrderedDict(sorted(ndict.items(), key=itemgetter(1)))
        edict = OrderedDict(sorted(edict.items(), key=itemgetter(1)))

        r_hyperedge_dict = {j: children[j] for j in range(len(children))}
        k_hyperedge_dict = {i: uidset[i] for i in range(len(uidset))}

        r_hyperedge_dict = OrderedDict(
            sorted(r_hyperedge_dict.items(), key=itemgetter(0))
        )
        k_hyperedge_dict = OrderedDict(
            sorted(k_hyperedge_dict.items(), key=itemgetter(0))
        )

        if len(ndict) != 0:

            # if index:
            #     rowdict = {v: k for k, v in ndict.items()}
            #     coldict = {v: k for k, v in edict.items()}

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
                    (data, (rows, cols)),
                    shape=(len(r_hyperedge_dict), len(k_hyperedge_dict)),
                )
            else:
                # Create an np.matrix
                MP = np.zeros((len(children), len(uidset)), dtype=int)
                for e in k_hyperedge_dict:
                    for n in r_hyperedge_dict:
                        if r_hyperedge_dict[n] <= k_hyperedge_dict[e]:
                            MP[ndict[n], edict[e]] = 1
            if index:
                return ndict, edict, MP
            else:
                return MP
        else:
            if index:
                return {}, {}, np.zeros(1)
            else:
                return np.zeros(1)

    def incidence_matrix(
        self, rank, to_rank, incidence_type="up", weight=None, sparse=True, index=False
    ):
        """
        An incidence matrix indexed by r-ranked hyperedges k-ranked hyperedges
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
        Incidence_matrix method  is a method for generating the incidence matrix of a combinatorial complex.
        An incidence matrix is a matrix that describes the relationships between the hyperedges
        of a complex. In this case, the incidence_matrix method generates a matrix where
        the rows correspond to the hyperedges of the complex and the columns correspond to the faces
        of the complex. The entries in the matrix are either 0 or 1,
        depending on whether a hyperedge contains a given face or not.
        For example, if hyperedge i contains face j, then the entry in the ith
        row and jth column of the matrix will be 1, otherwise it will be 0.

        To generate the incidence matrix, the incidence_matrix method first creates
        a dictionary where the keys are the faces of the complex and the values are
        the hyperedges that contain that face. This allows the method to quickly look up
        which hyperedges contain a given face. The method then iterates over the hyperedges in
        the HyperEdgeView instance, and for each hyperedge, it checks which faces it contains.
        For each face that the hyperedge contains, the method increments the corresponding entry
        in the matrix. Finally, the method returns the completed incidence matrix.
        """
        if rank == to_rank:
            raise ValueError("incidence must be computed for k!=r, got equal r and k.")
        if to_rank is None:
            if incidence_type == "up":
                children = self.skeleton(rank)
                uidset = self.skeleton(rank + 1, level="upper")
            elif incidence_type == "down":
                uidset = self.skeleton(rank)
                children = self.skeleton(rank - 1, level="lower")
            raise TopoNetXError("incidence_type must be 'up' or 'down' ")
        else:
            assert (
                rank != to_rank
            )  # incidence is defined between two skeletons of different ranks
            if (
                rank < to_rank
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(rank)
                uidset = self.skeleton(to_rank)

            elif (
                rank > to_rank
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(to_rank)
                uidset = self.skeleton(rank)
        return self._incidence_matrix_helper(children, uidset, sparse, index)

    def adjacency_matrix(self, rank, via_rank, s=1, weights=False, index=False):
        """Compute the adjacency matrix.

        An adjacency matrix for the RankedEntitySet of the rank-ranked considering
        their adjacency with respect to via_rank-ranked entities rank < via_rank

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
        """
        if via_rank is not None:
            assert rank < via_rank  # rank k must be smaller than rank r

        if index:
            B, row, col = self.incidence_matrix(
                rank, via_rank, incidence_type="up", sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                rank, via_rank, incidence_type="up", sparse=True, index=index
            )

        weights = False  # Currently weighting is not supported
        if weights is False:
            A = B.dot(B.transpose())
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, row
        return A

    def coadjacency_matrix(self, rank, via_rank, s=1, weights=False, index=False):
        """Compute the coadjacency matrix.

        A coadjacency matrix for the RankedEntitySet of the rank-ranked considering
        their adjacency with respect to via-rank ranked entities
        rank > via_rank

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
        """
        # TODO : None case
        assert rank < via_rank  # rank k must be larger than rank r
        if index:
            B, row, col = self.incidence_matrix(
                rank, via_rank, incidence_type="down", sparse=True, index=index
            )
        else:
            B = self.incidence_matrix(
                via_rank, rank, incidence_type="down", sparse=True, index=index
            )
        weights = False  # Currently weighting is not supported
        if weights is False:
            A = B.T.dot(B)
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, col
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


class SimplexView:
    """
    The SimplexView class is used to provide a view into a subset of the nodes
    in a simplex, allowing for efficient computations on the subset of nodes.
    These classes are used in conjunction with the SimplicialComplex class
    to perform computations on simplicial complexes.

    Parameters
    ----------
    name : str

    Examples
    --------
         >>> SC = SimplexView()
         >>> SC.insert_simplex ( (1,2,3,4),weight=1 )
         >>> SC.insert_simplex ( (2,3,4,1) )
         >>> SC.insert_simplex ( (1,2,3,4) )
         >>> SC.insert_simplex ( (1,2,3,6) )
         >>> c1=Simplex((1,2,3,4,5))
         >>> SC.insert_simplex(c1)
         #>>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        self.max_dim = -1
        self.faces_dict = []

    def __getitem__(self, simplex):
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
        if isinstance(simplex, Simplex):
            if simplex.nodes in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex.nodes]
        elif isinstance(simplex, Iterable):
            simplex = frozenset(simplex)
            if simplex in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex]
            else:
                raise KeyError(f"input {simplex} is not in the simplex dictionary")

        elif isinstance(simplex, Hashable):

            if frozenset({simplex}) in self:

                return self.faces_dict[0][frozenset({simplex})]

    # Set methods
    @property
    def shape(self):
        if len(self.faces_dict) == 0:
            print("Complex is empty.")
        else:
            return [len(self.faces_dict[i]) for i in range(len(self.faces_dict))]

    def __len__(self):
        if len(self.faces_dict) == 0:
            return 0
        else:
            return np.sum(self.shape)

    def __iter__(self):
        all_simplices = []
        for i in range(len(self.faces_dict)):
            all_simplices = all_simplices + list(self.faces_dict[i].keys())
        return iter(all_simplices)

    def __contains__(self, e):

        if len(self.faces_dict) == 0:
            return False

        if isinstance(e, Iterable):
            if len(e) - 1 > self.max_dim:
                return False
            elif len(e) == 0:
                return False
            else:
                return frozenset(e) in self.faces_dict[len(e) - 1]

        elif isinstance(e, Simplex):
            if len(e) - 1 > self.max_dim:
                return False
            elif len(e) == 0:
                return False
            else:
                return e.nodes in self.faces_dict[len(e) - 1]

        elif isinstance(e, Hashable):
            if isinstance(e, Iterable):
                if len(e) - 1 > self.max_dim:
                    return False
                elif len(e) == 0:
                    return False
            else:
                return frozenset({e}) in self.faces_dict[0]
        else:
            return False

    def __repr__(self):
        """C
        String representation of simplices
        Returns
        -------
        str
        """
        all_simplices = []
        for i in range(len(self.faces_dict)):
            all_simplices = all_simplices + [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"

    def __str__(self):
        """
        String representation of simplices

        Returns
        -------
        str
        """
        all_simplices = []
        for i in range(len(self.faces_dict)):
            all_simplices = all_simplices + [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"
