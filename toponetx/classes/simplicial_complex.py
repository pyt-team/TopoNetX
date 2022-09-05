"""

Simplicial Complex Class

"""
from collections import Hashable, OrderedDict
from collections.abc import Iterable
from itertools import combinations
from warnings import warn

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spl
from hypernetx import Hypergraph
from networkx import Graph
from scipy.linalg import fractional_matrix_power
from scipy.sparse import coo_matrix, csr_matrix, diags, dok_matrix, eye
from sklearn.preprocessing import normalize

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.ranked_entity import RankedEntity, RankedEntitySet
from toponetx.classes.simplex import Simplex, SimplexView
from toponetx.exception import TopoNetXError

try:
    from gudhi import SimplexTree
except ImportError:
    warn(
        "gudhi library is not installed."
        + " Default computing protocol will be set for 'normal'.\n"
        + " gudhi can be installed using: 'pip install gudhi'",
        stacklevel=2,
    )


__all__ = ["SimplicialComplex"]


class SimplicialComplex:
    """Class for construction boundary operators, Hodge Laplacians,
    higher order (co)adjacency operators from collection of
    simplices.

    Parameters
    ----------
    simplices : list of maximal simplices that define the simplicial complex
    name : hashable, optional, default: None
        If None then a placeholder '_'  will be inserted as name
    maxdimension: constrain the max dimension of the input inserted simplices

    mode : computational mode, available options are "normal" or "gudhi".
    default is normal

    Example
    =======
    >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])


    """

    def __init__(self, simplices=None, name=None, mode="normal", *attr):
        if simplices is not None:

            if not isinstance(simplices, Iterable):
                raise TypeError(
                    f"Input simplices must be given as Iterable, got {type(simplices)}."
                )

        if isinstance(simplices, Graph):
            _simplices = []
            for e in simplices.edges:
                _simplices.append(e)
            for e in simplices.nodes:
                _simplices.append([e])
            simplices = _simplices

        self.mode = mode
        if name is None:
            self.name = ""
        else:
            self.name = name

        self._simplex_set = SimplexView()
        if self.mode == "gudhi":
            try:
                from gudhi import SimplexTree
            except ImportError:
                warn(
                    "gudhi library is not installed."
                    + "normal mode will be used for computations",
                    stacklevel=2,
                )
            self.mode = "normal"

        if self.mode == "normal":
            if simplices is not None:
                if isinstance(simplices, Iterable):
                    self._simplex_set.add_simplices_from(simplices)

        elif self.mode == "gudhi":
            st = SimplexTree()
            if simplices is not None:
                for s in simplices:
                    st.insert(s)
                self._simplex_set._build_faces_dict_from_gudhi_tree(st)

        else:
            raise ValueError(f" Import modes must be 'normal' and 'gudhi', got {mode}")

    @property
    def shape(self):
        """
        (number of simplices[i], for i in range(0,dim(Sc))  )

        Returns
        -------
        tuple

        """
        if len(self._simplex_set.faces_dict) == 0:
            print("Simplicial Complex is empty.")
        else:
            return [
                len(self._simplex_set.faces_dict[i])
                for i in range(len(self._simplex_set.faces_dict))
            ]

    @property
    def dim(self):
        """
        dimension of the simplicial complex is the highest dimension of any simplex in the complex
        """
        return self._simplex_set.max_dim

    @property
    def nodes(self):

        return self.skeleton(0)

    @property
    def simplices(self):
        """
        set of all simplices
        """
        return [
            self._simplex_set.faces_dict[i].keys()
            for i in range(len(self._simplex_set.faces_dict))
        ]

    def get_simplex_id(self, simplex):
        if simplex in self:
            return self[simplex]["id"]

    def is_maximal(self, simplex):
        if simplex in self:
            return self[simplex]["is_maximal"]

    def get_maximal_simplices(self, simplex):
        return self[simplex]["membership"]

    def skeleton(self, n):
        """
        Returns
        -------
        set of simplices of dimesnsion n
        """
        if n < len(self._simplex_set.faces_dict):
            return list(self._simplex_set.faces_dict[n].keys())
        else:
            raise ValueError(f"input {n} exceeds max dim")

    def __str__(self):
        """
        String representation of SC

        Returns
        -------
        str

        """
        return f"simplicial Complex with shape {self.shape} and dimension {self.dim}"

    def __repr__(self):
        """
        String representation of simplicial complex

        Returns
        -------
        str

        """
        return f"SimplicialComplex(name={self.name})"

    def __len__(self):
        """
        Number of simplices

        Returns
        -------
        int

        """

        return np.sum(self.shape)

    def __getitem__(self, simplex):
        if simplex in self:
            return self._simplex_set[simplex]
        else:
            raise KeyError("simplex is not in the simplicial complex")

    def __setitem__(self, simplex, **attr):
        if simplex in self:
            self._simplex_set.__setitem__(simplex, **attr)
        else:
            raise KeyError("simplex is not in the simplicial complex")

    def __iter__(self):
        """
        Iterate over all faces of the simplicial complex

        Returns
        -------
        dict_keyiterator

        """

        all_simplices = []
        for i in range(len(self._simplex_set.faces_dict)):
            all_simplices = all_simplices + list(self._simplex_set.faces_dict[i].keys())
        return iter(all_simplices)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.face_set

        Parameters
        ----------
        item : tuple, list

        """
        return item in self._simplex_set

    @staticmethod
    def get_boundaries(simplices, max_dim=None):
        """
        Parameters
        ----------
        simplices : list
            DESCRIPTION. list or of simplices, typically integers.
        max_dim : constrain the max dimension of faces
        Returns
        -------
        faceset : set
            DESCRIPTION. list of tuples or all faces of the input list of simplices
        """

        # valid in normal mode and can be used as a static method on any face
        # TODO, do for gudhi mode as well.
        if not isinstance(simplices, Iterable):
            raise TypeError(
                f"Input simplices must be given as a list or tuple, got {type(simplices)}."
            )

        faceset = set()
        for simplex in simplices:
            numnodes = len(simplex)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex, r):
                    if max_dim is None:
                        faceset.add(tuple(sorted(face)))
                    elif len(face) <= max_dim + 1:
                        faceset.add(tuple(sorted(face)))
        return faceset

    def remove_maximal_simplex(self, simplex):
        self._simplex_set.remove_maximal_simplex(simplex)

    def add_simplex(self, simplex, **attr):
        self._simplex_set.insert_simplex(simplex, **attr)

    def add_simplices_from(self, simplices):
        for s in simplices:
            self.add_simplex(s)

    def set_simplex_attributes(self, values, name=None):
        """

            Parameters
            ----------
            values : TYPE
                DESCRIPTION.
            name : TYPE, optional
                DESCRIPTION. The default is None.

            Returns
            -------
            None.

            Example
            ------

            After computing some property of the simplex of a simplicial complex, you may want
            to assign a simplex attribute to store the value of that property for
            each simplex:

            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,2,3,4])
            >>> SC.add_simplex([1,2,4])
            >>> SC.add_simplex([3,4,8])
            >>> d={(1,2,3):'red',(1,2,4):'blue'}
            >>> SC.set_simplex_attributes(d,name='color')
            >>> SC[(1,2,3)]['color']
            'red'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update simplex attributes::

            Examples
            --------
            >>> CC = SimplicialComplex()
            >>> CC.add_simplex([1,3,4])
            >>> CC.add_simplex([1,2,3])
            >>> CC.add_simplex([1,2,4])
            >>> d={ (1,3,4): { 'color':'red','attr2':1 },(1,2,4): {'color':'blue','attr2':3 } }
            >>> CC.set_simplex_attributes(d)
            >>> CC[(1,3,4)]['color']
            'red'

        Note : If the dict contains simplices that are not in `self.simplices`, they are
        silently ignored.

        """

        if name is not None:
            # if `values` is a dict using `.items()` => {simplex: value}

            for simplex, value in values.items():
                try:
                    self[simplex][name] = value
                except KeyError:
                    pass

        else:

            for simplex, d in values.items():
                try:
                    self[simplex].update(d)
                except KeyError:
                    pass
            return

    def get_node_attributes(self, name):
        """Get node attributes from combintorial complex

        Parameters
        ----------

        name : string
           Attribute name

        Returns
        -------
        Dictionary of attributes keyed by node.

        Examples
        --------
            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,2,3,4])
            >>> SC.add_simplex([1,2,4])
            >>> SC.add_simplex([3,4,8])
            >>> d={(1):'red',(2):'blue',(3):"black"}
            >>> SC.set_simplex_attributes(d,name='color')
            >>> SC.get_node_attributes('color')
            >>>
        'blue'

        """
        return {tuple(n): self[n][name] for n in self.skeleton(0) if name in self[n]}

    def get_simplex_attributes(self, name, k=None):
        """Get node attributes from simplical complex

        Parameters
        ----------

        name : string
           Attribute name

        k : integer rank of the k-cell
        Returns
        -------
        Dictionary of attributes keyed by cell or k-cells if k is not None

        Examples
        --------
            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,2,3,4])
            >>> SC.add_simplex([1,2,4])
            >>> SC.add_simplex([3,4,8])
            >>> d={(1,2):'red',(2,3):'blue',(3,4):"black"}
            >>> SC.set_simplex_attributes(d,name='color')
            >>> SC.get_simplex_attributes('color')

        """

        if k is None:

            return {n: self[n][name] for n in self if name in self[n]}
        else:
            return {n: self[n][name] for n in self.skeleton(k) if name in self[n]}

    @staticmethod
    def get_edges_from_matrix(matrix):
        """
        Parameters
        ----------
        matrix : numpy or scipy array

        Returns
        -------
        edges : list of indices where the operator is not zero

        Rational:
        -------
         Most operaters (e.g. adjacencies/(co)boundary maps) that describe
         connectivity of the simplicial complex
         can be described as a graph whose nodes are the simplices used to
         construct the operator and whose edges correspond to the entries
         in the matrix where the operator is not zero.

         This property implies that many computations on simplicial complexes
         can be reduced to graph computations.

        """
        rows, cols = np.where(np.sign(np.abs(matrix)) == 1)
        edges = zip(rows.tolist(), cols.tolist())
        return edges

    # ---------- operators ---------------#

    def incidence_matrix(self, d, signed=True, weights=None, index=False):
        """
        get the boundary map using gudhi
        """

        if d == 0:
            boundary = dok_matrix(
                (1, len(self._simplex_set.faces_dict[d].items())), dtype=np.float32
            )
            boundary[0, 0 : len(self._simplex_set.faces_dict[d].items())] = 1
            return boundary.tocsr()
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in self._simplex_set.faces_dict[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex["id"])
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(self._simplex_set.faces_dict[d - 1][face]["id"])
        assert len(values) == (d + 1) * len(self._simplex_set.faces_dict[d])
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(
                len(self._simplex_set.faces_dict[d - 1]),
                len(self._simplex_set.faces_dict[d]),
            ),
        )
        if index:
            if signed:
                return (
                    list(self._simplex_set.faces_dict[d - 1].keys()),
                    list(self._simplex_set.faces_dict[d].keys()),
                    boundary,
                )
            else:
                return (
                    list(self._simplex_set.faces_dict[d - 1].keys()),
                    list(self._simplex_set.faces_dict[d].keys()),
                    abs(boundary),
                )
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def coincidence_matrix(self, d, signed=True, weights=None, index=False):

        if index:
            idx_faces, idx_simplices, boundary = self.incidence_matrix(
                d, signed, index
            ).T
            return idx_faces, idx_simplices, boundary.T
        else:
            return self.incidence_matrix(d, signed, index).T

    def hodge_laplacian_matrix(self, d, signed=True, weights=None, index=False):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.incidence_matrix(d + 1)
            B = self.incidence_matrix(d)
            L = B_next @ B_next.transpose() + B.transpose() @ B
        elif d == self.maxdim:
            B = self.incidence_matrix(d)
            L = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.maxdim} (maximal dimension simplices), got {d}"
            )
        if signed:
            return L
        else:
            return abs(L)

    def up_laplacian_matrix(self, d, weights=None, signed=True):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension simplices-1), got {d}"
            )
        if signed:
            return L_up
        else:
            return abs(L_up)

    def down_laplacian_matrix(self, d, weights=None, signed=True):
        if d <= self.maxdim and d > 0:
            B = self.incidence_matrix(d)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.maxdim} (maximal dimension simplices), got {d}."
            )
        if signed:
            return L_down
        else:
            return abs(L_down)

    def adjacency_matrix(self, d, weights=None, signed=False):

        L_up = self.up_laplacian_matrix(d, signed)
        L_up.setdiag(0)

        if signed:
            return L_up
        else:
            return abs(L_up)

    def coadjacency_matrix(self, d, weights=None, signed=False):

        L_down = self.down_laplacian_matrix(d, signed)
        L_down.setdiag(0)
        if signed:
            return L_down
        else:
            return abs(L_down)

    def k_hop_incidence_matrix(self, d, k):
        Bd = self.incidence_matrix(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.maxdim:
            return Bd @ np.power(coAd, k)
        elif d == 0:
            return Bd @ np.power(Ad, k)
        else:
            return Bd @ np.power(Ad, k) + Bd @ np.power(coAd, k)

    def k_hop_coincidence_matrix(self, d, k):
        BTd = self.coincidence_matrix(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.maxdim:
            return np.power(coAd, k) @ BTd
        elif d == 0:
            return np.power(Ad, k) @ BTd
        else:
            return np.power(Ad, k) @ BTd + np.power(coAd, k) @ BTd

    def add_elements_from_nx_graph(self, G):
        _simplices = []
        for e in G.edges:
            _simplices.append(e)
        for e in G.nodes:
            _simplices.append([e])

        self.add_simplices_from(_simplices)

    @staticmethod
    def from_nx_graph(G):
        return SimplicialComplex(G)

    def is_connected(self):
        g = nx.Graph()

        for e in self.skeleton(1):
            e = list(e)
            g.add_edge(e[0], e[1])
        for n in self.skeleton(0):
            g.add_node(list(n)[0])
        return nx.is_connected(g)

    def to_hypergraph(self):
        graph = []
        for i in range(1, self.dim + 1):
            edge = [list(j) for j in self.skeleton(i)]
            graph = graph + edge
        return Hypergraph(graph, static=True)

    def to_combinatorialcomplex(self):
        graph = []
        for i in range(1, self.dim + 1):
            edge = [
                RankedEntity(uid=tuple(j), elements=list(j), rank=len(j) - 1)
                for j in self.skeleton(i)
            ]
            graph = graph + edge
        RES = RankedEntitySet("", graph)
        return CombinatorialComplex(RES)
