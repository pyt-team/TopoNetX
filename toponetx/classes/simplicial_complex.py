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
from toponetx.classes.simplex import NodeView, Simplex, SimplexView
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
    -simplices : list of maximal simplices that define the simplicial complex
    -name : hashable, optional, default: None
        If None then a placeholder ''  will be inserted as name
    -mode : computational mode, available options are "normal" or "gudhi".
        default is 'normal'


    Note:
    -----
    A simplicial complex is determined by its maximal simplices, simplices that are not
    contained in any other simplices. If a maximal simplex is inserted, all faces of this
    simplex will be inserted automatically.

    Examples
    -------
    Example 1
        # defining a simplicial complex using a set of maximal simplices.
        >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])


    Example 2
        # compatabiltiy with networkx
        >>> G = Graph() # networkx graph
        >>> G.add_edge(0,1,weight=4)
        >>> G.add_edge(0,3)
        >>> G.add_edge(0,4)
        >>> G.add_edge(1,4)
        >>> SC = SimplicialComplex(simplices=G)
        >>> SC.add_simplex([1,2,3])
        >>> SC.simplices


    """

    def __init__(self, simplices=None, name=None, mode="normal", **attr):

        self.mode = mode
        if name is None:
            self.name = ""
        else:
            self.name = name

        self._simplex_set = SimplexView()

        self.complex = dict()  # dictionary for simplicial complex attributes
        self.complex.update(attr)

        if simplices is not None:

            if not isinstance(simplices, Iterable):
                raise TypeError(
                    f"Input simplices must be given as Iterable, got {type(simplices)}."
                )

        if isinstance(simplices, Graph):

            _simplices = []
            for n in simplices:  # simplices is a networkx graph
                _simplices.append(([n], simplices.nodes[n]))
            for e in simplices.edges:
                u, v = e
                _simplices.append((e, simplices.get_edge_data(u, v)))

            simplices = []
            for s in _simplices:
                s1 = Simplex(s[0], **s[1])
                simplices.append(s1)

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
        self._node_set = NodeView(self._simplex_set)

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
        return self._node_set

    @property
    def simplices(self):
        """
        set of all simplices
        """
        return self._simplex_set

    def get_simplex_id(self, simplex):
        if simplex in self:
            return self[simplex]["id"]

    def is_maximal(self, simplex):
        if simplex in self:
            return self[simplex]["is_maximal"]

    def get_maximal_simplices_of_simplex(self, simplex):
        return self[simplex]["membership"]

    def skeleton(self, n):
        """
        Returns
        -------
        set of simplices of dimesnsion n
        """
        if n < len(self._simplex_set.faces_dict):
            return list(self._simplex_set.faces_dict[n].keys())
        elif n < 0:
            raise ValueError(f"input must be a postive integer, got {n}")
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

    def incidence_matrix(self, d, signed=True, weight=None, index=False):
        """
        getting the matrix that correpodnds to the boundary matrix of the input SC.

        Examples
        --------
            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,2,3,4])
            >>> SC.add_simplex([1,2,4])
            >>> SC.add_simplex([3,4,8])
            >>> B1 = SC.incidence_matrix(1)
            >>> B2 = SC.incidence_matrix(2)

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

    def coincidence_matrix(self, d, signed=True, weight=None, index=False):

        if index:
            idx_faces, idx_simplices, boundary = self.incidence_matrix(
                d, signed=signed, weight=weight, index=index
            ).T
            return idx_faces, idx_simplices, boundary.T
        else:
            return self.incidence_matrix(d, signed, index).T

    def hodge_laplacian_matrix(self, d, signed=True, weight=None, index=False):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L = B_next @ B_next.transpose()
        elif d < self.dim:
            B_next = self.incidence_matrix(d + 1)
            B = self.incidence_matrix(d)
            L = B_next @ B_next.transpose() + B.transpose() @ B
        elif d == self.dim:
            B = self.incidence_matrix(d)
            L = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.dim} (maximal dimension simplices), got {d}"
            )
        if signed:
            return L
        else:
            return abs(L)

    def normalized_laplacian_matrix(self, d, weight=None):
        r"""Returns the normalized Laplacian matrix of G.

            The normalized graph Laplacian is the matrix

            .. math::

                N_d = D^{-1/2} L_d D^{-1/2}

            where `L` is the simplcial complex Laplacian and `D` is the diagonal matrix of
            d^th simplices degrees.

            Parameters
            ----------
            d : int
                dimension of the hodge laplacian matrix


            weight : string or None, optional (default='weight')
               The edge data key used to compute each value in the matrix.
               If None, then each edge has weight 1.

            Returns
            -------
            N : Scipy sparse matrix
              The normalized hodge Laplacian matrix of complex.

            -----


        Example 1

            >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
            >>> NL1= SC.normalized_hodge_laplacian_matrix(1)
            >>> NL1


        """
        import numpy as np
        import scipy as sp
        import scipy.sparse  # call as sp.sparse

        L = self.hodge_laplacian_matrix(d)
        m, n = L.shape
        diags_ = abs(L).sum(axis=1)

        with sp.errstate(divide="ignore"):
            diags_sqrt = 1.0 / np.sqrt(diags_)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt.T, 0, m, n, format="csr"))

        return sp.sparse.csr_matrix(DH @ (L @ DH))

    def up_laplacian_matrix(self, d, signed=True, weight=None, index=False):
        if d == 0:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        elif d < self.dim:
            B_next = self.incidence_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.dim-1} (maximal dimension simplices-1), got {d}"
            )
        if signed:
            return L_up
        else:
            return abs(L_up)

    def down_laplacian_matrix(self, d, signed=True, weight=None, index=False):
        if d <= self.dim and d > 0:
            B = self.incidence_matrix(d)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.dim} (maximal dimension simplices), got {d}."
            )
        if signed:
            return L_down
        else:
            return abs(L_down)

    def adjacency_matrix(self, d, signed=False, weight=None, index=False):

        L_up = self.up_laplacian_matrix(d, signed=signed)
        L_up.setdiag(0)

        if signed:
            return L_up
        else:
            return abs(L_up)

    def coadjacency_matrix(self, d, signed=False, weight=None, index=False):

        L_down = self.down_laplacian_matrix(d, signed=signed)
        L_down.setdiag(0)
        if signed:
            return L_down
        else:
            return abs(L_down)

    def k_hop_incidence_matrix(self, d, k):
        Bd = self.incidence_matrix(d, signed=True)
        if d < self.dim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.dim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.dim:
            return Bd @ np.power(coAd, k)
        elif d == 0:
            return Bd @ np.power(Ad, k)
        else:
            return Bd @ np.power(Ad, k) + Bd @ np.power(coAd, k)

    def k_hop_coincidence_matrix(self, d, k):
        BTd = self.coincidence_matrix(d, signed=True)
        if d < self.dim and d >= 0:
            Ad = self.adjacency_matrix(d, signed=True)
        if d <= self.dim and d > 0:
            coAd = self.coadjacency_matrix(d, signed=True)
        if d == self.dim:
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

    def restrict_to_simplices(self, cellset, name=None):
        """
        Constructs a combinatorial complex using a subset of the cells in combinatorial complex

        Parameters
        ----------
        cellset: iterable of hashables or RankedEntities
            A subset of elements of the combinatorial complex  cells

        name: str, optional

        Returns
        -------
        new Combinatorial Complex : CombinatorialComplex

        Example

        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((1,2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC1= SC.restrict_to_simplices([c1, (2,4) ])
        >>> SC1.simplices
        CellView([Cell(1, 2, 3)])



        """
        RNS = []
        for i in cellset:
            if i in self:
                RNS.append(i)

        SC = SimplicialComplex(simplices=RNS, name=name)

        return SC

    def restrict_to_nodes(self, nodeset, name=None):
        """
        Constructs a new combinatorial complex  by restricting the cells in the combintorial complex to
        the nodes referenced by nodeset.

        Parameters
        ----------
        nodeset: iterable of hashables
            References a subset of elements of self.nodes

        name: string, optional, default: None

        Returns
        -------
        new Simplicial Complex : SimplicialComplex

        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((1,2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC.restrict_to_nodes([1,2,3,4])

        """

        simplices = []
        nodeset = set(nodeset)
        for i in range(1, self.dim + 1):
            for s in self.skeleton(i):
                if s.issubset(nodeset):
                    simplices.append(s)
        all_sim = simplices + list([frozenset({i}) for i in nodeset if i in self.nodes])

        return SimplicialComplex(all_sim, name=name)

    def get_all_maximal_simplices(self):
        """
        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC.get_all_maximal_simplices()
        """
        maxmimals = []
        for s in self:
            if self.is_maximal(s):
                maxmimals.append(s)
        return maxmimals

    @staticmethod
    def from_nx_graph(G):
        """
        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge('1', '2', weight=2)
        >>> G.add_edge('3', '4', weight=4)
        >>> SC = SimplicialComplex.from_nx_graph(G)
        >>> SC[('1','2')]['weight']

        """
        return SimplicialComplex(G, name=G.name)

    def is_connected(self):
        g = nx.Graph()
        for e in self.skeleton(1):
            e = list(e)
            g.add_edge(e[0], e[1])
        for n in self.skeleton(0):
            g.add_node(list(n)[0])
        return nx.is_connected(g)

    def to_hypergraph(self):
        """
        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC.to_hypergraph()
        """
        graph = []
        for i in range(1, self.dim + 1):
            edge = [list(j) for j in self.skeleton(i)]
            graph = graph + edge
        return Hypergraph(graph, static=True)

    def to_combinatorialcomplex(self):
        """
        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC.to_combinatorialcomplex()
        """
        graph = []
        for i in range(1, self.dim + 1):
            edge = [
                RankedEntity(uid=tuple(j), elements=list(j), rank=len(j) - 1)
                for j in self.skeleton(i)
            ]
            graph = graph + edge
        RES = RankedEntitySet("", graph)
        return CombinatorialComplex(RES)
