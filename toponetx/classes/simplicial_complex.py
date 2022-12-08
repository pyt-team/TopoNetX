"""

Simplicial Complex Class

"""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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

from toponetx.classes.node_view import NodeView
from toponetx.classes.ranked_entity import (
    DynamicCell,
    Node,
    RankedEntity,
    RankedEntitySet,
)
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


# from toponetx.classes.cell_complex import CellComplex
# from toponetx.classes.combinatorial_complex import CombinatorialComplex
# from toponetx.classes.dynamic_combinatorial_complex import DynamicCombinatorialComplex

__all__ = ["SimplicialComplex"]


class SimplicialComplex:
    """Class for construction boundary operators, Hodge Laplacians,
    higher order (co)adjacency operators from collection of
    simplices.

    A simplicial complex is a topological space of a specific kind, constructed by
    "gluing together" points, line segments, triangles, and their higher-dimensional
    counterparts. It is a generalization of the notion of a triangle in a triangulated surface,
    or a tetrahedron in a tetrahedralized 3-dimensional manifold. Simplicial complexes are the
    basic objects of study in combinatorial topology.

    For example, a triangle is a simplicial complex because it is a collection of three
    points that are connected to each other in a specific way. Similarly, a tetrahedron is a
    simplicial complex because it is a collection of four points that are connected to each
    other in a specific way. These simplices can be thought of as the "building blocks" of a
    simplicial complex, and the complex itself is constructed by combining these building blocks
    in a specific way. For example, a 2-dimensional simplicial complex could be a collection of
    triangles that are connected to each other to form a surface, while a 3-dimensional simplicial
    complex could be a collection of tetrahedra that are connected to each other to form a solid object.


    The SimplicialComplex class is a class for representing simplicial complexes,
    which are a type of topological space constructed by "gluing together" points, line segments,
    triangles, and higher-dimensional counterparts. The class provides methods for computing boundary
    operators, Hodge Laplacians, and higher-order adjacency operators on the simplicial complex.
    It also allows for compatibility with the NetworkX and gudhi libraries.

    main features:
    --------------
    1. The SimplicialComplex class allows for the dynamic construction of simplicial complexes,
        enabling users to add or remove simplices from the complex after its initial creation.
    2. The class provides methods for computing boundary operators, Hodge Laplacians,
        and higher-order adjacency operators on the simplicial complex.
    3. The class is compatible with the gudhi library, allowing users to leverage the powerful
        algorithms and data structures provided by this package.
    4. The class supports the attachment of arbitrary attributes and data to simplices,
        enabling users to store and manipulate additional information about these objects.
    5. The class has robust error handling and input validation, ensuring reliable and easy use of the class.

    Parameters
    ----------
    -simplices : list, optional,  default: None
                list of maximal simplices that define the simplicial complex
    -name : hashable, optional, default: None
        If None then a placeholder '' will be inserted as name
    -mode : string, optional, default 'normal'.
        computational mode, available options are "normal" or "gudhi".
        default is 'normal'.

        Note : When ghudi is selected additioanl structure
        obtained from the simplicial tree is stored.
        this creates an additional reduannt storage
        but it can be used for access the simplicial
        tree of the complex.


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

        if self.mode == "normal":
            if simplices is not None:
                if isinstance(simplices, Iterable):
                    self._simplex_set.add_simplices_from(simplices)

        elif self.mode == "gudhi":

            self.st = SimplexTree()
            if simplices is not None:

                if isinstance(simplices, Iterable):
                    for s in simplices:
                        self.st.insert(s)
                    self._simplex_set.build_faces_dict_from_gudhi_tree(self.st)

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
    def maxdim(self):
        """
        dimension of the simplicial complex is the highest dimension of any simplex in the complex
        """
        return self._simplex_set.max_dim

    @property
    def nodes(self):
        return NodeView(self._simplex_set.faces_dict, cell_type=Simplex)

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
        int, total number of simplices in all dimensions
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
    def get_boundaries(simplices, min_dim=None, max_dim=None):
        """
        Parameters
        ----------
        simplices : list
            DESCRIPTION. list or of simplices, typically integers.
        min_dim : int, constrain the max dimension of faces
        max_dim : int, constrain the max dimension of faces
        Returns
        -------
        faceset : set
            DESCRIPTION. list of tuples or all faces at all levels (subsets) of the input list of simplices
        """

        if not isinstance(simplices, Iterable):
            raise TypeError(
                f"Input simplices must be given as a list or tuple, got {type(simplices)}."
            )

        faceset = set()
        for simplex in simplices:
            numnodes = len(simplex)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex, r):
                    if max_dim is None and min_dim is None:
                        faceset.add(frozenset(sorted(face)))
                    elif max_dim is not None and min_dim is not None:
                        if len(face) <= max_dim + 1 and len(face) >= min_dim + 1:
                            faceset.add(frozenset(sorted(face)))
                    elif max_dim is not None and min_dim is None:
                        if len(face) <= max_dim + 1:
                            faceset.add(frozenset(sorted(face)))
                    elif max_dim is None and min_dim is not None:
                        if len(face) >= min_dim + 1:
                            faceset.add(frozenset(sorted(face)))

        return faceset

    def remove_maximal_simplex(self, simplex):
        self._simplex_set.remove_maximal_simplex(simplex)

    def add_node(self, node, **attr):
        self.add_simplex(node, **attr)

    def add_simplex(self, simplex, **attr):
        self._simplex_set.insert_simplex(simplex, **attr)

    def add_simplices_from(self, simplices):
        for s in simplices:
            self.add_simplex(s)

    def get_cofaces(self, simplex, codimension):
        """
        Parameters
        ----------
        simplex : list, tuple or simplex
            DESCRIPTION. the n simplex represented by a list of its nodes
        codimension : int
            DESCRIPTION. The codimension. If codimension = 0, all cofaces are returned


        Returns
        -------
        TYPE
            list of tuples(simplex).


        """
        entire_tree = self.get_boundaries(
            self.get_maximal_simplices_of_simplex(simplex)
        )
        return [
            i
            for i in entire_tree
            if frozenset(simplex).issubset(i) and len(i) - len(simplex) >= codimension
        ]

    def get_star(self, simplex):
        """
        Parameters
        ----------
        simplex : list, tuple or simplex
            DESCRIPTION. the n simplex represented by a list of its nodes


        Returns
        -------
        TYPE
            list of tuples(simplex),

        Note : return of this function is
            same as get_cofaces(simplex,0) .

        """
        return self.get_cofaces(simplex, 0)

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
            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,3,4])
            >>> SC.add_simplex([1,2,3])
            >>> SC.add_simplex([1,2,4])
            >>> d={ (1,3,4): { 'color':'red','attr2':1 },(1,2,4): {'color':'blue','attr2':3 } }
            >>> SC.set_simplex_attributes(d)
            >>> SC[(1,3,4)]['color']
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
        rows, cols = np.where(np.sign(np.abs(matrix)) > 0)
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

        simplex_dict_d = {
            simplex: i
            for i, simplex in enumerate(self._simplex_set.faces_dict[d].keys())
        }
        simplex_dict_d_minus_1 = {
            simplex: i
            for i, simplex in enumerate(self._simplex_set.faces_dict[d - 1].keys())
        }
        for simplex, idx_simplex in simplex_dict_d.items():
            # for simplex, idx_simplex in self._simplex_set.faces_dict[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(simplex_dict_d_minus_1[face])
        assert len(values) == (d + 1) * len(simplex_dict_d)
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(
                len(simplex_dict_d_minus_1),
                len(simplex_dict_d),
            ),
        )
        if index:
            if signed:
                return (
                    list(simplex_dict_d_minus_1.keys()),
                    list(simplex_dict_d.keys()),
                    boundary,
                )
            else:
                return (
                    list(simplex_dict_d_minus_1.keys()),
                    list(simplex_dict_d.keys()),
                    abs(boundary),
                )
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def coincidence_matrix(self, d, signed=True, weight=None, index=False):

        """
        Note : also called the coboundary matrix
        """

        if index:
            idx_faces, idx_simplices, boundary = self.incidence_matrix(
                d, signed=signed, weight=weight, index=index
            ).T
            return idx_faces, idx_simplices, boundary.T
        else:
            return self.incidence_matrix(d, signed=signed, weight=weight, index=index).T

    def hodge_laplacian_matrix(self, d, signed=True, weight=None, index=False):
        """
        An hodge-laplacian matrix for the simplicial complex

        Parameters
        ----------
        d : int, dimension of the Laplacian matrix.

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False

        index : boolean, optional, default False
                indicates wheather to return the indices that define the incidence matrix


        Returns
        -------
        Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list



          Examples
          --------
              >>> SC = SimplicialComplex()
              >>> SC.add_simplex([1,2,3,4])
              >>> SC.add_simplex([1,2,4])
              >>> SC.add_simplex([3,4,8])
              >>> L1 = SC.hodge_laplacian_matrix(1)
        """
        if d == 0:
            row, column, B_next = self.incidence_matrix(
                d + 1, weight=weight, index=True
            )
            L = B_next @ B_next.transpose()
            if not signed:
                L = abs(L)
            if index:
                return row, L
            else:
                return L
        elif d < self.dim:
            row, column, B_next = self.incidence_matrix(
                d + 1, weight=weight, index=True
            )
            row, column, B = self.incidence_matrix(d, weight=weight, index=True)
            L = B_next @ B_next.transpose() + B.transpose() @ B
            if not signed:
                L = abs(L)
            if index:
                return column, L
            else:
                return L

        elif d == self.dim:
            row, column, B = self.incidence_matrix(d, weight=weight, index=True)
            L = B.transpose() @ B
            if not signed:
                L = abs(L)
            if index:
                return column, L
            else:
                return L

        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.dim} (maximal dimension simplices), got {d}"
            )
        if not signed:
            L = abs(L)
        else:
            return abs(L)

    def normalized_laplacian_matrix(self, d, weight=None):
        r"""Returns the normalized hoodge Laplacian matrix of G.

            The normalized hodge Laplacian is the matrix

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
            >>> NL1= SC.normalized_laplacian_matrix(1)
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
        """

        Parameters
        ----------
        d : int, dimension of the up Laplacian matrix.

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

        index : boolean, optional, default False
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension
        Returns
        -------
        up Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            dependeing on the input dimension
        """

        weight = None  # this feature is not supported in this version

        if d == 0:
            row, col, B_next = self.incidence_matrix(d + 1, weight=weight, index=True)
            L_up = B_next @ B_next.transpose()
        elif d < self.maxdim:
            row, col, B_next = self.incidence_matrix(d + 1, weight=weight, index=True)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension cells-1), got {d}"
            )
        if not signed:
            L_up = abs(L_up)

        if index:
            return row, L_up
        else:
            return L_up

    def down_laplacian_matrix(self, d, signed=True, weight=None, index=False):
        """

        Parameters
        ----------
        d : int, dimension of the down Laplacian matrix.

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.

        weight : bool, default=False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.cells.cell_weight dictionary values.

        index : boolean, optional, default False
            list identifying rows with simplices used to index the hodge Laplacian matrix
            dependeing on the input dimension
        Returns
        -------
        down Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with simplices used to index the hodge Laplacian matrix
            dependeing on the input dimension




        """
        weight = None  # this feature is not supported in this version

        if d <= self.maxdim and d > 0:
            row, column, B = self.incidence_matrix(d, weight=weight, index=True)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.maxdim} (maximal dimension cells), got {d}."
            )
        if not signed:
            L_down = abs(L_down)
        if index:
            return row, L_down
        else:
            return L_down

    def adjacency_matrix(self, d, signed=False, weight=None, index=False):
        """
        The method takes a d parameter, which is the dimension of the simplicial complex,
        and two optional parameters: signed and weight. The signed parameter determines whether
        the adjacency matrix should be signed or unsigned, and the weight parameter allows for
        specifying weights for the edges in the adjacency matrix. The index parameter determines
        whether the method should return the matrix indices along with the adjacency matrix.

        Examples
        --------
            >>> SC = SimplicialComplex()
            >>> SC.add_simplex([1,2,3,4])
            >>> SC.add_simplex([1,2,4])
            >>> SC.add_simplex([3,4,8])
            >>> A1 = SC.adjacency_matrix(1)

        """

        weight = None  # this feature is not supported in this version

        ind, L_up = self.up_laplacian_matrix(
            d, signed=signed, weight=weight, index=True
        )
        L_up.setdiag(0)

        if not signed:
            L_up = abs(L_up)
        if index:
            return ind, L_up
        else:
            return L_up

    def coadjacency_matrix(self, d, signed=False, weight=None, index=False):

        weight = None  # this feature is not supported in this version

        ind, L_down = self.down_laplacian_matrix(
            d, signed=signed, weight=weight, index=True
        )
        L_down.setdiag(0)
        if not signed:
            L_down = abs(L_down)
        if ind:
            return index, L_down
        else:
            return L_down

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
        Constructs a simplicial complex using a subset of the simplices
        in simplicial complex

        Parameters
        ----------
        cellset: iterable of hashables or simplices
            A subset of elements of the simplicial complex

        name: str, optional

        Returns
        -------
        new simplicial Complex : SimplicialComplex

        Example

        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((1,2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC1= SC.restrict_to_simplices([c1, (2,4) ])
        >>> SC1.simplices

        """
        RNS = []
        for i in cellset:
            if i in self:
                RNS.append(i)

        SC = SimplicialComplex(simplices=RNS, name=name)

        return SC

    def restrict_to_nodes(self, nodeset, name=None):
        """
        Constructs a new simplicial complex  by restricting the simplices in the
        simplicial complex to the nodes referenced by nodeset.

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
                maxmimals.append(tuple(s))
        return maxmimals

    @staticmethod
    def from_spharpy(mesh):
        """
        >>> import spharapy.trimesh as tm
        >>> import spharapy.spharabasis as sb
        >>> import spharapy.datasets as sd
        >>> mesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])

        >>> SC = SimplicialComplex.from_spharpy(mesh)

        """

        vertices = np.array(mesh.vertlist)
        SC = SimplicialComplex(mesh.trilist)

        first_ind = np.min(mesh.trilist)

        if first_ind == 0:

            SC.set_simplex_attributes(
                dict(zip(range(len(vertices)), vertices)), name="position"
            )
        else:  # first index starts at 1.

            SC.set_simplex_attributes(
                dict(zip(range(first_ind, len(vertices) + first_ind), vertices)),
                name="position",
            )

        return SC

    @staticmethod
    def from_gudhi(tree):
        """
        >>> from gudhi import SimplexTree
        >>> tree = SimplexTree()
        >>> tree.insert([1,2,3,5])
        >>> SC = SimplicialComplex.from_gudhi(tree)
        """
        SC = SimplicialComplex()
        SC._simplex_set.build_faces_dict_from_gudhi_tree(tree)
        return SC

    @staticmethod
    def from_trimesh(mesh):
        """
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                               faces=[[0, 1, 2]],
                               process=False)
        >>> SC = SimplicialComplex.from_trimesh(mesh)
        >>> print(SC.nodes0
        >>> Sprint(C.simplices)
        >>> SC[(0)]['position']

        """
        # try to see the index of the first vertex

        SC = SimplicialComplex(mesh.faces)

        first_ind = np.min(mesh.faces)

        if first_ind == 0:

            SC.set_simplex_attributes(
                dict(zip(range(len(mesh.vertices)), mesh.vertices)), name="position"
            )
        else:  # first index starts at 1.

            SC.set_simplex_attributes(
                dict(
                    zip(range(first_ind, len(mesh.vertices) + first_ind), mesh.vertices)
                ),
                name="position",
            )

        return SC

    @staticmethod
    def load_mesh(file_path, process=False, force=None):
        """

        Parameters
        ----------

            file_path: str, the source of the data to be loadeded

            process : bool, trimesh will try to process the mesh before loading it.

            force: (str or None)
                options: 'mesh' loader will "force" the result into a mesh through concatenation
                         None : will not force the above.

        Return
        -------
            SimplicialComplex
                the output simplicial complex stores the same structure stored in the mesh input file.

        Note:
        -------
            mesh files supported : obj, off, glb


        >>> SC = SimplicialComplex.load_mesh("C:/temp/stanford-bunny.obj")

        >>> SC.nodes

        """
        import trimesh

        mesh = trimesh.load_mesh(file_path, process=process, force=None)
        return SimplicialComplex.from_trimesh(mesh)

    def is_triangular_mesh(self):

        if self.dim <= 2:

            lst = self.get_all_maximal_simplices()
            for i in lst:
                if len(i) == 2:  # gas edges that are not part of a face
                    return False
            return True
        else:
            return False

    def to_trimesh(self, vertex_position_name="position"):

        import trimesh

        if not self.is_triangular_mesh():
            raise TopoNetXError(
                "input simplicial complex has dimension higher than 2 and hence it cannot be converted to a trimesh object"
            )
        else:

            vertices = list(
                dict(
                    sorted(self.get_node_attributes(vertex_position_name).items())
                ).values()
            )

            return trimesh.Trimesh(
                faces=self.get_all_maximal_simplices(), vertices=vertices, process=False
            )

    def to_spharapy(self, vertex_position_name="position"):

        """
        >>> import spharapy.trimesh as tm
        >>> import spharapy.spharabasis as sb
        >>> import spharapy.datasets as sd
        >>> mesh = tm.TriMesh([[0, 1, 2]],[[0, 0, 0], [0, 0, 1], [0, 1, 0]] )
        >>> SC = SimplicialComplex.from_spharpy(mesh)
        >>> mesh2 = SC.to_spharapy()
        >>> mesh2.vertlist == mesh.vertlist
        >>> mesh2.trilist == mesh.trilist
        """

        import spharapy.trimesh as tm

        if not self.is_triangular_mesh():
            raise TopoNetXError(
                "input simplicial complex has dimension higher than 2 and hence it cannot be converted to a trimesh object"
            )

        else:

            vertices = list(
                dict(
                    sorted(self.get_node_attributes(vertex_position_name).items())
                ).values()
            )

            return tm.TriMesh(self.get_all_maximal_simplices(), vertices)

    def laplace_beltrami_operator(self, mode="inv_euclidean"):

        """Compute a laplacian matrix for a triangular mesh
        The method creates a laplacian matrix for a triangular
        mesh using different weighting function.
        Parameters
        ----------
        mode : {'unit', 'inv_euclidean', 'half_cotangent'}, optional
            The methods for determining the edge weights. Using the option
            'unit' all edges of the mesh are weighted by unit weighting
            function, the result is an adjacency matrix. The option
            'inv_euclidean' results in edge weights corresponding to the
            inverse Euclidean distance of the edge lengths. The option
            'half_cotangent' uses the half of the cotangent of the two angles
            opposed to an edge as weighting function. the default weighting
            function is 'inv_euclidean'.
        Returns
        -------
        laplacianmatrix : array, shape (n_vertices, n_vertices)
            Matrix, which contains the discrete laplace operator for data
            defined at the vertices of a triangular mesh. The number of
            vertices of the triangular mesh is n_points.

        """

        mesh = self.to_spharapy()
        return mesh.laplacianmatrix(mode=mode)

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

        """
        Note: a simplicial complex is connected iff its 1-skeleton graph is connected.

        """

        g = nx.Graph()
        for e in self.skeleton(1):
            e = list(e)
            g.add_edge(e[0], e[1])
        for n in self.skeleton(0):
            g.add_node(list(n)[0])
        return nx.is_connected(g)

    def to_cell_complex(self):
        """
        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,4))
        >>> c3= Simplex((2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> SC.to_cell_complex()
        """
        from toponetx.classes.cell_complex import CellComplex

        return CellComplex(self.get_all_maximal_simplices())

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

    def to_combinatorial_complex(self, dynamic=True):
        """

        Parameters:
            dynamic:bool, optional, default is false
            when True returns DynamicCombinatorialComplex
            when False returns CombinatorialComplex

        Example
        >>> c1= Simplex((1,2,3))
        >>> c2= Simplex((1,2,3))

        >>> c3= Simplex((1,2,4))
        >>> c4= Simplex((2,5))
        >>> SC = SimplicialComplex([c1,c2,c3])
        >>> CC = SC.to_combinatorial_complex()
        """

        from toponetx.classes.combinatorial_complex import CombinatorialComplex
        from toponetx.classes.dynamic_combinatorial_complex import (
            DynamicCombinatorialComplex,
        )

        if dynamic:
            graph = []
            for i in range(1, self.dim + 1):
                edge = [
                    DynamicCell(elements=list(j), rank=len(j) - 1, **self[j])
                    for j in self.skeleton(i)
                ]
                graph = graph + edge
            RES = RankedEntitySet("", graph, safe_insert=False)
            return DynamicCombinatorialComplex(RES)
        else:
            CC = CombinatorialComplex()
            for i in range(1, self.dim + 1):
                for j in self.skeleton(i):
                    CC.add_cell(j, rank=len(j) - 1, **self[j])
            return CC
