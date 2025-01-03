"""Class for creation and manipulation of simplicial complexes.

The class also supports attaching arbitrary attributes and data to cells.
"""

from collections.abc import Collection, Generator, Hashable, Iterable, Iterator
from itertools import combinations
from typing import Any, Generic, TypeVar

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from typing_extensions import Self, deprecated

from toponetx.classes.complex import Complex
from toponetx.classes.reportviews import NodeView, SimplexView
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplex_trie import SimplexTrie

try:
    from gudhi import SimplexTree
except ImportError:
    SimplexTree = None

try:
    from hypernetx import Hypergraph
except ImportError:
    Hypergraph = None

__all__ = ["SimplicialComplex"]

ElementType = TypeVar(
    "ElementType", bound=Hashable
)  # TODO: Also bound to SupportsLessThanT but that is not accessible?


class SimplicialComplex(Complex, Generic[ElementType]):
    """Class representing a simplicial complex.

    Class for construction boundary operators, Hodge Laplacians,
    higher order (co)adjacency operators from a collection of
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
    It also allows for compatibility with NetworkX and the GUDHI library.

    Features:

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
    simplices : iterable, optional
        Iterable of maximal simplices that define the simplicial complex.
    **kwargs : keyword arguments, optional
        Attributes to add to the complex as key=value pairs.

    Attributes
    ----------
    complex : dict
        A dictionary that can be used to store additional information about the complex.

    Notes
    -----
    A simplicial complex is determined by its maximal simplices, simplices that are not
    contained in any other simplices. If a maximal simplex is inserted, all faces of this
    simplex will be inserted automatically.

    Examples
    --------
    Define a simplicial complex using a set of maximal simplices:

    >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])

    TopoNetX is also compatible with NetworkX, allowing users to create a simplicial complex from a NetworkX graph.
    Existing node and edge attributes are copied to the simplicial complex:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, weight=4)
    >>> G.add_edges_from([(0, 3), (0, 4), (1, 4)])
    >>> SC = tnx.SimplicialComplex(simplices=G)
    >>> SC.add_simplex([1, 2, 3])
    >>> SC.simplices
    SimplexView([(0,), (1,), (3,), (4,), (2,), (0, 1), (0, 3), (0, 4), (1, 4), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    >>> SC[(0, 1)]["weight"]
    4
    """

    _simplex_trie: SimplexTrie[ElementType]

    def __init__(self, simplices=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self._simplex_trie = SimplexTrie()

        if isinstance(simplices, nx.Graph):
            _simplices: dict[tuple[Hashable, ...], Any] = {}
            for simplex, data in simplices.nodes(data=True):
                _simplices[(simplex,)] = data
            for u, v, data in simplices.edges(data=True):
                _simplices[(u, v)] = data

            simplices = []
            for simplex, data in _simplices.items():
                simplices.append(Simplex(simplex, **data))
            self.add_simplices_from(simplices)
        elif isinstance(simplices, Iterable):
            self.add_simplices_from(simplices)
        elif simplices is not None:
            raise TypeError(
                f"Input simplices must be given as Iterable, got {type(simplices)}."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of simplicial complex, i.e., the number of simplices for each dimension.

        Returns
        -------
        tuple of ints
            This gives the number of cells in each rank.

        Examples
        --------
        >>> SC = SimplicialComplex([[0, 1], [1, 2, 3], [2, 3, 4]])
        >>> SC.shape
        (5, 5, 2)
        """
        return tuple(self._simplex_trie.shape)

    @property
    def dim(self) -> int:
        """Maximal dimension of the simplices in this complex.

        Returns
        -------
        int
            The dimension of the simplicial complex.

        Examples
        --------
        >>> SC = SimplicialComplex([[0, 1], [1, 2, 3], [2, 3, 4]])
        >>> SC.dim
        2
        """
        return len(self._simplex_trie.shape) - 1

    @property
    @deprecated(
        "`SimplicialComplex.maxdim` is deprecated and will be removed in the future, use `SimplicialComplex.dim` instead."
    )
    def maxdim(self) -> int:
        """Maximum dimension of the simplicial complex.

        This is the highest dimension of any simplex in the complex.

        Returns
        -------
        int
            The maximum dimension of the simplicial complex.
        """
        return self.dim

    @property
    def nodes(self) -> NodeView:
        """Return the list of nodes in the simplicial complex.

        Returns
        -------
        NodeView
            A NodeView object representing the nodes of the simplicial complex.
        """
        return NodeView(
            {
                frozenset(node.elements): node.attributes
                for node in self._simplex_trie.skeleton(0)
            },
            Simplex,
        )

    @property
    def simplices(self) -> SimplexView:
        """Set of all simplices in the simplicial complex.

        Returns
        -------
        SimplexView
            A SimplexView object representing the set of all simplices in the simplicial complex.
        """
        return SimplexView(self._simplex_trie)

    def is_maximal(self, simplex: Iterable[ElementType]) -> bool:
        """Check if simplex is maximal, i.e., not a face of any other simplex in the complex.

        Parameters
        ----------
        simplex : Iterable
            The Simplex to check.

        Returns
        -------
        bool
            True if the given simplex is maximal in this simplicial complex, False
            otherwise.

        Raises
        ------
        ValueError
            If simplex is not in the simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([[1, 2, 3]])
        >>> SC.is_maximal([1, 2, 3])
        True
        >>> SC.is_maximal([1, 2])
        False
        """
        return self._simplex_trie.is_maximal(tuple(sorted(simplex)))

    def get_maximal_simplices_of_simplex(
        self, simplex: Iterable[Hashable]
    ) -> set[Simplex[ElementType]]:
        """Get maximal simplices of simplex.

        Parameters
        ----------
        simplex : Iterable
            The simplex for which to compute the maximal simplices.

        Returns
        -------
        set of Simplex
            Set of maximal simplices of the given simplex.
        """
        return {
            node.simplex
            for node in self._simplex_trie.cofaces(tuple(sorted(simplex)))
            if self._simplex_trie.is_maximal(node.elements)
        }

    def skeleton(self, rank: int) -> list[Simplex[ElementType]]:
        """Compute the rank-skeleton of the simplicial complex containing simplices of given rank.

        Parameters
        ----------
        rank : int
            The rank of the skeleton to compute.

        Returns
        -------
        list of Simplex
            Simplices of rank `rank` in the simplicial complex.
        """
        return sorted(node.simplex for node in self._simplex_trie.skeleton(rank))

    def __str__(self) -> str:
        """Return a detailed string representation of the simplicial complex.

        Returns
        -------
        str
            A string representation containing information about the shape and dimension
            of the simplicial complex.
        """
        return f"Simplicial Complex with shape {self.shape} and dimension {self.dim}"

    def __repr__(self) -> str:
        """Return a string representation of the simplicial complex.

        Returns
        -------
        str
            A string representation of the simplicial complex.
        """
        return "SimplicialComplex()"

    def __len__(self) -> int:
        """Compute number of simplices.

        Returns
        -------
        int
            Number of vertices in the complex.
        """
        return self.shape[0]

    def __getitem__(self, simplex: Iterable[ElementType]) -> dict[Hashable, Any]:
        """Get the data associated with the given simplex.

        Parameters
        ----------
        simplex : Any
            The simplex to retrieve.

        Returns
        -------
        dict[Hashable, Any]
            The data associated with the given simplex.

        Raises
        ------
        KeyError
            If the simplex is not present in the simplicial complex.
        """
        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex = (simplex,)
        return self._simplex_trie[tuple(sorted(simplex))].attributes

    def __iter__(self) -> Iterator[Simplex[ElementType]]:
        """Iterate over all simplices (faces) of the simplicial complex.

        Returns
        -------
        Iterator[frozenset[Hashable]]
            An iterator over all simplices in the simplicial complex.

        Raises
        ------
        KeyError
            If the simplex is not present in the simplicial complex.
        """
        return iter(self.simplices)

    def __contains__(self, atom: Any) -> bool:
        """Check whether this simplicial complex contains the given atom.

        Parameters
        ----------
        atom : Any
            The atom to be checked.

        Returns
        -------
        bool
            Returns `True` if this simplicial complex contains the atom, else `False`.
        """
        if not isinstance(atom, Iterable):
            return (atom,) in self._simplex_trie
        return tuple(sorted(atom)) in self._simplex_trie

    @staticmethod
    def get_boundaries(
        simplices: Iterable, min_dim=None, max_dim=None
    ) -> set[frozenset]:
        """Get boundaries of the given simplices.

        Parameters
        ----------
        simplices : Iterable
            Iterable of simplices for which to compute the boundaries.
        min_dim : int, optional
            Constrain the max dimension of faces.
        max_dim : int, optional
            Constrain the max dimension of faces.

        Returns
        -------
        set of frozensets
            Set of simplices that are boundaries of the given simplices. If `min_dim` or `max_dim` are given, only
            simplices with dimension in the given range are returned.
        """
        if not isinstance(simplices, Iterable):
            raise TypeError(
                f"Input simplices must be given as a list or tuple, got {type(simplices)}."
            )

        face_set = set()
        for simplex in simplices:
            start = (
                min(max_dim + 1, len(simplex)) if max_dim is not None else len(simplex)
            )
            end = min_dim if min_dim is not None else 0

            for r in range(start, end, -1):
                for face in combinations(simplex, r):
                    face_set.add(frozenset(face))

        return face_set

    def remove_maximal_simplex(self, simplex: Collection) -> None:
        """Remove maximal simplex from simplicial complex.

        Parameters
        ----------
        simplex : Collection
            The simplex to be removed from the simplicial complex.

        Raises
        ------
        KeyError
            If simplex is not in simplicial complex.
        ValueError
            If simplex is not maximal.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex((1, 2, 3, 4), weight=1)
        >>> SC.add_simplex((1, 2, 3, 4, 5))
        >>> SC.remove_maximal_simplex((1, 2, 3, 4, 5))
        """
        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex_ = (simplex,)
        elif isinstance(simplex, Iterable):
            simplex_ = sorted(simplex)
        else:
            raise TypeError("`simplex` must be Hashable or Iterable.")

        node = self._simplex_trie.find(simplex_)
        if node is None:
            raise ValueError(
                f"Simplex {simplex_} is not a part of the simplicial complex."
            )
        if len(node.children) != 0:
            raise ValueError(f"Simplex {simplex_} is not maximal.")

        self._simplex_trie.remove_simplex(simplex_)

    def remove_nodes(self, node_set: Iterable[Hashable]) -> None:
        """Remove the given nodes from the simplicial complex.

        Any simplices that become invalid due to the removal of nodes are also removed.

        Parameters
        ----------
        node_set : Iterable
            The nodes to be removed from the simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([(1, 2), (2, 3), (3, 4)])
        >>> SC.remove_nodes([1])
        >>> SC.simplices
        SimplexView([(2,), (3,), (4,), (2, 3), (3, 4)])
        """
        for node in node_set:
            self._simplex_trie.remove_simplex((node,))

    def add_node(self, node: Hashable, **kwargs) -> None:
        """Add node to simplicial complex.

        Parameters
        ----------
        node : Hashable or Simplex
            The node to be added to the simplicial complex.
        **kwargs : keyword arguments, optional
            Additional attributes to be associated with the node.
        """
        if not isinstance(node, Hashable):
            raise TypeError(f"Input node must be Hashable, got {type(node)} instead.")

        if isinstance(node, Simplex):
            if len(node) != 1:
                raise ValueError(
                    f"Input node must be a singleton Simplex, got {node} instead."
                )
            self.add_simplex(node, **kwargs)
        else:
            self.add_simplex([node], **kwargs)

    def add_simplex(
        self, simplex: Iterable[ElementType] | ElementType, **kwargs
    ) -> None:
        """Add simplex to simplicial complex.

        In case sub-simplices are missing, they are added without attributes to the
        simplicial complex to fulfill the simplex requirements.

        If the given simplex is already part of the simplicial complex, its attributes
        will be updated by the provided ones.

        Parameters
        ----------
        simplex : Collection
            The simplex to be added to the simplicial complex.

            If a `Simplex` object is given, its attributes will be copied to the
            simplicial complex. `kwargs` take precedence over the attributes of the
            `Simplex` object.
        **kwargs : keyword arguments, optional
            Additional attributes to be associated with the simplex.
        """
        Simplex.validate_attributes(kwargs)

        # Support some short-hand calls for adding nodes. The user does not have to
        # provide a single-element list but can give the node directly.
        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex = [simplex]
        if isinstance(simplex, str):
            simplex = [simplex]

        if not isinstance(simplex, Iterable) and not isinstance(simplex, Simplex):
            raise TypeError(
                f"Input simplex must be Iterable or Simplex, got {type(simplex)} instead."
            )

        if isinstance(simplex, Simplex):
            simplex_ = simplex.elements
            kwargs = simplex._attributes | kwargs
        else:
            simplex_ = frozenset(simplex)
            if len(simplex_) != len(simplex):
                raise ValueError("a simplex cannot contain duplicate nodes")

        self._simplex_trie.insert(sorted(simplex_), **kwargs)

    def add_simplices_from(self, simplices) -> None:
        """Add simplices from iterable to simplicial complex.

        Parameters
        ----------
        simplices : iterable
            Iterable of simplices to be added to the simplicial complex.
        """
        # for simplex in simplices:
        #     if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
        #         simplex = [simplex]
        #     if isinstance(simplex, str):
        #         simplex = [simplex]
        #
        #     if not isinstance(simplex, Iterable) and not isinstance(simplex, Simplex):
        #         raise TypeError(
        #             f"Input simplex must be Iterable or Simplex, got {type(simplex)} instead."
        #         )
        #
        #     if isinstance(simplex, Simplex):
        #         simplex_ = simplex.elements
        #         attr = simplex._properties
        #     else:
        #         simplex_ = frozenset(simplex)
        #         if len(simplex_) != len(simplex):
        #             raise ValueError("a simplex cannot contain duplicate nodes")
        #         attr = {}
        #
        #     self._simplex_trie.insert_raw(sorted(simplex_), **attr)
        # self._simplex_trie.restore_simplex_property()
        for s in simplices:
            self.add_simplex(s)

    def get_cofaces(
        self, simplex: Iterable[ElementType], codimension: int
    ) -> list[Simplex[ElementType]]:
        """Get cofaces of simplex.

        Parameters
        ----------
        simplex : Iterable of AtomType
            The simplex for which to compute the cofaces.
        codimension : int
            The codimension of the returned cofaces. If zero, all cofaces are returned.

        Returns
        -------
        list of tuples
            The cofaces of the given simplex.
        """
        simplex = tuple(sorted(simplex))
        # TODO: This can be optimized inside the simplex trie.
        cofaces = self._simplex_trie.cofaces(simplex)
        return [
            coface.simplex
            for coface in cofaces
            if len(coface) - len(simplex) >= codimension
        ]

    def get_star(self, simplex: Iterable[ElementType]) -> list[Simplex[ElementType]]:
        """Get star.

        Notes
        -----
        This function is equivalent to calling `get_cofaces(simplex, 0)`.

        Parameters
        ----------
        simplex : Iterable of AtomType
            The simplex for which to compute the star.

        Returns
        -------
        list of tuples
            The star of the given simplex.
        """
        return self.get_cofaces(simplex, 0)

    def set_simplex_attributes(self, values, name: str | None = None) -> None:
        """Set simplex attributes.

        Parameters
        ----------
        values : dict
            Either provide a mapping from simplices to values or a dict of dicts. In the former case, the attribute
            `name` for each simplex is set to the corresponding value. In the latter case, the entire dictionary
            is used to update the attributes of the simplices.
        name : str, optional
            The name of the attribute to set.

        Notes
        -----
        If the dict contains simplices that are not in `self.simplices`, they are
        silently ignored.

        Examples
        --------
        After computing some property of the simplex of a simplicial complex, you may want
        to assign a simplex attribute to store the value of that property for
        each simplex:

        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> d = {(1, 2, 3): "red", (1, 2, 4): "blue"}
        >>> SC.set_simplex_attributes(d, name="color")
        >>> SC[(1, 2, 3)]["color"]
        'red'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update simplex attributes:

        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 3, 4])
        >>> SC.add_simplex([1, 2, 3])
        >>> SC.add_simplex([1, 2, 4])
        >>> d = {
        ...     (1, 3, 4): {"color": "red", "attr2": 1},
        ...     (1, 2, 4): {"color": "blue", "attr2": 3},
        ... }
        >>> SC.set_simplex_attributes(d)
        >>> SC[(1, 3, 4)]["color"]
        'red'
        """
        if name is not None:
            # if `values` is a dict using `.items()` => {simplex: value}
            for simplex, value in values.items():
                if simplex in self.simplices:
                    self[simplex][name] = value
        else:
            for simplex, d in values.items():
                if simplex in self.simplices:
                    self[simplex].update(d)

    def get_node_attributes(self, name: str) -> dict[Hashable, Any]:
        """Get node attributes from simplicial complex.

        Parameters
        ----------
        name : str
           Attribute name.

        Returns
        -------
        dict
            Dictionary mapping each node to the value of the given attribute name.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> SC.set_simplex_attributes({1: "red", 2: "blue", 3: "black"}, name="color")
        >>> SC.get_node_attributes("color")
        {1: 'red', 2: 'blue', 3: 'black'}
        """
        return {
            node.label: node.attributes[name]
            for node in self._simplex_trie.skeleton(0)
            if name in node.attributes
        }

    def get_simplex_attributes(
        self, name: str, rank: int | None = None
    ) -> dict[tuple[ElementType, ...], Any]:
        """Get node attributes from simplical complex.

        Parameters
        ----------
        name : str
           Attribute name.
        rank : int, optional
            Restrict the returned attribute values to simplices of a specific rank.

        Returns
        -------
        dict[tuple[Hashable, ...], Any]
            Dictionary mapping each simplex to the value of the given attribute name.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> d = {(1, 2): "red", (2, 3): "blue", (3, 4): "black"}
        >>> SC.set_simplex_attributes(d, name="color")
        >>> SC.get_simplex_attributes("color")
        {(1, 2): 'red', (2, 3): 'blue', (3, 4): 'black'}
        """
        if rank is None:
            return {
                n.elements: self.simplices[n][name]
                for n in self.simplices
                if name in self.simplices[n]
            }
        return {
            n.elements: self.simplices[n][name]
            for n in self.skeleton(rank)
            if name in self[n]
        }

    @staticmethod
    def get_edges_from_matrix(matrix):
        """Get edges from matrix.

        Parameters
        ----------
        matrix : numpy or scipy array
            The matrix to get the edges from.

        Returns
        -------
        list of int
            List of indices where the operator is not zero.

        Notes
        -----
        Most operaters (e.g. adjacencies/(co)boundary maps) that describe
        connectivity of the simplicial complex
        can be described as a G whose nodes are the simplices used to
        construct the operator and whose edges correspond to the entries
        in the matrix where the operator is not zero.

        This property implies that many computations on simplicial complexes
        can be reduced to G computations.
        """
        rows, cols = np.where(np.sign(np.abs(matrix)) > 0)
        return zip(rows.tolist(), cols.tolist(), strict=True)

    def incidence_matrix(
        self, rank, signed: bool = True, weight: str | None = None, index: bool = False
    ) -> csr_matrix | tuple[dict, dict, csr_matrix]:
        """Compute incidence matrix of the simplicial complex.

        Getting the matrix that correpodnds to the boundary matrix of the input SC.

        Parameters
        ----------
        rank : int
            For which rank of simplices to compute the incidence matrix.
        signed : bool, default=True
            Whether to return the signed or unsigned incidence matrix.
        weight : str, optional
            The name of the simplex attribute to use as weights for the incidence
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the incidence matrix.

        Returns
        -------
        row_indices, col_indices : dict
            Dictionary assigning each row and column of the incidence matrix to a
            simplex. Only returned if `index` is True.
        incidence_matrix : scipy.sparse.csr.csr_matrix
            The incidence matrix.

        Raises
        ------
        ValueError
            If rank is negative or larger than the dimension of the simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> B1 = SC.incidence_matrix(1)
        >>> B2 = SC.incidence_matrix(2)
        """
        if rank < 0:
            raise ValueError(f"Rank must be non-negative, got {rank}.")
        if rank > self.dim:
            raise ValueError(
                f"Rank cannot be larger than the dimension of the complex, got {rank}."
            )

        # TODO: Get rid of this special case...
        if rank == 0:
            boundary = dok_matrix((1, self._simplex_trie.shape[rank]), dtype=np.float32)
            boundary[0, 0 : self._simplex_trie.shape[rank]] = 1

            if index:
                simplex_dict_d = {
                    simplex.elements: i for i, simplex in enumerate(self.skeleton(0))
                }
                return {}, simplex_dict_d, boundary.tocsr()
            return boundary.tocsr()
        idx_simplices, idx_faces, values = [], [], []

        simplex_dict_d = {
            tuple(sorted(simplex)): i for i, simplex in enumerate(self.skeleton(rank))
        }
        simplex_dict_d_minus_1 = {
            tuple(sorted(simplex)): i
            for i, simplex in enumerate(self.skeleton(rank - 1))
        }
        for simplex, idx_simplex in simplex_dict_d.items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = frozenset(simplex).difference({left_out})
                idx_faces.append(simplex_dict_d_minus_1[tuple(sorted(face))])

        if len(values) != (rank + 1) * len(simplex_dict_d):
            raise RuntimeError("Error in computing the incidence matrix.")

        boundary = csr_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(
                len(simplex_dict_d_minus_1),
                len(simplex_dict_d),
            ),
        )

        if not signed:
            boundary = abs(boundary)
        if index:
            return simplex_dict_d_minus_1, simplex_dict_d, boundary
        return boundary

    def coincidence_matrix(
        self, rank, signed: bool = True, weight=None, index: bool = False
    ):
        """Compute coincidence matrix of the simplicial complex.

        This is also called the coboundary matrix.

        Parameters
        ----------
        rank : int
            For which rank of simplices to compute the coincidence matrix.
        signed : bool, default=True
            Whether to return the signed or unsigned coincidence matrix.
        weight : str, optional
            The name of the simplex attribute to use as weights for the coincidence
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the coincidence
            matrix.

        Returns
        -------
        row_indices, col_indices : dict
            Dictionary assigning each row and column of the coincidence matrix to a
            simplex. Only returned if `index` is True.
        coincidence_matrix : scipy.sparse.csr.csr_matrix
            The coincidence matrix.
        """
        if index:
            idx_faces, idx_simplices, boundary = self.incidence_matrix(
                rank, signed=signed, weight=weight, index=True
            )
            return idx_faces, idx_simplices, boundary.T

        return self.incidence_matrix(rank, signed=signed, weight=weight, index=False).T

    def hodge_laplacian_matrix(
        self, rank: int, signed: bool = True, weight=None, index: bool = False
    ):
        """Compute hodge-laplacian matrix for the simplicial complex.

        Parameters
        ----------
        rank : int
            Dimension of the Laplacian matrix.
        signed : bool, default=True
            Whether to return the signed or unsigned hodge laplacian.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Indicates whether to return the indices that define the hodge laplacian
            matrix.

        Returns
        -------
        index : list
            List assigning each row and column of the laplacian matrix to a simplex.
            Only available when `index` is True.
        laplacian : scipy.sparse.csr.csr_matrix
            The hodge laplacian matrix of rank `rank`.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> L1 = SC.hodge_laplacian_matrix(1)
        """
        if rank == 0:
            row, column, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            L_hodge = B_next @ B_next.transpose()
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return row, L_hodge
            return L_hodge
        if rank < self.dim:
            row, column, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_hodge = B_next @ B_next.transpose() + B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return column, L_hodge
            return L_hodge
        if rank == self.dim:
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_hodge = B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return column, L_hodge
            return L_hodge

        raise ValueError(
            f"Rank should be larger than 0 and <= {self.dim} (maximal dimension simplices), got {rank}"
        )

    def dirac_operator_matrix(
        self,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute dirac operator matrix matrix.

        Parameters
        ----------
        signed : bool, default=False
            Whether the returned dirac matrix should be signed (i.e., respect
            orientations) or unsigned.
        weight : str, optional
            The name of the simplex attribute to use as weights for the dirac operator
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the dirac operator matrix.

        Returns
        -------
        row_indices, col_indices : dict
            List identifying rows and columns of the dirac operator matrix. Only
            returned if `index` is True.
        scipy.sparse.csr.csc_matrix
            The dirac operator matrix of this simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> SC.dirac_operator_matrix()
        """
        from scipy.sparse import bmat

        index_set = []
        incidence = {}
        for i in range(self.dim + 1):
            _, indexi, Bi = self.incidence_matrix(i, weight=weight, index=True)
            index_set.append(indexi)
            incidence[(i, i + 1)] = Bi
        dirac = []
        for i in range(self.dim + 1):
            row = []
            for j in range(self.dim + 1):
                if (i, j) in incidence:
                    row.append(incidence[(i + 1, j + 1)])
                elif (j, i) in incidence:
                    row.append(incidence[(j + 1, i + 1)].T)
                else:
                    row.append(None)
            dirac.append(row)

        dirac_mat = bmat(dirac)
        if index:
            d = {}
            shift = 0
            for i in index_set:
                i = {k: v + shift for k, v in i.items()}
                d.update(i)
                shift = len(d)

            if signed:
                return d, dirac_mat
            return d, abs(dirac_mat)

        if signed:
            return dirac_mat
        return abs(dirac_mat)

    def normalized_laplacian_matrix(self, rank: int, weight: str | None = None):
        """Return the normalized hodge Laplacian matrix of simplicial complex .

        The normalized hodge Laplacian is the matrix

        .. math::
            N_d = D^{-1/2} L_d D^{-1/2}

        where `L` is the simplicial complex Laplacian and `D` is the diagonal matrix of
        simplices of rank d.

        Parameters
        ----------
        rank : int
            Rank of the hodge laplacian matrix.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.

        Returns
        -------
        Scipy sparse matrix
            The normalized hodge Laplacian matrix.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        >>> SC.normalized_laplacian_matrix(1)
        """
        import scipy as sp

        L_hodge = self.hodge_laplacian_matrix(rank)
        m, n = L_hodge.shape
        diags_ = abs(L_hodge).sum(axis=1)

        with np.errstate(divide="ignore"):
            diags_sqrt = 1.0 / np.sqrt(diags_)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        diags_sqrt = sp.sparse.csr_array(
            sp.sparse.spdiags(diags_sqrt.T, 0, m, n, format="csr")
        )

        return sp.sparse.csr_matrix(diags_sqrt @ (L_hodge @ diags_sqrt))

    def up_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ):
        """Compute the up Laplacian matrix of the simplicial complex.

        Parameters
        ----------
        rank : int
            Rank of the up Laplacian matrix.
        signed : bool
            Whether to return the signed or unsigned up laplacian.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the laplacian.

        Returns
        -------
        row_indices, col_indices : list
            List identifying the rows and columns of the laplacian matrix with the
            simplices of this complex. Only returned if `index` is True.
        up_laplacian : scipy.sparse.csr.csr_matrix
            The upper laplacian matrix of this simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        >>> SC.up_laplacian_matrix(1)
        """
        if weight is not None:
            raise ValueError("`weight` is not supported in this version")

        if rank < self.dim and rank >= 0:
            row, col, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            L_up = B_next @ B_next.transpose()
        else:
            raise ValueError(
                f"Rank should larger than 0 and <= {self.dim - 1} (maximal dimension cells-1), got {rank}"
            )
        if not signed:
            L_up = abs(L_up)

        if index:
            return row, L_up
        return L_up

    def down_laplacian_matrix(
        self, rank, signed: bool = True, weight=None, index: bool = False
    ):
        """Compute the down Laplacian matrix of the simplicial complex.

        Parameters
        ----------
        rank : int
            Rank of the down Laplacian matrix.
        signed : bool
            Whether to return the signed or unsigned down laplacian.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the laplacian.

        Returns
        -------
        indices : dict
            Dictionary assigning each row and column of the laplacian matrix to a
            simplex. Only returned if `index` is True.
        down_laplacian : scipy.sparse.csr.csr_matrix
            The down laplacian matrix of this simplicial complex.
        """
        if weight is not None:
            raise ValueError("`weight` is not supported in this version")

        if self.dim >= rank > 0:
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {self.dim} (maximal dimension cells), got {rank}."
            )
        if not signed:
            L_down = abs(L_down)
        if index:
            return column, L_down
        return L_down

    def adjacency_matrix(
        self, rank, signed: bool = False, weight: str | None = None, index: bool = False
    ):
        """Compute the adjacency matrix of the simplicial complex.

        Parameters
        ----------
        rank : int
            Rank of the adjacency matrix.
        signed : bool
            Whether to return the signed or unsigned adjacency matrix.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the adjacency
            matrix.

        Returns
        -------
        indices : dict
            Dictionary mapping each row and column of the coadjacency matrix to a
            simplex. Only returned if `index` is True.
        adjacency_matrix : scipy.sparse.csr.csr_matrix
            The adjacency matrix of rank `rank` of this simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> adj1 = SC.adjacency_matrix(1)
        """
        if weight is not None:
            raise ValueError("`weight` is not supported in this version")

        ind, L_up = self.up_laplacian_matrix(
            rank, signed=signed, weight=weight, index=True
        )
        L_up.setdiag(0)

        if not signed:
            L_up = abs(L_up)
        if index:
            return ind, L_up
        return L_up

    def coadjacency_matrix(
        self, rank: int, signed: bool = False, weight=None, index: bool = False
    ):
        """Compute the coadjacency matrix of the simplicial complex.

        Parameters
        ----------
        rank : int
            Rank of the coadjacency matrix.
        signed : bool
            Whether to return the signed or unsigned coadjacency matrix.
        weight : str, optional
            The name of the simplex attribute to use as weights for the hodge laplacian
            matrix. If `None`, the matrix is binary.
        index : bool, default=False
            Whether to return the indices of the rows and columns of the coadjacency
            matrix.

        Returns
        -------
        indices : list
            List identifying the rows and columns of the coadjacency matrix with the
            simplices of the simplicial complex. Only returned if `index` is True.
        coadjacency_matrix : scipy.sparse.csr.csr_matrix
            The coadjacency matrix of this simplicial complex.
        """
        if weight is not None:
            raise ValueError("`weight` is not supported in this version")

        ind, L_down = self.down_laplacian_matrix(
            rank, signed=signed, weight=weight, index=True
        )
        L_down.setdiag(0)
        if not signed:
            L_down = abs(L_down)
        if index:
            return ind, L_down
        return L_down

    def add_elements_from_nx_graph(self, G: nx.Graph) -> None:
        """Add elements from a networkx graph to this simplicial complex.

        Parameters
        ----------
        G : networkx.Graph
            A networkx graph instance.
        """
        _simplices = [[n] for n in G.nodes] + list(G.edges)
        self.add_simplices_from(_simplices)

    def restrict_to_simplices(self, cell_set) -> Self:
        """Construct a simplicial complex using a subset of the simplices.

        Parameters
        ----------
        cell_set : iterable of hashables or simplices
            A subset of elements of the simplicial complex that should be in the new
            simplicial complex.

        Returns
        -------
        SimplicialComplex
            New simplicial complex restricted to the simplices in `cell_set`.

        Examples
        --------
        >>> c1 = tnx.Simplex((1, 2, 3))
        >>> c2 = tnx.Simplex((1, 2, 4))
        >>> c3 = tnx.Simplex((1, 2, 5))
        >>> SC = tnx.SimplicialComplex([c1, c2, c3])
        >>> SC1 = SC.restrict_to_simplices([c1, (2, 4)])
        >>> SC1.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (2, 3), (2, 4), (1, 2, 3)])
        """
        rns = [cell for cell in cell_set if cell in self.simplices]
        return self.__class__(simplices=rns)

    def restrict_to_nodes(self, node_set) -> "SimplicialComplex[ElementType]":
        """Construct a new simplicial complex by restricting the simplices.

        The simplices are restricted to the nodes referenced by node_set.

        Parameters
        ----------
        node_set : iterable of hashables
            A subset of nodes of the simplicial complex to restrict to.

        Returns
        -------
        SimplicialComplex
            A new simplicial complex restricted to the nodes in `node_set`.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([(1, 2, 3), (1, 2, 4), (1, 2, 5)])
        >>> new_complex = SC.restrict_to_nodes([1, 2, 3, 4])
        >>> new_complex.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (1, 2, 3), (1, 2, 4)])
        """
        node_set = set(node_set)
        simplices = [
            s
            for rank in range(1, self.dim + 1)
            for s in self.skeleton(rank)
            if set(s).issubset(node_set)
        ]
        all_sim = simplices + [frozenset({i}) for i in node_set if i in self.nodes]

        return SimplicialComplex(all_sim)

    def get_all_maximal_simplices(self) -> Generator[Simplex[ElementType], None, None]:
        """Get all maximal simplices of this simplicial complex.

        A simplex is maximal if it is not a face of any other simplex in the complex.

        Returns
        -------
        list of tuples
            List of maximal simplices in this simplicial complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex([(1, 2), (1, 2, 3), (1, 2, 4), (2, 5)])
        >>> SC.get_all_maximal_simplices()
        [(2, 5), (1, 2, 3), (1, 2, 4)]
        """
        for node in self._simplex_trie:
            simplex = node.simplex
            if self.is_maximal(simplex):
                yield simplex

    @classmethod
    def from_spharapy(cls, mesh) -> Self:
        """Import from sharpy.

        Parameters
        ----------
        mesh : spharapy.trimesh.TriMesh
            The input spharapy object.

        Returns
        -------
        SimplicialComplex
            The resulting SimplicialComplex.

        Examples
        --------
        >>> import spharapy.trimesh as tm
        >>> import spharapy.spharabasis as sb
        >>> import spharapy.datasets as sd
        >>> mesh = tm.TriMesh(
        ...     [[0, 1, 2]], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ... )

        >>> SC = tnx.SimplicialComplex.from_spharapy(mesh)
        """
        vertices = np.array(mesh.vertlist)
        SC = cls(mesh.trilist)

        first_ind = np.min(mesh.trilist)

        if first_ind == 0:
            SC.set_simplex_attributes(dict(enumerate(vertices)), name="position")
        else:  # first index starts at 1.
            SC.set_simplex_attributes(
                dict(enumerate(vertices, first_ind)),
                name="position",
            )

        return SC

    @classmethod
    @deprecated("`SimplicialComplex.from_spharpy` is deprecated and will be removed in the future, use `SimplicialComplex.from_spharapy` instead.")
    def from_spharpy(cls, mesh) -> Self:
        """Import from sharpy.

        Parameters
        ----------
        mesh : spharapy.trimesh.TriMesh
            The input spharapy object.

        Returns
        -------
        SimplicialComplex
            The resulting SimplicialComplex.
        """
        return cls.from_spharapy(mesh)

    def to_hasse_graph(self) -> nx.DiGraph:
        """Create the hasse graph corresponding to this simplicial complex.

        Returns
        -------
        nx.DiGraph
            A NetworkX Digraph representing the Hasse graph of the Simplicial Complex.

        Examples
        --------
        >>> SC = tnx.SimplicialComplex()
        >>> SC.add_simplex([0, 1, 2])
        >>> G = SC.to_hasse_graph()
        """
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(tuple(sorted(n)))
        for i in range(1, self.dim + 1):
            for c in self.skeleton(i):
                G.add_node(tuple(sorted(c)))
                for f in combinations(c, len(c) - 1):
                    G.add_edge(tuple(sorted(f)), tuple(sorted(c)))
        return G

    @classmethod
    def from_gudhi(cls, tree: SimplexTree) -> Self:
        """Import from gudhi.

        Parameters
        ----------
        tree : gudhi.SimplexTree
            The input gudhi simplex tree.

        Returns
        -------
        SimplicialComplex
            The resulting SimplicialComplex.

        Examples
        --------
        >>> from gudhi import SimplexTree
        >>> tree = SimplexTree()
        >>> _ = tree.insert([1, 2, 3, 5])
        >>> SC = tnx.SimplicialComplex.from_gudhi(tree)
        """
        SC = cls()
        for simplex, _ in tree.get_skeleton(tree.dimension()):
            SC.add_simplex(simplex)
        return SC

    @classmethod
    def from_trimesh(cls, mesh) -> Self:
        """Import from trimesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The input trimesh object.

        Returns
        -------
        SimplicialComplex
            The resulting SimplicialComplex.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(
        ...     vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        ...     faces=[[0, 1, 2]],
        ...     process=False,
        ... )
        >>> SC = tnx.SimplicialComplex.from_trimesh(mesh)
        >>> print(SC.nodes)
        >>> print(SC.simplices)
        >>> SC[(0)]["position"]
        """
        SC = cls(mesh.faces)

        first_ind = np.min(mesh.faces)

        if first_ind == 0:
            SC.set_simplex_attributes(
                dict(enumerate(mesh.vertices)),
                name="position",
            )
        else:  # first index starts at 1.
            SC.set_simplex_attributes(
                dict(enumerate(mesh.vertices, first_ind)),
                name="position",
            )

        return SC

    @classmethod
    def load_mesh(cls, file_path, process: bool = False) -> Self:
        """Load a mesh.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The source of the data to be loaded.
        process : bool
            Whether trimesh should try to process the mesh before loading it.

        Returns
        -------
        SimplicialComplex
            The output simplicial complex stores the same structure stored in the mesh input file.

        Notes
        -----
        mesh files supported : obj, off, glb

        Examples
        --------
        >>> SC = tnx.SimplicialComplex.load_mesh("C:/temp/stanford-bunny.obj")
        >>> SC.nodes
        """
        import trimesh

        mesh = trimesh.load_mesh(file_path, process=process, force=None)
        return cls.from_trimesh(mesh)

    def is_triangular_mesh(self) -> bool:
        """Check if the simplicial complex is a triangular mesh.

        Returns
        -------
        bool
            True if the simplicial complex can be converted to a triangular mesh,
            False otherwise.
        """
        if self.dim > 2:
            return False

        lst = self.get_all_maximal_simplices()
        return all(len(i) != 2 for i in lst)

    def to_trimesh(self, vertex_position_name: str = "position"):
        """Convert simplicial complex to trimesh object.

        Parameters
        ----------
        vertex_position_name : str, default="position"
            The simplex attribute name that contains the vertex positions.

        Returns
        -------
        trimesh.Trimesh
            The trimesh object corresponding to this simplicial complex.
        """
        import trimesh

        if not self.is_triangular_mesh():
            raise RuntimeError(
                "input simplicial complex has dimension higher than 2 and hence it cannot be converted to a trimesh object"
            )

        vertices = np.array(
            list(
                dict(
                    sorted(self.get_node_attributes(vertex_position_name).items())
                ).values()
            )
        )
        faces = [simplex.elements for simplex in self.get_all_maximal_simplices()]

        return trimesh.Trimesh(faces=faces, vertices=vertices, process=False)

    def to_spharapy(self, vertex_position_name: str = "position"):
        """Convert to spharapy.

        Parameters
        ----------
        vertex_position_name : str, default="position"
            The simplex attribute name that contains the vertex positions.

        Returns
        -------
        spharapy.trimesh.TriMesh
            The spharapy object corresponding to this simplicial complex.

        Raises
        ------
        RuntimeError
            If package `spharapy` is not installed.

        Examples
        --------
        >>> import spharapy.trimesh as tm
        >>> import spharapy.spharabasis as sb
        >>> import spharapy.datasets as sd
        >>> mesh = tm.TriMesh([[0, 1, 2]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        >>> SC = tnx.SimplicialComplex.from_spharapy(mesh)
        >>> mesh2 = SC.to_spharapy()
        >>> mesh2.vertlist == mesh.vertlist
        >>> mesh2.trilist == mesh.trilist
        """
        try:
            import spharapy.trimesh as tm
        except ImportError as e:
            raise RuntimeError(
                "Package `spharapy` is required for this function."
            ) from e

        if not self.is_triangular_mesh():
            raise RuntimeError(
                "Simplicial complex has dimension higher than 2 and cannot be converted to a trimesh object."
            )

        vertices = list(
            dict(
                sorted(self.get_node_attributes(vertex_position_name).items())
            ).values()
        )
        triangles = [simplex.elements for simplex in self.get_all_maximal_simplices()]

        return tm.TriMesh(triangles, vertices)

    def laplace_beltrami_operator(self, mode: str = "inv_euclidean"):
        """Compute a laplacian matrix for a triangular mesh.

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
        numpy.ndarray, shape (n_vertices, n_vertices)
            Matrix, which contains the discrete laplace operator for data
            defined at the vertices of a triangular mesh. The number of
            vertices of the triangular mesh is n_points.
        """
        mesh = self.to_spharapy()
        return mesh.laplacianmatrix(mode=mode)

    @classmethod
    def from_nx(cls, G: nx.Graph) -> Self:
        """Convert from netwrokx graph.

        Parameters
        ----------
        G : nx.Graph
            The graph to convert to a simplicial complex.

        Returns
        -------
        SimplicialComplex
            The simplicial complex.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge(1, 2, weight=2)
        >>> G.add_edge(3, 4, weight=4)
        >>> SC = tnx.SimplicialComplex.from_nx(G)
        >>> SC[(1, 2)]["weight"]
        2
        """
        return cls(G)

    def is_connected(self) -> bool:
        """Check if the simplicial complex is connected.

        Returns
        -------
        bool
            True if the simplicial complex is connected, False otherwise.

        Notes
        -----
        A simplicial complex is connected iff its 1-skeleton is connected.
        """
        return nx.is_connected(self.graph_skeleton())

    @classmethod
    def simplicial_closure_of_hypergraph(cls, H) -> Self:
        """Compute the simplicial complex closure of a hypergraph.

        Parameters
        ----------
        H : hyernetx hypergraph
            The hypergraph to compute the simplicial complex closure of.

        Returns
        -------
        SimplicialComplex
            Simplicial complex closure of the hypergraph.

        Examples
        --------
        >>> import hypernetx as hnx
        >>> hg = hnx.Hypergraph([[1, 2, 3, 4], [1, 2, 3]], static=True)
        >>> sc = tnx.SimplicialComplex.simplicial_closure_of_hypergraph(hg)
        >>> sc.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)])
        """
        edges = H.edges
        lst = [edges[e] for e in edges]
        return cls(lst)

    def to_cell_complex(self):
        """Convert a simplicial complex to a cell complex.

        Returns
        -------
        toponetx.classes.CellComplex
            The cell complex corresponding to this simplicial complex.

        Examples
        --------
        >>> c1 = tnx.Simplex((1, 2, 3))
        >>> c2 = tnx.Simplex((1, 2, 4))
        >>> c3 = tnx.Simplex((2, 5))
        >>> SC = tnx.SimplicialComplex([c1, c2, c3])
        >>> SC.to_cell_complex()
        """
        from toponetx.classes.cell_complex import CellComplex

        return CellComplex(
            simplex.elements for simplex in self.get_all_maximal_simplices()
        )

    def to_hypergraph(self) -> Hypergraph:
        """Convert a simplicial complex to a hypergraph.

        Returns
        -------
        Hypergraph
            The hypergraph corresponding to this simplicial complex.

        Examples
        --------
        >>> c1 = tnx.Simplex((1, 2, 3))
        >>> c2 = tnx.Simplex((1, 2, 4))
        >>> c3 = tnx.Simplex((2, 5))
        >>> SC = tnx.SimplicialComplex([c1, c2, c3])
        >>> SC.to_hypergraph()
        Hypergraph({'e0': [1, 2], 'e1': [1, 3], 'e2': [1, 4], 'e3': [2, 3], 'e4': [2, 4], 'e5': [2, 5], 'e6': [1, 2, 3], 'e7': [1, 2, 4]},name=)
        """
        hyperedges = [
            list(cell)
            for rank in range(1, self.dim + 1)
            for cell in self.skeleton(rank)
        ]
        return Hypergraph(hyperedges, static=True)

    def to_combinatorial_complex(self):
        """Convert a simplicial complex to a combinatorial complex.

        Returns
        -------
        toponetx.classes.CombinatorialComplex
            The combinatorial complex equivalent to this simplicial complex.

        Examples
        --------
        >>> c1 = tnx.Simplex((1, 2, 3))
        >>> c2 = tnx.Simplex((1, 2, 3))
        >>> c3 = tnx.Simplex((1, 2, 4))
        >>> SC = tnx.SimplicialComplex([c1, c2, c3])
        >>> CCC = SC.to_combinatorial_complex()
        """
        from toponetx.classes.combinatorial_complex import CombinatorialComplex

        CCC = CombinatorialComplex()
        for rank in range(1, self.dim + 1):
            for cell in self.skeleton(rank):
                CCC.add_cell(cell, rank=len(cell) - 1, **self[cell])
        return CCC

    def graph_skeleton(self) -> nx.Graph:
        """Return the graph-skeleton of this simplicial complex.

        The graph-skeleton consists of the 0 and 1-simplices (i.e., nodes and edges)
        of the simplicial complex.

        Returns
        -------
        nx.Graph
            The graph-skeleton of this simplicial complex.
        """
        G = nx.Graph()
        for node in self.skeleton(0):
            print("node", node, type(node))
            G.add_node(next(iter(node)), **self[node])
        for edge in self.skeleton(1):
            G.add_edge(*edge, **self[edge])
        return G

    def clone(self) -> Self:
        """Return a copy of the simplicial complex.

        The clone method by default returns an independent shallow copy of the
        simplicial complex. Use Python's `copy.deepcopy` for new containers.

        Returns
        -------
        SimplicialComplex
            A shallow copy of this simplicial complex.
        """
        return self.__class__(self.simplices)
