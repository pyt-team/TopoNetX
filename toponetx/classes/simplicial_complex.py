"""Class for creation and manipulation of simplicial complexes.

The class also supports attaching arbitrary attributes and data to cells.
"""

from collections.abc import Hashable, Iterable, Iterator
from itertools import chain, combinations
from warnings import warn

import networkx as nx
import numpy as np
from gudhi import SimplexTree
from hypernetx import Hypergraph
from scipy.sparse import coo_matrix, dok_matrix

from toponetx.classes.complex import Complex
from toponetx.classes.reportviews import NodeView, SimplexView
from toponetx.classes.simplex import Simplex
from toponetx.exception import TopoNetXError

__all__ = ["SimplicialComplex"]


class SimplicialComplex(Complex):
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

    Features
    --------
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
    simplices : list, optional
        list of maximal simplices that define the simplicial complex
    name : hashable, optional
        If None then a placeholder '' will be inserted as name
    kwargs : keyword arguments, optional
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

    >>> SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])

    TopoNetX is also compatible with NetworkX, allowing users to create a simplicial complex from a NetworkX graph.
    Existing node and edge attributes are copied to the simplicial complex:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, weight=4)
    >>> G.add_edges_from([(0, 3), (0, 4), (1, 4)])
    >>> SC = SimplicialComplex(simplices=G)
    >>> SC.add_simplex([1, 2, 3])
    >>> SC.simplices
    SimplexView([(0,), (1,), (3,), (4,), (2,), (0, 1), (0, 3), (0, 4), (1, 4), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    >>> SC[(0, 1)]["weight"]
    4
    """

    def __init__(self, simplices=None, name: str = "", **kwargs) -> None:
        super().__init__(name, **kwargs)

        self._simplex_set = SimplexView()

        if isinstance(simplices, nx.Graph):
            _simplices = {}
            for simplex, data in simplices.nodes(
                data=True
            ):  # `simplices` is a networkx graph
                _simplices[(simplex,)] = data
            for u, v, data in simplices.edges(data=True):
                _simplices[(u, v)] = data

            simplices = []
            for simplex, data in _simplices.items():
                s1 = Simplex(simplex, **data)
                simplices.append(s1)
            self.add_simplices_from(simplices)
        elif isinstance(simplices, Iterable):
            self.add_simplices_from(simplices)
        elif simplices is not None:
            raise TypeError(
                f"Input simplices must be given as Iterable, got {type(simplices)}."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of simplicial complex.

        (number of simplices[i], for i in range(0,dim(Sc))  )

        Returns
        -------
        tuple of ints
        """
        return self._simplex_set.shape

    @property
    def dim(self) -> int:
        """Dimension.

        This is the highest dimension of any simplex in the complex.
        """
        return self._simplex_set.max_dim

    @property
    def maxdim(self) -> int:
        """Maximum dimension.

        This is the highest dimension of any simplex in the complex
        """
        warn(
            "`SimplicialComplex.maxdim` is deprecated and will be removed in the future, use `SimplicialComplex.max_dim` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._simplex_set.max_dim

    @property
    def nodes(self):
        """Nodes."""
        return NodeView(self._simplex_set.faces_dict, cell_type=Simplex)

    @property
    def simplices(self) -> SimplexView:
        """Set of all simplices."""
        return self._simplex_set

    def is_maximal(self, simplex: Iterable) -> bool:
        """Check if simplex is maximal.

        Parameters
        ----------
        simplex : Iterable
            simplex to check

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If simplex is not in the simplicial complex.

        Examples
        --------
        >>> SC = SimplicialComplex([[1, 2, 3]])
        >>> SC.is_maximal([1, 2, 3])
        True
        >>> SC.is_maximal([1, 2])
        False
        """
        if simplex not in self:
            raise ValueError(f"Simplex {simplex} is not in the simplicial complex.")
        return self[simplex]["is_maximal"]

    def get_maximal_simplices_of_simplex(self, simplex):
        """Get maximal simplices of simplex."""
        return self[simplex]["membership"]

    def skeleton(self, rank):
        """Compute skeleton.

        Returns
        -------
        Set of simplices of dimension n.
        """
        if rank < len(self._simplex_set.faces_dict) and rank >= 0:
            return sorted(
                tuple(sorted(i)) for i in self._simplex_set.faces_dict[rank].keys()
            )
        if rank < 0:
            raise ValueError(f"input must be a postive integer, got {rank}")
        raise ValueError(f"input {rank} exceeds max dim")

    def __str__(self) -> str:
        """Return detailed string representation."""
        return f"Simplicial Complex with shape {self.shape} and dimension {self.dim}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SimplicialComplex(name='{self.name}')"

    def __len__(self) -> int:
        """Compute number of simplices.

        Returns
        -------
        int
            Number of vertices in the complex.
        """
        return len(self.skeleton(0))

    def __getitem__(self, simplex):
        """Get simplex."""
        if simplex in self:
            return self._simplex_set[simplex]
        else:
            raise KeyError("simplex is not in the simplicial complex")

    def __iter__(self) -> Iterator:
        """Iterate over all faces of the simplicial complex.

        Returns
        -------
        dict_keyiterator
        """
        return chain.from_iterable(self._simplex_set.faces_dict)

    def __contains__(self, item) -> bool:
        """Return boolean indicating if item is in self.face_set.

        Parameters
        ----------
        item : tuple, list
        """
        return item in self._simplex_set

    def _update_faces_dict_length(self, simplex) -> None:
        if len(simplex) > len(self._simplex_set.faces_dict):
            diff = len(simplex) - len(self._simplex_set.faces_dict)
            for _ in range(diff):
                self._simplex_set.faces_dict.append(dict())

    def _update_faces_dict_entry(self, face, simplex, maximal_faces) -> None:
        """Update faces dictionary entry.

        Parameters
        ----------
        face :  an iterable, typically a list, tuple, set or a Simplex
        simplex : an iterable, typically a list, tuple, set or a Simplex

        Notes
        -----
        the input `face` is a face of the input `simplex`.
        """
        face = frozenset(face)
        k = len(face)

        if face not in self._simplex_set.faces_dict[k - 1]:
            if k == len(simplex):
                self._simplex_set.faces_dict[k - 1][face] = {
                    "is_maximal": True,
                    "membership": set(),
                }
            else:
                self._simplex_set.faces_dict[k - 1][face] = {
                    "is_maximal": False,
                    "membership": {simplex},
                }
        else:
            if k != len(simplex):
                self._simplex_set.faces_dict[k - 1][face]["membership"].add(simplex)
                if self._simplex_set.faces_dict[k - 1][face]["is_maximal"]:
                    maximal_faces.add(face)
                    self._simplex_set.faces_dict[k - 1][face]["is_maximal"] = False
                else:
                    # make sure all children of previous maximal simplices do not have that membership anymore
                    self._simplex_set.faces_dict[k - 1][face][
                        "membership"
                    ] -= maximal_faces

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
            constrain the max dimension of faces
        max_dim : int, optional
            constrain the max dimension of faces

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

    def remove_maximal_simplex(self, simplex) -> None:
        """Remove maximal simplex from simplicial complex.

        Parameters
        ----------
        simplex : Iterable

        Raises
        ------
        KeyError
            If simplex is not in simplicial complex.
        ValueError
            If simplex is not maximal.

        Examples
        --------
        >>> SC = SimplicialComplex()
        >>> SC.add_simplex((1, 2, 3, 4), weight=1)
        >>> SC.add_simplex((1, 2, 3, 4, 5))
        >>> SC.remove_maximal_simplex((1, 2, 3, 4, 5))
        """
        if isinstance(simplex, Iterable):
            if not isinstance(simplex, Simplex):
                simplex_ = frozenset(simplex)
            else:
                simplex_ = simplex.elements
        if simplex_ in self._simplex_set.faces_dict[len(simplex_) - 1]:
            if self.is_maximal(simplex):
                del self._simplex_set.faces_dict[len(simplex_) - 1][simplex_]
                faces = Simplex(simplex_).faces
                for s in faces:
                    if len(s) == len(simplex_):
                        continue
                    else:
                        s = s.elements
                        self._simplex_set.faces_dict[len(s) - 1][s]["membership"] -= {
                            simplex_
                        }
                        if (
                            len(
                                self._simplex_set.faces_dict[len(s) - 1][s][
                                    "membership"
                                ]
                            )
                            == 0
                            and len(s) == len(simplex) - 1
                        ):
                            self._simplex_set.faces_dict[len(s) - 1][s][
                                "is_maximal"
                            ] = True

                if (
                    len(self._simplex_set.faces_dict[len(simplex_) - 1]) == 0
                    and len(simplex_) - 1 == self._simplex_set.max_dim
                ):
                    del self._simplex_set.faces_dict[len(simplex_) - 1]
                    self._simplex_set.max_dim = len(self._simplex_set.faces_dict) - 1

            else:
                raise ValueError(
                    "Only maximal simplices can be deleted, input simplex is not maximal"
                )
        else:
            raise KeyError("simplex is not a part of the simplicial complex")

    def remove_nodes(self, node_set: Iterable[Hashable]) -> None:
        """Remove the given nodes from the simplicial complex.

        Any simplices that become invalid due to the removal of nodes are also removed.

        Parameters
        ----------
        node_set : Iterable
            The nodes to be removed from the simplicial complex.

        Examples
        --------
        >>> SC = SimplicialComplex([(1, 2), (2, 3), (3, 4)])
        >>> SC.remove_nodes([1])
        >>> SC.simplices
        SimplexView([(2,), (3,), (4,), (2, 3), (3, 4)])
        """
        removed_simplices = set()
        for simplex in self:
            if any(node in simplex for node in node_set):
                removed_simplices.add(simplex)

        # Delete the simplices from largest to smallest. This way they are maximal when they are deleted.
        for simplex in sorted(removed_simplices, key=len, reverse=True):
            self.remove_maximal_simplex(simplex)

    def add_node(self, node: Hashable, **attr) -> None:
        """Add node to simplicial complex.

        Parameters
        ----------
        node : Hashable or Simplex
            a Hashable or singleton Simplex representing a node in a simplicial complex.
        """
        if not isinstance(node, Hashable):
            raise TypeError(f"Input node must be Hashable, got {type(node)} instead.")

        if isinstance(node, Simplex):
            if len(node) != 1:
                raise ValueError(
                    f"Input node must be a singleton Simplex, got {node} instead."
                )
            self.add_simplex(node, **attr)
        else:
            self.add_simplex([node], **attr)

    def add_simplex(self, simplex, **attr) -> None:
        """Add simplex to simplicial complex.

        Parameters
        ----------
        simplex : Hashable or Iterable or Simplex or str
            a Hashable or Iterable or Simplex in a simplicial complex.
        """
        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex = [simplex]
        if isinstance(simplex, str):
            simplex = [simplex]
        if isinstance(simplex, Iterable) or isinstance(simplex, Simplex):
            if not isinstance(simplex, Simplex):
                simplex_ = frozenset(simplex)
                if len(simplex_) != len(simplex):
                    raise ValueError("a simplex cannot contain duplicate nodes")
            else:
                simplex_ = simplex.elements
            self._update_faces_dict_length(simplex_)

            if (
                simplex_ in self._simplex_set.faces_dict[len(simplex_) - 1]
            ):  # simplex is already in the complex, just update the properties if needed
                self._simplex_set.faces_dict[len(simplex_) - 1][simplex_].update(attr)
                return

            if self._simplex_set.max_dim < len(simplex) - 1:
                self._simplex_set.max_dim = len(simplex) - 1

            numnodes = len(simplex_)
            maximal_faces = set()

            for r in range(numnodes, 0, -1):
                for face in combinations(simplex_, r):
                    self._update_faces_dict_entry(face, simplex_, maximal_faces)
            self._simplex_set.faces_dict[len(simplex_) - 1][simplex_].update(attr)
            if isinstance(simplex, Simplex):
                self._simplex_set.faces_dict[len(simplex_) - 1][simplex_].update(
                    simplex._properties
                )
            else:
                self._simplex_set.faces_dict[len(simplex_) - 1][simplex_].update(attr)

    def add_simplices_from(self, simplices) -> None:
        """Add simplices from iterable to simplicial complex."""
        for s in simplices:
            self.add_simplex(s)

    def get_cofaces(self, simplex, codimension):
        """Get cofaces of simplex.

        Parameters
        ----------
        simplex : list, tuple or simplex
            DESCRIPTION. the n simplex represented by a list of its nodes
        codimension : int
            DESCRIPTION. The codimension. If codimension = 0, all cofaces are returned

        Returns
        -------
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
        """Get star.

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

        Examples
        --------
        After computing some property of the simplex of a simplicial complex, you may want
        to assign a simplex attribute to store the value of that property for
        each simplex:

        >>> SC = SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> d = {(1, 2, 3): 'red', (1, 2, 4): 'blue'}
        >>> SC.set_simplex_attributes(d, name='color')
        >>> SC[(1, 2, 3)]['color']
        'red'

        If you provide a dictionary of dictionaries as the second argument,
        the entire dictionary will be used to update simplex attributes:

        >>> SC = SimplicialComplex()
        >>> SC.add_simplex([1, 3, 4])
        >>> SC.add_simplex([1, 2, 3])
        >>> SC.add_simplex([1, 2, 4])
        >>> d = {(1, 3, 4): {'color': 'red', 'attr2': 1}, (1, 2, 4): {'color': 'blue', 'attr2': 3}}
        >>> SC.set_simplex_attributes(d)
        >>> SC[(1, 3, 4)]['color']
        'red'

        Notes
        -----
        If the dict contains simplices that are not in `self.simplices`, they are
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

    def get_node_attributes(self, name: str):
        """Get node attributes from combinatorial complex.

        Parameters
        ----------
        name : str
           Attribute name.

        Returns
        -------
        Dictionary of attributes keyed by node.

        Examples
        --------
        >>> SC = SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> SC.set_simplex_attributes({1: "red", 2: "blue", 3: "black"}, name="color")
        >>> SC.get_node_attributes("color")
        {(1,): 'red', (2,): 'blue', (3,): 'black'}
        """
        return {tuple(n): self[n][name] for n in self.skeleton(0) if name in self[n]}

    def get_simplex_attributes(self, name: str, rank=None):
        """Get node attributes from simplical complex.

        Parameters
        ----------
        name : str
           Attribute name
        rank : int
            rank of the cell

        Returns
        -------
        Dictionary of attributes keyed by cell or k-cells of (rank=k) if rank is not None

        Examples
        --------
        >>> SC = SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> d={(1, 2): "red", (2, 3): "blue", (3, 4): "black"}
        >>> SC.set_simplex_attributes(d, name="color")
        >>> SC.get_simplex_attributes("color")
        {frozenset({1, 2}): 'red', frozenset({2, 3}): 'blue', frozenset({3, 4}): 'black'}
        """
        if rank is None:
            return {n: self[n][name] for n in self if name in self[n]}
        return {n: self[n][name] for n in self.skeleton(rank) if name in self[n]}

    @staticmethod
    def get_edges_from_matrix(matrix):
        """Get edges from matrix.

        Parameters
        ----------
        matrix : numpy or scipy array

        Returns
        -------
        edges : list of indices where the operator is not zero

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
        edges = zip(rows.tolist(), cols.tolist())
        return edges

    def incidence_matrix(
        self, rank, signed: bool = True, weight: str | None = None, index: bool = False
    ):
        """Compute incidence matrix of the simplicial complex.

        Getting the matrix that correpodnds to the boundary matrix of the input SC.

        Examples
        --------
        >>> SC = SimplicialComplex()
        >>> SC.add_simplex([1, 2, 3, 4])
        >>> SC.add_simplex([1, 2, 4])
        >>> SC.add_simplex([3, 4, 8])
        >>> B1 = SC.incidence_matrix(1)
        >>> B2 = SC.incidence_matrix(2)
        """
        if rank < 0:
            raise ValueError(f"input dimension d must be positive integer, got {rank}")
        if rank > self.dim:
            raise ValueError(
                f"input dimenion cannat be larger than the dimension of the complex, got {rank}"
            )

        if rank == 0:
            boundary = dok_matrix(
                (1, len(self._simplex_set.faces_dict[rank].items())), dtype=np.float32
            )
            boundary[0, 0 : len(self._simplex_set.faces_dict[rank].items())] = 1
            return boundary.tocsr()
        idx_simplices, idx_faces, values = [], [], []

        simplex_dict_d = {simplex: i for i, simplex in enumerate(self.skeleton(rank))}
        simplex_dict_d_minus_1 = {
            simplex: i for i, simplex in enumerate(self.skeleton(rank - 1))
        }
        for simplex, idx_simplex in simplex_dict_d.items():
            # for simplex, idx_simplex in self._simplex_set.faces_dict[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = frozenset(simplex).difference({left_out})
                idx_faces.append(simplex_dict_d_minus_1[tuple(sorted(face))])
        assert len(values) == (rank + 1) * len(simplex_dict_d)
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
                    simplex_dict_d_minus_1,
                    simplex_dict_d,
                    boundary,
                )
            else:
                return (
                    simplex_dict_d_minus_1,
                    simplex_dict_d,
                    abs(boundary),
                )
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def coincidence_matrix(
        self, rank, signed: bool = True, weight=None, index: bool = False
    ):
        """Compute coincidence matrix of the simplicial complex.

        This is also called the coboundary matrix.
        """
        if index:
            idx_faces, idx_simplices, boundary = self.incidence_matrix(
                rank, signed=signed, weight=weight, index=True
            )
            return idx_faces, idx_simplices, boundary.T
        else:
            return self.incidence_matrix(
                rank, signed=signed, weight=weight, index=False
            ).T

    def hodge_laplacian_matrix(
        self, rank, signed: bool = True, weight=None, index: bool = False
    ):
        """Compute hodge-laplacian matrix for the simplicial complex.

        Parameters
        ----------
        d : int
            Dimension of the Laplacian matrix.
        signed : bool, default=True
            Whether to return the signed or unsigned hodge laplacian.
            This is useful when one needs to obtain higher-order
            adjacency matrices from the hodge-laplacian
            typically higher-order adjacency matrices' entries are
            typically positive.
        weight : bool, default=False
        index : bool, default=False
            Indicates whether to return the indices that define the incidence matrix.

        Returns
        -------
        Laplacian : scipy.sparse.csr.csr_matrix
        when index is true:
            return also a list : list

        Examples
        --------
        >>> SC = SimplicialComplex()
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
            else:
                return L_hodge
        elif rank < self.dim:
            row, column, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_hodge = B_next @ B_next.transpose() + B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return column, L_hodge
            else:
                return L_hodge
        elif rank == self.dim:
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_hodge = B.transpose() @ B
            if not signed:
                L_hodge = abs(L_hodge)
            if index:
                return column, L_hodge
            else:
                return L_hodge
        else:
            raise ValueError(
                f"Rank should be larger than 0 and <= {self.dim} (maximal dimension simplices), got {rank}"
            )
        if not signed:
            L_hodge = abs(L_hodge)
        else:
            return abs(L_hodge)

    def normalized_laplacian_matrix(self, rank, weight=None):
        """Return the normalized hodge Laplacian matrix of G.

        The normalized hodge Laplacian is the matrix

        .. math::
            N_d = D^{-1/2} L_d D^{-1/2}

        where `L` is the simplicial complex Laplacian and `D` is the diagonal matrix of
        simplices of rank d.

        Parameters
        ----------
        rank : int
            Rank of the hodge laplacian matrix
        weight : str or None, optional (default='weight')
            The edge data key used to compute each value in the matrix.
            If None, then each edge has weight 1.

        Returns
        -------
        Scipy sparse matrix
            The normalized hodge Laplacian matrix.

        Examples
        --------
        >>> SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        >>> SC.normalized_laplacian_matrix(1)
        """
        import numpy as np
        import scipy as sp

        L_hodge = self.hodge_laplacian_matrix(rank)
        m, n = L_hodge.shape
        diags_ = abs(L_hodge).sum(axis=1)

        with sp.errstate(divide="ignore"):
            diags_sqrt = 1.0 / np.sqrt(diags_)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        diags_sqrt = sp.sparse.csr_array(
            sp.sparse.spdiags(diags_sqrt.T, 0, m, n, format="csr")
        )

        return sp.sparse.csr_matrix(diags_sqrt @ (L_hodge @ diags_sqrt))

    def up_laplacian_matrix(
        self, rank, signed: bool = True, weight: str | None = None, index: bool = False
    ):
        """Compute the up Laplacian matrix of the simplicial complex.

        Parameters
        ----------
        rank : int, rank of the up Laplacian matrix.

        signed : bool, is true return absolute value entry of the Laplacian matrix
                       this is useful when one needs to obtain higher-order
                       adjacency matrices from the hodge-laplacian
                       typically higher-order adjacency matrices' entries are
                       typically positive.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : boolean, optional, default False
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            depending on the input dimension

        Returns
        -------
        up Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with nodes,edges or cells used to index the hodge Laplacian matrix
            depending on the input dimension
        """
        if weight is not None:
            raise ValueError("`weight` is not supported in this version")

        if rank == 0:
            row, col, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            L_up = B_next @ B_next.transpose()
        elif rank < self.dim:
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
            is true return absolute value entry of the Laplacian matrix
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
            depending on the input dimension.

        Returns
        -------
        down Laplacian : scipy.sparse.csr.csr_matrix

        when index is true:
            return also a list : list
            list identifying rows with simplices used to index the hodge Laplacian matrix
            depending on the input dimension
        """
        weight = None  # this feature is not supported in this version

        if rank <= self.dim and rank > 0:
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

        The method takes a rank parameter, which is the rank of the simplicial complex,
        and two optional parameters: signed and weight. The signed parameter determines whether
        the adjacency matrix should be signed or unsigned, and the weight parameter allows for
        specifying weights for the edges in the adjacency matrix. The index parameter determines
        whether the method should return the matrix indices along with the adjacency matrix.

        Examples
        --------
        >>> SC = SimplicialComplex()
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
        self, rank, signed: bool = False, weight=None, index: bool = False
    ):
        """Compute the coadjacency matrix of the simplicial complex."""
        weight = None  # this feature is not supported in this version

        ind, L_down = self.down_laplacian_matrix(
            rank, signed=signed, weight=weight, index=True
        )
        L_down.setdiag(0)
        if not signed:
            L_down = abs(L_down)
        if index:
            return ind, L_down
        return L_down

    def add_elements_from_nx_graph(self, G) -> None:
        """Add elements from a networkx graph to self."""
        _simplices = []
        for edge in G.edges:
            _simplices.append(edge)
        for node in G.nodes:
            _simplices.append([node])

        self.add_simplices_from(_simplices)

    def restrict_to_simplices(self, cell_set, name: str = "") -> "SimplicialComplex":
        """Construct a simplicial complex using a subset of the simplices.

        Parameters
        ----------
        cell_set: iterable of hashables or simplices
            A subset of elements of the simplicial complex
        name: str, optional

        Returns
        -------
        SimplicialComplex
            New simplicial complex instance restricted to the simplices in `cell_set`

        Examples
        --------
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 4))
        >>> c3 = Simplex((1, 2, 5))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> SC1 = SC.restrict_to_simplices([c1, (2, 4)])
        >>> SC1.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (2, 3), (2, 4), (1, 2, 3)])
        """
        rns = []
        for cell in cell_set:
            if cell in self:
                rns.append(cell)

        return SimplicialComplex(simplices=rns, name=name)

    def restrict_to_nodes(self, node_set, name: str = ""):
        """Construct a new simplicial complex by restricting the simplices.

        The simplices are restricted to the nodes referenced by node_set.

        Parameters
        ----------
        node_set: iterable of hashables
            References a subset of elements of self.nodes
        name: str, optional
            The name of the new simplicial complex.

        Returns
        -------
        new Simplicial Complex : SimplicialComplex

        Examples
        --------
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 4))
        >>> c3 = Simplex((1, 2, 5))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> new_complex = SC.restrict_to_nodes([1, 2, 3, 4])
        >>> new_complex.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (1, 2, 3), (1, 2, 4)])
        """
        simplices = []
        node_set = set(node_set)
        for rank in range(1, self.dim + 1):
            for s in self.skeleton(rank):
                if set(s).issubset(node_set):
                    simplices.append(s)
        all_sim = simplices + list(
            [frozenset({i}) for i in node_set if i in self.nodes]
        )

        return SimplicialComplex(all_sim, name=name)

    def get_all_maximal_simplices(self):
        """Get all maximal simplices.

        Examples
        --------
        >>> c0 = Simplex((1, 2))
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 4))
        >>> c3 = Simplex((2, 5))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> SC.get_all_maximal_simplices()
        [(2, 5), (1, 2, 3), (1, 2, 4)]
        """
        maximals = []
        for s in self:
            if self.is_maximal(s):
                maximals.append(tuple(s))
        return maximals

    @classmethod
    def from_spharpy(cls, mesh) -> "SimplicialComplex":
        """Import from sharpy.

        Examples
        --------
        >>> import spharapy.trimesh as tm
        >>> import spharapy.spharabasis as sb
        >>> import spharapy.datasets as sd
        >>> mesh = tm.TriMesh([[0, 1, 2]], [[1., 0., 0.], [0., 2., 0.],
        ...                                        [0., 0., 3.]])

        >>> SC = SimplicialComplex.from_spharpy(mesh)
        """
        vertices = np.array(mesh.vertlist)
        SC = cls(mesh.trilist)

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

    @classmethod
    def from_gudhi(cls, tree: SimplexTree) -> "SimplicialComplex":
        """Import from gudhi.

        Examples
        --------
        >>> from gudhi import SimplexTree
        >>> tree = SimplexTree()
        >>> _ = tree.insert([1,2,3,5])
        >>> SC = SimplicialComplex.from_gudhi(tree)
        """
        SC = cls()
        for simplex, _ in tree.get_skeleton(tree.dimension()):
            SC.add_simplex(simplex)
        return SC

    @classmethod
    def from_trimesh(cls, mesh) -> "SimplicialComplex":
        """Import from trimesh.

        Examples
        --------
        >>> import trimesh
        >>> mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        >>> SC = SimplicialComplex.from_trimesh(mesh)
        >>> print(SC.nodes)
        >>> print(SC.simplices)
        >>> SC[(0)]['position']
        """
        SC = cls(mesh.faces)

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

    @classmethod
    def load_mesh(
        cls, file_path, process: bool = False, force: bool | None = None
    ) -> "SimplicialComplex":
        """Load a mesh.

        Parameters
        ----------
        file_path: str or pathlib.Path
            the source of the data to be loadeded

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

        Examples
        --------
        >>> SC = SimplicialComplex.load_mesh("C:/temp/stanford-bunny.obj")
        >>> SC.nodes
        """
        import trimesh

        mesh = trimesh.load_mesh(file_path, process=process, force=None)
        return cls.from_trimesh(mesh)

    def is_triangular_mesh(self) -> bool:
        """Check if the simplicial complex is a triangular mesh."""
        if self.dim <= 2:

            lst = self.get_all_maximal_simplices()
            for i in lst:
                if len(i) == 2:  # gas edges that are not part of a face
                    return False
            return True
        else:
            return False

    def to_trimesh(self, vertex_position_name: str = "position"):
        """Convert simplicial complex to trimesh object."""
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

    def to_spharapy(self, vertex_position_name: str = "position"):
        """Convert to sharapy.

        Examples
        --------
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
        laplacianmatrix : array, shape (n_vertices, n_vertices)
            Matrix, which contains the discrete laplace operator for data
            defined at the vertices of a triangular mesh. The number of
            vertices of the triangular mesh is n_points.
        """
        mesh = self.to_spharapy()
        return mesh.laplacianmatrix(mode=mode)

    @classmethod
    def from_nx_graph(cls, G) -> "SimplicialComplex":
        """Convert from netwrokx graph.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge(1, 2, weight=2)
        >>> G.add_edge(3, 4, weight=4)
        >>> SC = SimplicialComplex.from_nx_graph(G)
        >>> SC[(1, 2)]["weight"]
        2
        """
        return cls(G, name=G.name)

    def is_connected(self):
        """Check if the simplicial complex is connected.

        Notes
        -----
        A simplicial complex is connected iff its 1-skeleton G is connected.
        """
        G = nx.Graph()
        for edge in self.skeleton(1):
            edge = list(edge)
            G.add_edge(edge[0], edge[1])
        for node in self.skeleton(0):
            G.add_node(list(node)[0])
        return nx.is_connected(G)

    @classmethod
    def simplicial_closure_of_hypergraph(cls, H) -> "SimplicialComplex":
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
        >>> sc = SimplicialComplex.simplicial_closure_of_hypergraph(hg)
        >>> sc.simplices
        SimplexView([(1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (1, 2, 3, 4)])
        """
        edges = H.edges
        lst = []
        for e in edges:
            lst.append(edges[e])
        return cls(lst)

    def to_cell_complex(self):
        """Convert a simplicial complex to a cell complex.

        Examples
        --------
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 4))
        >>> c3 = Simplex((2, 5))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> SC.to_cell_complex()
        """
        from toponetx.classes.cell_complex import CellComplex

        return CellComplex(self.get_all_maximal_simplices())

    def to_hypergraph(self):
        """Convert a simplicial complex to a hyperG.

        Examples
        --------
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 4))
        >>> c3 = Simplex((2, 5))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> SC.to_hypergraph()
        Hypergraph({'e0': [1, 2], 'e1': [1, 3], 'e2': [1, 4], 'e3': [2, 3], 'e4': [2, 4], 'e5': [2, 5], 'e6': [1, 2, 3], 'e7': [1, 2, 4]},name=)
        """
        G = []
        for rank in range(1, self.dim + 1):
            edge = [list(cell) for cell in self.skeleton(rank)]
            G = G + edge
        return Hypergraph(G, static=True)

    def to_combinatorial_complex(self):
        """Convert a simplicial complex to a combinatorial complex.

        Parameters
        ----------
        dynamic: bool, optional, default is false
            when True returns DynamicCombinatorialComplex
            when False returns CombinatorialComplex

        Examples
        --------
        >>> c1 = Simplex((1, 2, 3))
        >>> c2 = Simplex((1, 2, 3))
        >>> c3 = Simplex((1, 2, 4))
        >>> SC = SimplicialComplex([c1, c2, c3])
        >>> CC = SC.to_combinatorial_complex()
        """
        from toponetx.classes.combinatorial_complex import CombinatorialComplex

        CC = CombinatorialComplex()
        for rank in range(1, self.dim + 1):
            for cell in self.skeleton(rank):
                CC.add_cell(cell, rank=len(cell) - 1, **self[cell])
        return CC

    def clone(self) -> "SimplicialComplex":
        """Return a copy of the simplicial complex.

        The clone method by default returns an independent shallow copy of the simplicial complex. Use Python’s
        `copy.deepcopy` for new containers.

        Returns
        -------
        SimplicialComplex
        """
        return SimplicialComplex(self.simplices, name=self.name)
