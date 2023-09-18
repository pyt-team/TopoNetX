"""Path complex."""
from collections.abc import Hashable, Iterable, Iterator, Sequence
from itertools import chain
from typing import Any

import networkx as nx
import numpy as np
import scipy as sp
from hypernetx import Hypergraph
from networkx.classes.reportviews import EdgeView, NodeView

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.complex import Complex
from toponetx.classes.path import Path
from toponetx.classes.reportviews import PathView

__all__ = ["PathComplex"]


class PathComplex(Complex):
    """A class representing a path complex based on simple paths as proposed in (https://arxiv.org/abs/2308.06838). The original path complex is defined in (https://arxiv.org/pdf/1207.2834.pdf).

    A path complex contains elementary p-paths that span the space of simple paths. Path complexes are a topological structure whose
    building blocks are paths, which are essentially different from simplicial complexes and cell complexes. If certain conditions are met, path complexes can generalize
    simplicial complexes.

    For example, a triangle with three vertices 1, 2, and 3 can be represented as a simplicial complex (1, 2, 3). Similarly, it can be represented as a path complex (1, 2, 3) whose
    boundary 1-paths are (1, 2), (2, 3), and (1, 3). Another example is a simple path (1, 2, 3). In this case, we cannot lift the simple path to a 2-dimensional simplicial complex, as
    triangle does not exist. However, we can still represent the simple path as a path complex (1, 2, 3) whose boundary 1-paths are (1, 2) and (2, 3) (1-path (1,3) does not exist).

    Parameters
    ----------
    paths : nx.Graph or Iterable[Sequence[Hashable]]
        The paths in the path complex. If a graph is provided, the path complex will be constructed from the graph, and allowed paths are automatically computed.
    name : str, optional
        A name for the path complex.
    reserve_sequence_order : bool, default=False
        If True, reserve the order of the sub-sequence of nodes in the elementary p-path. Else, the sub-sequence of nodes in the elementary p-path will
        be reversed if the first index is larger than the last index.
    allowed_paths : Iterable[tuple[Hashable]], optional
        An iterable of allowed boundaries.
        If None and the input is a Graph, allowed paths are automatically computed by enumerating all simple paths in the graph whose length is less than or equal to max_rank.
        If None and the input is not a Graph, allowed paths contain the input paths, their truncated subsequences (sub-sequences where the first
        or the last index is omitted), and any sub-sequences of the truncated subsequences in a recursive manner.
    max_rank : int, default=3
        The maximal length of a path in the path complex.
    attr: keyword arguments, optional
        Additional attributes to be associated with the path complex.

    Notes
    -----
    - By the definition established by (https://arxiv.org/pdf/1207.2834.pdf), a path complex P is a non-empty collection of elementary p-paths such that for any sequence
    of vertices in a finite non-empty set V that belong to P, the truncated sequence of vertices (which is sometimes referred to as obvious boundaries) also belongs to P.
    The truncated sequences are sub-sequences of the original elementary p-path where the first or the last index is omitted. For instance, if we have an elementary p-path (1, 2, 3),
    the truncated sequences are (1,2) and (2,3).
    - Our path complex is different from the path complex defined in (https://arxiv.org/pdf/1207.2834.pdf). In our path complex, elementary p-paths span the space of simple paths.
    The path complex originally proposed has elementary p-paths that span the space of boundary-invariant paths.
    - The path complex is a simplicial complex if certain conditions are met (https://arxiv.org/pdf/1207.2834.pdf).

    Examples
    --------
    >>> PC = PathComplex([(1, 2, 3)])
    >>> PC.paths
    PathView([(1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3)])
    >>> PC.add_paths_from([(1, 2, 4), (1, 2, 5), (4,5)])
    >>> PC.paths
    PathView([(1,), (2,), (3,), (4,), (5,), (1, 2), (2, 3), (2, 4), (2, 5), (4, 5), (1, 2, 3), (1, 2, 4), (1, 2, 5)])
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    >>> PC = PathComplex(G)
    >>> PC.paths
    PathView([(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 3, 2), (1, 2, 3), (2, 1, 3)])
    """

    def __init__(
        self,
        paths: nx.Graph | Iterable[Sequence[Hashable]] = None,
        name: str = "",
        reserve_sequence_order: bool = False,
        allowed_paths: Iterable[tuple[Hashable]] = None,
        max_rank: int = 3,
        **kwargs,
    ) -> None:

        super().__init__(name=name, **kwargs)

        self._path_set = PathView()
        self._G = nx.Graph()
        self._reserve_sequence_order = reserve_sequence_order
        if allowed_paths is not None:
            if len(allowed_paths) > 0:
                for i in range(
                    len(allowed_paths)
                ):  # make sure list does not contain list
                    if isinstance(allowed_paths[i], list):
                        allowed_paths[i] = tuple(allowed_paths[i])
                self._allowed_paths = set(allowed_paths)
            else:
                self._allowed_paths = set()
        else:
            self._allowed_paths = set()

        if isinstance(paths, nx.Graph):
            # compute allowed_paths in order to construct boundary incidence matrix/adj matrix.
            self._G = paths
            if self._allowed_paths is not None:
                self._allowed_paths.update(
                    self.compute_allowed_paths(
                        paths,
                        reserve_sequence_order=reserve_sequence_order,
                        max_rank=max_rank,
                    )
                )

            # get feature of nodes and edges if available
            for path, data in paths.nodes(data=True):
                self.add_path(path, **data)
            for u, v, data in paths.edges(
                data=True
            ):  # so far, path complex only supports undirected graph
                if (str(u) > str(v)) and not reserve_sequence_order:
                    u, v = v, u
                self.add_path((u, v), **data)

            # add all simple paths
            self.add_paths_from(self._allowed_paths)

        elif isinstance(paths, list) or isinstance(paths, tuple):
            if len(paths) > 0:
                if (
                    isinstance(paths[0], int)
                    or isinstance(paths[0], str)
                    or isinstance(paths[0], list)
                ):
                    paths = [tuple(path) for path in paths]
            self.add_paths_from(set(paths))
        elif paths is not None:
            raise TypeError(
                "Input paths must be a graph or an iterable of paths as lists or tuples."
            )

    def add_paths_from(self, paths: Iterable[Sequence[Hashable] | Path]) -> None:
        """
        Add elementary paths from an iterable of elementary paths.

        An elementary p-path is a sequence of nodes (n1, ..., np) where p is the length of the sequence. In a path complex,
        for every elementary p-path in the path complex, their truncated sequences (n2, ..., np) and (n1, ..., np-1) are also in the path complex.

        Parameters
        ----------
        paths : Iterable[Sequence[Hashable] or Path]
            an iterable of paths as lists, tuples, or Path objects.

        Examples
        --------
        >>> PC = PathComplex([(1, 2, 3)])
        >>> PC.paths
        PathView([(1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3)])
        >>> PC.add_paths_from([(1, 2, 4), (1, 2, 5), (4,5)])
        >>> PC.paths
        PathView([(1,), (2,), (3,), (4,), (5,), (1, 2), (2, 3), (2, 4), (2, 5), (4, 5), (1, 2, 3), (1, 2, 4), (1, 2, 5)])
        """
        if isinstance(paths, Hashable):
            raise TypeError("Paths must be a iterable of paths as lists, tuples.")
        paths_clone = paths.copy()
        for p in paths_clone:
            self.add_path(p)

    def add_path(self, path: Hashable | Sequence[Hashable] | Path, **attr) -> None:
        """
        Add an elementary path to the path complex.

        An elementary p-path is a sequence of nodes (n1, ..., np) where p is the length of the sequence. In a path complex,
        for every elementary p-path in the path complex, their truncated sequences (n2, ..., np) and (n1, ..., np-1) are also in the path complex.

        This method automatically initializes any obvious sub-paths (sub-paths where the first or last index is omitted) of the elementary path if not available.
        In order to add non-obvious sub-paths, manually add the sub-paths.

        Parameters
        ----------
        path : Hashable or Sequence[Hashable] or Path
            a Hashable or Sequence[Hashable] or Path representing a path in a path complex.
        attr : keyword arguments, optional

        Examples
        --------
        >>> PC = PathComplex([(1, 2, 3)])
        >>> PC.paths
        PathView([(1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3)])
        >>> PC.add_path((1, 2, 4))
        >>> PC.paths
        PathView([(1,), (2,), (3,), (4,), (1, 2), (2, 3), (2, 4), (1, 2, 3), (1, 2, 4)])

        """
        new_paths = set()
        if isinstance(path, int) or isinstance(path, str):
            path = [
                path,
            ]
        if isinstance(path, list) or isinstance(path, tuple) or isinstance(path, Path):
            if not isinstance(path, Path):  # path is a list or tuple
                path_ = tuple(path)
                if len(path) != len(set(path)):
                    raise ValueError(
                        "An elementary p-path cannot contain duplicate nodes."
                    )
                if (
                    len(path_) > 1
                    and str(path_[0]) > str(path_[-1])
                    and not self._reserve_sequence_order
                ):
                    raise ValueError(
                        "An elementary p-path must have the first index smaller than the last index, got {}".format(
                            path
                        )
                    )
            else:  # path is a Path object
                path_ = path.elements
            self._update_faces_dict_length(
                path_
            )  # add dict corresponding to the path dimension

            if (
                self._path_set.max_dim < len(path_) - 1
            ):  # update max dimension for PathView()
                self._path_set.max_dim = len(path_) - 1

            if (
                path_ in self._path_set.faces_dict[len(path_) - 1]
            ):  # path is already in the complex, just update the properties if needed
                self._update_attributes(path, **attr)
                return
            else:
                # update sub-paths
                for length in range(len(path_), 0, -1):
                    for i in range(0, len(path_) - length + 1):
                        sub_path = path_[i : i + length]
                        if not self._reserve_sequence_order and str(sub_path[0]) > str(
                            sub_path[-1]
                        ):
                            sub_path = sub_path[::-1]
                        sub_path = tuple(sub_path)

                        # add to _G
                        if len(sub_path) == 1:
                            self._G.add_node(sub_path[0])
                        elif len(sub_path) == 2:
                            self._G.add_edge(sub_path[0], sub_path[1])

                        # expand _path_set if necessary. keep track of newly added paths to expend _allowed_paths
                        new_path = self._update_faces_dict_entry(sub_path)
                        if new_path is not None:
                            new_paths.add(new_path)
                # update allowed paths
                if len(new_paths) > 0:
                    self._allowed_paths.update(new_paths)

                self._update_attributes(path, **attr)

    @property
    def dim(self) -> int:
        """Dimension.

        Returns
        -------
        int
            This is the highest dimension of any elementary p-path in the complex.
        """
        return self._path_set.max_dim

    @property
    def nodes(self) -> NodeView:
        """Nodes.

        Returns
        -------
        NodeView
            A view of all nodes in the path complex.
        """
        return self._G.nodes

    @property
    def edges(self) -> EdgeView:
        """Edges.

        Returns
        -------
        EdgeView
            A view of all edges in the path complex.
        """
        return self._G.edges

    @property
    def paths(self) -> PathView:
        """
        Set of all elementary p-paths.

        Returns
        -------
        PathView
            A view of all elementary p-paths in the path complex.
        """
        return self._path_set

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of path complex.

        (number of elementary p-paths for each dimension d, for d in range(0,dim(Pc)))

        Returns
        -------
        tuple of ints
        """
        return self._path_set.shape

    def clone(self) -> "PathComplex":
        """Return a copy of the path complex.

        The clone method by default returns an independent shallow copy of the path complex. Use Python’s
        `copy.deepcopy` for new containers.

        Returns
        -------
        PathComplex
        """
        return PathComplex(list(self.paths), name=self.name)

    def skeleton(self, rank: int) -> set[tuple[Hashable]]:
        """Compute skeleton.

        Returns
        -------
        set[tuple[Hashable]]
            Set of elementary p-paths of dimension specified by `rank`.
        """
        if rank < len(self._path_set.faces_dict) and rank >= 0:
            tmp = (path for path in self._path_set.faces_dict[rank].keys())
            return sorted(
                tmp, key=lambda x: tuple(map(str, x))
            )  # lexicographic comparison
        if rank < 0:
            raise ValueError(f"input must be a postive integer, got {rank}")
        raise ValueError(f"input {rank} exceeds max dim")

    def add_node(self, node: Hashable | Path, **attr) -> None:
        """Add node to the path complex.

        Parameters
        ----------
        node : Hashable or Path
            a Hashable or singleton Path representing a node in a path complex.
        """
        if not isinstance(node, Hashable):
            raise TypeError(f"Input node must be Hashable, got {type(node)} instead.")

        if isinstance(node, Path):
            if len(node) != 1:
                raise ValueError(
                    f"Input node must be a singleton Path, got {node} instead."
                )
            self.add_path(node, **attr)
        else:
            self.add_path([node], **attr)

    def remove_nodes(self, node_set: Iterable[Hashable]) -> None:
        """
        Remove nodes from the path complex.

        Parameters
        ----------
        node_set : Iterable[Hashable]
            An iterable of nodes to be removed.
        """
        removed_paths = set()
        for path in self:  # iterate over all paths
            if any(
                node in path for node in node_set
            ):  # if any node in node_set is in the path, remove the path
                removed_paths.add(path)

        for path in removed_paths:
            self._remove_path(path)

        self._G.remove_nodes_from(node_set)

    def incidence_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute incidence matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the incidence matrix.
        signed : bool, default=True
            If True, return signed incidence matrix. Else, return absolute incidence matrix.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            If True, return incidence matrix with indices. Else, return incidence matrix without indices.

        Returns
        -------
        If `index` is True, return a tuple of (idx_p_minus_1, idx_p, incidence_matrix).
        If `index` is False, return incidence_matrix.
        """
        if rank < 0:
            raise ValueError(f"input dimension d must be positive integer, got {rank}")
        if rank > self.dim:
            raise ValueError(
                f"input dimenion cannat be larger than the dimension of the complex, got {rank}"
            )
        if rank == 0:
            boundary = sp.sparse.lil_matrix((0, len(self.nodes)))
            if index:
                node_index = {
                    tuple([node]): i
                    for i, node in enumerate(sorted(self.nodes, key=lambda x: str(x)))
                }
                return {}, node_index, abs(boundary.tocoo())
            else:
                return abs(boundary.tocoo())
        else:
            idx_p_minus_1, idx_p, values = [], [], []
            path_minus_1_dict = {
                path: i for i, path in enumerate(self.skeleton(rank - 1))
            }  # path2idx dict
            path_dict = {
                path: i for i, path in enumerate(self.skeleton(rank))
            }  # path2idx dict
            for path, idx_path in path_dict.items():
                for i, _ in enumerate(path):
                    boundary_path = path[0:i] + path[(i + 1) :]
                    if not self._reserve_sequence_order and str(boundary_path[0]) > str(
                        boundary_path[-1]
                    ):
                        boundary_path = boundary_path[::-1]
                    boundary_path = tuple(boundary_path)
                    if boundary_path in self._allowed_paths:
                        idx_p_minus_1.append(path_minus_1_dict[boundary_path])
                        idx_p.append(idx_path)
                        values.append((-1) ** i)
            boundary = sp.sparse.coo_matrix(
                (values, (idx_p_minus_1, idx_p)),
                dtype=np.float32,
                shape=(
                    len(path_minus_1_dict),
                    len(path_dict),
                ),
            )
        if index:
            if signed:
                return (
                    path_minus_1_dict,
                    path_dict,
                    boundary,
                )
            else:
                return (
                    path_minus_1_dict,
                    path_dict,
                    abs(boundary),
                )
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def up_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute up laplacian matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the up laplacian matrix.
        signed : bool, default=True
            If True, return signed up laplacian matrix. Else, return absolute up laplacian matrix.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            If True, return up laplacian matrix with indices. Else, return up laplacian matrix without indices.

        Returns
        -------
        If `index` is True, return a tuple of (idx_p, up_laplacian_matrix).
        If `index` is False, return up_laplacian_matrix.
        """
        if weight is not None:
            raise ValueError("Weighted Laplacian is not supported in this version.")

        if 0 <= rank < self.dim:
            row, col, B_next = self.incidence_matrix(
                rank + 1, weight=weight, index=True
            )
            L_up = B_next @ B_next.transpose()
        else:
            raise ValueError(
                f"Rank should larger than 0 and <= {self.dim - 1} (maximal dimension-1), got {rank}."
            )
        if not signed:
            L_up = abs(L_up)

        if index:
            return row, L_up.tolil()
        else:
            return L_up.tolil()

    def down_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute down laplacian matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the down laplacian matrix.
        signed : bool, default=True
            If True, return signed down laplacian matrix. Else, return absolute down laplacian matrix.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            If True, return down laplacian matrix with indices. Else, return down laplacian matrix without indices.

        Returns
        -------
        If `index` is True, return a tuple of (idx_p, down_laplacian_matrix).
        If `index` is False, return down_laplacian_matrix.
        """
        if weight is not None:
            raise ValueError("Weighted Laplacian is not supported in this version.")

        if rank <= self.dim and rank > 0:
            row, column, B = self.incidence_matrix(rank, weight=weight, index=True)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {self.dim} (maximal dimension), got {rank}."
            )
        if not signed:
            L_down = abs(L_down)
        if index:
            return row, L_down.tolil()
        else:
            return L_down.tolil()

    def hodge_laplacian_matrix(
        self,
        rank: int,
        signed: bool = True,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute Hodge Laplacian matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the Hodge Laplacian matrix.
        signed : bool, default=False
            If True, return signed Hodge Laplacian matrix. Else, return absolute Hodge Laplacian matrix.
        weight : str, optional
            If not given, all nonzero entries are 1.
        index : bool, default=False
            If True, return Hodge Laplacian matrix with indices. Else, return Hodge Laplacian matrix without indices.

        Returns
        -------
        Laplacian: scipy.sparse.lil_matrix
        when index is True, return a tuple of (idx_p, Laplacian)
        when index is False, return Laplacian
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

    def adjacency_matrix(
        self,
        rank: int,
        signed: bool = False,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute adjacency matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the adjacency matrix.
        signed : bool, default=False
            If True, return signed adjacency matrix. Else, return absolute adjacency matrix.
        index : bool, default=False
            If True, return adjacency matrix with indices. Else, return adjacency matrix without indices.

        Returns
        -------
        If `index` is True, return a tuple of (idx_p, adjacency_matrix).
        If `index` is False, return adjacency_matrix.
        """
        ind, L_up = self.up_laplacian_matrix(rank, signed=signed, index=True)
        L_up.setdiag(0)

        if not signed:
            L_up = abs(L_up)
        if index:
            return ind, L_up
        return L_up

    def coadjacency_matrix(
        self,
        rank: int,
        signed: bool = False,
        weight: str | None = None,
        index: bool = False,
    ) -> tuple[dict, sp.sparse.lil_matrix] | sp.sparse.lil_matrix:
        """
        Compute coadjacency matrix of the path complex.

        Parameters
        ----------
        rank : int
            The dimension of the coadjacency matrix.
        signed : bool, default=False
            If True, return signed coadjacency matrix. Else, return absolute coadjacency matrix.
        index : bool, default=False
            If True, return coadjacency matrix with indices. Else, return coadjacency matrix without indices.

        Returns
        -------
        If `index` is True, return a tuple of (idx_p, coadjacency_matrix).
        If `index` is False, return coadjacency_matrix.
        """
        ind, L_down = self.down_laplacian_matrix(rank, signed=signed, index=True)
        L_down.setdiag(0)
        if not signed:
            L_down = abs(L_down)
        if index:
            return ind, L_down
        return L_down

    def to_hypergraph(self) -> Hypergraph:
        """Return a hypergraph representation of the path complex."""
        G = []
        for rank in range(1, self.dim + 1):
            edge = [list(path) for path in self.skeleton(rank)]
            G = G + edge
        return Hypergraph(G, static=True)

    def restrict_to_nodes(self, node_set: Iterable[Hashable], name: str = ""):
        """Return a new path complex restricted to a subset of nodes."""
        if len(node_set) == 0:
            raise ValueError("Input node_set cannot be empty.")
        node_set = set(node_set)
        new_paths = []
        for path in self:
            if all(node in node_set for node in path):
                new_paths.append(path)
        return PathComplex(new_paths, name=name)

    def restrict_to_paths(self, path_set: Iterable[Sequence[Hashable]], name: str = ""):
        """Return a new path complex restricted to a subset of paths."""
        new_path_set = []
        if len(path_set) == 0:
            raise ValueError("Input path_set cannot be empty.")
        else:
            for path in path_set:
                new_path_set.append(tuple(path))
        new_path_set = set(new_path_set)
        new_paths = []
        for path in self:
            if path in new_path_set:
                new_paths.append(path)
        return PathComplex(new_paths, name=name)

    def get_node_attributes(self, name: str) -> dict[tuple[Hashable], Any]:
        """Get node attributes from a path complex.

        Parameters
        ----------
        name : str
            The name of the node attribute.

        Returns
        -------
        dict[tuple[Hashable], Any]
            A dictionary of node and its associated value for the given attribute name.

        Examples
        --------
        >>> PX = PathComplex()
        >>> PX.add_node(0)
        >>> PX.add_node(1, heat=55)
        >>> PX.add_node(2, heat=66)
        >>> PX.add_node(3, color="red")
        >>> PX.add_node(2, color="blue")
        >>> PX.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        >>> PX.get_node_attributes("heat")
        {(1, ): 55, (2, ): 66}
        >>> PX.get_node_attributes("color")
        {(2, ): "blue", (3, ): "red"}
        """
        return {tuple(n): self[n][name] for n in self.skeleton(0) if name in self[n]}

    def set_node_attributes(
        self,
        values: dict[Sequence[Hashable], Hashable | dict[Hashable, Any]],
        name: str | None = None,
    ) -> None:
        """
        Set node attributes for a path complex.

        Parameters
        ----------
        values : dict[Sequence[Hashable], Hashable | dict[Hashable, Any]]
            A dictionary of node and its associated attribute values.
        name : str
            The name of the node attribute.

        Examples
        --------
        >>> PX = PathComplex()
        >>> PX.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        >>> PX.set_node_attributes({(1, ): {"heat": 55, "color": "red"}, (2, ): {"heat": 66, "color": "blue"}})
        >>> PX[1]
        {'heat': 55, 'color': 'red'}
        >>> PX[2]
        {'heat': 66, 'color': 'blue'}
        >>> PX.set_node_attributes({(1, ): 58, (2, ): 60}, name="heat")
        >>> PX[1]
        {'heat': 58, 'color': 'red'}
        >>> PX[2]
        {'heat': 60, 'color': 'blue'}
        """
        if name is None:
            for node, value in values.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        self[node][k] = v
                        if isinstance(node, tuple) or isinstance(node, list):
                            if len(node) == 1:
                                self._G.nodes[node[0]][k] = v
                            else:
                                raise ValueError("Input node must be a singleton.")
                        elif isinstance(node, int) or isinstance(node, str):
                            self._G.nodes[node][k] = v
                else:
                    raise TypeError(
                        "Input values must be a dictionary of node and its associated attribute values."
                    )
        else:
            for node, value in values.items():
                self[node][name] = value
                if isinstance(node, tuple) or isinstance(node, list):
                    if len(node) == 1:
                        self._G.nodes[node[0]][name] = value
                    else:
                        raise ValueError("Input node must be a singleton.")
                elif isinstance(node, int) or isinstance(node, str):
                    self._G.nodes[node][name] = value

    def get_edge_attributes(self, name: str) -> dict[tuple[Hashable], Hashable]:
        """Get edge attributes from a path complex.

        Parameters
        ----------
        name : str
            The name of the edge attribute.

        Returns
        -------
        dict[tuple[Hashable], Hashable]
            A dictionary of edge and its associated value for the given attribute name.

        Examples
        --------
        >>> PX = PathComplex()
        >>> PX.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        >>> PX.add_path([0, 1], weight=32)
        >>> PX.add_path([1, 2], weight=98)
        >>> PX.add_path([1, 3], color="red")
        >>> PX.add_path([2, 3], color="blue")
        >>> PX.get_edge_attributes("weight")
        {(0, 1): 32, (1, 2): 98}
        >>> PX.get_edge_attributes("color")
        {(1, 3): "red", (2, 3): "blue"}
        """
        return {tuple(e): self[e][name] for e in self.skeleton(1) if name in self[e]}

    def set_edge_attributes(
        self,
        values: dict[Sequence[Hashable], Hashable | dict[Hashable, Any]],
        name: str | None = None,
    ) -> None:
        """
        Set edge attributes for a path complex.

        Parameters
        ----------
        values : dict[Sequence[Hashable], Hashable | dict[Hashable, Any]]
            A dictionary of edge and its associated attribute values.
        name : str
            The name of the edge attribute.

        Examples
        --------
        >>> PX = PathComplex()
        >>> PX.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        >>> PX.add_path([0, 1], weight=32)
        >>> PX.add_path([1, 2], weight=98)
        >>> PX.add_path([1, 3], color="red")
        >>> PX.add_path([2, 3], color="blue")
        >>> PX.get_edge_attributes("weight")
        {(0, 1): 32, (1, 2): 98}
        >>> PX.set_edge_attributes({(0, 1): 33}, name="weight")
        >>> PX.get_edge_attributes("weight")
        {(0, 1): 33, (1, 2): 98}
        >>> PX.set_edge_attributes({(1, 3): {"weight": 55, "color": "yellow"}, (2, 3): {"weight": 66, "color": "blue"}})
        >>> PX[1, 3]
        {'color': 'yellow', 'weight': 55}
        >>> PX[2, 3]
        {'color': 'blue', 'weight': 66}
        """
        if name is None:
            for edge, value in values.items():
                if len(edge) != 2:
                    raise ValueError("Input edge must be a pair.")
                if isinstance(value, dict):
                    for k, v in value.items():
                        self[edge][k] = v
                        if isinstance(edge, tuple) or isinstance(edge, list):
                            self._G.edges[edge[0], edge[1]][k] = v
                else:
                    raise TypeError(
                        "Input values must be a dictionary of edge and its associated attribute values."
                    )
        else:
            for edge, value in values.items():
                if len(edge) != 2:
                    raise ValueError("Input edge must be a pair.")
                self[edge][name] = value
                if isinstance(edge, tuple) or isinstance(edge, list):
                    self._G.edges[edge[0], edge[1]][name] = value

    def _remove_path(self, path: tuple[Hashable]) -> None:
        del self._path_set.faces_dict[len(path) - 1][path]
        self._allowed_paths.remove(path)
        if (
            len(self._path_set.faces_dict[len(path) - 1]) == 0
            and self._path_set.max_dim == len(path) - 1
        ):  # update max dimension for PathView() if highest dimension is empty
            self._path_set.max_dim -= 1

    def _update_faces_dict_length(self, path: tuple[Hashable]) -> None:
        if len(path) > len(self._path_set.faces_dict):
            diff = len(path) - len(self._path_set.faces_dict)
            for _ in range(diff):
                self._path_set.faces_dict.append(dict())

    def _update_faces_dict_entry(self, path: tuple[Hashable]):
        dim = len(path) - 1
        if path not in self._path_set.faces_dict[dim]:  # Not in faces_dict
            self._path_set.faces_dict[dim][path] = dict()
            return path
        else:
            return None

    def _update_attributes(self, path, **attr):
        if not isinstance(path, Path):  # path is a list or tuple
            path_ = tuple(path)
        else:  # path is a Path object
            path_ = path.elements
        if isinstance(path, Path):  # update attributes for PathView() and _G
            self._path_set.faces_dict[len(path_) - 1][path_].update(path._properties)
            if len(path_) == 1:
                self._G.add_node(path_[0], **path._properties)
            elif len(path_) == 2:
                self._G.add_edge(path_[0], path_[1], **path._properties)
        else:
            self._path_set.faces_dict[len(path_) - 1][path_].update(attr)
            if len(path_) == 1:
                self._G.add_node(path_[0], **attr)
            elif len(path_) == 2:
                self._G.add_edge(path_[0], path_[1], **attr)

    def __contains__(self, item: Sequence[Hashable] | Hashable) -> bool:
        """Return boolean indicating if item is in self._path_set.

        Parameters
        ----------
        item : Sequence[Hashable] | Hashable
        """
        return item in self._path_set

    def __getitem__(self, item: Sequence[Hashable] | Hashable):
        """
        Get the elementary p-path.

        Parameters
        ----------
        item : Sequence[Hashable] | Hashable
            An elementary p-path or a node in the path complex.
        """
        if item in self:
            return self._path_set[item]
        else:
            raise KeyError("The elementary p-path is not in the path complex")

    def __iter__(self) -> Iterator:
        """Iterate over all faces of the path complex.

        Returns
        -------
        dict_keyiterator
        """
        return chain.from_iterable(self._path_set.faces_dict)

    def __len__(self) -> int:
        """Return the number of elementary p-paths in the path complex.

        Returns
        -------
        int
        """
        return len(list(self.__iter__()))

    def __str__(self) -> str:
        """Return detailed string representation."""
        return f"Path Complex with shape {self.shape} and dimension {self.dim}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PathComplex(name='{self.name}')"

    @staticmethod
    def compute_allowed_paths(
        graph: nx.Graph, reserve_sequence_order: bool = False, max_rank: int = 3
    ) -> set[list | tuple]:
        """
        Compute allowed paths from a graph.

        Allowed paths are automatically computed by enumerating all simple paths in the graph whose length is less than
        or equal to max_rank. Allowed paths allow us to restrict the boundary of an elementary p-path to only sequences that exist in the graph.

        Parameters
        ----------
        graph : nx.Graph
            A graph.
        reserve_sequence_order : bool, default=False
            If True, reserve the order of the sub-sequence of nodes in the elmentary p-path.
            Else, the sub-sequence of nodes in the elementary p-path will be reversed if the first index is larger than the last index.
        max_rank : int, default=3
            The maximal length of a path in the path complex.

        Returns
        -------
        set[list | tuple]
            A set of allowed paths.

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1, 2), (2, 3), (1, 3), (0, 1)])
        >>> allowed_paths = PathComplex.compute_allowed_paths(G, max_rank = 2)
        >>> allowed_paths
        {(0, 1), (1, 3), (1, 2), (2,), (1, 3, 2), (0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 1, 3), (2, 3), (1,), (0,), (3,)}
        """
        allowed_paths = list()
        all_nodes_list = list(
            tuple([node]) for node in sorted(graph.nodes, key=lambda x: str(x))
        )
        all_edges_list = list()
        for edge in graph.edges:
            if not reserve_sequence_order and str(edge[0]) > str(edge[1]):
                edge = edge[::-1]
            all_edges_list.append(edge)
        allowed_paths.extend(all_nodes_list)
        allowed_paths.extend(all_edges_list)

        node_ls = list(graph.nodes)
        for src_idx in range(len(node_ls)):
            for tgt_idx in range(src_idx + 1, len(node_ls)):
                all_simple_paths = list(
                    nx.all_simple_paths(
                        graph,
                        source=node_ls[src_idx],
                        target=node_ls[tgt_idx],
                        cutoff=max_rank,
                    )
                )

                for i in range(len(all_simple_paths)):
                    path = all_simple_paths[i]
                    if not reserve_sequence_order:
                        all_simple_paths[i] = (
                            path[::-1] if str(path[0]) > str(path[-1]) else path
                        )
                    all_simple_paths[i] = tuple(all_simple_paths[i])

                if len(all_simple_paths) > 0:
                    allowed_paths.extend(all_simple_paths)
        return set(allowed_paths)
