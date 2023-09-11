"""TODO: docstring."""
from collections.abc import Hashable, Iterable, Iterator
from itertools import chain, combinations
from typing import Any, List, Set, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np
import scipy as sp
from hypernetx import Hypergraph

from toponetx.classes.complex import Complex
from toponetx.classes.path import Path
from toponetx.classes.reportviews import NodeView, PathView
from toponetx.exception import TopoNetXError

__all__ = ["PathComplex"]


class PathComplex(Complex):
    """TODO: docstring."""

    def __init__(
        self,
        paths=None,
        name: str = "",
        reserve_sequence_order: bool = False,
        allowed_paths: List[Union[List, Tuple]] = None,
        max_rank: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self._path_set = PathView()
        self.reserve_sequence_order = reserve_sequence_order
        self.allowed_paths = allowed_paths

        if isinstance(paths, nx.Graph):
            # compute allowed_paths in order to construct boundary incidence matrix/adj matrix.
            self.allowed_paths = self.compute_allowed_paths(
                paths, reserve_sequence_order=reserve_sequence_order, max_rank=max_rank
            )

            # get feature of nodes and edges if available
            for path, data in paths.nodes(data=True):
                self.add_path((path,), data)
            for u, v, data in paths.edges(
                data=True
            ):  # so far, path complex only supports undirected graph
                if (u > v) and not reserve_sequence_order:
                    u, v = v, u
                self.add_path((u, v), data)

            self.add_paths_from(
                self.allowed_paths, update_features=False
            )  # we don't want to erase the features of 0-paths and 1-paths. Assume  higher dimensional paths have no features.

        # elif isinstance(paths, Iterable[Union[List, Tuple]]):
        #     # TODO: implement this
        elif paths is not None:
            raise TypeError(
                "Input paths must be a graph or an iterable of paths as lists or tuples."
            )

    def add_paths_from(
        self, paths: Set[Union[List, Tuple, "Path"]], update_features: bool = False
    ) -> None:
        """TODO: docstring."""
        for p in paths:
            self.add_path(p, update_features=update_features)

    def add_path(
        self,
        path: Union[Hashable, List, Tuple, "Path"],
        update_features: bool = True,
        **attr,
    ) -> None:
        """TODO: docstring."""
        if isinstance(path, int) or isinstance(path, str):
            path = [
                path,
            ]
        if isinstance(path, List) or isinstance(path, Tuple) or isinstance(path, Path):
            if not isinstance(path, Path):  # path is a list or tuple
                path_ = tuple(path)
                if len(path) != len(set(path)):
                    raise ValueError("A p-path cannot contain duplicate nodes.")
                if (
                    len(path_) > 1
                    and path_[0] > path_[-1]
                    and not self.reserve_sequence_order
                ):
                    raise ValueError(
                        "A p-path must have the first index smaller than the last index, got {}".format(
                            path
                        )
                    )
            else:
                path_ = path.elements
            self._update_faces_dict_length(
                path_
            )  # add dict corresponding to the path dimension

            if (
                self._path_set.max_dim < len(path) - 1
            ):  # update max dimension for PathView()
                self._path_set.max_dim = len(path) - 1

            if (
                path_ in self._path_set.faces_dict[len(path_) - 1]
            ):  # path is already in the complex, just update the properties if needed
                if update_features:
                    self._path_set.faces_dict[len(path_) - 1][path_].update(attr)
                return

            # update sub-sequence of the path to _path_set.faces_dict if not available
            for length in range(len(path), 0, -1):
                for i in range(0, len(path) - length + 1):
                    sub_path = tuple(path[i : i + length])
                    self._update_faces_dict_entry(sub_path)

            if isinstance(path, Path):  # update attrbiutes for PathView()
                self._path_set.faces_dict[len(path_) - 1][path_] = path._properties
            else:
                self._path_set.faces_dict[len(path_) - 1][path_] = attr

    @property
    def dim(self) -> int:
        """Dimension.

        This is the highest dimension of any p-path in the complex.
        """
        return self._path_set.max_dim

    @property
    def nodes(self):
        """Nodes."""
        return NodeView(self._simplex_set.faces_dict, cell_type=Path)

    @property
    def paths(self) -> PathView:
        """Set of all p-paths."""
        return self._path_set

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of path complex.

        (number of p-paths[i], for i in range(0,dim(Pc)))

        Returns
        -------
        tuple of ints
        """
        return self._path_set.shape

    def clone(self) -> "PathComplex":
        """Return a copy of the simplicial complex.

        The clone method by default returns an independent shallow copy of the simplicial complex. Use Python’s
        `copy.deepcopy` for new containers.

        Returns
        -------
        SimplicialComplex
        """
        return PathComplex(self.paths, name=self.name)

    def skeleton(self, rank):
        """Compute skeleton.

        Returns
        -------
        Set of p-paths of dimension n.
        """
        if rank < len(self._path_set.faces_dict) and rank >= 0:
            return sorted(path for path in self._simplex_set.faces_dict[rank].keys())
        if rank < 0:
            raise ValueError(f"input must be a postive integer, got {rank}")
        raise ValueError(f"input {rank} exceeds max dim")

    def add_node(self, node: Union[Hashable, Path], **attr) -> None:
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
        """TODO: docstring."""
        removed_paths = set()
        for path in self:  # iterate over all paths
            if any(
                node in path for node in node_set
            ):  # if any node in node_set is in the path, remove the path
                removed_paths.add(path)

        for path in removed_paths:
            self._remove_path(path)

    def incidence_matrix(self, rank, signed: bool = True, index: bool = False):
        """TODO: docstring."""
        if rank < 0:
            raise ValueError(f"input dimension d must be positive integer, got {rank}")
        if rank > self.dim:
            raise ValueError(
                f"input dimenion cannat be larger than the dimension of the complex, got {rank}"
            )
        if rank == 0:
            A = sp.sparse.lil_matrix(0, len(self.nodes))
            if index:
                node_index = {node: i for i, node in enumerate(sorted(self.nodes))}
                return {}, node_index, abs(A.asformat("csr"))
            else:
                return abs(A.asformat("csr"))
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
                    boundary_path = tuple(path[0:i] + path[(i + 1) :])
                    if boundary_path in self.allowed_paths:
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
                    idx_p_minus_1,
                    idx_p,
                    boundary,
                )
            else:
                return (
                    idx_p_minus_1,
                    idx_p,
                    abs(boundary),
                )
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def coincidence_matrix(self, rank, signed: bool = True, index: bool = False):
        """Compute coincidence matrix of the path complex.

        This is also called the coboundary matrix.
        """
        if index:
            idx_faces, idx_paths, boundary = self.incidence_matrix(
                rank, signed=signed, index=True
            )
            return idx_faces, idx_paths, boundary.T
        else:
            return self.incidence_matrix(rank, signed=signed, index=False).T

    def up_laplacian_matrix(self, rank: int, signed: bool = True, index: bool = False):
        """TODO: docstring."""
        if rank == 0:
            row, col, B_next = self.incidence_matrix(rank + 1, index=True)
            L_up = B_next @ B_next.transpose()
        elif rank < self.dim:
            row, col, B_next = self.incidence_matrix(rank + 1, index=True)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"Rank should larger than 0 and <= {self.dim - 1} (maximal dimension-1), got {rank}."
            )
        if not signed:
            L_up = abs(L_up)

        if index:
            return row, L_up
        else:
            return L_up

    def down_laplacian_matrix(
        self, rank: int, signed: bool = True, index: bool = False
    ):
        """TODO: docstring."""
        if rank <= self.dim and rank > 0:
            row, column, B = self.incidence_matrix(rank, index=True)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {self.dim} (maximal dimension), got {rank}."
            )
        if not signed:
            L_down = abs(L_down)
        if index:
            return row, L_down
        else:
            return L_down

    def adjacency_matrix(self, rank: int, signed: bool = False, index: bool = False):
        """TODO: docstring."""
        ind, L_up = self.up_laplacian_matrix(rank, signed=signed, index=True)
        L_up.setdiag(0)

        if not signed:
            L_up = abs(L_up)
        if index:
            return ind, L_up
        return L_up

    def coadjacency_matrix(self, rank: int, signed: bool = False, index: bool = False):
        """TODO: docstring."""
        ind, L_down = self.down_laplacian_matrix(rank, signed=signed, index=True)
        L_down.setdiag(0)
        if not signed:
            L_down = abs(L_down)
        if index:
            return ind, L_down
        return L_down

    def _remove_path(self, path: Tuple) -> None:
        del self._path_set.faces_dict[len(path) - 1][path]
        if (
            len(self._path_set.faces_dict[len(path) - 1]) == 0
            and self._path_set.max_dim == len(path) - 1
        ):  # update max dimension for PathView() if highest dimension is empty
            self._path_set.max_dim -= 1

    def _update_faces_dict_length(self, path) -> None:
        if len(path) > len(self._path_set.faces_dict):
            diff = len(path) - len(self._path_set.faces_dict)
            for _ in range(diff):
                self._path_set.faces_dict.append(dict())

    def _update_faces_dict_entry(self, path) -> None:
        dim = len(path) - 1
        if path not in self._path_set.faces_dict[dim]:  # Not in faces_dict
            self._path_set.faces_dict[dim][path] = dict()

    def __contains__(self, item) -> bool:
        """Return boolean indicating if item is in self._path_set.

        Parameters
        ----------
        item : tuple, list
        """
        return item in self._path_set

    def __getitem__(self, item):
        """Get p-path."""
        if item in self:
            return self._path_set[item]
        else:
            raise KeyError("path is not in the path complex")

    def __iter__(self) -> Iterator:
        """Iterate over all faces of the path complex.

        Returns
        -------
        dict_keyiterator
        """
        return chain.from_iterable(self._path_set.faces_dict)

    def __len__(self) -> int:
        """Return the number of p-paths in the path complex.

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
    ) -> Set[Union[List, Tuple]]:
        """TODO: docstring."""
        allowed_paths = list()
        all_nodes_list = list(tuple([node]) for node in graph.nodes)
        all_edges_list = list(tuple(edge) for edge in graph.edges)
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
                        all_simple_paths[i] = path[::-1] if path[0] > path[-1] else path
                    all_simple_paths[i] = tuple(all_simple_paths[i])

                if len(all_simple_paths) > 0:
                    allowed_paths.extend(all_simple_paths)
        return set(allowed_paths)


if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1)])
    pc = PathComplex(G)
    print(pc._path_set.faces_dict)
