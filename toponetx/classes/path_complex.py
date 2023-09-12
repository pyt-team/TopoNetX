"""TODO: docstring."""
from collections.abc import Hashable, Iterable, Iterator
from itertools import chain
from typing import List, Set, Tuple, Union

import networkx as nx
import numpy as np
import scipy as sp

from toponetx.classes.complex import Complex
from toponetx.classes.path import Path
from toponetx.classes.reportviews import NodeView, PathView

__all__ = ["PathComplex"]


class PathComplex(Complex):
    """TODO: docstring."""

    def __init__(
        self,
        paths=None,
        name: str = "",
        reserve_sequence_order: bool = False,
        allowed_paths: List[Tuple] = None,
        max_rank: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self._path_set = PathView()
        self.reserve_sequence_order = reserve_sequence_order
        if allowed_paths is not None:
            self.allowed_paths = set(allowed_paths)
        else:
            self.allowed_paths = set()

        if isinstance(paths, nx.Graph):
            # compute allowed_paths in order to construct boundary incidence matrix/adj matrix.
            if self.allowed_paths is not None:
                self.allowed_paths = self.compute_allowed_paths(
                    paths,
                    reserve_sequence_order=reserve_sequence_order,
                    max_rank=max_rank,
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
            self.add_paths_from(self.allowed_paths)

        elif isinstance(paths, list) or isinstance(paths, tuple):
            if len(paths) > 0:
                paths = [tuple(path) for path in paths]
                self.add_paths_from(set(paths))
        elif paths is not None:
            raise TypeError(
                "Input paths must be a graph or an iterable of paths as lists or tuples."
            )

    def add_paths_from(self, paths: Set[Union[List, Tuple, "Path"]]) -> None:
        """TODO: docstring."""
        if isinstance(paths, Hashable):
            raise TypeError("Paths must be an iterable of paths as lists or tuples.")
        paths_clone = paths.copy()
        for p in paths_clone:
            self.add_path(p)

    def add_path(self, path: Union[Hashable, List, Tuple, "Path"], **attr) -> None:
        """TODO: docstring."""
        new_paths = set()
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
                    and str(path_[0]) > str(path_[-1])
                    and not self.reserve_sequence_order
                ):
                    raise ValueError(
                        "A p-path must have the first index smaller than the last index, got {}".format(
                            path
                        )
                    )
            else:
                path_ = path.elements
            new_path = self._update_faces_dict_length(
                path_
            )  # add dict corresponding to the path dimension

            if (
                self._path_set.max_dim < len(path_) - 1
            ):  # update max dimension for PathView()
                self._path_set.max_dim = len(path_) - 1

            if (
                path_ in self._path_set.faces_dict[len(path_) - 1]
            ):  # path is already in the complex, just update the properties if needed
                if isinstance(path, Path):  # update attrbiutes for PathView()
                    self._path_set.faces_dict[len(path_) - 1][path_].update(
                        path._properties
                    )
                else:
                    self._path_set.faces_dict[len(path_) - 1][path_].update(attr)
                return

            # update sub-sequence of the path to _path_set.faces_dict if not available
            # the sub-sequences are obvious sub-paths. there may exist some non-obvious sub-paths (when cliques are involved).
            # for such case, manually add the sub-paths to the allowed_paths if needed.
            for length in range(len(path_), 0, -1):
                for i in range(0, len(path_) - length + 1):
                    sub_path = path_[i : i + length]
                    if not self.reserve_sequence_order and str(sub_path[0]) > str(
                        sub_path[-1]
                    ):
                        sub_path = sub_path[::-1]
                    sub_path = tuple(sub_path)
                    new_path = self._update_faces_dict_entry(sub_path)
                    if new_path is not None:
                        new_paths.add(new_path)
            # update allowed paths
            if len(new_paths) > 0:
                self.allowed_paths.update(new_paths)

            if isinstance(path, Path):  # update attrbiutes for PathView()
                self._path_set.faces_dict[len(path_) - 1][path_].update(
                    path._properties
                )
            else:
                self._path_set.faces_dict[len(path_) - 1][path_].update(attr)

    @property
    def dim(self) -> int:
        """Dimension.

        This is the highest dimension of any p-path in the complex.
        """
        return self._path_set.max_dim

    @property
    def nodes(self):
        """Nodes."""
        return NodeView(
            self._path_set.faces_dict, cell_type=Path
        )  # TODO: fix NodeView class as frozenset is too restricted for Path

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
            tmp = (path for path in self._path_set.faces_dict[rank].keys())
            return sorted(
                tmp, key=lambda x: tuple(map(str, x))
            )  # lexicographic comparison
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
            boundary = sp.sparse.lil_matrix((0, len(self.nodes)))
            if index:
                node_index = {
                    node: i
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
                    if not self.reserve_sequence_order and str(boundary_path[0]) > str(
                        boundary_path[-1]
                    ):
                        boundary_path = boundary_path[::-1]
                    boundary_path = tuple(boundary_path)
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
            return row, L_up.tolil()
        else:
            return L_up.tolil()

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
        self.allowed_paths.remove(path)
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

    def _update_faces_dict_entry(self, path):
        dim = len(path) - 1
        if path not in self._path_set.faces_dict[dim]:  # Not in faces_dict
            self._path_set.faces_dict[dim][path] = dict()
            return path
        else:
            return None

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


if __name__ == "__main__":
    # G = nx.Graph()
    # G.add_nodes_from([2, 3, 67, 89])
    # G.add_edges_from([(67, 89), (2, 89), (2, 3), (3, 89)])
    # pc = PathComplex(G)
    # print(pc._path_set.faces_dict)

    # row0, col0, B0 = pc.incidence_matrix(0, index=True)
    # print(row0, col0)
    # print(B0.todense())

    # row1, col1, B1 = pc.incidence_matrix(1, index=True)
    # print(row1, col1)
    # print(B1.todense())

    # row2, col2, B2 = pc.incidence_matrix(2, index=True)
    # print(row2, col2)
    # print(B2.todense())

    # row3, col3, B3 = pc.incidence_matrix(3, index=True)
    # print(row3, col3)
    # print(B3.todense())

    # # pc.add_paths_from([(0, 2), (0, 4, 5), (6, 7)])
    # # print(pc._path_set.faces_dict)
    # # print("AAAAA")
    # # row2, col2, B2 = pc.incidence_matrix(2, index=True)
    # # print(row2, col2)
    # # print(B2.todense())

    # # row3, col3, B3 = pc.incidence_matrix(3, index=True)
    # # print(row3, col3)
    # # print(B3.todense())

    # print("+++++++++++++")
    # print("Shape", pc.shape)
    # print("Dim", pc.dim)
    # print("Nodes", pc.nodes)
    # print("Skeleton(0)", pc.skeleton(0))
    # print("Skeleton(1)", pc.skeleton(1))
    # print("Skeleton(2)", pc.skeleton(2))
    # print("Skeleton(3)", pc.skeleton(3))
    # print("Paths", pc.paths)
    # print("Number of paths", len(pc))
    # print("test 0,1,2 in pc", (0, 1, 2) in pc)
    # print("+++++++++++++")
    # path = Path([0, 5, 4])
    # pc.add_path(path)
    # print("AAAAA")
    # row2, col2, B2 = pc.incidence_matrix(2, index=True)
    # print(row2, col2)
    # print(B2.todense())

    # row3, col3, B3 = pc.incidence_matrix(3, index=True)
    # print(row3, col3)
    # print(B3.todense())
    # print("+++++++++++++")
    # row0, A0 = pc.up_laplacian_matrix(0, index=True, signed=False)
    # print(row0)
    # print(A0.todense())
    # row1, A1 = pc.up_laplacian_matrix(1, index=True)
    # print(row1)
    # print(A1.todense())
    # row2, A2 = pc.up_laplacian_matrix(2, index=True)
    # print(row2)
    # print(A2.todense())

    # print("+++++++++++++")
    # row0, A0 = pc.adjacency_matrix(0, index=True)
    # print(row0)
    # print(A0.todense())
    # row1, A1 = pc.adjacency_matrix(1, index=True)
    # print(row1)
    # print(A1.todense())
    # row2, A2 = pc.adjacency_matrix(2, index=True)
    # print(row2)
    # print(A2.todense())

    # print("+++++++++++++")
    # # pc.remove_nodes([1,2])
    # # print(pc.paths)
    # # print(pc.dim)

    # pc = PathComplex([[0, 1, 2], [1, 2], [1, 3], [0]])
    # print(pc.paths)
    # print(pc.dim)

    # pc.add_path(4, weight=53)
    # print(pc.paths)
    # print(pc[4])
    # print(pc[1, 2])

    # pc.add_path(4, weight=21)
    # print(pc[4])

    # print(pc.remove_nodes([4]))
    # print(pc.paths)

    # pc.add_path([1, 2], weight=3)
    # print(pc[[1, 2]])

    # path = pc[[1, 2]]
    # path.update(weight=5)
    # print(pc[[1, 2]])

    # path = Path([1, 2], weight=6)
    # pc.add_path(path)
    # print(pc[[1, 2]])
    # print(pc.paths)

    PX = PathComplex([[1, 2, 3], [1, 2, 4]])
    print(PX.nodes.nodes)
    print((1,) in PX.nodes.nodes)

    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    PX = PathComplex(G)

    PX = PathComplex([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
    row, col, B = PX.incidence_matrix(2, signed=False, index=True)
    print(row)
    print(col)
    print(PX.incidence_matrix(2, signed=False).todense())
