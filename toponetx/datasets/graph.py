"""Various examples of named graphs represented as complexes."""

from typing import Literal, Union, overload

import networkx as nx
import numpy as np

from toponetx import CellComplex, SimplicialComplex
from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors
from toponetx.transform.graph_to_simplicial_complex import graph_2_clique_complex

__all__ = ["karate_club"]


@overload
def karate_club(
    complex_type: Literal["cell"] = ..., feat_dim: int = ...
) -> CellComplex:
    ...


@overload
def karate_club(
    complex_type: Literal["simplicial"] = ..., feat_dim: int = ...
) -> SimplicialComplex:
    ...


def karate_club(
    complex_type: Literal["cell", "simplicial"] = "simplicial", feat_dim: int = 2
) -> Union[CellComplex, SimplicialComplex]:
    """Load the karate club as featured cell/simplicial complex.

    Parameters
    ----------
    complex_type : {'simplicial','cell'}, default='simplicial'
        The type of complex to loaded.
    feat_dim : int, default=2
        The number of eigenvectors to be attached to the simplices/cells
        of the output complex.

    Returns
    -------
    When input is "simplicial":
           a SimplicialComplex obtained from karate club with the following features
        "node_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to nodes.
        "edge_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to edges.
        "face_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to faces.
        "tetrahedron_feat": the first feat_dim Hodge Laplacian eigenvectors attached to tetrahedron.
    When input is "cell":
            a CellComplex obtained from karate club with the following features
        "node_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to nodes.
        "edge_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to edges.
        "cell_feat":
            - its value is the first feat_dim Hodge Laplacian eigenvectors attached to cells.

    Raises
    ------
    ValueError
        If complex_type is not one of the supported values.

    Note
    -----
    A featured simplicial complex is returned as the clique complex of the graph.
    A featured cell complex is returned as the cell complex obtained by adding the
    independent cycles of graph.

    """
    if complex_type == "simplicial":
        g = nx.karate_club_graph()
        sc = graph_2_clique_complex(g)

        _, nodes_feat = hodge_laplacian_eigenvectors(
            sc.hodge_laplacian_matrix(0), feat_dim
        )
        _, edges_feat = hodge_laplacian_eigenvectors(
            sc.hodge_laplacian_matrix(1), feat_dim
        )
        _, faces_feat = hodge_laplacian_eigenvectors(
            sc.hodge_laplacian_matrix(2), feat_dim
        )
        _, tetrahedron_feat = hodge_laplacian_eigenvectors(
            sc.hodge_laplacian_matrix(3), feat_dim
        )

        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(0), np.array(nodes_feat))), name="node_feat"
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(1), np.array(edges_feat))), name="edge_feat"
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(2), np.array(faces_feat))), name="face_feat"
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(3), np.array(tetrahedron_feat))),
            name="tetrahedron_feat",
        )

        return sc

    if complex_type == "cell":
        g = nx.karate_club_graph()
        cycles = nx.cycle_basis(g)
        cx = CellComplex(g)
        cx.add_cells_from(cycles, rank=2)

        _, nodes_feat = hodge_laplacian_eigenvectors(
            cx.hodge_laplacian_matrix(0), feat_dim
        )
        _, edges_feat = hodge_laplacian_eigenvectors(
            cx.hodge_laplacian_matrix(1), feat_dim
        )
        _, cells_feat = hodge_laplacian_eigenvectors(
            cx.hodge_laplacian_matrix(2), feat_dim
        )

        cx.set_cell_attributes(
            dict(zip(cx.skeleton(0), np.array(nodes_feat))), name="node_feat", rank=0
        )
        cx.set_cell_attributes(
            dict(zip(cx.skeleton(1), np.array(edges_feat))), name="edge_feat", rank=1
        )
        cx.set_cell_attributes(
            dict(zip(cx.skeleton(2), np.array(cells_feat))), name="cell_feat", rank=2
        )
        return cx

    raise ValueError(f"complex_type must be 'simplicial' or 'cell' got {complex_type}")
