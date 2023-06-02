"""Various examples of named graphs represented as complexes."""

import networkx as nx
import numpy as np

from toponetx import CellComplex
from toponetx.algorithms.spectrum import (
    hodge_laplacian_eigenvectors,
    set_hodge_laplacian_eigenvector_attrs,
)
from toponetx.transform.graph_to_simplicial_complex import graph_2_clique_complex

__all__ = ["karate_club"]


def karate_club(complex_type="simplicial", feat_dim=2):
    """Load the karate club as featured cell/simplicial complex.

    Parameters
    ----------
    complex_type : str, optional
        The type of complex to load. Supported values are
        "simplicial" and "cell".
        The default is "simplicial".
    feat_dim : int, optional
        The number of eigenvectors to be attached to the simplices/cells of the output complex.

    Returns
    -------
    When input is "simplicial":
        A python dictionary with the following keys :
        "complex": with value being a SimplicialComplex obtained from karate club.
        "node_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to nodes.
        "edge_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to edges.
        "face_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to faces.
        "tet_feat": the first feat_dim Hodge Laplacian eigenvectors attached to tetrohedron.
    When input is "cell":
        A python dictionary with the following keys :
        "complex": with value being a CellComplex obtained from karate club.
        "node_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to nodes.
        "edge_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to edges.
        "cell_feat": with value being the first feat_dim Hodge Laplacian eigenvectors attached to cells.

    Raises
    ------
    ValueError
        If complex_type is not one of the supported values.

    Note
    -----
    A featured simplicial complex is returned as the clique complex of the graph.
    A featured cell complex is returned as the cell complex obtained by adding the independent cycles of graph.

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
        _, tet_feat = hodge_laplacian_eigenvectors(
            sc.hodge_laplacian_matrix(3), feat_dim
        )

        data = {
            "complex": sc,
            "node_feat": np.array(nodes_feat),
            "edge_feat": np.array(edges_feat),
            "face_feat": np.array(faces_feat),
            "tet_feat": np.array(tet_feat),
        }
        return data

    elif complex_type == "cell":
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

        data = {
            "complex": cx,
            "node_feat": np.array(nodes_feat),
            "edge_feat": np.array(edges_feat),
            "cell_feat": np.array(cells_feat),
        }
        return data

    raise ValueError(f"complex_type must be 'simplicial' or 'cell' got {complex_type}")
