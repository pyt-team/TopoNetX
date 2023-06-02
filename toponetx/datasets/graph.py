"""Various examples of named graphs represented as complexes."""

import networkx as nx
import numpy as np

from toponetx import CellComplex
from toponetx.algorithms.spectrum import (
    hodge_laplacian_eigenvectors,
    set_hodge_laplacian_eigenvector_attrs,
)
from toponetx.transform.graph_to_simplicial_complex import graph_2_clique_complex

__all__ = ["karate_club_complex"]


def karate_club_complex(complex_type="simplicial complex"):
    """Load the karate club as complex.

        simplicial complex is returned as the clique complex of the graph
        cell complex is return as the cell complex obtained by adding the independent cycles of graph

    Parameters
    ----------
    complex_type : str, optional
        The type of complex to load. Supported values are
        "simplicial complex" and "cell complex".
        The default is "simplicial complex".

    Returns
    -------
    SimplicialComplex or CellComplex
        The loaded complex of the specified type.

    Raises
    ------
    ValueError
        If complex_type is not one of the supported values.
    """
    if complex_type == "simplicial complex":
        g = nx.karate_club_graph()
        sc = graph_2_clique_complex(g)  # lift to sc

        _, nodes_feat = hodge_laplacian_eigenvectors(sc.hodge_laplacian_matrix(0), 2)
        _, edges_feat = hodge_laplacian_eigenvectors(sc.hodge_laplacian_matrix(1), 2)
        _, faces_feat = hodge_laplacian_eigenvectors(sc.hodge_laplacian_matrix(2), 2)
        _, tet_feat = hodge_laplacian_eigenvectors(sc.hodge_laplacian_matrix(3), 2)

        data = {
            "complex": sc,
            "node_feat": np.array(nodes_feat),
            "edge_feat": np.array(edges_feat),
            "face_feat": np.array(faces_feat),
            "tet_feat": np.array(tet_feat),
        }
        return data

    elif complex_type == "cell complex":
        g = nx.karate_club_graph()
        cycles = nx.cycle_basis(g)
        cx = CellComplex(g)  # lift to graph
        cx.add_cells_from(cycles, rank=2)  # add basis cycles

        _, nodes_feat = hodge_laplacian_eigenvectors(cx.hodge_laplacian_matrix(0), 2)
        _, edges_feat = hodge_laplacian_eigenvectors(cx.hodge_laplacian_matrix(1), 2)
        _, cells_feat = hodge_laplacian_eigenvectors(cx.hodge_laplacian_matrix(2), 2)

        data = {
            "complex": cx,
            "node_feat": np.array(nodes_feat),
            "edge_feat": np.array(edges_feat),
            "cell_feat": np.array(cells_feat),
        }
        return data

    else:
        raise ValueError("cmplex_type must be 'simplicial complex' or 'cell complex'")
