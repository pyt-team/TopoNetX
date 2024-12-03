"""Various examples of named graphs represented as complexes."""

from pathlib import Path
from typing import Literal, overload

import networkx as nx
import numpy as np
import requests

from toponetx.algorithms.spectrum import hodge_laplacian_eigenvectors
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.transform.graph_to_simplicial_complex import graph_to_clique_complex

__all__ = ["coauthorship", "karate_club"]

DIR = Path(__file__).parent


@overload
def karate_club(
    complex_type: Literal["cell"], feat_dim: int = ...
) -> CellComplex:  # numpydoc ignore=GL08
    ...


@overload
def karate_club(
    complex_type: Literal["simplicial"], feat_dim: int = ...
) -> SimplicialComplex:  # numpydoc ignore=GL08
    ...


def karate_club(
    complex_type: Literal["cell", "simplicial"], feat_dim: int = 2
) -> CellComplex | SimplicialComplex:
    """Load the karate club as featured cell/simplicial complex.

    Parameters
    ----------
    complex_type : {'simplicial','cell'}
        The type of complex to load.
    feat_dim : int, default=2
        The number of eigenvectors to be attached to the simplices/cells
        of the output complex.

    Returns
    -------
    SimplicialComplex or CellComplex
        When input is "simplicial":
        A SimplicialComplex obtained from karate club graph by lifting the graph to its
        clique complex. The simplicial complex comes with the following features:

        - "node_feat": its value is the first feat_dim Hodge Laplacian eigenvectors attached to nodes.
        - "edge_feat": its value is the first feat_dim Hodge Laplacian eigenvectors attached to edges.
        - "face_feat": its value is the first feat_dim Hodge Laplacian eigenvectors attached to faces.
        - "tetrahedron_feat": the first feat_dim Hodge Laplacian eigenvectors attached to tetrahedron.

        When input is "cell":
        A CellComplex obtained from karate club by lifting the graph to a cell obtained
        obtained from the graph by adding the independent homology cycles in the graph.
        The cell complex comes with the following features:

        - "node_feat": its value is the first feat_dim Hodge Laplacian eigenvectors
          attached to nodes.
        - "edge_feat": its value is the first feat_dim Hodge Laplacian eigenvectors
          attached to edges.
        - "cell_feat": its value is the first feat_dim Hodge Laplacian eigenvectors
          attached to cells.

    Raises
    ------
    ValueError
        If complex_type is not one of the supported values.

    Notes
    -----
    A featured simplicial complex is returned as the clique complex of the graph.
    A featured cell complex is returned as the cell complex obtained by adding the
    independent cycles of graph.
    """
    if complex_type == "simplicial":
        g = nx.karate_club_graph()
        sc = graph_to_clique_complex(g)

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
            dict(zip(sc.skeleton(0), np.array(nodes_feat), strict=True)),
            name="node_feat",
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(1), np.array(edges_feat), strict=True)),
            name="edge_feat",
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(2), np.array(faces_feat), strict=True)),
            name="face_feat",
        )
        sc.set_simplex_attributes(
            dict(zip(sc.skeleton(3), np.array(tetrahedron_feat), strict=True)),
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
            dict(zip(cx.skeleton(0), np.array(nodes_feat), strict=True)),
            name="node_feat",
            rank=0,
        )
        cx.set_cell_attributes(
            dict(zip(cx.skeleton(1), np.array(edges_feat), strict=True)),
            name="edge_feat",
            rank=1,
        )
        cx.set_cell_attributes(
            dict(zip(cx.skeleton(2), np.array(cells_feat), strict=True)),
            name="cell_feat",
            rank=2,
        )
        return cx

    raise ValueError(f"complex_type must be 'simplicial' or 'cell' got {complex_type}")


def coauthorship() -> SimplicialComplex:
    """Load the coauthorship network as a simplicial complex.

    The coauthorship network is a simplicial complex where a paper with k authors is
    represented by a (k-1)-simplex.
    The dataset is pre-processed as in [1]_. From the
    Semantic Scholar Open Research Corpus 80 papers with number of citations between 5
    and 10 were sampled.

    The papers constitute simplices in the complex, which is completed with
    sub-simplices (seen as collaborations between subsets of authors) to form a
    simplicial complex.
    An attribute named *citations* is added to each simplex, corresponding to the sum
    of citations of all papers on which the authors represented by the simplex
    collaborated. The resulting simplicial complex is of dimension 10 and contains
    24552 simplices in total. See [1]_ for a more detailed description of the dataset.

    Returns
    -------
    SimplicialComplex
        The simplicial complex comes with the attribute *citations*, the number of
        citations attributed to the given collaborations of k authors.

    References
    ----------
    .. [1] Stefania Ebli, Michael Defferrard and Gard Spreemann. Simplicial Neural
        Networks. Topological Data Analysis and Beyond workshop at NeurIPS.
        https://arxiv.org/abs/2010.03633
    """
    dataset_file = DIR / "coauthorship.npy"

    if not dataset_file.exists():
        r = requests.get(
            "https://github.com/pyt-team/topological-datasets/raw/main/resources/coauthorship.npy",
            timeout=10,
        )
        with dataset_file.open("wb") as f:
            f.write(r.content)

    coauthorship = np.load(dataset_file, allow_pickle=True)

    simplices = []
    for dim in range(len(coauthorship) - 1, -1, -1):
        simplices += [list(el) for el in coauthorship[dim]]

    sc = SimplicialComplex(simplices)

    for i in range(len(coauthorship)):
        dic = {tuple(sorted(k)): v for k, v in coauthorship[i].items()}
        sc.set_simplex_attributes(dic, name="citations")

    return sc
