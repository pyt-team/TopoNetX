"""Methods to lift a graph to a simplicial complex."""

from itertools import combinations, takewhile

import networkx as nx
from typing_extensions import deprecated

from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = [
    "graph_to_clique_complex",
    "graph_to_neighbor_complex",
    "weighted_graph_to_vietoris_rips_complex",
]


def graph_to_neighbor_complex(G: nx.Graph) -> SimplicialComplex:
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.

    Returns
    -------
    toponetx.classes.SimplicialComplex
        The neighbor complex of the graph.

    Notes
    -----
    This type of simplicial complexes can have very large dimension (max degree of the
    graph) and it is a function of the distribution of the valency of the graph.
    """
    simplices = [[*list(G.neighbors(node)), node] for node in G]
    return SimplicialComplex(simplices)


def graph_to_clique_complex(
    G: nx.Graph, max_rank: int | None = None
) -> SimplicialComplex:
    """Get the clique complex of a graph.

    Parameters
    ----------
    G : networks.Graph
        Input graph.
    max_rank : int, optional
        The maximum rank of the simplices in the output clique complex.

    Returns
    -------
    SimplicialComplex
        The clique simplicial complex of rank `max_rank` of the graph G.
    """
    cliques = nx.enumerate_all_cliques(G)

    # `nx.enumerate_all_cliques` returns cliques in ascending order of size. Abort calling the generator once we reach
    # cliques larger than the requested max dimension.
    cliques = takewhile(
        lambda clique: max_rank is None or len(clique) <= max_rank + 1, cliques
    )

    SC = SimplicialComplex(cliques)

    # copy attributes of the input graph
    for node in G.nodes:
        SC[[node]].update(G.nodes[node])
    # if the resulting complex has dimension 1 or higher, copy the attributes of the edges
    if SC.dim >= 1:
        for edge in G.edges:
            SC[edge].update(G.edges[edge])
    SC.complex.update(G.graph)

    return SC


@deprecated(
    "`graph_2_neighbor_complex` is deprecated and will be removed in a future version, use `graph_to_neighbor_complex` instead."
)
def graph_2_neighbor_complex(G) -> SimplicialComplex:
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.

    Returns
    -------
    toponetx.classes.SimplicialComplex
        The neighbor complex of the graph.

    Notes
    -----
    This type of simplicial complexes can have very large dimension (max degree of the
    graph) and it is a function of the distribution of the valency of the graph.
    """
    return graph_to_neighbor_complex(G)


@deprecated(
    "`graph_2_clique_complex` is deprecated and will be removed in a future version, use `graph_to_clique_complex` instead."
)
def graph_2_clique_complex(
    G: nx.Graph, max_rank: int | None = None
) -> SimplicialComplex:
    """Get the clique complex of a graph.

    Parameters
    ----------
    G : networks.Graph
        Input graph.
    max_rank : int, optional
        The maximum rank of the simplices in the output clique complex.

    Returns
    -------
    SimplicialComplex
        The clique simplicial complex of rank `max_rank` of the graph `G`.
    """
    return graph_to_clique_complex(G, max_rank)


def weighted_graph_to_vietoris_rips_complex(
    G: nx.Graph, r: float, max_dim: int | None = None
):
    r"""Get the Vietoris-Rips complex of radius r of a weighted undirected graph.

    The Vietoris-Rips complex of radius `r` is the clique complex given by the cliques
    of `G` whose nodes have pairwise distances less or equal than `r`. All vertices are
    added to the Vietoris-Rips complex regardless of the radius introduced.

    If `G` is a clique weighted by a dissimilarity function d that satisfies
    \max_v d(v, v) <= \min d(u,v) for u != v, and r >= d(v, v) for all nodes v,
    then the Vietoris-Rips complex of radius `r` is the usual Vietoris-Rips abstract
    simplicial complex of radius `r` for point clouds with dissimilarities.

    Parameters
    ----------
    G : networkx.Graph
        Weighted undirected input graph. The weights of the edges must be in the attribute 'weight'.
    r : float
        The radius for the Vietoris-Rips simplicial complex computation.
    max_dim : int, optional
        The max dimension of the cliques in the output clique complex.

    Returns
    -------
    SimplicialComplex
        The Vietoris-Rips simplicial complex of dimension max_dim of the graph G.
    """

    def is_in_vr_complex(clique):  # numpydoc ignore=GL08
        edges = combinations(clique, 2)
        return all(G[u][v]["weight"] <= r for u, v in edges)

    all_cliques = nx.enumerate_all_cliques(G)
    possible_cliques = takewhile(
        lambda face: max_dim is None or len(face) <= max_dim, all_cliques
    )
    vr_cliques = filter(is_in_vr_complex, possible_cliques)
    return SimplicialComplex(list(vr_cliques))
