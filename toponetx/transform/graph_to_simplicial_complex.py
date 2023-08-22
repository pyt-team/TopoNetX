"""Methods to lift a graph to a simplicial complex."""
from itertools import takewhile
from warnings import warn

import networkx as nx

from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = [
    "graph_to_clique_complex",
    "graph_to_neighbor_complex",
]


def graph_to_neighbor_complex(G: nx.Graph) -> SimplicialComplex:
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.

    Returns
    -------
    SimplicialComplex
        The neighbor complex of the graph.

    Notes
    -----
    This type of simplicial complexes can have very large dimension ( dimension = max_i(len (G.neighbors(i))) )
    and it is a function of the distribution of the valency of the graph.
    """
    simplices = []
    for node in G:
        # each simplex is the node and its n-hop neighbors
        simplices.append(list(G.neighbors(node)) + [node])
    return SimplicialComplex(simplices)


def graph_to_clique_complex(
    G: nx.Graph, max_dim: int | None = None
) -> SimplicialComplex:
    """Get the clique complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    max_dim : int, optional
        The max dimension of the cliques in
        the output clique complex.
        The default is None indicate max dimension.

    Returns
    -------
    SimplicialComplex
        The clique simplicial complex of dimension dim of the graph G.
    """
    cliques = nx.enumerate_all_cliques(G)

    # `nx.enumerate_all_cliques` returns cliques in ascending order of size. Abort calling the generator once we reach
    # cliques larger than the requested max dimension.
    if max_dim is not None:
        cliques = takewhile(lambda clique: len(clique) <= max_dim, cliques)

    return SimplicialComplex(cliques)


def graph_2_neighbor_complex(G) -> SimplicialComplex:
    warn(
        "`graph_2_neighbor_complex` is deprecated and will be removed in a future version, use `graph_to_neighbor_complex` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return graph_to_neighbor_complex(G)


def graph_2_clique_complex(
    G: nx.Graph, max_dim: int | None = None
) -> SimplicialComplex:
    warn(
        "`graph_2_clique_complex` is deprecated and will be removed in a future version, use `graph_to_clique_complex` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return graph_to_clique_complex(G, max_dim)
