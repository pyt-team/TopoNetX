"""Methods to lift a graph to a simplicial complex."""

__all__ = [
    "graph_2_clique_complex",
    "graph_2_neighbor_complex",
]


import networkx as nx

from toponetx import SimplicialComplex


def graph_2_neighbor_complex(G):
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
    neighbors = []
    for i in G.nodes():
        N = list(G.neighbors(i)) + [i]  # each simplex is the node and its n-hop nbhd
        neighbors.append(N)
    return SimplicialComplex(neighbors)


def graph_2_clique_complex(G, max_dim=None):
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
    if max_dim is None:
        lst = nx.enumerate_all_cliques(G)
        return SimplicialComplex(list(lst))

    lst = filter(lambda face: len(face) <= max_dim, nx.enumerate_all_cliques(G))
    return SimplicialComplex(list(lst))
