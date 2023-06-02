"""Methods to lift a graph to a simplicial complex."""

__all__ = [
    "graph_2_clique_complex",
    "graph_2_neighbor_complex",
    "weighted_graph_2_Vietoris_Rips_complex"
]

import itertools

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


def weighted_graph_2_Vietoris_Rips_complex(G, r, max_dim=None):
    """Get the Vietoris-Rips complex of radius r of a weighted undirected graph. The Vietoris-Rips complex of radius
    r is the clique complex given by the cliques of G whose nodes have pairwise distances less or equal than r. All
    vertices are added to the Vietoris-Rips complex regardless the radius introduced. If G is a clique weighted by a
    dissimilarity function d that satisfies max_v d(v, v) <= min d(u,v) for u != v, and r >= d(v, v) for all nodes v,
    then the Vietoris-Rips complex of radius r is the usual Vietoris-Rips abstract simplicial complex of radius r for
    point clouds with dissimilarities.

    Parameters
    ----------
    G : networkx graph
        Weighted undirected input graph. The weights of the edges must be in the attribute 'weight'.
    r : float
        The radius for the Vietoris-Rips simplicial complex computation.
    max_dim : int, optional
        The max dimension of the cliques in
        the output clique complex.
        The default is None indicate max dimension.

    Returns
    -------
    SimplicialComplex
        The Vietoris-Rips simplicial complex of dimension max_dim of the graph G.
        """

    def is_in_VR_complex(clique):
        # Check if all edge weights are less or equal than r
        edges = itertools.combinations(clique, 2)
        edge_weights_lower_than_r = all(G[u][v]['weight'] <= r for u, v in edges)
        return edge_weights_lower_than_r

    all_cliques = nx.enumerate_all_cliques(G)
    possible_cliques = all_cliques if max_dim is None else filter(lambda face: len(face) <= max_dim, all_cliques)
    vr_cliques = filter(is_in_VR_complex, possible_cliques)
    return SimplicialComplex(list(vr_cliques))
