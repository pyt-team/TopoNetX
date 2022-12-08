# -*- coding: utf-8 -*-
"""
@author: Mustafa Hajij
"""

import networkx as nx
from node2vec import Node2Vec

from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)
from toponetx.utils.structure import sparse_array_to_neighborhood_dict


class Cell2Vec(Node2Vec):
    """
     Cell2Vec is a class that extends the Node2Vec class
    and provides additional functionality for generating node embeddings for simplicial, cell, combinatorial,
    or dynamic combinatorial complexes. The Cell2Vec class takes as input a simplicial, cell, combinatorial,
    or dynamic combinatorial complex, and uses the adjacency matrix or coadjacency matrix of the complex to
    create a graph object using the networkx library. The Cell2Vec class then uses this graph object to generate
    node embeddings using the Node2Vec algorithm. The Cell2Vec class allows users to specify the type of adjacency
    or coadjacency matrix to use for the graph (e.g. "adj" for adjacency matrix or "coadj" for coadjacency matrix),
    as well as the dimensions of the neighborhood to use for the matrix (e.g. the "r" and "k" values for the matrix).
    Additionally, users can specify the dimensions of the node embeddings to generate, the length and number of
    random walks to use for the node2vec algorithm, and the number of workers to use for parallelization.


    Parameters
    ==========
    cmplex: This is the combinatorial complex data structure that the Cell2Vec class will operate on.
        It must be an instance of one of the following classes: SimplicialComplex, CellComplex,
        CombinatorialComplex, or DynamicCombinatorialComplex.

    neighborhood_type: This parameter specifies the type of neighborhood to use when creating
        the graph representation of the combinatorial complex. The options are 'adj' (for adjacency) and
        'coadj' (for co-adjacency).

    neighborhood_dim: This parameter specifies the dimensions of the neighborhood to use when creating
        the graph representation of the combinatorial complex.
        It should be a dictionary with two keys: 'r' and 'k'.
        The values of these keys depend on the neighborhood_type parameter.
        If neighborhood_type is 'adj', then 'r' should be set to the radius of the
        neighborhood (an integer), and 'k' should be set to -1.
        If neighborhood_type is 'coadj', then 'r' should be set to -1, and 'k'
        should be set to the number of edges in the neighborhood (an integer).

    dimensions: This is the number of dimensions to use for the node embeddings.

    walk_length: This is the length of each random walk.

    num_walks: This is the number of random walks to generate.

    workers: This is the number of worker threads to use for parallel computation
    """

    def __init__(
        self,
        cmplex,
        neighborhood_type="adj",
        neighborhood_dim={"r": 0, "k": -1},
        dimensions=64,
        walk_length=30,
        num_walks=200,
        workers=4,
    ):
        if isinstance(cmplex, SimplicialComplex) or isinstance(cmplex, CellComplex):
            if neighborhood_type == "adj":
                A = cmplex.adjacency_matrix(neighborhood_dim["r"])
            else:
                A = cmplex.coadjacency_matrix(neighborhood_dim["r"])
        elif isinstance(cmplex, CombinatorialComplex) or isinstance(
            cmplex, DynamicCombinatorialComplex
        ):
            if neighborhood_type == "adj":
                A = cmplex.adjacency_matrix(
                    neighborhood_dim["r"], neighborhood_dim["k"]
                )
            else:
                A = cmplex.coadjacency_matrix(
                    neighborhood_dim["k"], neighborhood_dim["r"]
                )
        else:
            ValueError(
                "input cmplex must be SimplicialComplex,CellComplex,CombinatorialComplex, or DynamicCombinatorialComplex "
            )
        G = nx.from_numpy_matrix(A)
        Node2Vec.__init__(
            self,
            graph=G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
        )
