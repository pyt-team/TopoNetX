# -*- coding: utf-8 -*-
"""
@author: Mustafa Hajij
"""

import networkx as nx
from karateclub import DeepWalk, Node2Vec

from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)
from toponetx.utils.structure import sparse_array_to_neighborhood_dict

# from node2vec import Node2Vec


def _neighbohood_from_complex(
    cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}
):

    if isinstance(cmplex, SimplicialComplex) or isinstance(cmplex, CellComplex):
        if neighborhood_type == "adj":
            ind, A = cmplex.adjacency_matrix(neighborhood_dim["r"], index=True)

        else:
            ind, A = cmplex.coadjacency_matrix(neighborhood_dim["r"], index=True)
    elif isinstance(cmplex, CombinatorialComplex) or isinstance(
        cmplex, DynamicCombinatorialComplex
    ):
        if neighborhood_type == "adj":
            ind, A = cmplex.adjacency_matrix(
                neighborhood_dim["r"], neighborhood_dim["k"], index=True
            )
        else:
            ind, A = cmplex.coadjacency_matrix(
                neighborhood_dim["k"], neighborhood_dim["r"], index=True
            )
    else:
        ValueError(
            "input cmplex must be SimplicialComplex,CellComplex,CombinatorialComplex, or DynamicCombinatorialComplex "
        )

    return ind, A


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

    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        super().__init__(
            walk_number=walk_number,
            walk_length=walk_length,
            p=p,
            q=q,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            min_count=min_count,
            seed=seed,
        )

        self.A = []
        self.ind = []

    def fit(self, cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):

        self.ind, self.A = _neighbohood_from_complex(
            cmplex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(Cell2Vec, self).fit(g)

    def get_embedding(self, get_dic=False):
        emb = super(Cell2Vec, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb


class DeepCell(DeepWalk):
    """

    Parameters
    ==========

    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        super().__init__(
            walk_number=walk_number,
            walk_length=walk_length,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            min_count=min_count,
            seed=seed,
        )

        self.A = []
        self.ind = []

    def fit(self, cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        self.ind, self.A = _neighbohood_from_complex(
            cmplex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(DeepCell, self).fit(g)

    def get_embedding(self, get_dic=False):
        emb = super(DeepCell, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb
