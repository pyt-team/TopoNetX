# -*- coding: utf-8 -*-
"""
@author: Mustafa Hajij
"""

import networkx as nx
from karateclub import Diff2Vec

from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)

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


class CellDiff2Vec(Diff2Vec):
    """

    Parameters
    ==========

    """

    def __init__(
        self,
        diffusion_number: int = 10,
        diffusion_cover: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        super().__init__(
            diffusion_number=diffusion_number,
            diffusion_cover=diffusion_cover,
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

        super(CellDiff2Vec, self).fit(g)

    def get_embedding(self, get_dic=False):
        emb = super(CellDiff2Vec, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb
