# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:02:53 2022

@author: musta
"""

import networkx as nx
import numpy as np
from karateclub import GLEE

from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)


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


class HOGLEE(GLEE):
    """

    Parameters
    ==========

    """

    def __init__(self, dimensions: int = 3, seed: int = 42):

        super().__init__(dimensions=dimensions, seed=seed)

        self.A = []
        self.ind = []

    def fit(self, cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        self.ind, self.A = _neighbohood_from_complex(
            cmplex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(HOGLEE, self).fit(g)

    def get_embedding(self, get_dic=True):
        emb = super(HOGLEE, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb
