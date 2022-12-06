# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 00:38:45 2022

@author: Mustafa Hajij
"""

import random

from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.utils.structure import sparse_array_to_neighborhood_dict


def _TraverseAndSelect(
    length, num_walks, hyperedges, vertexMemberships, alpha=1.0, beta=0
):
    """
    Generate random walks on a combinatorial complex using the Traverse and Select algorithm.

    This function generates random walks of a given length on a given combinatorial complex.
    The Traverse and Select algorithm is used to decide which cell in a  combinatorial complex to traverse at each step.
    The length and number of walks can be specified, as well as the combinatorial complex structure and vertex membership information.
    The alpha and beta parameters control the probabilities of traversing a cell or a vertex, respectively.

    :param length: The length of each random walk.
    :type length: int
    :param num_walks: The number of random walks to generate.
    :type num_walks: int
    :param hyperedges: A dictionary representing the structure of the combinatorial complex.
    :type hyperedges: dict
    :param vertexMemberships: A dictionary indicating the hyperedges that each vertex belongs to.
    :type vertexMemberships: dict
    :param alpha: The probability of traversing a hyperedge. Default is 1.0.
    :type alpha: float
    :param beta: The probability of traversing a vertex. Default is 0.
    :type beta: float
    :return: The generated random walks.
    :rtype: list of list of str
    """

    walksTAS = []
    for hyperedge_index in hyperedges:
        walk_hyperedge = []
        for _ in range(num_walks):
            curr_vertex = random.choice(hyperedges[hyperedge_index])

            curr_hyperedge_num = hyperedge_index
            curr_hyperedge = hyperedge_index
            for i in range(length):
                proba = (float(alpha) / len(vertexMemberships[curr_vertex])) + beta
                if random.random() < proba:
                    adjacent_vertices = hyperedges[curr_hyperedge]
                    curr_vertex = random.choice(adjacent_vertices)
                walk_hyperedge.append(str(curr_hyperedge_num))
                adjacent_hyperedges = vertexMemberships[curr_vertex]
                curr_hyperedge_num = random.choice(adjacent_hyperedges)
                curr_hyperedge = curr_hyperedge_num
            walksTAS.append(walk_hyperedge)
    return walksTAS


def _SubsampleAndTraverse(
    length, num_walks, hyperedges, vertexMemberships, alpha=1.0, beta=0
):
    """
    Generate random walks on a combinatorial complex using the Subsample and Traverse algorithm.

    This function generates random walks of a given length on a given combinatorial complex.
    The Subsample and Traverse algorithm is used to decide which hyperedge to traverse at each step.
    The length and number of walks can be specified, as well as the combinatorial complex structure and vertex membership information.
    The alpha and beta parameters control the probabilities of traversing a hyperedge or a vertex, respectively.

    :param length: The length of each random walk.
    :type length: int
    :param num_walks: The number of random walks to generate.
    :type num_walks: int
    :param hyperedges: A dictionary representing the structure of the combinatorial complex.
    :type hyperedges: dict
    :param vertexMemberships: A dictionary indicating the hyperedges that each vertex belongs to.
    :type vertexMemberships: dict
    :param alpha: The probability of traversing a hyperedge. Default is 1.0.
    :type alpha: float
    :param beta: The probability of traversing a vertex. Default is 0.
    :type beta: float
    :return: The generated random walks.
    :rtype: list of list of str
    """
    walksSAT = []
    for hyperedge_index in hyperedges:
        walk_vertex = []
        curr_vertex = random.choice(hyperedges[hyperedge_index])
        for _ in range(num_walks):
            hyperedge_num = hyperedge_index
            curr_hyperedge = hyperedge_num
            for i in range(length):
                proba = (float(alpha) / len(hyperedges[curr_hyperedge])) + beta
                if random.random() < proba:
                    adjacent_hyperedges = vertexMemberships[curr_vertex]
                    hyperedge_num = random.choice(adjacent_hyperedges)
                    curr_hyperedge = hyperedge_num
                walk_vertex.append(str(curr_vertex))
                curr_vertex = random.choice(hyperedges[curr_hyperedge])
            walksSAT.append(walk_vertex)
    return walksSAT
