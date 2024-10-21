"""Utils loading datasets."""

import networkx as nx
import numpy as np
import pandas as pd
import requests


def load_ppi():
    """Load the protein-protein-interaction graph.

    In the graph, high interaction score is represented as low weight.

    Returns
    -------
    networkx.Graph
        Graph with nodes that are proteins and edges corresponding to their
        chemical interactions.

    References
    ----------
    https://towardsdatascience.com/visualizing-protein-networks-in-python-58a9b51be9d5
    """
    protein_list = [
        "TPH1",
        "COMT",
        "SLC18A2",
        "HTR1B",
        "HTR2C",
        "HTR2A",
        "MAOA",
        "TPH2",
        "HTR1A",
        "HTR7",
        "SLC6A4",
        "GABBR2",
        "POMC",
        "GNAI3",
        "NPY",
        "ADCY1",
        "PDYN",
        "GRM2",
        "GRM3",
        "GABBR1",
    ]
    proteins = "%0d".join(protein_list)
    url = f"https://string-db.org/api/tsv/network?identifiers={proteins}&species=9606"
    r = requests.get(url, timeout=10)

    lines = r.text.split("\n")
    data = [line.split("\t") for line in lines]
    protein_df = pd.DataFrame(data[1:-1], columns=data[0])

    interactions = protein_df[["preferredName_A", "preferredName_B", "score"]]

    G = nx.Graph(name="Protein Interaction Graph")
    interactions = np.array(interactions)
    for interaction in interactions:
        protein_a = interaction[0]
        protein_b = interaction[1]
        interaction_weight = float(interaction[2])
        G.add_weighted_edges_from([(protein_a, protein_b, interaction_weight)])
    return G
