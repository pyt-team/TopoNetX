"""Generators for random cell complexes."""

import networkx as nx

from toponetx.classes import CellComplex
from toponetx.transform.graph_to_cell_complex import homology_cycle_cell_complex

__all__ = [
    "np_cell_complex",
]


def np_cell_complex(n: int, p: float, seed=None) -> CellComplex:
    """Generate $CC_{(n,p)}$ complex.

    In an $(n,p)$ cell complex, a sample from an Erdős-Rényi graph is drawn, also called $G_{n,p}$,
    and all basis cycles are identified with a cell of a cell complex.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        The probability of an edge between two nodes.
    seed : int or random_state (optional)
        Indicator of random number generation state.

    Returns
    -------
    CellComplex
        An (n,p) cell complex.
    """
    graph = nx.gnp_random_graph(n, p, seed)
    return homology_cycle_cell_complex(graph)
