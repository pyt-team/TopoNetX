"""Methods to lift a graph to a cell complex."""

__all__ = [
    "homology_cycle_cell_complex",
]


import networkx as nx

from toponetx import CellComplex


def homology_cycle_cell_complex(G):
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.

    Returns
    -------
    CellComplex
        the cell complex obtained by adding the homology cycles of the graph
    """
    cycles = nx.cycle_basis(G)
    cx = CellComplex(G)  # lift to graph
    cx.add_cells_from(cycles, rank=2)  # add basis cycles
    return cx
