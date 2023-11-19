"""Methods to lift a graph to a cell complex."""

__all__ = [
    "homology_cycle_cell_complex",
]


import networkx as nx

from toponetx.classes.cell_complex import CellComplex


def homology_cycle_cell_complex(G) -> CellComplex:
    """Get the cell complex obtained by adding homology cycles of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.

    Returns
    -------
    CellComplex
        The cell complex obtained by adding the homology cycles of the graph.
    """
    cycles = nx.cycle_basis(G)
    cx = CellComplex(G)
    cx.add_cells_from(cycles, rank=2)
    return cx
