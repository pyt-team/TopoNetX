"""Generators for some classic cell complexes."""

from toponetx.classes import CellComplex
from toponetx.classes.cell import Cell

__all__ = ["pyrmaid_complex", "single_cell_complex"]


def single_cell_complex(n: int) -> CellComplex:
    """Generate a cell complex with a single cell of length n.

    Parameters
    ----------
    n : int
        Length that is provided for the purposes of generating a cell complex.

    Returns
    -------
    CellComplex
        Returns a cell complex with a single cell of length n.
    """
    c = Cell(list(range(1, n)))

    return CellComplex([c])


def pyrmaid_complex(n: int) -> CellComplex:
    """Generate a pyrmaid cell complex with a base cell of length n.

    Parameters
    ----------
    n : int
        Length that is provided for the purposes of generating a pyrmaid cell complex.

    Returns
    -------
    CellComplex
        Returns a pyrmaid cell complex with a base cell of length n.
    """
    c = Cell(list(range(1, n)))
    cc = CellComplex([c])
    for i in range(1, n - 1):
        cc.add_cell([0, i, i + 1], rank=2)
    cc.add_cell([0, n - 1, 1], rank=2)
    return cc
