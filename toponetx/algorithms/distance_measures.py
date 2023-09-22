"""Module to distance measures on topological domains."""
from collections.abc import Collection, Hashable, Iterable, Iterator
from warnings import warn

import networkx as nx
import numpy as np

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.exception import TopoNetXError

__all__ = ["node_diameters", "cell_diameters", "diameter", "cell_diameter"]


def node_diameters(complex):
    """Return the node diameters of the connected components in cell complex.

    Returns
    -------
    diameters : list
        List of the diameters of the s-components.
    components : list
        List of the s-component nodes.

    Example
    -------
    >>> CC = CellComplex()
    >>> CC.add_cell([2,3,4],rank=2)
    >>> CC.add_cell([5,6,7],rank=2)
    >>> list(node_diameters(CC))
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(node_diameters(CCC))
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(node_diameters(CHG))
    """
    node_dict, A = complex.node_to_all_cell_adjacnecy_matrix(index=True)
    if isinstance(complex, ColoredHyperGraph) and not isinstance(
        complex, CombinatorialComplex
    ):
        node_dict = {v: k[0] for k, v in node_dict.items()}
    else:
        node_dict = {v: k for k, v in node_dict.items()}

    G = nx.from_scipy_sparse_array(A)
    diams = []
    comps = []
    for c in nx.connected_components(G):
        diamc = nx.diameter(G.subgraph(c))
        temp = set()
        for e in c:
            temp.add(node_dict[e])
        comps.append(temp)
        diams.append(diamc)
    return diams, comps


def cell_diameters(complex, s=1):
    """Return the cell diameters of the s_cell_connected component subgraphs.

    Parameters
    ----------
    complex : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.

    Returns
    -------
    list of diameters : list
        List of cell_diameters for s-cell component subcomplexes in the cell complex.

    list of component : list
        List of the cell uids in the s-cell component subcomplexes.

    Example
    -------
    >>> CC = CellComplex()
    >>> CC.add_cell([2,3,4],rank=2)
    >>> CC.add_cell([5,6,7],rank=2)
    >>> list(cell_diameters(CC))
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(cell_diameters(CCC))
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(cell_diameters(CHG))
    """
    if not isinstance(complex, (CellComplex, CombinatorialComplex, ColoredHyperGraph)):
        raise ValueError("Input complex {complex} is not supported.")
    coldict, A = complex.all_cell_to_node_coadjacnecy_matrix(index=True)
    coldict = {v: k for k, v in coldict.items()}

    G = nx.from_scipy_sparse_array(A)
    diams = []
    comps = []
    for c in nx.connected_components(G):
        diamc = nx.diameter(G.subgraph(c))
        temp = set()
        for e in c:
            temp.add(coldict[e])
        comps.append(temp)
        diams.append(diamc)
    return diams, comps


def diameter(complex) -> int:
    """Return length of the longest shortest s-walk between nodes.

    Parameters
    ----------
    complex : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.

    Returns
    -------
    diameter : int

    Raises
    ------
    TopoNetXError
        If the cell complex is not s-cell-connected

    Notes
    -----
    Two nodes are s-adjacent if they share s cells.
    Two nodes v_start and v_end are s-walk connected if there is a sequence of
    nodes v_start, v_1, v_2, ... v_n-1, v_end such that consecutive nodes
    are s-adjacent. If the cell complex is not connected, an error will be raised.

    Example
    -------
    >>> CC = CellComplex()
    >>> CC.add_cell([2,3,4],rank=2)
    >>> CC.add_cell([5,6,7],rank=2)
    >>> CC.add_cell([2,5],rank=2)
    >>> diameter(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> diameter(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> diameter(CHG)

    """
    if not isinstance(complex, (CellComplex, CombinatorialComplex, ColoredHyperGraph)):
        raise ValueError("Input complex {complex} is not supported.")
    A = complex.node_to_all_cell_adjacnecy_matrix()
    G = nx.from_scipy_sparse_array(A)
    if nx.is_connected(G):
        return nx.diameter(G)
    raise TopoNetXError("cc is not connected.")


def cell_diameter(complex, s: int = None) -> int:
    """Return the length of the longest shortest s-walk between cells.

    Parameters
    ----------
    complex : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.

    Return
    ------
    cell_diameter : int

    Raises
    ------
    TopoNetXError
        If cell complex is not s-cell-connected

    Notes
    -----
    Two cells are s-coadjacent if they share s nodes.
    Two nodes e_start and e_end are s-walk connected if there is a sequence of
    cells (one or two dimensional) e_start, e_1, e_2, ... e_n-1, e_end such that consecutive cells
    are s-coadjacent. If the cell complex is not connected, an error will be raised.
    Example
    -------
    >>> CC = CellComplex()
    >>> CC.add_cell([2,3,4],rank=2)
    >>> CC.add_cell([5,6,7],rank=2)
    >>> CC.add_cell([2,5],rank=1)
    >>> cell_diameter(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> cell_diameter(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> cell_diameter(CHG)
    """
    if not isinstance(complex, (CellComplex, CombinatorialComplex, ColoredHyperGraph)):
        raise ValueError("Input complex {complex} is not supported.")
    A = complex.all_cell_to_node_coadjacnecy_matrix()
    G = nx.from_scipy_sparse_array(A)
    if nx.is_connected(G):
        return nx.diameter(G)
    raise TopoNetXError(f"cell complex is not s-connected. s={s}")
