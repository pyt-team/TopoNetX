"""Module to distance measures on topological domains."""

from collections.abc import Hashable

import networkx as nx

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex

__all__ = ["cell_diameter", "cell_diameters", "diameter", "node_diameters"]

# In this module, only cell complexes, combinatorial complexes and colored
# hypergraphs are supported.
ComplexType = CellComplex | CombinatorialComplex | ColoredHyperGraph


def node_diameters(domain: ComplexType) -> tuple[list[int], list[set[Hashable]]]:
    """Return the node diameters of the connected components in cell complex.

    Parameters
    ----------
    domain : CellComplex or CombinaorialComplex or ColoredHyperGraph
        The complex to be used to generate the node diameters for.

    Returns
    -------
    diameters : list
        List of the diameters of the s-components.
    components : list
        List of the s-component nodes.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> tnx.node_diameters(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> tnx.node_diameters(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> tnx.node_diameters(CHG)
    """
    node_dict, A = domain.node_to_all_cell_adjacnecy_matrix(index=True)
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


def cell_diameters(domain: ComplexType, s: int = 1) -> tuple[list[int], list[set[int]]]:
    """Return the cell diameters of the s_cell_connected component subgraphs.

    Parameters
    ----------
    domain : CellComplex or CombinatorialComplex or ColoredHyperGraph
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.

    Returns
    -------
    list of diameters : list
        List of cell_diameters for s-cell component subcomplexes in the cell complex.

    list of component : list
        List of the cell uids in the s-cell component subcomplexes.

    Examples
    --------
    >>> CC = CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> tnx.cell_diameters(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> tnx.cell_diameters(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> tnx.cell_diameters(CHG)
    """
    if not isinstance(domain, CellComplex | CombinatorialComplex | ColoredHyperGraph):
        raise TypeError(f"Input complex {domain} is not supported.")
    coldict, A = domain.all_cell_to_node_coadjacency_matrix(index=True)
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


def diameter(domain: ComplexType) -> int:
    """Return length of the longest shortest s-walk between nodes.

    Parameters
    ----------
    domain : CellComplex or CombinatorialComplex or ColoredHyperGraph
        Supported complexes are cell/combintorial and hypegraphs.

    Returns
    -------
    int
        The diameter of the longest shortest s-walk between nodes.

    Raises
    ------
    RuntimeError
        If the cell complex is not s-cell-connected

    Notes
    -----
    Two nodes are s-adjacent if they share s cells.
    Two nodes v_start and v_end are s-walk connected if there is a sequence of
    nodes v_start, v_1, v_2, ... v_n-1, v_end such that consecutive nodes
    are s-adjacent. If the cell complex is not connected, an error will be raised.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> CC.add_cell([2, 5], rank=2)
    >>> tnx.diameter(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> tnx.diameter(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> tnx.diameter(CHG)
    """
    if not isinstance(domain, CellComplex | CombinatorialComplex | ColoredHyperGraph):
        raise TypeError(f"Input complex {domain} is not supported.")
    A = domain.node_to_all_cell_adjacnecy_matrix()
    G = nx.from_scipy_sparse_array(A)
    if nx.is_connected(G):
        return nx.diameter(G)
    raise RuntimeError("cc is not connected.")


def cell_diameter(domain: ComplexType, s: int = 1) -> int:
    """Return the length of the longest shortest s-walk between cells.

    Parameters
    ----------
    domain : CellComplex or CombinatorialComplex or ColoredHyperGraph
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, default=1
        The number of intersections between pairwise consecutive cells.

    Returns
    -------
    int
        Returns the length of the longest shortest s-walk between cells.

    Raises
    ------
    RuntimeError
        If cell complex is not s-cell-connected

    Notes
    -----
    Two cells are s-coadjacent if they share s nodes.
    Two nodes e_start and e_end are s-walk connected if there is a sequence of
    cells (one or two dimensional) e_start, e_1, e_2, ... e_n-1, e_end such that consecutive cells
    are s-coadjacent. If the cell complex is not connected, an error will be raised.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> CC.add_cell([2, 5], rank=1)
    >>> tnx.cell_diameter(CC)
    >>> CCC = CC.to_combinatorial_complex()
    >>> tnx.cell_diameter(CCC)
    >>> CHG = CC.to_colored_hypergraph()
    >>> tnx.cell_diameter(CHG)
    """
    if not isinstance(domain, CellComplex | CombinatorialComplex | ColoredHyperGraph):
        raise TypeError(f"Input complex {domain} is not supported.")
    A = domain.all_cell_to_node_coadjacency_matrix()
    G = nx.from_scipy_sparse_array(A)
    if nx.is_connected(G):
        return nx.diameter(G)
    raise RuntimeError(f"cell complex is not s-connected. s={s}")
