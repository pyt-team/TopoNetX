"""Module to compute connected components on topological domains."""
from collections.abc import Generator, Hashable

import networkx as nx

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.complex import Complex

__all__ = [
    "s_connected_components",
    "s_component_subcomplexes",
    "connected_components",
    "connected_component_subcomplexes",
]


def s_connected_components(
    domain: Complex, s: int = 1, cells: bool = True, return_singletons: bool = False
) -> Generator[set[Hashable] | set[tuple[Hashable, ...]], None, None]:
    """Return generator for the s-connected components.

    Parameters
    ----------
    domain : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.
    cells : bool, optional
        If True will return cell components, if False will return node components.
    return_singletons : bool, optional
        When True, returns singletons connected components.

    Notes
    -----
    If cells=True, this method returns the s-cell-connected components as
    lists of lists of cell uids.
    An s-cell-component has the property that for any two cells e1 and e2
    there is a sequence of cells starting with e1 and ending with e2
    such that pairwise adjacent cells in the sequence intersect in at least
    s nodes. If s=1 these are the path components of the cell complex.

    If cells=False this method returns s-node-connected components.
    A list of sets of uids of the nodes which are s-walk connected.
    Two nodes v1 and v2 are s-walk-connected if there is a
    sequence of nodes starting with v1 and ending with v2 such that pairwise
    adjacent nodes in the sequence share s cells. If s=1 these are the
    path components of the cell complex .

    Yields
    ------
    set[Hashable] or set[tuple[Hashable, ...]]
        Returns sets of uids of the cells (or nodes) in the s-cells(node)
        components of the complex.

    Examples
    --------
    >>> CC = CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(s_connected_components(CC, s=1, cells=False))
    >>> #  [{2, 3, 4}, {5, 6, 7}]
    >>> list(s_connected_components(CC, s=1, cells=True))
    >>> # [{(2, 3), (2, 3, 4), (2, 4), (3, 4)},
    >>> # {(5, 6), (5, 6, 7), (5, 7), (6, 7)}]
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(s_connected_components(CHG, s=1, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(s_connected_components(CC, s=1, cells=False))
    >>> # [{2, 3, 4, 5, 6, 7}]
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(s_connected_components(CCC, s=1, cells=False))
    """
    if not isinstance(domain, (CellComplex, ColoredHyperGraph, CombinatorialComplex)):
        raise TypeError(f"Input complex {domain} is not supported.")

    if cells:
        cell_dict, A = domain.all_cell_to_node_coadjacency_matrix(s=s, index=True)
        cell_dict = {v: k for k, v in cell_dict.items()}
        G = nx.from_scipy_sparse_array(A)

        for c in nx.connected_components(G):
            if not return_singletons and len(c) == 1:
                continue
            if isinstance(domain, CellComplex):
                yield {cell_dict[n] for n in c}
            else:
                yield {tuple(cell_dict[n]) for n in c}

    else:
        node_dict, A = domain.node_to_all_cell_adjacnecy_matrix(s=s, index=True)
        if isinstance(domain, ColoredHyperGraph) and not isinstance(
            domain, CombinatorialComplex
        ):
            node_dict = {v: k[0] for k, v in node_dict.items()}
        else:
            node_dict = {v: k for k, v in node_dict.items()}
        G = nx.from_scipy_sparse_array(A)
        for c in nx.connected_components(G):
            if not return_singletons:
                if len(c) == 1:
                    continue
            if isinstance(domain, CellComplex):
                yield {node_dict[n] for n in c}
            else:
                yield {tuple(node_dict[n])[0] for n in c}


def s_component_subcomplexes(
    domain: Complex, s: int = 1, cells: bool = True, return_singletons: bool = False
) -> Generator[Complex, None, None]:
    """Return a generator for the induced subcomplexes of s_connected components.

    Removes singletons unless return_singletons is set to True.

    Parameters
    ----------
    domain : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    s : int, optional
        The number of intersections between pairwise consecutive cells.
    cells : bool, optional
        Determines if cell or node components are desired. Returns
        subcomplexes equal to the cell complex restricted to each set of nodes(cells) in the
        s-connected components or s-cell-connected components.
    return_singletons : bool, optional
        When True, returns singletons connected components.

    Yields
    ------
    Complex
        Returns subcomplexes generated by the cells (or nodes) in the
        s-cell(node) components of complex.

    Examples
    --------
    >>> CC = CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(s_component_subcomplexes(CC, 1, cells=False))
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(s_component_subcomplexes(CCC, s=1, cells=False))
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(s_component_subcomplexes(CHG, s=1, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(s_component_subcomplexes(CC, s=1, cells=False))
    """
    for idx, c in enumerate(
        s_connected_components(
            domain, s=s, cells=cells, return_singletons=return_singletons
        )
    ):
        if cells:
            yield domain.restrict_to_cells(list(c))
        else:
            yield domain.restrict_to_nodes(list(c))


def connected_components(
    domain: Complex, cells: bool = False, return_singletons: bool = True
) -> Generator[set[Hashable] | set[tuple[Hashable, ...]], None, None]:
    """Compute s-connected components with s=1.

    Same as s_connected_component` with s=1, but nodes returned.

    Parameters
    ----------
    domain : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    cells : bool, optional
        If True will return cell components, if False will return node components.
    return_singletons : bool, optional
        When True, returns singletons connected components.

    Yields
    ------
    set[Hashable] | set[tuple[Hashable, ...]], None, None
        Yields subcomplexes generated by the cells (or nodes) in the
        cell(node) components of complex.

    See Also
    --------
    s_connected_components
        Method implemented in the library for s-connected-components.

    Examples
    --------
    >>> CC = CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(connected_components(CC, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(CC.connected_components(CC, cells=False))
    """
    yield from s_connected_components(domain, s=1, cells=cells, return_singletons=True)


def connected_component_subcomplexes(
    domain: Complex, return_singletons: bool = True
) -> Generator[Complex, None, None]:
    """Compute connected component subcomplexes with s=1.

    Same as :meth:`s_component_subcomplexes` with s=1.

    Parameters
    ----------
    domain : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    return_singletons : bool, optional
        When True, returns singletons connected components.

    Yields
    ------
    Complex
        The subcomplexes obtained as the resstriction of the input complex on its connected components.

    See Also
    --------
    s_component_subcomplexes
        Method implemented in the library to get a generator for s-component subcomplexes.

    Examples
    --------
    >>> CC = CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(connected_component_subcomplexes(CC))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(connected_component_subcomplexes(CC))
    """
    yield from s_component_subcomplexes(domain, return_singletons=return_singletons)
