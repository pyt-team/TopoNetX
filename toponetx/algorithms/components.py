"""Module to compute connected components on topological domains."""

from collections.abc import Generator, Hashable
from typing import Literal, TypeVar, overload

import networkx as nx

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import CombinatorialComplex

__all__ = [
    "connected_component_subcomplexes",
    "connected_components",
    "s_component_subcomplexes",
    "s_connected_components",
]

# In this module, only cell complexes, combinatorial complexes and colored
# hypergraphs are supported. We bound the type variable to these types.
ComplexType = CellComplex | CombinatorialComplex | ColoredHyperGraph
ComplexTypeVar = TypeVar("ComplexTypeVar", bound=ComplexType)


@overload
def s_connected_components(
    domain: ComplexType,
    s: int,
    cells: Literal[True] = ...,
    return_singletons: bool = ...,
) -> Generator[set[tuple[Hashable, ...]], None, None]:  # numpydoc ignore=GL08
    pass


@overload
def s_connected_components(
    domain: ComplexType, s: int, cells: Literal[False], return_singletons: bool = ...
) -> Generator[set[Hashable], None, None]:  # numpydoc ignore=GL08
    pass


@overload
def s_connected_components(
    domain: ComplexType, s: int, cells: bool, return_singletons: bool = ...
) -> Generator[
    set[Hashable] | set[tuple[Hashable, ...]], None, None
]:  # numpydoc ignore=GL08
    pass


def s_connected_components(
    domain: ComplexType, s: int, cells: bool = True, return_singletons: bool = False
) -> Generator[set[Hashable] | set[tuple[Hashable, ...]], None, None]:
    """Return generator for the s-connected components.

    Parameters
    ----------
    domain : CellComplex or CombinatorialComplex or ColoredHyperGraph
        The domain on which to compute the s-connected components.
    s : int
        The number of intersections between pairwise consecutive cells.
    cells : bool, default=True
        If True will return cell components, if False will return node components.
    return_singletons : bool, default=False
        When True, returns singleton connected components.

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
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(tnx.s_connected_components(CC, s=1, cells=False))
    [{2, 3, 4}, {5, 6, 7}]
    >>> list(tnx.s_connected_components(CC, s=1, cells=True))
    [{(2, 3), (2, 3, 4), (2, 4), (3, 4)}, {(5, 6), (5, 6, 7), (5, 7), (6, 7)}]
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(tnx.s_connected_components(CHG, s=1, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(tnx.s_connected_components(CC, s=1, cells=False))
    [{2, 3, 4, 5, 6, 7}]
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(tnx.s_connected_components(CCC, s=1, cells=False))
    """
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
        node_dict = {v: k for k, v in node_dict.items()}

        G = nx.from_scipy_sparse_array(A)
        for c in nx.connected_components(G):
            if not return_singletons and len(c) == 1:
                continue
            if isinstance(domain, CellComplex):
                yield {node_dict[n] for n in c}
            else:
                yield {next(iter(node_dict[n])) for n in c}


def s_component_subcomplexes(
    domain: ComplexTypeVar,
    s: int = 1,
    cells: bool = True,
    return_singletons: bool = False,
) -> Generator[ComplexTypeVar, None, None]:
    """Return a generator for the induced subcomplexes of s_connected components.

    Removes singletons unless return_singletons is set to True.

    Parameters
    ----------
    domain : CellComplex or CombinatorialComplex or ColoredHyperGraph
        The domain for which to compute the the s-connected subcomplexes.
    s : int, default=1
        The number of intersections between pairwise consecutive cells.
    cells : bool, default=True
        Determines if cell or node components are desired. Returns
        subcomplexes equal to the cell complex restricted to each set of nodes(cells) in the
        s-connected components or s-cell-connected components.
    return_singletons : bool, default=False
        When True, returns singletons connected components.

    Yields
    ------
    Complex
        Returns subcomplexes generated by the cells (or nodes) in the
        s-cell(node) components of complex.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(tnx.s_component_subcomplexes(CC, 1, cells=False))
    >>> CCC = CC.to_combinatorial_complex()
    >>> list(tnx.s_component_subcomplexes(CCC, s=1, cells=False))
    >>> CHG = CC.to_colored_hypergraph()
    >>> list(tnx.s_component_subcomplexes(CHG, s=1, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(tnx.s_component_subcomplexes(CC, s=1, cells=False))
    """
    for c in s_connected_components(
        domain, s=s, cells=cells, return_singletons=return_singletons
    ):
        if cells:
            yield domain.restrict_to_cells(list(c))
        else:
            yield domain.restrict_to_nodes(list(c))


@overload
def connected_components(
    domain: ComplexType, cells: Literal[True] = ..., return_singletons: bool = ...
) -> Generator[set[tuple[Hashable, ...]], None, None]:  # numpydoc ignore=GL08
    pass


@overload
def connected_components(
    domain: ComplexType, cells: Literal[False], return_singletons: bool = ...
) -> Generator[set[Hashable], None, None]:  # numpydoc ignore=GL08
    pass


@overload
def connected_components(
    domain: ComplexType, cells: bool, return_singletons: bool = ...
) -> Generator[
    set[Hashable] | set[tuple[Hashable, ...]], None, None
]:  # numpydoc ignore=GL08
    pass


def connected_components(
    domain: ComplexType, cells: bool = False, return_singletons: bool = True
) -> Generator[set[Hashable] | set[tuple[Hashable, ...]], None, None]:
    """Compute s-connected components with s=1.

    Same as s_connected_component` with s=1, but nodes returned.

    Parameters
    ----------
    domain : Complex
        Supported complexes are cell/combintorial and hypegraphs.
    cells : bool, default=False
        If True will return cell components, if False will return node components.
    return_singletons : bool, default=True
        When True, returns singletons connected components.

    Yields
    ------
    set[Hashable] | set[tuple[Hashable, ...]]
        Yields subcomplexes generated by the cells (or nodes) in the
        cell(node) components of complex.

    See Also
    --------
    s_connected_components
        Method implemented in the library for s-connected-components.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(tnx.connected_components(CC, cells=False))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(tnx.CC.connected_components(CC, cells=False))
    """
    yield from s_connected_components(
        domain, s=1, cells=cells, return_singletons=return_singletons
    )


def connected_component_subcomplexes(
    domain: ComplexTypeVar, return_singletons: bool = True
) -> Generator[ComplexTypeVar, None, None]:
    """Compute connected component subcomplexes with s=1.

    Same as :meth:`s_component_subcomplexes` with s=1.

    Parameters
    ----------
    domain : CellComplex or CombinaorialComplex or ColoredHyperGraph
        The domain for which to compute the the connected subcomplexes.
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
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([2, 3, 4], rank=2)
    >>> CC.add_cell([5, 6, 7], rank=2)
    >>> list(tnx.connected_component_subcomplexes(CC))
    >>> CC.add_cell([4, 5], rank=1)
    >>> list(tnx.connected_component_subcomplexes(CC))
    """
    yield from s_component_subcomplexes(domain, return_singletons=return_singletons)
