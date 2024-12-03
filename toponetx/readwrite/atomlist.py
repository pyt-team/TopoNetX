"""Read and write complexes as a list of their atoms."""

from collections.abc import Generator, Hashable, Iterable
from itertools import combinations
from typing import Literal, overload

import networkx as nx

from toponetx.classes import CellComplex, SimplicialComplex

__all__ = [
    "generate_atomlist",
    "load_from_atomlist",
    "parse_atomlist",
    "write_atomlist",
]


def _atomlist_line(atom: Iterable[Hashable] | Hashable, attributes: dict) -> str:
    """Construct a single line of an atom list.

    Parameters
    ----------
    atom : iterable of hashable or hashable
        The atom to write.
    attributes : dict
        Attributes associated with the atom.

    Returns
    -------
    str
        The line of the atom list that represents the given atom.
    """
    line = " ".join(map(str, atom)) if isinstance(atom, Iterable) else str(atom)

    if len(attributes) > 0:
        line += " " + str(attributes)

    return line


def _generate_atomlist_simplicial(
    domain: SimplicialComplex,
) -> Generator[str, None, None]:
    """Generate an atom list from a simplicial complex.

    The list of atoms is truncated to only contain maximal simplices and simplices with user-defined attributes. All
    other simplices are implicitly contained by the simplex property already.

    Parameters
    ----------
    domain : SimplicialComplex
        The simplicial complex to be converted to an atom list.

    Yields
    ------
    str
        One line of the atom list, which corresponds to one atom of the complex together with its attributes.
    """
    for atom in domain.simplices:
        data = domain[atom].copy()
        data.pop("is_maximal", None)
        data.pop("membership", None)

        if len(data) == 0 and not domain.is_maximal(atom):
            continue

        yield _atomlist_line(atom, data)


def _generate_atomlist_cell(domain: CellComplex) -> Generator[str, None, None]:
    """Generate an atom list from a cell complex.

    The list of atoms is truncated to only contain maximal cells and cells with user-defined attributes. All
    other cells are implicitly contained already.
    We add a special `rank` attribute to cells of cardinality 2 that have rank 2 to differentiate them from edges.

    Parameters
    ----------
    domain : CellComplex
        The cell complex to be converted to an atom list.

    Yields
    ------
    str
        One line of the atom list, which corresponds to one atom of the complex together with its attributes.
    """
    for atom in domain.nodes:
        if len(domain.neighbors(atom)) == 0 or len(domain._G.nodes[atom]) > 0:
            yield _atomlist_line(atom, domain._G.nodes[atom])

    covered_edges = set()
    for cell in domain.cells:
        for edge in combinations(cell, 2):
            covered_edges.add(tuple(sorted(edge)))
    for atom in domain.edges:
        if len(domain._G.edges[atom]) > 0 or tuple(sorted(atom)) not in covered_edges:
            yield _atomlist_line(atom, domain._G.edges[atom])

    for atom in domain.cells:
        attributes = atom._attributes.copy()
        if len(atom) == 2:
            attributes["rank"] = 2
        yield _atomlist_line(atom, attributes)


def generate_atomlist(
    domain: CellComplex | SimplicialComplex,
) -> Generator[str, None, None]:
    """Generate an atom list from a complex.

    The list of atoms is truncated to only contain maximal atoms and atoms with user-defined attributes. All
    other atoms are implicitly contained already.
    For cell complexes, e add a special `rank` attribute to cells of cardinality 2 that have rank 2 to differentiate
    them from edges.

    Parameters
    ----------
    domain : CellComplex or SimplicialComplex
        The complex to be converted to an atom list.

    Yields
    ------
    str
        One line of the atom list, which corresponds to one atom of the complex together with its attributes.

    Examples
    --------
    Generate a list of atoms from a simplicial complex:

    >>> SC = tnx.SimplicialComplex()
    >>> SC.add_simplex((1,), weight=1.0)
    >>> SC.add_simplex((1, 2, 3), weight=4.0)
    >>> list(tnx.generate_atomlist(SC))
    ["1 {'weight': 1.0}", "1 2 3 {'weight': 4.0}"]

    Generate a list of atoms from a cell complex:

    >>> CC = tnx.CellComplex()
    >>> CC.add_cell((1, 2, 3), rank=2, weight=4.0)
    >>> list(tnx.generate_atomlist(CC))
    ["1 2 3 {'weight': 4.0}"]
    """
    if isinstance(domain, SimplicialComplex):
        yield from _generate_atomlist_simplicial(domain)
    elif isinstance(domain, CellComplex):
        yield from _generate_atomlist_cell(domain)
    else:
        raise TypeError(f"Expected a cell or simplicial complex, got {type(domain)}.")


@nx.utils.open_file(1, "wb")
def write_atomlist(
    domain: CellComplex | SimplicialComplex, path, encoding="utf-8"
) -> None:
    """Write an atom list to a file.

    Parameters
    ----------
    domain : CellComplex or SimplicialComplex
        The complex to be converted to an atom list.
    path : file or str
        File or filename to write. If a file is provided, it must be opened in `wb`
        mode. Filenames ending in .gz or .bz2 will be compressed.
    encoding : str, default="utf-8"
       Specify which encoding to use when writing file.

    Raises
    ------
    TypeError
        If the domain is not a cell or simplicial complex.
    """
    if not isinstance(domain, CellComplex | SimplicialComplex):
        raise TypeError(f"Expected a cell or simplicial complex, got {type(domain)}.")

    for line in generate_atomlist(domain):
        line += "\n"
        path.write(line.encode(encoding))


@overload
def load_from_atomlist(
    filepath: str, complex_type: Literal["cell"], nodetype=None, encoding="utf-8"
) -> CellComplex:  # numpydoc ignore=GL08
    pass


@overload
def load_from_atomlist(
    filepath: str, complex_type: Literal["simplicial"], nodetype=None, encoding="utf-8"
) -> SimplicialComplex:  # numpydoc ignore=GL08
    pass


@nx.utils.open_file(0, "rb")
def load_from_atomlist(
    path, complex_type: Literal["cell", "simplicial"], nodetype=None, encoding="utf-8"
) -> CellComplex | SimplicialComplex:
    """Load a complex from an atom list.

    Parameters
    ----------
    path : file or str
        File or filename to read. If a file is provided, it must be opened in `rb`
        mode. Filenames ending in .gz or .bz2 will be uncompressed.
    complex_type : {"cell", "simplicial"}
        The type of complex that should be constructed based on the atom list.
    nodetype : callable, optional
        Convert node data from strings to the specified type.
    encoding : str, default="utf-8"
         Specify which encoding to use when reading file.

    Returns
    -------
    CellComplex or SimplicialComplex
        The complex that was loaded from the atom list.

    Raises
    ------
    ValueError
        If the complex type is unknown.
    """
    return parse_atomlist(
        (line.decode(encoding) for line in path), complex_type, nodetype
    )


@overload
def parse_atomlist(
    lines: Iterable[str], complex_type: Literal["cell"], nodetype=None
) -> CellComplex:  # numpydoc ignore=GL08
    pass


@overload
def parse_atomlist(
    lines: Iterable[str], complex_type: Literal["simplicial"], nodetype=None
) -> SimplicialComplex:  # numpydoc ignore=GL08
    pass


def parse_atomlist(
    lines: Iterable[str], complex_type: Literal["cell", "simplicial"], nodetype=None
) -> CellComplex | SimplicialComplex:
    """Parse an atom list.

    Parameters
    ----------
    lines : iterable of str
        List of lines.
    complex_type : {"cell", "simplicial"}
        Complex type.
    nodetype : callable, optional
        Node type.

    Returns
    -------
    CellComplex or SimplicialComplex
        The complex that was parsed from the atom list.

    Raises
    ------
    ValueError
        If the complex type is unknown.
    """
    from ast import literal_eval

    domain: CellComplex | SimplicialComplex
    if complex_type == "cell":
        domain = CellComplex()
    elif complex_type == "simplicial":
        domain = SimplicialComplex()
    else:
        raise ValueError(f"Unknown complex type {complex_type}.")

    for line in lines:
        attributes_pos = line.find("{")
        if attributes_pos == -1:
            elements_str = line
            attributes = {}
        else:
            elements_str = line[:attributes_pos].strip()
            attributes = literal_eval(line[attributes_pos:])

        elements = elements_str.split(" ")
        elements = [e.strip() for e in elements]
        if nodetype is not None:
            elements = [nodetype(e) for e in elements]

        if isinstance(domain, CellComplex):
            if "rank" in attributes:
                rank = attributes.pop("rank")
            else:
                rank = min(len(elements) - 1, 2)

            if rank == 0:
                domain.add_node(elements[0], **attributes)
            else:
                domain.add_cell(elements, rank=rank, **attributes)
        elif isinstance(domain, SimplicialComplex):
            domain.add_simplex(elements, **attributes)

    return domain
