"""Module to compute spectra."""

from typing import Any, Literal

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import diags

from toponetx.classes import CombinatorialComplex
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.path_complex import PathComplex
from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = [
    "_normalize",
    "cell_complex_adjacency_spectrum",
    "cell_complex_hodge_laplacian_spectrum",
    "combinatorial_complex_adjacency_spectrum",
    "hodge_laplacian_eigenvectors",
    "laplacian_beltrami_eigenvectors",
    "laplacian_spectrum",
    "set_hodge_laplacian_eigenvector_attrs",
    "simplicial_complex_adjacency_spectrum",
    "simplicial_complex_hodge_laplacian_spectrum",
]


def _normalize(f: dict[Any, Any]) -> dict[Any, Any]:
    """Normalize.

    Parameters
    ----------
    f : callable
        A scalar function on nodes of a graph.

    Returns
    -------
    callable
        A normalized copy of f between [0,1].
    """
    minf = min(f.values())
    maxf = max(f.values())
    f_normalized = {}
    for key, value in f.items():
        if minf == maxf:
            f_normalized[key] = 0
        else:
            f_normalized[key] = (value - minf) / (maxf - minf)

    return f_normalized


def hodge_laplacian_eigenvectors(
    hodge_laplacian, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the first k eigenvectors of the hodge laplacian matrix.

    Parameters
    ----------
    hodge_laplacian : scipy sparse matrix
        Hodge laplacian.
    n_components : int
        Number of eigenvectors that should be computed.

    Returns
    -------
    numpy.ndarray
        First k eigevals and eigenvec associated with the hodge laplacian matrix.

    Examples
    --------
    >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
    >>> L1 = SC.hodge_laplacian_matrix(1)
    >>> vals, vecs = tnx.hodge_laplacian_eigenvectors(L1, 2)
    """
    Diag = diags(hodge_laplacian.diagonal())
    if Diag.shape[0] > 10:
        vals, vect = sparse.linalg.eigs(
            hodge_laplacian, n_components, M=Diag, which="SM"
        )  # the lowest k eigenstuff
    else:
        vals, vect = np.linalg.eig(hodge_laplacian.toarray())

    eigenvaluevector = [round(i, 15) for i in vals.real]
    eigenvectorstemp = vect.transpose().real
    mydict = {}
    for i in range(len(eigenvaluevector)):
        mydict[eigenvaluevector[i]] = eigenvectorstemp[i]
    eigenvaluevector.sort()
    finaleigenvectors = [mydict[val] for val in eigenvaluevector]
    return np.array(eigenvaluevector).T, np.array(finaleigenvectors).T


def set_hodge_laplacian_eigenvector_attrs(
    SC: SimplicialComplex,
    dim: int,
    n_components: int,
    laplacian_type: Literal["up", "down", "hodge"] = "hodge",
    normalized: bool = True,
) -> None:
    """Set the hodge laplacian eigenvectors as simplex attributes.

    Parameters
    ----------
    SC : SimplicialComplex
        The simplicial complex for which to compute the hodge laplacian eigenvectors.
    dim : int
        Dimension of the hodge laplacian to be computed.
    n_components : int
        The number of eigenvectors to be computed.
    laplacian_type : {"up", "down", "hodge"}, default="hodge"
        Type of hodge matrix to be computed.
    normalized : bool, default=True
        Normalize the eigenvector or not.

    Examples
    --------
    >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
    >>> tnx.set_hodge_laplacian_eigenvector_attrs(SC, 1, 2, "down")
    >>> SC.get_simplex_attributes("0.th_eigen", 1)
    """
    index = SC.skeleton(dim)
    if laplacian_type == "hodge":
        L = SC.hodge_laplacian_matrix(dim)
    elif laplacian_type == "up":
        L = SC.up_laplacian_matrix(dim)
    elif laplacian_type == "down":
        L = SC.down_laplacian_matrix(dim)
    else:
        raise ValueError(
            f"laplacian_type must be up, down or hodge, got {laplacian_type}"
        )
    _, vect = hodge_laplacian_eigenvectors(L, n_components)

    for i in range(len(vect)):
        d = dict(zip(index, vect[i], strict=True))
        if normalized:
            d = _normalize(d)
        SC.set_simplex_attributes(d, str(i) + ".th_eigen")


def laplacian_beltrami_eigenvectors(
    SC: SimplicialComplex, mode: str = "fem"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the first k eigenvectors of the laplacian beltrami matrix.

    Parameters
    ----------
    SC : toponetx.classes.SimplicialComplex
        The simplicial complex for which to compute the beltrami eigenvectors.
    mode : {"fem", "sphara"}, default="fem"
        Mode of the spharapy basis.

    Returns
    -------
    k_eigenvectors : numpy.ndarray
        First k Eigenvectors associated with the hodge laplacian matrix.
    k_eigenvals : numpy.ndarray
        First k Eigenvalues associated with the hodge laplacian matrix.

    Raises
    ------
    RuntimeError
        If package `spharapy` is not installed.

    Examples
    --------
    >>> SC = tnx.datasets.stanford_bunny("simplicial")
    >>> eigenvectors, eigenvalues = tnx.laplacian_beltrami_eigenvectors(SC)
    """
    try:
        import spharapy.spharabasis as sb
    except ImportError as e:
        raise RuntimeError("Package `spharapy` is required for this function.") from e

    mesh = SC.to_spharapy()
    sphara_basis = sb.SpharaBasis(mesh, mode=mode)
    eigenvectors, eigenvalues = sphara_basis.basis()
    return eigenvectors, eigenvalues


def set_laplacian_beltrami_eigenvectors(SC: SimplicialComplex) -> None:
    """Set the laplacian beltrami eigenvectors as simplex attributes.

    Parameters
    ----------
    SC : SimplicialComplex
        Complex.

    Examples
    --------
    >>> SC = tnx.datasets.stanford_bunny()
    >>> tnx.set_laplacian_beltrami_eigenvectors(SC)
    >>> vec1 = SC.get_simplex_attributes("1.laplacian_beltrami_eigenvectors")
    """
    index = SC.skeleton(0)
    vect, _ = laplacian_beltrami_eigenvectors(SC)
    for i in range(len(vect)):
        d = dict(zip(index, vect[:, i], strict=True))
        SC.set_simplex_attributes(d, str(i) + ".laplacian_beltrami_eigenvectors")


def laplacian_spectrum(matrix) -> np.ndarray:
    """Return eigenvalues of the Laplacian matrix.

    Parameters
    ----------
    matrix : scipy sparse matrix
        The Laplacian matrix.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.
    """
    return sp.linalg.eigvalsh(matrix.todense())


def cell_complex_hodge_laplacian_spectrum(
    CC: CellComplex, rank: int, weight: str | None = None
) -> np.ndarray:
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    CC : toponetx.classes.CellComplex
        The cell complex for which to compute the spectrum.
    rank : int
        Rank of the cells to compute the Hodge Laplacian for.
    weight : str, optional
        If None, then each cell has weight 1.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> CC.add_cell([2, 3, 4, 5], rank=2)
    >>> CC.add_cell([5, 6, 7, 8], rank=2)
    >>> tnx.cell_complex_hodge_laplacian_spectrum(CC, 1)
    """
    return laplacian_spectrum(CC.hodge_laplacian_matrix(rank=rank, weight=weight))


def simplicial_complex_hodge_laplacian_spectrum(
    SC: SimplicialComplex, rank: int, weight: str = "weight"
) -> np.ndarray:
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    SC : toponetx.classes.SimplicialComplex
        The simplicial complex for which to compute the spectrum.
    rank : int
        Rank of the Hodge Laplacian.
    weight : str or None, optional (default='weight')
        If None, then each cell has weight 1.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.

    Examples
    --------
    >>> SC = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
    >>> spectrum = tnx.simplicial_complex_hodge_laplacian_spectrum(SC, 1)
    """
    return laplacian_spectrum(SC.hodge_laplacian_matrix(rank=rank))


def path_complex_hodge_laplacian_spectrum(
    PC: PathComplex, rank: int, weight: str | None = "weight"
) -> np.ndarray:
    """Return eigenvalues of the Laplacian of PC.

    Parameters
    ----------
    PC : PathComplex
        PathComplex for which to return eigenvalues.
    rank : int
        Rank of the PathComplex.
    weight : str or None, default='weight'
        If None, then each cell has weight 1.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.
    """
    return laplacian_spectrum(
        PC.hodge_laplacian_matrix(rank=rank, signed=True, weight=weight)
    )


def cell_complex_adjacency_spectrum(CC: CellComplex, rank: int):
    """Return eigenvalues of the adjacency matrix of the cell complex.

    Parameters
    ----------
    CC : toponetx.classes.CellComplex
        The cell complex for which to compute the spectrum.
    rank : int
        Rank of the cells to take the adjacency from:
        - 0-cells are nodes
        - 1-cells are edges
        - 2-cells are polygons
        Currently, no cells of rank > 2 are supported.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.

    Examples
    --------
    >>> CC = tnx.CellComplex()
    >>> CC.add_cell([1, 2, 3, 4], rank=2)
    >>> CC.add_cell([2, 3, 4, 5], rank=2)
    >>> CC.add_cell([5, 6, 7, 8], rank=2)
    >>> tnx.cell_complex_adjacency_spectrum(CC, 1)
    """
    return laplacian_spectrum(CC.adjacency_matrix(rank=rank))


def simplicial_complex_adjacency_spectrum(
    SC: SimplicialComplex, rank: int, weight: str | None = None
) -> np.ndarray:
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    SC : toponetx.classes.SimplicialComplex
        The simplicial complex for which to compute the spectrum.
    rank : int
        Rank of the adjacency matrix.
    weight : str, optional
        If None, then each cell has weight 1.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.
    """
    return laplacian_spectrum(SC.adjacency_matrix(rank, weight=weight))


def path_complex_adjacency_spectrum(
    PC: PathComplex, dim: int, weight: str | None = None
) -> np.ndarray:
    """Return eigenvalues of the adjacency matrix of PC.

    Parameters
    ----------
    PC : PathComplex
        PathComplex for which adjacency matrix is computed.
    dim : int
        Dimension.
    weight : str, optional
        If None, then each cell has weight 1.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.
    """
    return laplacian_spectrum(PC.adjacency_matrix(rank=dim, signed=True, weight=weight))


def combinatorial_complex_adjacency_spectrum(
    CCC: CombinatorialComplex, rank: int, via_rank: int
) -> np.ndarray:
    """Return eigenvalues of the adjacency matrix of the combinatorial complex.

    Parameters
    ----------
    CCC : toponetx.classes.CombinatorialComplex
        The combinatorial complex for which to compute the spectrum.
    rank, via_rank : int
        Rank of the cells to compute the adjacency spectrum for.

    Returns
    -------
    numpy.ndarray
        Eigenvalues.

    Examples
    --------
    >>> CCC = tnx.CombinatorialComplex(cells=[[1, 2, 3], [2, 3], [0]], ranks=[2, 1, 0])
    >>> tnx.combinatorial_complex_adjacency_spectrum(CCC, 0, 2)
    """
    return laplacian_spectrum(CCC.adjacency_matrix(rank, via_rank))
