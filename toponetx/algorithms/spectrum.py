"""Module to compute spectra."""

import numpy as np
import scipy.sparse as sparse
from numpy import linalg as LA
from scipy.sparse import diags

from toponetx.classes import CombinatorialComplex
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.datasets.mesh import stanford_bunny

__all__ = [
    "hodge_laplacian_eigenvectors",
    "set_hodge_laplacian_eigenvector_attrs",
    "_normalize",
    "laplacian_beltrami_eigenvectors",
    "laplacian_spectrum",
    "cell_complex_hodge_laplacian_spectrum",
    "simplicial_complex_hodge_laplacian_spectrum",
    "cell_complex_adjacency_spectrum",
    "simplicial_complex_adjacency_spectrum",
    "combinatorial_complex_adjacency_spectrum",
]


def _normalize(f):
    """Normalize.

    Parameters
    ----------
    f : callable
        A scalar function on nodes of a graph.

    Returns
    -------
    f_normalized : callable
        A normalized copy of f between [0,1].
    """
    minf = min(f.values())
    maxf = max(f.values())
    f_normalized = {}
    for v in f.keys():

        if minf == maxf:
            f_normalized[v] = 0
        else:
            f_normalized[v] = (f[v] - minf) / (maxf - minf)

    return f_normalized


def hodge_laplacian_eigenvectors(hodge_laplacian, n_components):
    """Compute the first k eigenvectors of the hodge laplacian matrix.

    Parameters
    ----------
    hodge laplacian : scipy sparse matrix
        Hodge laplacian.
    n_components : int
        Number of eigenvectors one needs to output, if
        laplacian.shape[0]<=10, then all eigenvectors will be returned

    Returns
    -------
        First k eigevals and eigenvec associated with the hodge laplacian matrix.

    Examples
    --------
    >>> from toponetx import SimplicialComplex
    >>> SC = SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> row,column,B1 = SC.incidence_matrix(1,index=True)
    >>> L1 = SC.hodge_laplacian_matrix(1)
    >>> vals,vecs = hodge_laplacian_eigenvectors(L1,2)
    """
    Diag = diags(hodge_laplacian.diagonal())
    if Diag.shape[0] > 10:
        vals, vect = sparse.linalg.eigs(
            hodge_laplacian, n_components, M=Diag, which="SM"
        )  # the lowest k eigenstuff
    else:
        vals, vect = LA.eig(hodge_laplacian.toarray())

    eigenvaluevector = []
    for i in vals.real:
        eigenvaluevector.append(round(i, 15))
    eigenvectorstemp = vect.transpose().real
    mydict = {}
    for i in range(0, len(eigenvaluevector)):
        mydict[eigenvaluevector[i]] = eigenvectorstemp[i]
    eigenvaluevector.sort()
    finaleigenvectors = []
    for val in eigenvaluevector:
        finaleigenvectors.append(mydict[val])
    return [eigenvaluevector, finaleigenvectors]


def set_hodge_laplacian_eigenvector_attrs(
    cmplex, dim, n_components, laplacian_type="hodge", normalized=True
):
    """Set the hodge laplacian eigenvectors as simplex attributes.

    Parameters
    ----------
    cmplex : a SimplialComplex/CellComplex object
        Complex.
    dim : int
        Dimension of the hodge laplacian to be computed.
    n_components : int
        The number of eigenvectors to be computed
    laplacian_type : str
        Rype of hodge matrix to be computed,
        options : up, down, hodge
    normalized : bool
        Normalize the eigenvector or not.

    Examples
    --------
    >>> from toponetx import SimplicialComplex
    >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC = set_hodge_laplacian_eigenvector_attrs(SC,1,2,"down")
    >>> SC.get_simplex_attributes("0.th_eigen", 1)
    """
    index = cmplex.skeleton(dim)
    if laplacian_type == "hodge":
        L = cmplex.hodge_laplacian_matrix(dim)
    elif laplacian_type == "up":
        L = cmplex.up_laplacian_matrix(dim)
    elif laplacian_type == "down":
        L = cmplex.up_laplacian_matrix(dim)
    else:
        raise ValueError(
            f"laplacian_type must be up, down or hodge, got {laplacian_type}"
        )
    vals, vect = hodge_laplacian_eigenvectors(L, n_components)

    for i in range(len(vect)):
        d = dict(zip(index, vect[i]))
        if normalized:
            d = _normalize(d)
        cmplex.set_simplex_attributes(d, str(i) + ".th_eigen")


def laplacian_beltrami_eigenvectors(SC, mode="fem"):
    """Compute the first k eigenvectors of the laplacian beltrami matrix.

    Examples
    --------
    >>> SC = stanford_bunny()
    >>> eigenvectors, eigenvalues = laplacian_beltrami_eigenvectors(SC)
    """
    import spharapy.spharabasis as sb

    mesh = SC.to_spharapy()
    sphara_basis = sb.SpharaBasis(mesh, mode=mode)
    eigenvectors, eigenvalues = sphara_basis.basis()
    return eigenvectors, eigenvalues


def set_laplacian_beltrami_eigenvectors(cmplex):
    """Set the laplacian beltrami eigenvectors as simplex attributes.

    Parameters
    ----------
    cmplex : a SimplialComplex object
        Complex.

    Examples
    --------
    >>> from toponetx import SimplicialComplex
    >>> SC = SimplicialComplex.load_mesh("bunny.obj")
    >>> set_laplacian_beltrami_eigenvectors(SC)
    >>> vec1 = SC.get_simplex_attributes("1.laplacian_beltrami_eigenvectors")
    """
    index = cmplex.skeleton(0)
    vect, vals = laplacian_beltrami_eigenvectors(cmplex)
    for i in range(len(vect)):
        d = dict(zip(index, vect[:, i]))
        cmplex.set_simplex_attributes(d, str(i) + ".laplacian_beltrami_eigenvectors")


def laplacian_spectrum(matrix, weight="weight"):
    """Return eigenvalues of the Laplacian matrix.

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : str or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
        Eigenvalues.
    """
    import scipy as sp
    import scipy.linalg  # call as sp.linalg

    return sp.linalg.eigvalsh(matrix.todense())


def cell_complex_hodge_laplacian_spectrum(CX: CellComplex, rank: int, weight="weight"):
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    matrix : scipy sparse matrix
    weight : str or None, optional (default='weight')
        If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
        Eigenvalues.

    Examples
    --------
    >>> from toponetx import CellComplex
    >>> CX = CellComplex()
    >>> CX.add_cell([1,2,3,4],rank=2)
    >>> CX.add_cell([2,3,4,5],rank=2)
    >>> CX.add_cell([5,6,7,8],rank=2)
    >>> cell_complex_hodge_laplacian_spectrum(CX,1)
    """
    return laplacian_spectrum(CX.hodge_laplacian_matrix(rank=rank, weight=weight))


def simplicial_complex_hodge_laplacian_spectrum(
    SC: SimplicialComplex, rank, weight="weight"
):
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    matrix : scipy sparse matrix
    weight : str or None, optional (default='weight')
        If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
        Eigenvalues.

    Examples
    --------
    >>> from toponets import SimplicialComplex
    >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> spectrum=simplicial_complex_hodge_laplacian_spectrum(SC,1)
    """
    return laplacian_spectrum(SC.hodge_laplacian_matrix(rank=rank))


def cell_complex_adjacency_spectrum(CX: CellComplex, rank):
    """Return eigenvalues of the adjacency matrix of CX.

    Parameters
    ----------
    CX : CellComplex
    rank : int
        rank of the cells to take the adjacency from:
        - 0-cells are nodes
        - 1-cells are edges
        - 2-cells are polygons
        currently, no cells of rank > 2 are supported.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Examples
    --------
    >>> from toponetx import CellComplex
    >>> CX = CellComplex()
    >>> CX.add_cell([1,2,3,4],rank=2)
    >>> CX.add_cell([2,3,4,5],rank=2)
    >>> CX.add_cell([5,6,7,8],rank=2)
    >>> cell_complex_adjacency_spectrum(CX,1)
    """
    return laplacian_spectrum(CX.adjacency_matrix(rank=rank))


def simplicial_complex_adjacency_spectrum(
    SC: SimplicialComplex, dim: int, weight="weight"
):
    """Return eigenvalues of the Laplacian of G.

    Parameters
    ----------
    matrix : scipy sparse matrix
    weight : str or None, optional (default='weight')
        If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
        Eigenvalues.
    """
    return laplacian_spectrum(SC.adjacency_matrix(rank=dim, weight=weight))


def combinatorial_complex_adjacency_spectrum(CC: CombinatorialComplex, r, k):
    """Return eigenvalues of the adjacency matrix of CC.

    Parameters
    ----------
    matrix : scipy sparse matrix
    weight : str or None, default='weight'
        If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
        Eigenvalues

    Examples
    --------
    >>> from toponetx import CombinatorialComplex
    >>> CC = CombinatorialComplex(cells=[[1,2,3],[2,3], [0] ],ranks=[2,1,0] )
    >>> s = laplacian_spectrum( CC.adjacency_matrix( 0,2) )
    """
    return laplacian_spectrum(CC.adjacency_matrix(r, k))
