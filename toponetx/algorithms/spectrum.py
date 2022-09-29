# -*- coding: utf-8 -*-


import scipy.sparse as sparse
from numpy import linalg as LA
from scipy.sparse import diags

from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex


def hodge_laplacian_eigenvectors(laplacian, k):

    """
    Input
    ======
        laplacian : scipy sparse matrix representing the hodge laplacian
        k : int, the number of eigenvectors one needs to output, if
            laplacian.shape[0]<=10, then all eigenvectors will be returned
    output:
    =======
        first k eigevals and eigenvec associated with the laplacian matrix.
    example
    ========
    >>> SC = SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> row,column,B1 = SC.incidence_matrix(1,index=True)
    >>> L1 = SC.hodge_laplacian_matrix(1)
    >>> vals,vecs = hodge_laplacian_eigenvectors(L1,2)

    """

    Diag = diags(laplacian.diagonal())
    if Diag.shape[0] > 10:
        vals, vect = sparse.linalg.eigs(
            laplacian, k, M=Diag, which="SM"
        )  # the lowest k eigenstuff
    else:
        vals, vect = LA.eig(laplacian.toarray())

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


def hodge_laplacian_eigenvector_attrs(
    cmplex, dim, k, laplacian_type="hodge", normalized=True
):
    """
    input
    =====
        complex : a SimplialComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        k : int, the number of eigenvectors to be computed
        laplacian_type : str, type of hodge matrix to be computed,
                        options : up, down, hodge
        normalized: bool, normalize the eigenvector or not.
    output
    ======
        complex with eigecvectors are stored in the correpsonding skeleton on the k-th skeleton

    example
    ========
    >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC = hodge_laplacian_eigenvector_attrs(SC,1,2,"down")
    >>> SC.get_simplex_attributes("0.th_eigen")
    """
    index = cmplex.skeleton(dim)
    if laplacian_type == "hodge":
        L = cmplex.hodge_laplacian_matrix(dim)
    elif laplacian_type == "up":
        L = cmplex.up_laplacian_matrix(dim)
    elif laplacian_type == "down":
        L = cmplex.up_laplacian_matrix(dim)
    else:
        raise ValueError("type must be up, down or hodge")
    vals, vect = hodge_laplacian_eigenvectors(L, k)

    for i in range(len(vect)):
        d = dict(zip(index, vect[i]))
        if normalized:
            d = normalize(d)
        cmplex.set_simplex_attributes(d, str(i) + ".th_eigen")
    return cmplex


def normalize(f):
    """
    input f ascalar function on nodes of a graph
    output a  normalized copy of f between [0,1].
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


def laplacian_spectrum(matrix, weight="weight"):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    See Also
    --------
    """
    import scipy as sp
    import scipy.linalg  # call as sp.linalg

    return sp.linalg.eigvalsh(matrix.todense())


def cell_complex_hodge_laplacian_spectrum(CX: CellComplex, dim: int, weight="weight"):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    Example
    =======
            >>> CX = CellComplex()
            >>> CX.add_cell([1,2,3,4],rank=2)
            >>> CX.add_cell([2,3,4,5],rank=2)
            >>> CX.add_cell([5,6,7,8],rank=2)
            >>> cell_complex_hodge_laplacian_spectrum(CX,1)

    See Also
    --------
    """

    return laplacian_spectrum(CX.hodge_laplacian_matrix(d=dim, weight=weight))


def simplicial_complex_hodge_laplacian_spectrum(
    SC: SimplicialComplex, dim, weight="weight"
):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    Example
    ======
        >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
        >>> spectrum=simplicial_complex_hodge_laplacian_spectrum(SC,1)

    See Also
    --------
    """
    return laplacian_spectrum(SC.hodge_laplacian_matrix(d=dim))


def cell_complex_adjacency_spectrum(CX: CellComplex, dim, weight="weight"):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    Example
    =======
            >>> CX = CellComplex()
            >>> CX.add_cell([1,2,3,4],rank=2)
            >>> CX.add_cell([2,3,4,5],rank=2)
            >>> CX.add_cell([5,6,7,8],rank=2)
            >>> cell_complex_adjacency_spectrum(CX,1)

    See Also
    --------
    """
    return laplacian_spectrum(CX.adjacency_matrix(d=dim, weight=weight))


def simplcial_complex_adjacency_spectrum(
    SC: SimplicialComplex, dim: int, weight="weight"
):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    See Also
    --------
    """
    return laplacian_spectrum(SC.adjacency_matrix(d=dim, weight=weight))


def combintorial_complex_adjacency_spectrum(CC, r, k, weight="weight"):
    """Returns eigenvalues of the Laplacian of G

    Parameters
    ----------
    matrix : scipy sparse matrix

    weight : string or None, optional (default='weight')
       If None, then each cell has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----

    Example
    ======
        >>> CC = CombinatorialComplex(cells=[[1,2,3],[2,3], [0] ],ranks=[2,1,0] )
        >>> s= laplacian_spectrum( CC.adjacency_matrix( 0,2) )

    See Also
    --------
    """
    return laplacian_spectrum(CC.adjacency_matrix(r, k, weight=weight))
