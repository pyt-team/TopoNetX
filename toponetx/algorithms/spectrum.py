# -*- coding: utf-8 -*-


from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex

# from toponetx.classes.cell_complex import hodge_laplacian_matrix as cx_lap
# from toponetx.classes.simplicial_complex import hodge_laplacian_matrix as sc_lap
# from toponetx.classes.simplicial_complex import adjacency_matrix as cc_adj


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
