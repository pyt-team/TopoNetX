# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse
from numpy import linalg as LA
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from toponetx.algorithms.eigen_align import align_eigenvectors_kl
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = [
    "_sparse_spectral",
    "hodge_laplacian_eigenvectors",
    "set_hodge_laplacian_eigenvector_attrs",
    "set_spectral_embedding_attr",
    "set_spectral_embedding_attr_list_of_complexes",
    "normalize",
    "laplacian_beltrami_eigenvectors",
    "laplacian_spectrum",
    "cell_complex_hodge_laplacian_spectrum",
    "simplicial_complex_hodge_laplacian_spectrum",
    "cell_complex_adjacency_spectrum",
    "simplcial_complex_adjacency_spectrum",
    "combintorial_complex_adjacency_spectrum",
    "normalized_spectral_embedding",
    "set_normalized_spectral_embedding_attr",
]


def _sparse_spectral(L, dim=2):
    """
    Compute the spectral layout of a graph using the sparse eigenvalue solver from scipy.

    This function computes the eigenvectors of the Hodge Laplacian matrix of the given graph,
    and returns the eigenvalues and eigenvectors corresponding to the k smallest nonzero eigenvalues.

    :param L: The Hodge Laplacian matrix of the graph.
    :type L: scipy.sparse.csr_matrix
    :param dim: The number of dimensions in the layout. Default is 2.
    :type dim: int
    :return: The eigenvalues and eigenvectors of the Hodge Laplacian matrix.
    :rtype: tuple of ndarray
    """

    # Hodge Laplcian matrix A
    # Uses sparse eigenvalue solver from scipy
    # ref:
    # https://networkx.org/documentation/networkx-1.9/_modules/networkx/drawing/layout.html#spectral_layout

    try:
        from scipy.sparse.linalg.eigen import eigsh
    except ImportError:
        # scipy <0.9.0 names eigsh differently
        from scipy.sparse.linalg import eigen_symmetric as eigsh

    k = dim + 1
    # number of Lanczos vectors for ARPACK solver.What is the right scaling?
    ncv = max(2 * k + 1, int(np.sqrt(L.shape[9])))
    # return smallest k eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k, which="SM", ncv=ncv)
    index = np.argsort(eigenvalues)[1:k]  # 0 index is zero eigenvalue
    return eigenvalues, np.real(eigenvectors[:, index])


def hodge_laplacian_eigenvectors(laplacian, n_components):

    """
    Input
    ======
        laplacian : scipy sparse matrix representing the hodge laplacian
        n_components : int, the number of eigenvectors one needs to output, if
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
            laplacian, n_components, M=Diag, which="SM"
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


def set_hodge_laplacian_eigenvector_attrs(
    cmplex, dim, n_components, laplacian_type="hodge", normalized=True
):
    """
    input
    =====
        cmplex : a SimplialComplex/CellComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        laplacian_type : str, type of hodge matrix to be computed,
                        options : up, down, hodge
        normalized: bool, normalize the eigenvector or not.


    example
    ========
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
        raise ValueError("type must be up, down or hodge")
    vals, vect = hodge_laplacian_eigenvectors(L, n_components)

    for i in range(len(vect)):
        d = dict(zip(index, vect[i]))
        if normalized:
            d = normalize(d)
        cmplex.set_simplex_attributes(d, str(i) + ".th_eigen")


def normalized_spectral_embedding(cmplex, dim, n_components):
    """
    input
    =====
        cmplex : a SimplialComplex/CellComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        align : bool, indicates wheather the eigenvectors are to be aligned


    example
    ========
    >>> SC = SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> vect = normalized_spectral_embedding(SC,dim=0,n_components=2)
    """

    from sklearn.manifold import SpectralEmbedding

    embedding = SpectralEmbedding(affinity="precomputed", n_components=n_components)
    A = cmplex.adjacency_matrix(dim)
    vect = embedding.fit_transform(A)
    return vect


def set_normalized_spectral_embedding_attr(cmplex, dim, n_components):
    """
    input
    =====
        cmplex : a SimplialComplex/CellComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        align : bool, indicates wheather the eigenvectors are to be aligned


    example
    ========
    >>> SC = SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> set_normalized_spectral_embedding_attr(SC,0,2)
    >>> SC.get_simplex_attributes("0.normalized_spec_embedding",0)
    """

    index = cmplex.skeleton(dim)
    vect = normalized_spectral_embedding(cmplex, dim, n_components)
    for i in range(n_components):
        d = dict(zip(index, vect[:, i]))
        cmplex.set_simplex_attributes(d, str(i) + ".normalized_spec_embedding")


def spectral_embedding_unnormalized(cmplex, dim, n_components):
    """
    input
    =====
        cmplex : a SimplialComplex/CellComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        align : bool, indicates wheather the eigenvectors are to be aligned


    example
    ========
    >>> SC = SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> vect = compute_spectral_embedding(SC,dim=0,n_components=2)
    """

    L = cmplex.hodge_laplacian_matrix(dim)
    _, vect = eigsh(L, k=n_components, which="SM")
    n = L.shape[0]
    vect *= np.sqrt(n)
    return vect


def set_spectral_embedding_attr(cmplex, dim, n_components=2):
    """
    input
    =====
        cmplex : a SimplialComplex/CellComplex object
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        align : bool, indicates wheather the eigenvectors are to be aligned


    example
    ========
    >>> SC=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> set_spectral_embedding_attr(SC,dim=0,n_components=2)
    >>> SC.get_simplex_attributes("0.spec_embedding",0)
    """
    index = cmplex.skeleton(dim)
    vect = spectral_embedding_unnormalized(cmplex, dim, n_components)

    for i in range(n_components):
        d = dict(zip(index, vect[:, i]))
        cmplex.set_simplex_attributes(d, str(i) + ".spec_embedding")


def set_spectral_embedding_attr_list_of_complexes(
    cmplex_list, dim, n_components=2, align=True
):
    """
    input
    =====
        cmplex_list : a list of SimplialComplex/CellComplex objects
        dim : int, the dimension of the hodge laplacian to be computed
        n_components : int, the number of eigenvectors to be computed
        align : bool, indicates wheather the eigenvectors are to be aligned


    example1
    ========

    >>> SC1=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC2=SimplicialComplex([[1,2,3],[2,3,5],[0,1],[0,7]])
    >>> set_spectral_embedding_attr_list_of_complexes( [SC1,SC2] ,
                                                      dim=0,
                                                      n_components=4,
                                                      align =True)
    >>> SC1.get_simplex_attributes("1.spec_embedding",0)
    >>> SC2.get_simplex_attributes("1.spec_embedding",0)

    example2
    ========
    >>> SC1=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC2=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC3=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> SC4=SimplicialComplex([[1,2,3],[2,3,5],[0,1]])
    >>> set_spectral_embedding_attr_list_of_complexes( [SC1,SC2,SC3,SC4] ,
                                                      dim=0,
                                                      n_components=4,
                                                      align =True)
    >>> # must be equal
    >>> SC1.get_simplex_attributes("1.spec_embedding",0)
    >>> SC2.get_simplex_attributes("1.spec_embedding",0)
    >>> SC3.get_simplex_attributes("1.spec_embedding",0)
    >>> SC4.get_simplex_attributes("1.spec_embedding",0)
    """

    if not isinstance(cmplex_list, list):
        raise TypeError("input cmplex_list must be a list")
    if len(cmplex_list) == 1:
        return set_spectral_embedding_attr(cmplex_list[0], dim, n_components)

    if not align:
        for C in cmplex_list:
            set_spectral_embedding_attr(C, dim, n_components)
    else:
        all_eigen_vectors = []
        for cmplex in cmplex_list:
            vect = spectral_embedding_unnormalized(cmplex, dim, n_components)
            all_eigen_vectors.append(vect)

        index = cmplex_list[0].skeleton(dim)
        for i in range(n_components):
            d = dict(zip(index, all_eigen_vectors[0][:, i]))
            cmplex_list[0].set_simplex_attributes(d, str(i) + ".spec_embedding")

        for i in range(1, len(cmplex_list)):
            cmplex = cmplex_list[i]

            aligned_vect = align_eigenvectors_kl(
                all_eigen_vectors[0], all_eigen_vectors[i]
            )
            index = cmplex.skeleton(dim)
            for j in range(n_components):
                d = dict(zip(index, aligned_vect[:, j]))
                cmplex.set_simplex_attributes(d, str(j) + ".spec_embedding")


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


def laplacian_beltrami_eigenvectors(SC, mode="fem"):

    """
    >>> SC = SimplicialComplex.load_mesh("C:/Users/musta/OneDrive/Desktop/bunny.obj")
    >>> eigenvectors, eigenvalues = laplacian_beltrami_eigenvectors(SC)
    """

    import spharapy.spharabasis as sb

    mesh = SC.to_spharapy()
    sphara_basis = sb.SpharaBasis(mesh, mode=mode)
    eigenvectors, eigenvalues = sphara_basis.basis()
    return eigenvectors, eigenvalues


def set_laplacian_beltrami_eigenvectors(cmplex):
    """
    input
    =====
        cmplex : a SimplialComplex object
    example
    ========
    >>> SC = SimplicialComplex.load_mesh("C:/Users/musta/OneDrive/Desktop/bunny.obj")
    >>> set_laplacian_beltrami_eigenvectors(SC)
    >>> vec1 = SC.get_simplex_attributes("1.laplacian_beltrami_eigenvectors")
    """

    index = cmplex.skeleton(0)
    vect, vals = laplacian_beltrami_eigenvectors(cmplex)
    for i in range(len(vect)):
        d = dict(zip(index, vect[:, i]))
        cmplex.set_simplex_attributes(d, str(i) + ".laplacian_beltrami_eigenvectors")


# ---------------- laplacian spectrum for various complexes ---------------#


def laplacian_spectrum(matrix, weight="weight"):
    """Returns eigenvalues of the Laplacian matrix

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
