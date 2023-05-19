# -*- coding: utf-8 -*-

import unittest

import numpy as np
import scipy.sparse as sparse
from numpy import linalg as LA
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from toponetx.algorithms.spectrum import (
    _normalize,
    cell_complex_adjacency_spectrum,
    cell_complex_hodge_laplacian_spectrum,
    combintorial_complex_adjacency_spectrum,
    hodge_laplacian_eigenvectors,
    laplacian_beltrami_eigenvectors,
    laplacian_spectrum,
    set_hodge_laplacian_eigenvector_attrs,
    simplicial_complex_adjacency_spectrum,
    simplicial_complex_hodge_laplacian_spectrum,
)
from toponetx.classes.cell_complex import CellComplex, CombinatorialComplex
from toponetx.classes.simplicial_complex import SimplicialComplex


class TestStructure(unittest.TestCase):
    def test_normalize(self):
        f = {1: 2, 2: 5, 3: 1}
        expected_result = {1: 0.25, 2: 1.0, 3: 0.0}
        result = _normalize(f)
        assert result == expected_result
        pass

    def test_hodge_laplacian_eigenvectors(self):

        L = sparse.eye(3)  # Sample Laplacian matrix
        n_components = 2
        eigenvaluevector, eigenvectors = hodge_laplacian_eigenvectors(L, n_components)
        assert len(eigenvaluevector) == 3
        assert len(eigenvectors) == 3

    def test_set_hodge_laplacian_eigenvector_attrs(self):

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        set_hodge_laplacian_eigenvector_attrs(SC, 1, 2, "down")
        d = SC.get_simplex_attributes("0.th_eigen", 1)

        assert len(d) == len(SC.skeleton(1))

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        set_hodge_laplacian_eigenvector_attrs(SC, 1, 2, "up")
        d = SC.get_simplex_attributes("1.th_eigen", 1)

        assert len(d) == len(SC.skeleton(1))

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        set_hodge_laplacian_eigenvector_attrs(SC, 1, 2, "hodge")
        d = SC.get_simplex_attributes("2.th_eigen", 1)

        assert len(d) == len(SC.skeleton(1))

    def set_laplacian_beltrami_eigenvectors(self):

        pass

    def cell_complex_hodge_laplacian_spectrum(self):

        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        eigen = cell_complex_hodge_laplacian_spectrum(CX, 1)
        len(eigen) == len(CX.edges)

    def test_simplicial_complex_hodge_laplacian_spectrum(self):

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        spectrum = simplicial_complex_hodge_laplacian_spectrum(SC, 1)
        assert len(spectrum) == len(SC.skeleton(1))

    def test_cell_complex_adjacency_spectrum(self):

        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)

        spectrum = cell_complex_adjacency_spectrum(CX, 1)

        assert len(spectrum) == len(CX.skeleton(1))

    def test_simplcial_complex_adjacency_spectrum(self):

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        spectrum = simplicial_complex_adjacency_spectrum(SC, 1)
        assert len(spectrum) == len(SC.skeleton(1))

    def test_combintorial_complex_adjacency_spectrum(self):
        CC = CombinatorialComplex(cells=[[1, 2, 3], [2, 3], [0]], ranks=[2, 1, 0])
        s = combintorial_complex_adjacency_spectrum(CC, 0, 2)

        assert len(CC.nodes) == len(s)


if __name__ == "__main__":
    unittest.main()
