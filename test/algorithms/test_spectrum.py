"""Test spectrum module."""

import pytest
import scipy.sparse as sparse

from toponetx.algorithms.spectrum import (
    _normalize,
    cell_complex_adjacency_spectrum,
    cell_complex_hodge_laplacian_spectrum,
    combinatorial_complex_adjacency_spectrum,
    hodge_laplacian_eigenvectors,
    laplacian_beltrami_eigenvectors,
    path_complex_adjacency_spectrum,
    path_complex_hodge_laplacian_spectrum,
    set_hodge_laplacian_eigenvector_attrs,
    set_laplacian_beltrami_eigenvectors,
    simplicial_complex_adjacency_spectrum,
    simplicial_complex_hodge_laplacian_spectrum,
)
from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    PathComplex,
    SimplicialComplex,
)
from toponetx.datasets.mesh import stanford_bunny

try:
    import spharapy.trimesh as tm
except ImportError:
    tm = None


class TestSpectrum:
    """Test spectrum module."""

    def test_normalize(self):
        """Test normalize function."""
        f = {1: 2, 2: 5, 3: 1}
        expected_result = {1: 0.25, 2: 1.0, 3: 0.0}
        result = _normalize(f)
        assert result == expected_result

        f = {1: 1, 2: 1}
        expected_result = {1: 0, 2: 0}
        result = _normalize(f)
        assert result == expected_result

    def test_hodge_laplacian_eigenvectors(self):
        """Test hodge_laplacian_eigenvectors function."""
        L = sparse.eye(3)
        n_components = 2
        eigenvaluevector, eigenvectors = hodge_laplacian_eigenvectors(L, n_components)
        assert len(eigenvaluevector) == 3
        assert eigenvectors.shape == (3, 3)

        L = sparse.eye(29)
        n_components = 2
        eigenvaluevector, eigenvectors = hodge_laplacian_eigenvectors(L, n_components)
        assert len(eigenvaluevector) == 2
        assert eigenvectors.shape == (29, 2)

    @pytest.mark.skipif(
        tm is None, reason="Optional dependency 'spharapy' not installed."
    )
    def test_laplacian_beltrami_eigenvectors1(self):
        """Test laplacian_beltrami_eigenvectors."""
        sc = stanford_bunny("simplicial")
        _, eigenvalues = laplacian_beltrami_eigenvectors(sc)
        assert len(eigenvalues) == len(sc.nodes)

    @pytest.mark.skipif(
        tm is None, reason="Optional dependency 'spharapy' not installed."
    )
    def test_set_laplacian_beltrami_eigenvectors2(self):
        """Test laplacian_beltrami_eigenvectors."""
        SC = stanford_bunny("simplicial")
        set_laplacian_beltrami_eigenvectors(SC)
        vec1 = SC.get_simplex_attributes("1.laplacian_beltrami_eigenvectors")
        assert len(vec1) == len(SC.skeleton(0))

    @pytest.mark.skipif(
        tm is not None, reason="Optional dependency 'spharapy' installed."
    )
    def test_laplacian_beltrami_eigenvectors_missing_dependency(self):
        """Test laplacian_beltrami_eigenvectors for when `spharapy` is missing."""
        sc = stanford_bunny("simplicial")
        with pytest.raises(RuntimeError):
            _ = laplacian_beltrami_eigenvectors(sc)

    def test_set_hodge_laplacian_eigenvector_attrs(self):
        """Test set_hodge_laplacian_eigenvector_attrs function."""
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

        with pytest.raises(ValueError):
            SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
            set_hodge_laplacian_eigenvector_attrs(SC, 1, 2, "hi")

    def test_cell_complex_hodge_laplacian_spectrum(self):
        """Test cell_complex_hodge_laplacian_spectrum function."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        eigen = cell_complex_hodge_laplacian_spectrum(CX, 1)
        assert len(eigen) == len(CX.edges)

    def test_simplicial_complex_hodge_laplacian_spectrum(self):
        """Test simplicial_complex_hodge_laplacian_spectrum function."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        spectrum = simplicial_complex_hodge_laplacian_spectrum(SC, 1)
        assert len(spectrum) == len(SC.skeleton(1))

    def test_path_complex_hodge_laplacian_spectrum(self):
        """Test path_complex_hodge_laplacian_spectrum function."""
        PC = PathComplex(
            [[0, 1], [0, 1, 2], [0, 1, 3], [1, 2, 3], [1, 3, 2], [2, 1, 3]]
        )
        spectrum = path_complex_hodge_laplacian_spectrum(PC, 0)
        assert len(spectrum) == len(PC.skeleton(0))
        spectrum = path_complex_hodge_laplacian_spectrum(PC, 1)
        assert len(spectrum) == len(PC.skeleton(1))
        spectrum = path_complex_hodge_laplacian_spectrum(PC, 2)
        assert len(spectrum) == len(PC.skeleton(2))

    def test_cell_complex_adjacency_spectrum(self):
        """Test cell_complex_adjacency_spectrum function."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)

        spectrum = cell_complex_adjacency_spectrum(CX, 1)

        assert len(spectrum) == len(CX.skeleton(1))

    def test_simplicial_complex_adjacency_spectrum(self):
        """Test simplicial_complex_adjacency_spectrum function."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        spectrum = simplicial_complex_adjacency_spectrum(SC, 1)
        assert len(spectrum) == len(SC.skeleton(1))

    def test_path_complex_adjacency_spectrum(self):
        """Test path_complex_adjacency_spectrum function."""
        PC = PathComplex(
            [[0, 1], [0, 1, 2], [0, 1, 3], [1, 2, 3], [1, 3, 2], [2, 1, 3]]
        )
        spectrum = path_complex_hodge_laplacian_spectrum(PC, 0)
        assert len(spectrum) == len(PC.skeleton(0))
        spectrum = path_complex_adjacency_spectrum(PC, 1)
        assert len(spectrum) == len(PC.skeleton(1))

    def test_combinatorial_complex_adjacency_spectrum(self):
        """Test combinatorial_complex_adjacency_spectrum function."""
        CC = CombinatorialComplex(cells=[[1, 2, 3], [2, 3], [0]], ranks=[2, 1, 0])
        s = combinatorial_complex_adjacency_spectrum(CC, 0, 2)

        assert len(CC.nodes) == len(s)
