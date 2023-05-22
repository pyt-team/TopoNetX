"""Unit tests for the combinatorial complex class."""

import unittest

import networkx as nx

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge


class TestCombinatorialComplex(unittest.TestCase):
    """Test CombinatorialComplex class."""

    def test_init_empty_cc(self):
        """Test creation of an empty CC."""
        CC = CombinatorialComplex()
        assert len(CC) == 0

    def test_init_from_lists(self):
        """Test creation of a CC from a list of cells."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert len(CC.cells) == 6
        assert (1, 2, 3) in CC.cells
        assert (2, 3, 4) in CC.cells

    def test_init_from_abstract_cells(self):
        """Test creation of a CC from abstract cells."""
        y1 = HyperEdge(elements=[1, 2], rank=1)
        y2 = HyperEdge(elements=[2, 4], rank=1)
        y3 = HyperEdge(elements=[3, 5], rank=1)
        y4 = HyperEdge(elements=[4, 5], rank=1)
        y5 = HyperEdge(elements=[5, 7], rank=1)

        CC = CombinatorialComplex(cells=[y1, y2, y3, y4, y5])

        assert y1 in CC.cells
        assert y2 in CC.cells
        assert y3 in CC.cells
        assert y4 in CC.cells
        assert y5 in CC.cells

    def test_init_from_networkx_graph(self):
        """Test creation of a CC from a networkx graph."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 3)
        G.add_edge(0, 4)
        G.add_edge(1, 4)

        CC = CombinatorialComplex(cells=G)

        assert (0, 1) in CC.cells
        assert (0, 3) in CC.cells
        assert (0, 4) in CC.cells
        assert (1, 4) in CC.cells
        assert (0, 5) not in CC.cells

        G = nx.Graph()
        G.add_edge(5, 7)
        G.add_edge("a", "b")

        CC.from_networkx_graph(G)

        assert "a" in CC.cells

    def test_add_cell(self):
        """Test adding a cell to a CC."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2, 3], rank=2)

        assert (1, 2, 3) in CC.cells

    def test_add_cells_from(self):
        """Test adding multiple cells to a CC."""
        CC = CombinatorialComplex()
        CC.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=2)

        assert (2, 3, 4) in CC.cells
        assert (3, 4, 5) in CC.cells

    def test_remove_cell(self):
        """Test removing a cell from a CC."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CC.remove_cell([1, 2, 3])

        assert (1, 2, 3) not in CC.cells
        assert (2, 3, 4) in CC.cells

    def test_remove_cells(self):
        """Test removing multiple cells from a CC."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CC.remove_cells([[1, 2, 3], [2, 3, 4]])

        assert (1, 2, 3) not in CC.cells
        assert (2, 3, 4) not in CC.cells

    def test_incidence_matrix(self):
        """Test generating an incidence matrix."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)

        B = CC.incidence_matrix(rank=0, to_rank=2)
        assert B.shape == (4, 2)
        assert (B.T[0].todense() == [1, 1, 1, 0]).all()
        assert (B.T[1].todense() == [0, 1, 1, 1]).all()


if __name__ == "__main__":
    unittest.main()
