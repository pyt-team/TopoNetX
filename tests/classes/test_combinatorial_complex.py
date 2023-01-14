"""Unit tests for the combinatorial complex class."""

import unittest

import networkx as nx

from toponetx.classes.abstract_cell import AbstractCell
from toponetx.classes.combinatorial_complex import CombinatorialComplex


class TestCombinatorialComplex(unittest.TestCase):
    def test_init_empty_cc(self):
        # Test empty combinatorial complex
        cc = CombinatorialComplex()
        assert len(cc) == 0

    def test_init_from_lists(self):
        # Test combinatorial complex with cells
        cc = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert len(cc.cells) == 6
        assert (1, 2, 3) in cc.cells
        assert (2, 3, 4) in cc.cells

    def test_init_from_abstract_cells(self):
        """Test creation of a CC from nodes and cells."""

        y1 = AbstractCell(elements=[1, 2], rank=1)
        y2 = AbstractCell(elements=[2, 4], rank=1)
        y3 = AbstractCell(elements=[3, 5], rank=1)
        y4 = AbstractCell(elements=[4, 5], rank=1)
        y5 = AbstractCell(elements=[5, 7], rank=1)

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

        CC.from_networkx_graph(g)

        assert "a" in CC.cells

    def test_combinatorial_complex_add_cell(self):
        cc = CombinatorialComplex()

        # Test adding a cell
        cc.add_cell([1, 2, 3], rank=2)

        assert (1, 2, 3) in cc.cells

        # Test adding multiple cells
        cc.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=2)

        assert (1, 2, 3) in cc.cells
        assert (2, 3, 4) in cc.cells
        assert (3, 4, 5) in cc.cells

    def test_combinatorial_complex_remove_cell(self):
        cc = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)

        # Test removing a cell
        cc.remove_cell([1, 2, 3])

        assert (1, 2, 3) not in cc.cells
        assert (2, 3, 4) in cc.cells

        # Test removing multiple cells
        cc.remove_cells([[2, 3, 4]])

        assert (1, 2, 3) not in cc.cells
        assert (2, 3, 4) not in cc.cells

    def test_combinatorial_complex_incidence_matrix(self):
        cc = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)

        # Test generating incidence matrix
        im = cc.incidence_matrix(0, 2)
        assert im.shape == (4, 2)
        assert (im.T[0].todense() == [1, 1, 1, 0]).all()
        assert (im.T[1].todense() == [0, 1, 1, 1]).all()


if __name__ == "__main__":
    unittest.main()
