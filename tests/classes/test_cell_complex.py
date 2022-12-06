import unittest

import networkx as nx

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex


class TestCellComplex(unittest.TestCase):
    def test_cell_complex_initialization(self):
        # Test empty cell complex
        cc = CellComplex()
        self.assertEqual(len(cc.cells), 0)
        self.assertEqual(cc.is_regular, True)
        self.assertEqual(cc.dim, 0)

        # Test cell complex with cells

        c1 = Cell([1, 2, 3])
        c2 = Cell([1, 2, 3, 4])
        cc = CellComplex([c1, c2])
        assert c1 in cc.cells
        assert c2 in cc.cells
        self.assertEqual(cc.dim, 2)

        # Test cell complex with graph

        G = nx.Graph()
        G.add_edge(1, 0)
        G.add_edge(2, 0)
        G.add_edge(1, 2)
        cc = CellComplex(G)
        self.assertEqual(cc.dim, 1)

        # Test non-regular cell complex
        # allows for constructions of non-regular cells
        cc = CellComplex(regular=False)  
        # the "is_regular" method checks if any non-regular cells are added
        self.assertEqual(cc.is_regular, True)
        self.assertEqual(cc.dim, 0)

        # test non-regular cell complex
        CX = CellComplex(regular=False)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        CX.add_cell(c1)
        CX.add_cell([5, 6, 7, 8], rank=2)

        assert CX.is_regular == False


def test_CellComplex_add_cell():
    # Test adding a single cell
    cx = CellComplex()
    cx.add_cell([1, 2, 3, 4], rank=2)
    assert len(cx.cells) == 1

    # Test adding multiple cells
    cx = CellComplex()
    cx.add_cell([1, 2, 3, 4], rank=2)
    cx.add_cell([2, 3, 4, 5], rank=2)
    cx.add_cell([5, 6, 7, 8], rank=2)
    assert len(cx.cells) == 3


def test_CellComplex_add_cells_from():
    # Test adding cells from a list of cells
    cx = CellComplex()
    cells = [Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))]
    cx.add_cells_from(cells)
    assert len(cx.cells) == 2

    # Test adding cells from a list of cell lists
    cx = CellComplex()
    cell_lists = [[1, 2, 3, 4], [2, 3, 4, 5]]
    cx.add_cells_from(cell_lists, rank=2)
    assert len(cx.cells) == 2

    def test_CellComplex_remove_cell(self):
        # Test removing a single cell
        cx = CellComplex()
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.remove_cell([1, 2, 3, 4])
        assert len(cx.cells) == 0

        # Test removing multiple cells
        cx = CellComplex()
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        cx.remove_cell([1, 2, 3, 4])
        cx.remove_cell([2, 3, 4, 5])
        assert len(cx.cells) == 1

    # Test empty CellComplex
    def test_empty_cell_complex(self):
        cc = CellComplex()
        assert len(cc.cells) == 0
        assert len(cc.nodes) == 0
        assert len(cc.edges) == 0
        assert cc.is_regular == True

    # Test adding cells to CellComplex
    def test_add_cells_to_cell_complex(self):
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4, 5], rank=2)
        cc.add_cell([5, 6, 7, 8], rank=2)
        assert len(cc.cells) == 3
        assert len(cc.nodes) == 8
        assert len(cc.edges) == 10

    # Test instantiating CellComplex from list of Cells
    def test_instantiating_cell_complex_from_list_of_cells(self):
        c1 = Cell((1, 2, 3))
        c2 = Cell((1, 2, 3, 4))
        cc = CellComplex([c1, c2])
        assert len(cc.cells) == 2
        assert len(cc.nodes) == 4
        assert len(cc.edges) == 5

    # Test instantiating CellComplex from Graph
    def test_instantiating_cell_complex_from_graph(self):
        g = nx.Graph()
        g.add_edge(1, 0)
        g.add_edge(2, 0)
        g.add_edge(1, 2)
        cc = CellComplex(g)
        assert len(cc.cells) == 0
        assert len(cc.nodes) == 3
        assert len(cc.edges) == 3

    # Test adding cells from list of lists
    def test_add_cells_from_list_of_lists(self):
        cc = CellComplex()
        cc.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(cc.cells) == 2
        assert len(cc.nodes) == 4
        assert len(cc.edges) == 5

    # Test removing cells from CellComplex
    def test_remove_cells_from_cell_complex(self):
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4, 5], rank=2)
        cc.add_cell([5, 6, 7, 8], rank=2)
        cc.remove_cell([2, 3, 4, 5])
        assert len(cc.cells) == 2
        assert len(cc.nodes) == 8
        assert len(cc.edges) == 10


if __name__ == "__main__":
    unittest.main()
