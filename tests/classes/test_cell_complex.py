import unittest

import networkx as nx
import numpy as np

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex


class TestCellComplex(unittest.TestCase):
    def test_init(self):
        """Test empty cell complex and cell complex with cells."""
        # Test empty cell complex
        cx = CellComplex()
        self.assertEqual(len(cx.cells), 0)
        self.assertEqual(cx.is_regular, True)
        self.assertEqual(cx.dim, 0)

        # Test cell complex with cells
        c1 = Cell([1, 2, 3])
        c2 = Cell([1, 2, 3, 4])
        cx = CellComplex([c1, c2])
        assert c1 in cx.cells
        assert c2 in cx.cells
        self.assertEqual(cx.dim, 2)

    def test_init_with_nx_graph(self):
        """Test cell complex with networkx graph as input."""
        gr = nx.Graph()
        gr.add_edge(1, 0)
        gr.add_edge(2, 0)
        gr.add_edge(1, 2)
        cx = CellComplex(gr)
        self.assertEqual(cx.dim, 1)

    def test_is_regular_property(self):
        """Test is_regular property."""
        # Test non-regular cell complex
        # allows for constructions of non-regular cells
        cx = CellComplex(regular=False)
        # the "is_regular" method checks if any non-regular cells are added
        self.assertEqual(cx.is_regular, True)
        self.assertEqual(cx.dim, 0)

        # test non-regular cell complex
        CX = CellComplex(regular=False)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        CX.add_cell(c1)
        CX.add_cell([5, 6, 7, 8], rank=2)

        assert CX.is_regular == False

    def test_CellComplex_add_cell(self):
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

    def test_CellComplex_add_cells_from(self):
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
        cx = CellComplex()
        assert len(cx.cells) == 0
        assert len(cx.nodes) == 0
        assert len(cx.edges) == 0
        assert cx.is_regular == True

    # Test adding cells to CellComplex
    def test_add_cells_to_cell_complex(self):
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        assert len(cx.cells) == 3
        assert len(cx.nodes) == 8
        assert len(cx.edges) == 10

    # Test instantiating CellComplex from list of Cells
    def test_instantiating_cell_complex_from_list_of_cells(self):
        c1 = Cell((1, 2, 3))
        c2 = Cell((1, 2, 3, 4))
        cx = CellComplex([c1, c2])
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 4
        assert len(cx.edges) == 5

    # Test instantiating CellComplex from Graph
    def test_instantiating_cell_complex_from_graph(self):
        g = nx.Graph()
        g.add_edge(1, 0)
        g.add_edge(2, 0)
        g.add_edge(1, 2)
        cx = CellComplex(g)
        assert len(cx.cells) == 0
        assert len(cx.nodes) == 3
        assert len(cx.edges) == 3

    # Test adding cells from list of lists
    def test_add_cells_from_list_of_lists(self):
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 4
        assert len(cx.edges) == 5

    # Test removing cells from CellComplex
    def test_remove_cells_from_cell_complex(self):
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        cx.remove_cell([2, 3, 4, 5])
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 8
        assert len(cx.edges) == 10

    def test_add_cells_from(self):
        # Test adding multiple cells to an empty cell complex
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 3], [2, 3, 4]], rank=2)

        # Test adding multiple cells with duplicate vertices to a cell complex
        cx.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5]], rank=2)
        assert len(cx.cells) == 4

        # Test adding multiple cells with vertices that do not exist in the cell complex
        cx.add_cells_from([[4, 5, 6], [7, 8, 9]], rank=2)

        assert 6 in cx.nodes
        assert 9 in cx.nodes
        assert 8 in cx.nodes

    def test_incidence_matrix1(self):
        # Test computing the boundary matrix for the cell complex
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], rank=2)
        incidence_matrix = cx.incidence_matrix(2)
        assert incidence_matrix.shape == (10, 3)
        B1 = cx.incidence_matrix(1)
        assert B1.shape == (8, 10)

    def test_incidence_matrix2(self):

        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4], rank=2)
        cc.add_cell([3, 4, 5], rank=2)

        # Test the incidence matrix for the full cell complex
        inc_matrix_2 = cc.incidence_matrix(2, signed=False)
        assert inc_matrix_2.shape == (7, 3)
        assert (
            inc_matrix_2[:, 0].T.toarray()[0] == np.array([1, 1, 1, 0, 0, 0, 0])
        ).all()
        assert (
            inc_matrix_2[:, 1].T.toarray()[0] == np.array([0, 0, 1, 1, 1, 0, 0])
        ).all()
        assert (
            inc_matrix_2[:, 2].T.toarray()[0] == np.array([0, 0, 0, 0, 1, 1, 1])
        ).all()

        # Test the incidence matrix for the full cell complex
        inc_matrix_1 = cc.incidence_matrix(1, signed=False)
        assert inc_matrix_1.shape == (5, 7)
        assert (inc_matrix_1[:, 0].T.toarray()[0] == np.array([1, 1, 0, 0, 0])).all()
        assert (inc_matrix_1[:, 1].T.toarray()[0] == np.array([1, 0, 1, 0, 0])).all()
        assert (inc_matrix_1[:, 2].T.toarray()[0] == np.array([0, 1, 1, 0, 0])).all()

        inc_matrix_2_signed = cc.incidence_matrix(2, signed=True)
        inc_matrix_1_signed = cc.incidence_matrix(1, signed=True)

        # B1 * B2 == 0
        assert np.sum(inc_matrix_1_signed.dot(inc_matrix_2_signed).toarray()) == 0.0

        cc = CellComplex()
        inc_matrix = cc.incidence_matrix(1)
        assert inc_matrix.shape == (0, 0)

        cc = CellComplex()
        inc_matrix = cc.incidence_matrix(2)
        assert inc_matrix.shape == (0, 0)

    def test_cell_complex_clear(self):
        CX = CellComplex()
        CX.add_cells_from([[1, 2, 3, 4], [5, 6, 7, 8]], rank=2)
        CX.clear()
        assert len(CX.cells) == 0

    def test_incidence_matrix(self):
        # Test empty cell complex
        cc = CellComplex()
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(), np.zeros((0, 0))
        )

        # Test cell complex with only one cell
        cc.add_cell([1, 2, 3], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(), np.array([[1, -1, 1]]).T
        )

        # Test cell complex with multiple cells
        cc = CellComplex()
        cc.add_cell([2, 3, 4], rank=2)
        cc.add_cell([1, 3, 4], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(),
            np.array([[0, 0, 1, -1, 1], [1, -1, 0, 0, 1]]).T,
        )

        # Test non-regular cell complex
        cc = CellComplex(regular=False)
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(),
            np.array([[1, -1, 1, 0, 0], [0, 0, 1, -1, 1]]).T,
        )

    def test_feature_addition(self):

        CX = CellComplex()
        c1 = Cell((2, 3, 4), color="black")
        CX.add_cell(c1, weight=3)
        CX.add_cell([1, 2, 3, 4], rank=2, color="red")
        CX.add_cell([2, 3, 4, 5], rank=2, color="blue")
        CX.add_cell([5, 6, 7, 8], rank=2, color="green")

        assert CX.cells[(1, 2, 3, 4)]["color"] == "red"
        assert CX.cells[(2, 3, 4, 5)]["color"] == "blue"
        assert CX.cells[(5, 6, 7, 8)]["color"] == "green"

    def test_cellcomplex_adjacency_matrix(self):
        # Test adjacency matrix of empty CellComplex
        CX = CellComplex()
        np.testing.assert_array_equal(CX.adjacency_matrix(0), np.zeros((0, 0)))

        # Test adjacency matrix of CellComplex with one cell
        CX.add_cell([1, 2, 3], rank=2)
        A1 = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(CX.adjacency_matrix(0).todense(), A1)

        # Test adjacency matrix of CellComplex with multiple cells
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([4, 5, 6], rank=2)
        A = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(CX.adjacency_matrix(0).toarray(), A)


if __name__ == "__main__":
    unittest.main()
