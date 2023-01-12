"""Test cell complex class."""

import unittest

import networkx as nx
import numpy as np

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex


class TestCellComplex(unittest.TestCase):
    def test_init_empty_cell_complex(self):
        """Test empty cell complex."""
        cx = CellComplex()
        assert len(cx.cells) == 0
        assert len(cx.nodes) == 0
        assert len(cx.edges) == 0
        self.assertEqual(cx.dim, 0)
        assert cx.is_regular

    def test_init_cell_complex_with_list_of_cells(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 2, 3])
        c2 = Cell([1, 2, 3, 4])
        cx = CellComplex([c1, c2])
        assert c1 in cx.cells
        assert c2 in cx.cells
        self.assertEqual(cx.dim, 2)

        c1 = Cell((1, 2, 3))
        c2 = Cell((1, 2, 3, 4))
        cx = CellComplex([c1, c2])
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 4
        assert len(cx.edges) == 5

    def test_nodes_and_edges(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 3, 4])
        c2 = Cell([2, 3, 4])
        cx = CellComplex([c1, c2])
        assert set(cx.nodes) == {1, 2, 3, 4}
        assert set(cx.edges) == {(1, 3), (1, 4), (3, 2), (3, 4), (4, 2)}

    def test_init_networkx_graph(self):
        """Test cell complex with networkx graph as input."""
        gr = nx.Graph()
        gr.add_edge(1, 0)
        gr.add_edge(2, 0)
        gr.add_edge(1, 2)
        cx = CellComplex(gr)
        self.assertEqual(cx.dim, 1)
        assert len(cx.cells) == 0
        assert len(cx.nodes) == 3
        assert len(cx.edges) == 3

    def test_is_regular(self):
        """Test is_regular property."""
        # Test non-regular cell complex
        # allows for constructions of non-regular cells
        cx = CellComplex(regular=False)
        # the "is_regular" method checks if any non-regular cells are added
        self.assertEqual(cx.is_regular, True)
        self.assertEqual(cx.dim, 0)

        # test non-regular cell complex
        cx = CellComplex(regular=False)
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        cx.add_cell(c1)
        cx.add_cell([5, 6, 7, 8], rank=2)

        assert cx.is_regular is False

    def test_add_cell(self):
        """Test adding cells to a cell complex."""
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

        # Test adding cells to CellComplex
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        assert len(cx.cells) == 3
        assert len(cx.nodes) == 8
        assert len(cx.edges) == 10

    def test_add_cells_from(self):
        """Test adding cells from a list of cells or cell lists."""
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

        # Test adding cells from a list of lists
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 4
        assert len(cx.edges) == 5

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

    def test_add_cell_and_remove_cell(self):
        """Test removing one cell and several cells from a cell complex."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.remove_cell([1, 2, 3, 4])
        assert len(cx.cells) == 0

        cx = CellComplex()
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        cx.remove_cell([1, 2, 3, 4])
        cx.remove_cell([2, 3, 4, 5])
        assert len(cx.cells) == 1

        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4, 5], rank=2)
        cx.add_cell([5, 6, 7, 8], rank=2)
        cx.remove_cell([2, 3, 4, 5])
        assert len(cx.cells) == 2
        assert len(cx.nodes) == 8
        assert len(cx.edges) == 10

    def test_incidence_matrix_shape(self):
        """Test the shape of the incidence matrix for the cell complex."""
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], rank=2)
        incidence_matrix = cx.incidence_matrix(2)
        assert incidence_matrix.shape == (10, 3)
        B1 = cx.incidence_matrix(1)
        assert B1.shape == (8, 10)

    def test_incidence_matrix_empty_cell_complex(self):
        """Test the incidence matrix for an empty cell complex."""
        cc = CellComplex()
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(), np.zeros((0, 0))
        )

    def test_incidence_matrix_cell_complex_with_one_cell(self):
        """Test the incidence matrix for a cell complex with only one cell."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(2).toarray(), np.array([[1, -1, 1]]).T
        )

    def test_incidence_matrix_cell_complex_with_multiple_cells(self):
        """Test the incidence matrix for a cell complex with multiple cells."""
        # Test cell complex with multiple cells
        cc = CellComplex()
        cc.add_cell([2, 3, 4], rank=2)
        cc.add_cell([1, 3, 4], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(d=2).toarray(),
            np.array([[0, 0, 1, -1, 1], [1, -1, 0, 0, 1]]).T,
        )

        # Test non-regular cell complex
        cc = CellComplex(regular=False)
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4], rank=2)
        np.testing.assert_array_equal(
            cc.incidence_matrix(d=2).toarray(),
            np.array([[1, -1, 1, 0, 0], [0, 0, 1, -1, 1]]).T,
        )

    def test_incidence_matrix_unsigned_and_signed(self):
        """Test incidence matrix for the cell complex."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4], rank=2)
        cc.add_cell([3, 4, 5], rank=2)

        # Test the incidence matrix for the full cell complex
        inc_matrix_d2 = cc.incidence_matrix(d=2, signed=False)
        assert inc_matrix_d2.shape == (7, 3)
        assert (
            inc_matrix_d2[:, 0].T.toarray()[0] == np.array([1, 1, 1, 0, 0, 0, 0])
        ).all()
        assert (
            inc_matrix_d2[:, 1].T.toarray()[0] == np.array([0, 0, 1, 1, 1, 0, 0])
        ).all()
        assert (
            inc_matrix_d2[:, 2].T.toarray()[0] == np.array([0, 0, 0, 0, 1, 1, 1])
        ).all()

        # Test the incidence matrix for the full cell complex
        inc_matrix_d1 = cc.incidence_matrix(d=1, signed=False)
        assert inc_matrix_d1.shape == (5, 7)
        assert (inc_matrix_d1[:, 0].T.toarray()[0] == np.array([1, 1, 0, 0, 0])).all()
        assert (inc_matrix_d1[:, 1].T.toarray()[0] == np.array([1, 0, 1, 0, 0])).all()
        assert (inc_matrix_d1[:, 2].T.toarray()[0] == np.array([0, 1, 1, 0, 0])).all()

        inc_matrix_d2_signed = cc.incidence_matrix(d=2, signed=True)
        inc_matrix_d1_signed = cc.incidence_matrix(d=1, signed=True)

        # B1 * B2 == 0
        assert np.sum(inc_matrix_d1_signed.dot(inc_matrix_d2_signed).toarray()) == 0.0

        cc = CellComplex()
        inc_matrix_d1 = cc.incidence_matrix(d=1)
        assert inc_matrix_d1.shape == (0, 0)

        cc = CellComplex()
        inc_matrix_d2 = cc.incidence_matrix(d=2)
        assert inc_matrix_d2.shape == (0, 0)

    def test_clear(self):
        """Test the clear method of the cell complex."""
        cx = CellComplex()
        cx.add_cells_from([[1, 2, 3, 4], [5, 6, 7, 8]], rank=2)
        cx.clear()
        assert len(cx.cells) == 0

    def test_add_cell_with_color_feature(self):
        """Test adding a cell with a color feature."""
        cx = CellComplex()
        c1 = Cell((2, 3, 4), color="black")
        cx.add_cell(c1, weight=3)
        cx.add_cell([1, 2, 3, 4], rank=2, color="red")
        cx.add_cell([2, 3, 4, 5], rank=2, color="blue")
        cx.add_cell([5, 6, 7, 8], rank=2, color="green")

        assert cx.cells[(1, 2, 3, 4)]["color"] == "red"
        assert cx.cells[(2, 3, 4, 5)]["color"] == "blue"
        assert cx.cells[(5, 6, 7, 8)]["color"] == "green"

    def test_adjacency_matrix_empty_cell_complex(self):
        """Test adjacency matrix for an empty cell complex."""
        cx = CellComplex()
        np.testing.assert_array_equal(cx.adjacency_matrix(0), np.zeros((0, 0)))

    def test_adjacency_matrix_cell_complex_with_one_cell(self):
        """Test adjacency matrix for a cell complex with one cell."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(cx.adjacency_matrix(0).todense(), adj)

    def test_adjacency_matrix_cell_complex_with_multiple_cell(self):
        """Test adjacency matrix for a cell complex with multiple cells."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4], rank=2)
        cx.add_cell([4, 5, 6], rank=2)
        adj = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(cx.adjacency_matrix(d=0).toarray(), adj)

    def test_up_laplacian_matrix_empty_cell_complex(self):
        """Test up laplacian matrix for an empty cell complex."""
        cx = CellComplex()
        np.testing.assert_array_equal(cx.up_laplacian_matrix(d=0), np.zeros((0, 0)))

    def test_up_laplacian_matrix_and_incidence_matrix(self):
        cx = CellComplex()
        cx.add_cell([2, 3, 4], rank=2)
        cx.add_cell([1, 3, 4], rank=2)

        inc_matrix_d1 = cx.incidence_matrix(d=1)
        up_lap_d0 = cx.up_laplacian_matrix(d=0)
        expected = inc_matrix_d1.dot(inc_matrix_d1.T)
        np.testing.assert_array_equal(up_lap_d0.toarray(), expected.toarray())

        inc_matrix_d2 = cx.incidence_matrix(d=2)
        up_lap_d1 = cx.up_laplacian_matrix(d=1)
        expected = inc_matrix_d2.dot(inc_matrix_d2.T)
        np.testing.assert_array_equal(up_lap_d1.toarray(), expected.toarray())

    def test_down_laplacian_matrix_and_incidence_matrix(self):
        cx = CellComplex()
        cx.add_cell([2, 3, 4], rank=2)
        cx.add_cell([1, 3, 4], rank=2)

        inc_matrix_d1 = cx.incidence_matrix(d=1)
        down_lap_d1 = cx.down_laplacian_matrix(d=1)
        expected = (inc_matrix_d1.T).dot(inc_matrix_d1)
        np.testing.assert_array_equal(down_lap_d1.toarray(), expected.toarray())

        inc_matrix_d2 = cx.incidence_matrix(d=2)
        down_lap_d2 = cx.down_laplacian_matrix(d=2)
        expected = (inc_matrix_d2.T).dot(inc_matrix_d2)
        np.testing.assert_array_equal(down_lap_d2.toarray(), expected.toarray())

    def test_hodge_laplacian_and_up_down_laplacians(self):
        cx = CellComplex()
        cx.add_cell([2, 3, 4], rank=2)
        cx.add_cell([1, 3, 4], rank=2)

        up_lap_d1 = cx.up_laplacian_matrix(d=1)
        down_lap_d1 = cx.down_laplacian_matrix(d=1)
        hodge_lap_d1 = cx.hodge_laplacian_matrix(d=1)
        expected = up_lap_d1 + down_lap_d1
        np.testing.assert_array_equal(hodge_lap_d1.toarray(), expected.toarray())


if __name__ == "__main__":
    unittest.main()
