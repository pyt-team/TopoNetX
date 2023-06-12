"""Test cell complex class."""

import unittest

import networkx as nx
import numpy as np

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.exception import TopoNetXError


class TestCellComplex(unittest.TestCase):
    """Test cell complex class."""

    def test_init_empty_cell_complex(self):
        """Test empty cell complex."""
        CX = CellComplex()
        assert len(CX.cells) == 0
        assert len(CX.nodes) == 0
        assert len(CX.edges) == 0
        assert CX.dim == 0
        assert CX.is_regular

    def test_init_cell_complex_with_list_of_cells(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 2, 3])
        c2 = Cell([1, 2, 3, 4])
        CX = CellComplex([c1, c2])
        assert c1 in CX.cells
        assert c2 in CX.cells
        assert CX.dim == 2

        c1 = Cell((1, 2, 3))
        c2 = Cell((1, 2, 3, 4))
        CX = CellComplex([c1, c2])
        assert len(CX.cells) == 2
        assert len(CX.nodes) == 4
        assert len(CX.edges) == 5

    def test_nodes_and_edges(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 3, 4])
        c2 = Cell([2, 3, 4])
        CX = CellComplex([c1, c2])
        assert set(CX.nodes) == {1, 2, 3, 4}
        assert set(CX.edges) == {(1, 3), (1, 4), (3, 2), (3, 4), (4, 2)}

    def test_init_networkx_graph(self):
        """Test cell complex with networkx graph as input."""
        gr = nx.Graph()
        gr.add_edge(1, 0)
        gr.add_edge(2, 0)
        gr.add_edge(1, 2)
        CX = CellComplex(gr)
        self.assertEqual(CX.dim, 1)
        assert len(CX.cells) == 0
        assert len(CX.nodes) == 3
        assert len(CX.edges) == 3

    def test_clone(self):
        """Test CellComplex.clone()."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2, color="blue")
        CX2 = CX.clone()
        assert CX2 is not CX
        assert CX2.cells[(1, 2, 3, 4)] is not CX.cells[(1, 2, 3, 4)]
        CX2.remove_node(1)
        assert 1 in CX.nodes
        assert [1, 2, 3, 4] in CX.cells

    def test_is_regular(self):
        """Test is_regular property."""
        # Test non-regular cell complex
        # allows for constructions of non-regular cells
        CX = CellComplex(regular=False)
        # the "is_regular" method checks if any non-regular cells are added
        self.assertEqual(CX.is_regular, True)
        self.assertEqual(CX.dim, 0)

        # test non-regular cell complex
        CX = CellComplex(regular=False)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        CX.add_cell(c1)
        CX.add_cell([5, 6, 7, 8], rank=2)

        assert CX.is_regular is False

    def test_add_cell(self):
        """Test adding cells to a cell complex."""
        # Test adding a single cell
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        assert len(CX.cells) == 1

        # Test adding multiple cells
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        assert len(CX.cells) == 3

        # Test adding cells to CellComplex
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        assert len(CX.cells) == 3
        assert len(CX.nodes) == 8
        assert len(CX.edges) == 10

    def test_add_cells_from(self):
        """Test adding cells from a list of cells or cell lists."""
        # Test adding cells from a list of cells
        CX = CellComplex()
        cells = [Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))]
        CX.add_cells_from(cells)
        assert len(CX.cells) == 2

        # Test adding cells from a list of cell lists
        CX = CellComplex()
        cell_lists = [[1, 2, 3, 4], [2, 3, 4, 5]]
        CX.add_cells_from(cell_lists, rank=2)
        assert len(CX.cells) == 2

        # Test adding cells from a list of lists
        CX = CellComplex()
        CX.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(CX.cells) == 2
        assert len(CX.nodes) == 4
        assert len(CX.edges) == 5

        # Test adding multiple cells to an empty cell complex
        CX = CellComplex()
        CX.add_cells_from([[1, 2, 3], [2, 3, 4]], rank=2)

        # Test adding multiple cells with duplicate vertices to a cell complex
        CX.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5]], rank=2)
        assert len(CX.cells) == 4

        # Test adding multiple cells with vertices that do not exist in the cell complex
        CX.add_cells_from([[4, 5, 6], [7, 8, 9]], rank=2)

        assert 6 in CX.nodes
        assert 9 in CX.nodes
        assert 8 in CX.nodes

    def test_add_cell_and_remove_cell(self):
        """Test removing one cell and several cells from a cell complex."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.remove_cell([1, 2, 3, 4])
        assert len(CX.cells) == 0

        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        CX.remove_cell([1, 2, 3, 4])
        CX.remove_cell([2, 3, 4, 5])
        assert len(CX.cells) == 1

        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        CX.remove_cell([2, 3, 4, 5])
        assert len(CX.cells) == 2
        assert len(CX.nodes) == 8
        assert len(CX.edges) == 10

    def test_incidence_matrix_shape(self):
        """Test the shape of the incidence matrix for the cell complex."""
        CX = CellComplex()
        CX.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], rank=2)
        row_index, col_index, B = CX.incidence_matrix(2, index=True)
        assert B.shape == (10, 3)
        assert len(row_index) == 10
        assert len(col_index) == 3
        row_index, col_index, B = CX.incidence_matrix(1, index=True)
        assert B.shape == (8, 10)
        assert len(row_index) == 8
        assert len(col_index) == 10
        row_index, col_index, B = CX.incidence_matrix(0, index=True)
        assert B.shape == (0, 8)
        assert len(row_index) == 0
        assert len(col_index) == 8

    def test_incidence_matrix_empty_cell_complex(self):
        """Test the incidence matrix for an empty cell complex."""
        CX = CellComplex()
        np.testing.assert_array_equal(
            CX.incidence_matrix(2).toarray(), np.zeros((0, 0))
        )

    def test_incidence_matrix_cell_complex_with_one_cell(self):
        """Test the incidence matrix for a cell complex with only one cell."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        np.testing.assert_array_equal(
            CX.incidence_matrix(2).toarray(), np.array([[1, -1, 1]]).T
        )

    def test_incidence_matrix_cell_complex_with_multiple_cells(self):
        """Test the incidence matrix for a cell complex with multiple cells."""
        CX = CellComplex()
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([1, 3, 4], rank=2)
        np.testing.assert_array_equal(
            CX.incidence_matrix(rank=2).toarray(),
            np.array([[0, 0, 1, -1, 1], [1, -1, 0, 0, 1]]).T,
        )

        # Test non-regular cell complex
        CX = CellComplex(regular=False)
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_cell([2, 3, 4], rank=2)
        np.testing.assert_array_equal(
            CX.incidence_matrix(rank=2).toarray(),
            np.array([[1, -1, 1, 0, 0], [0, 0, 1, -1, 1]]).T,
        )

    def test_incidence_matrix_unsigned_and_signed(self):
        """Test incidence matrix for the cell complex."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([3, 4, 5], rank=2)

        # Test the incidence matrix for the full cell complex
        B2 = CX.incidence_matrix(rank=2, signed=False)
        assert B2.shape == (7, 3)
        assert (B2[:, 0].T.toarray()[0] == np.array([1, 1, 1, 0, 0, 0, 0])).all()
        assert (B2[:, 1].T.toarray()[0] == np.array([0, 0, 1, 1, 1, 0, 0])).all()
        assert (B2[:, 2].T.toarray()[0] == np.array([0, 0, 0, 0, 1, 1, 1])).all()

        # Test the incidence matrix for the full cell complex
        B1 = CX.incidence_matrix(rank=1, signed=False)
        assert B1.shape == (5, 7)
        assert (B1[:, 0].T.toarray()[0] == np.array([1, 1, 0, 0, 0])).all()
        assert (B1[:, 1].T.toarray()[0] == np.array([1, 0, 1, 0, 0])).all()
        assert (B1[:, 2].T.toarray()[0] == np.array([0, 1, 1, 0, 0])).all()

        B2_signed = CX.incidence_matrix(rank=2, signed=True)
        B1_signed = CX.incidence_matrix(rank=1, signed=True)

        # B1 * B2 == 0
        assert np.sum(B1_signed.dot(B2_signed).toarray()) == 0.0

        CX = CellComplex()
        B1 = CX.incidence_matrix(rank=1)
        assert B1.shape == (0, 0)

        CX = CellComplex()
        B2 = CX.incidence_matrix(rank=2)
        assert B2.shape == (0, 0)

    def test_clear(self):
        """Test the clear method of the cell complex."""
        CX = CellComplex()
        CX.add_cells_from([[1, 2, 3, 4], [5, 6, 7, 8]], rank=2)
        CX.clear()
        assert len(CX.cells) == 0

    def test_add_cell_with_color_feature(self):
        """Test adding a cell with a color feature."""
        CX = CellComplex()
        c1 = Cell((2, 3, 4), color="black")
        CX.add_cell(c1, weight=3)
        CX.add_cell([1, 2, 3, 4], rank=2, color="red")
        CX.add_cell([2, 3, 4, 5], rank=2, color="blue")
        CX.add_cell([5, 6, 7, 8], rank=2, color="green")

        assert CX.cells[(1, 2, 3, 4)]["color"] == "red"
        assert CX.cells[(2, 3, 4, 5)]["color"] == "blue"
        assert CX.cells[(5, 6, 7, 8)]["color"] == "green"

    def test_adjacency_matrix_empty_cell_complex(self):
        """Test adjacency matrix for an empty cell complex."""
        CX = CellComplex()
        np.testing.assert_array_equal(CX.adjacency_matrix(0), np.zeros((0, 0)))

    def test_adjacency_matrix_cell_complex_with_one_cell(self):
        """Test adjacency matrix for a cell complex with one cell."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(CX.adjacency_matrix(0).todense(), A)

    def test_adjacency_matrix_cell_complex_with_multiple_cell(self):
        """Test adjacency matrix for a cell complex with multiple cells."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
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
        np.testing.assert_array_equal(CX.adjacency_matrix(rank=0).toarray(), A)

    def test_up_laplacian_matrix_empty_cell_complex(self):
        """Test up laplacian matrix for an empty cell complex."""
        CX = CellComplex()
        np.testing.assert_array_equal(CX.up_laplacian_matrix(rank=0), np.zeros((0, 0)))

    def test_up_laplacian_matrix_and_incidence_matrix(self):
        """Test up laplacian matrix and incidence matrix for a cell complex."""
        CX = CellComplex()
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([1, 3, 4], rank=2)

        B1 = CX.incidence_matrix(rank=1)
        L0_up = CX.up_laplacian_matrix(rank=0)
        expected = B1.dot(B1.T)
        np.testing.assert_array_equal(L0_up.toarray(), expected.toarray())

        B2 = CX.incidence_matrix(rank=2)
        L1_up = CX.up_laplacian_matrix(rank=1)
        expected = B2.dot(B2.T)
        np.testing.assert_array_equal(L1_up.toarray(), expected.toarray())

    def test_down_laplacian_matrix_and_incidence_matrix(self):
        """Test down laplacian matrix and incidence matrix for a cell complex."""
        CX = CellComplex()
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([1, 3, 4], rank=2)

        B1 = CX.incidence_matrix(rank=1)
        L1_down = CX.down_laplacian_matrix(rank=1)
        expected = (B1.T).dot(B1)
        np.testing.assert_array_equal(L1_down.toarray(), expected.toarray())

        B2 = CX.incidence_matrix(rank=2)
        L2_down = CX.down_laplacian_matrix(rank=2)
        expected = (B2.T).dot(B2)
        np.testing.assert_array_equal(L2_down.toarray(), expected.toarray())

    def test_hodge_laplacian_and_up_down_laplacians(self):
        """Test hodge laplacian matrix and up and down laplacians for a cell complex."""
        CX = CellComplex()
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([1, 3, 4], rank=2)

        L1_up = CX.up_laplacian_matrix(rank=1)
        L1_down = CX.down_laplacian_matrix(rank=1)
        L1_hodge = CX.hodge_laplacian_matrix(rank=1)
        expected = L1_up + L1_down
        np.testing.assert_array_equal(L1_hodge.toarray(), expected.toarray())

    def test_init_empty_abstrcct_cell(self):
        """Test empty cell complex."""
        cc = CellComplex([], rank=0)
        assert len(cc.nodes) == 0

    def test_init_nonempty_abstrcct_cell(self):
        """Test non-empty cell complex cells."""
        cc1 = CellComplex((1, 2, 3))
        cc2 = CellComplex(("a", "b", "c", "d"))
        assert len(cc1.nodes) == 3
        assert len(cc2.nodes) == 4

    def test_add_single_cell(self):
        """Test adding single cell to cell complex."""
        cc = CellComplex()
        cc1 = CellComplex()
        cc.add_cell((0, 1, 2, 3, 4), rank=2)
        cc1.add_cell(Cell((0, 1, 2, 3, 4)), rank=2)
        assert len(cc.cells) == 1
        assert cc.shape == (5, 5, 1)  # five nodes, five edges and one 2-cell

    def test_add_many_cells(self):
        """Test adding many cells to cell complex."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_cell([2, 3, 4, 5], rank=2)
        cc.add_cell([5, 6, 7, 8], rank=2)
        assert len(cc.cells) == 3
        assert cc.shape == (8, 10, 3)

    def test_add_single_node(self):
        """Test adding single node to empty cell complex."""
        cc = CellComplex()
        cc.add_node(1)
        assert len(cc.cells) == 0
        assert len(cc.nodes) == 1

    def test_add_many_nodes(self):
        """Test adding many nodes to empty cell complex."""
        cc = CellComplex()
        cc._add_nodes_from([1, 2, 3])
        assert len(cc.cells) == 0
        assert len(cc.nodes) == 3

    def test_add_node_to_cell(self):
        """Test add node multiple times to cell complex."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        cc.add_node(1)
        assert len(cc.cells) == 1
        assert len(cc.nodes) == 3

    def test_remove_cell(self):
        """Test remove cell from cell complex."""
        cc = CellComplex()
        cell = [1, 2, 3]
        cc.add_cell(cell, rank=2)
        cc.remove_cell(cell)
        assert len(cc.cells) == 0
        assert len(cc.nodes) == 3

    def test_remove_many_cells(self):
        """Test remove cells from cell complex."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3, 4], rank=2)
        cc.add_cell([2, 3, 4, 5], rank=2)
        cc.add_cell([5, 6, 7, 8], rank=2)
        cc.remove_cell([1, 2, 3, 4])
        cc.remove_cell([2, 3, 4, 5])
        assert len(cc.cells) == 1

    def test_remove_node(self):
        """Test remove node from cell complex."""
        cc = CellComplex()
        cell = [1, 2, 3]
        cc.add_cell(cell, rank=2)
        cc.remove_node(1)  # removing node removes cells attached to it.
        assert len(cc.cells) == 0
        assert len(cc.edges) == 1
        assert len(cc.nodes) == 2

        with self.assertRaises(TopoNetXError):
            cc.remove_node(980)

    def test_add_cells_from_list_cells(self):
        """Test adding cells from a list of cells or cell lists."""
        cc = CellComplex()
        cells = [Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))]
        cc.add_cells_from(cells)
        assert len(cc.cells) == 2

    def test_add_cells_from_list_of_cell_lists(self):
        """Test adding cells from a list of cell lists."""
        cc = CellComplex()
        cell_lists = [[1, 2, 3, 4], [2, 3, 4, 5]]
        cc.add_cells_from(cell_lists, rank=2)
        assert len(cc.cells) == 2

    def test_add_cells_from_list_of_lists(self):
        """Test adding cells from a list of lists."""
        cc = CellComplex()
        cc.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(cc.cells) == 2
        assert len(cc.nodes) == 4

    def test_incidence_matrix_empty_abstrcct_cell(self):
        """Test the incidence matrix for an empty cell complex."""
        cc = CellComplex()
        B = cc.incidence_matrix(2)
        np.testing.assert_array_equal(B.toarray(), np.zeros((0, 0)))

    def test_incidence_matrix_abstract_cell_with_one_cell(self):
        """Test the incidence matrix for abstract cell with only one cell."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        B = cc.incidence_matrix(2)
        np.testing.assert_array_equal(B.toarray(), np.array([[1, -1, 1]]).T)

    def test_adjacency_matrix_empty_abstrcct_cell(self):
        """Test adjccency matrix for an empty cell complex."""
        cc = CellComplex()
        np.testing.assert_array_equal(cc.adjacency_matrix(0), np.zeros((0, 0)))

    def test_adjccency_matrix_abstrcct_cell_with_one_cell(self):
        """Test adjccency matrix for an abstrcct cell with one cell."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(cc.adjacency_matrix(0).todense(), A)

    def test_coadjccency_matrix_abstrcct_cell_with_one_cell(self):
        """Test coadjccency matrix for an abstrcct cell with one cell."""
        cc = CellComplex()
        cc.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(cc.coadjacency_matrix(1).todense(), A)

    def test_repr(self):
        """Test the string representation of the cell complex."""
        cx = CellComplex(name="test")
        self.assertEqual(repr(cx), "CellComplex(name=test)")

    def test_len(self):
        """Test the length of the cell complex."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        self.assertEqual(len(cx), 3)

    def test_iter(self):
        """Test the iterator of the cell complex."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3], rank=2)
        cx.add_cell([2, 3, 4], rank=2)
        self.assertEqual(set(iter(cx)), {1, 2, 3, 4})

    def test_cell_equivalence_class(self):
        """Test the cell equivalence class method."""
        cx = CellComplex()
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.add_cell([2, 3, 4, 1], rank=2)
        cx.add_cell([1, 2, 3, 4], rank=2)
        cx.add_cell([1, 2, 3, 6], rank=2)
        cx.add_cell([3, 4, 1, 2], rank=2)
        cx.add_cell([4, 3, 2, 1], rank=2)
        cx.add_cell([1, 2, 7, 3], rank=2)
        c1 = Cell((1, 2, 3, 4, 5))
        cx.add_cell(c1, rank=2)

        equivalence_classes = cx._cell_equivalence_class()

        self.assertEqual(len(equivalence_classes), 4)

    def test__remove_equivalent_cells(self):
        """Test the remove equivalent cells method."""
        cx = CellComplex()
        cx.add_cell((1, 2, 3, 4), rank=2)
        cx.add_cell((2, 3, 4, 1), rank=2)
        cx.add_cell((1, 2, 3, 4), rank=2)
        cx.add_cell((1, 2, 3, 6), rank=2)
        cx.add_cell((3, 4, 1, 2), rank=2)
        cx.add_cell((4, 3, 2, 1), rank=2)
        cx.add_cell((1, 2, 7, 3), rank=2)
        c1 = Cell((1, 2, 3, 4, 5))
        cx.add_cell(c1, rank=2)
        cx._remove_equivalent_cells()
        assert len(cx.cells) == 4

    def test_get_cell_attributes2(self):
        """Test the remove equivalent cells method."""
        import networkx as nx

        G = nx.path_graph(3)

        d = {
            ((1, 2, 3, 4), 0): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
        }
        CX = CellComplex(G)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CX.add_cell([3, 4, 8], rank=2)
        CX.set_cell_attributes(d, rank=2)
        cell_color = CX.get_cell_attributes("color", 2)
        assert cell_color == {((1, 2, 3, 4), 0): "red", (1, 2, 4): "blue"}

    def test_remove_cells(self):
        """Test remove cells."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CX.add_cell([3, 4, 8], rank=2)
        CX.remove_cells([(1, 2, 3, 4), (1, 2, 4)])

        assert len(CX.cells) == 1

    def test_degree(self):
        """Test the degree of the cell complex."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)

        assert CX.degree(1) == 2
        assert CX.degree(2) == 3
        assert CX.degree(3) == 2
        assert CX.degree(4) == 3
        assert CX.degree(5) == 4
        assert CX.degree(6) == 2
        assert CX.degree(7) == 2
        assert CX.degree(8) == 2

    def test_size(self):
        """Test the size of the cell complex."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)

        c = Cell([1, 2, 3, 4])
        assert CX.size((1, 2, 3, 4)) == 4
        assert CX.size((2, 3, 4, 5)) == 4
        assert CX.size((5, 6, 7, 8)) == 4

        assert CX.size((1, 2, 3, 4), node_set=[1, 2, 3]) == 3
        assert CX.size((2, 3, 4, 5), node_set=[3, 4, 5]) == 3
        assert CX.size((5, 6, 7, 8), node_set=[6, 7, 8]) == 3
        assert CX.size(c, node_set=[1, 2, 3]) == 3
        with self.assertRaises(KeyError):
            CX.size((1, 2, 3, 4, 5, 5, 6))

    def test_insert_cell(self):
        """Test inserting a cell into the cell complex."""
        CX = CellComplex()
        CX._insert_cell([1, 2, 3, 4], rank=2)
        CX._insert_cell([2, 3, 4, 5], rank=2)
        CX._insert_cell([5, 6, 7, 8], rank=2)

        assert len(CX.cells) == 3
        assert [cell.elements for cell in CX.cells] == [
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            (5, 6, 7, 8),
        ]

    def test_delete_cell(self):
        """Test deleting a cell from the cell complex."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)

        CX._delete_cell((1, 2, 3, 4))
        assert len(CX.cells) == 2
        assert [cell.elements for cell in CX.cells] == [(2, 3, 4, 5), (5, 6, 7, 8)]

        CX._delete_cell((2, 3, 4, 5))
        assert len(CX.cells) == 1
        assert [cell.elements for cell in CX.cells] == [(5, 6, 7, 8)]

    def test_shape(self):
        """Test the shape of the cell complex."""
        CX = CellComplex()
        assert CX.shape == (0, 0, 0)

        CX.add_node(1)
        assert CX.shape == (1, 0, 0)

        CX.add_cell([1, 2], rank=1)
        assert CX.shape == (2, 1, 0)

        CX.add_cell([1, 2, 3, 4], rank=2)
        assert CX.shape == (4, 4, 1)

    def test_skeleton(self):
        """Test the skeleton of the cell complex."""
        CX = CellComplex()
        CX.add_node(1)
        CX.add_cell([1, 2], rank=1)
        CX.add_cell([1, 2, 3, 4], rank=2)

        nodes = CX.skeleton(0)
        assert len(nodes) == 4

        edges = CX.skeleton(1)
        assert len(edges) == 4

        cells = CX.skeleton(2)
        assert len(cells) == 1

    def test_filtration(self):
        """Test the filtration of the cell complex."""
        CX = CellComplex([[1, 2, 3], [4, 5]])
        test_filtration = {3: 2, 1: 4, (2, 3): 3.4, (1, 2, 3): 7.6}
        CX.set_filtration(test_filtration, "test")
        assert CX.get_filtration("test") == test_filtration

    def test_to_hypergraph(self):
        """Test the conversion of a cell complex to a hypergraph."""
        CX = CellComplex([[1, 2, 3], [4, 5]])
        hx = CX.to_hypergraph()
        assert set(hx.nodes) == set(CX.nodes)
        assert len(hx.edges) == len(CX.edges) + len(CX.cells)

    def test_restrict_to_nodes(self):
        """Test restricting a cell complex to a subset of nodes."""
        CX = CellComplex([[1, 2, 3], [3, 4, 5], [1, 4]])
        CX.set_filtration({(1, 2, 3): 1, (3, 4, 5): 2, (1, 4): 0, 3: 1}, "test")
        restricted = CX.restrict_to_nodes({1, 2, 3, 4})
        assert (1, 2, 3) in restricted.cells
        assert (1, 4) in restricted.edges
        assert (3, 4) in restricted.edges
        assert (3, 4, 5) not in restricted.cells
        assert (4, 5) not in restricted.edges
        assert restricted.get_filtration("test") == {(1, 2, 3): 1, (1, 4): 0, 3: 1}

    def test_get_cell_attributes(self):
        """Unit test for the get_cell_attributes method."""
        CX = CellComplex()
        d = {
            ((1, 2, 3, 4), 0): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
        }
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CX.add_cell([3, 4, 8], rank=2)
        CX.set_cell_attributes(d, 2)
        attributes = CX.get_cell_attributes("color", 2)
        self.assertEqual(attributes, {((1, 2, 3, 4), 0): "red", (1, 2, 4): "blue"})

    def test_remove_equivalent_cells(self):
        """Unit test for the remove_equivalent_cells method."""
        CX = CellComplex()

        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 1], rank=2)
        CX.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CX.add_cell([3, 4, 8], rank=2)
        self.assertEqual(len(CX.cells), 5)
        CX.remove_equivalent_cells()
        self.assertEqual(len(CX.cells), 3)

    def test_is_insertable_cycle(self):
        """Unit test for the is_insertable_cycle method."""
        CX = CellComplex()

        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([3, 4, 5], rank=2)
        self.assertTrue(CX.is_insertable_cycle([1, 2, 3, 4]))
        self.assertFalse(CX.is_insertable_cycle([1, 2, 3, 4, 1, 2]))
        self.assertFalse(CX.is_insertable_cycle([1, 2, 1]))

    def test_incidence_matrix(self):
        """Unit test for the incidence_matrix method."""
        CX = CellComplex()

        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([3, 4, 5], rank=2)
        B1 = CX.incidence_matrix(1)
        B2 = CX.incidence_matrix(2)
        self.assertEqual(
            B1.dot(B2).todense().tolist(),
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )

    def test_restrict_to_cells(self):
        """Test restricting a cell complex to a subset of cells and edges."""
        CX = CellComplex([[1, 2, 3, 4], [3, 4, 5], [1, 2, 3, 6]])
        CX.add_cell((1, 2, 3, 4), rank=2)
        CX.set_filtration(
            {(1, 2, 3, 4): 1, (1, 2): 2, (3, 5): 3, (4, 5): 4, (3, 4, 5): 5}, "test"
        )
        restricted = CX.restrict_to_cells(
            {(1, 2, 3, 4), (3, 5), CX.cells.raw((1, 2, 3, 6))}
        )
        assert len(restricted.cells[(1, 2, 3, 4)]) == 2
        assert (1, 2, 3, 4) in restricted.cells
        assert (1, 2, 3, 6) in restricted.cells
        assert (3, 4, 5) not in restricted.cells
        assert (3, 5) in restricted.edges
        assert (4, 5) not in restricted.edges
        assert restricted.get_filtration("test") == {
            ((1, 2, 3, 4), 0): 1,
            ((1, 2, 3, 4), 1): 1,
            (2, 1): 2,
            (3, 5): 3,
        }

    def test_is_connected(self):
        """Test is connected."""
        # connected cell complex
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4, 5], rank=2)
        CX.add_cell([1, 2, 7, 8], rank=2)
        assert CX.is_connected() is True

        # disconnected cell complex
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4, 5], rank=2)
        CX.add_cell([7, 8, 9, 10], rank=2)
        assert CX.is_connected() is False

    def test_singletons(self):
        """Test singletons."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        CX.add_node(0)
        CX.add_node(10)
        assert CX.singletons() == [0, 10]

    def test_remove_singletons(self):
        """Test remove singletons."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)
        CX.add_cell([5, 6, 7, 8], rank=2)
        CX.add_node(0)
        CX.add_node(10)
        CX.remove_singletons()
        assert CX.singletons() == []

    def test_contains__(self):
        """Test __contains__."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)

        assert 1 in CX
        assert 2 in CX
        assert 3 in CX
        assert 4 in CX
        assert 5 in CX
        assert (1, 2, 3, 4, 5) not in CX

    def test_neighbors(self):
        """Test neighbors of nodes."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3, 4], rank=2)
        CX.add_cell([2, 3, 4, 5], rank=2)

        assert list(CX.neighbors(1)) == [2, 4]

    def test_insert_cell_2(self):
        """Test for the _insert_cell method."""
        CX = CellComplex()
        CX._insert_cell([1, 2, 3], color="red")
        self.assertEqual(len(CX._cells._cells), 1)
        CX._insert_cell((1, 2, 3), attr1=5)
        self.assertEqual(len(CX._cells._cells), 1)
        CX._insert_cell(Cell([1, 2, 3], name="Cell 1"))
        self.assertEqual(len(CX._cells._cells), 1)
        CX._insert_cell([1, 2, 4])
        self.assertEqual(len(CX._cells._cells), 2)

    def test_insert_cell_invalid_input(self):
        """Test _insert_cell method with invalid input."""
        CX = CellComplex()
        with self.assertRaises(TypeError):
            CX._insert_cell("invalid_input")
        with self.assertRaises(TypeError):
            CX._insert_cell(123)
        with self.assertRaises(TypeError):
            CX._insert_cell({"elements": [1, 2, 3]})

    def test_delete_cell_2(self):
        """Test for the _delete_cell method."""
        CX = CellComplex()
        CX._insert_cell([1, 2, 3], color="red")
        CX._insert_cell([4, 5, 6], color="blue")
        CX._delete_cell((1, 2, 3))
        self.assertEqual(len(CX._cells._cells), 1)
        self.assertEqual(len(CX.cells), 1)
        CX._delete_cell((4, 5, 6))
        self.assertEqual(len(CX._cells._cells), 0)
        self.assertEqual(len(CX.cells), 0)

    def test_delete_cell_invalid_input(self):
        """Test for the _delete_cell method with invalid input."""
        CX = CellComplex()
        with self.assertRaises(KeyError):
            CX._delete_cell((1, 2, 3))

    def test_number_of_nodes(self):
        """Test for the number_of_nodes method."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_node(4)
        CX.add_node(5)
        self.assertEqual(CX.number_of_nodes(), 5)
        self.assertEqual(CX.number_of_nodes([1, 2, 3]), 3)
        self.assertEqual(CX.number_of_nodes([1, 6]), 1)

    def test_number_of_edges(self):
        """Test for the number_of_edges method."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_node(4)
        CX.add_node(5)
        self.assertEqual(CX.number_of_edges(), 3)
        self.assertEqual(CX.number_of_edges([(1, 2), (2, 3)]), 2)
        self.assertEqual(CX.number_of_edges([(4, 5), (6, 7)]), 0)

    def test_number_of_cells(self):
        """Test for the number_of_cells method."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_node(4)
        CX.add_node(5)
        self.assertEqual(CX.number_of_cells(), 1)
        self.assertEqual(CX.number_of_cells([(4, 5), (6, 7)]), 0)

    def test_order(self):
        """Test for the order method."""
        CX = CellComplex()
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_node(4)
        CX.add_node(5)
        self.assertEqual(CX.order(), 5)

    def test_neighbors_error(self):
        """Test to check for ValueError in neighbors method."""
        CX = CellComplex()
        with self.assertRaises(KeyError):
            CX.neighbors(10)

    def test_node_attributes(self):
        """Test the setting and retrieval of node attributes."""
        G = nx.path_graph(3)
        CX = CellComplex(G)
        CX.add_cell([1, 2, 3, 4], rank=2)
        d = {1: {"color": "red", "attr2": 1}, 2: {"color": "blue", "attr2": 3}}
        CX.set_node_attributes(d)
        assert CX.get_node_attributes("color") == {1: "red", 2: "blue"}

    def test_edge_attributes(self):
        """Test the setting and retrieval of edge attributes."""
        G = nx.path_graph(3)
        CX = CellComplex(G)
        CX.add_cell([1, 2, 3, 4], rank=2)
        d = {
            (1, 2): {"color": "red", "attr2": 1},
            (2, 3): {"color": "blue", "attr2": 3},
        }
        CX.set_edge_attributes(d)
        assert CX.get_edge_attributes("color") == {(1, 2): "red", (2, 3): "blue"}

    def test_hodge_laplacian_matrix(self):
        """Test the hodge_laplacian_matrix method of CellComplex."""
        CX = CellComplex()

        # Add cells to the complex for testing
        CX.add_cell([1, 2, 3], rank=2)
        CX.add_cell([2, 3, 4], rank=2)
        CX.add_cell([3, 4, 5], rank=2)
        CX.add_cell([4, 5, 6], rank=2)

        # Test case 1: Rank is 0
        rank = 0
        signed = True
        weight = None
        index = False

        result = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        assert result.shape == (6, 6)

        # Test case 2: Rank is 0 with index=True
        index = True

        result, index_list = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        assert len(result) == 6

        # Test case 3: Rank is 1 and maxdim is 2
        rank = 1

        result = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        # Test case 4: Rank is 1 and maxdim is 2 with index=True
        index = True

        index_list, result = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        # Assert the index_list is of type list
        self.assertIsInstance(index_list, dict)

        # Test case 5: Rank is 2 and maxdim is 2
        rank = 2

        result = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        # Test case 6: Rank is 2 and maxdim is 2 with index=True
        index = True

        index_list, result = CX.hodge_laplacian_matrix(rank, signed, weight, index)

        # Assert the index_list is of type list
        self.assertIsInstance(index_list, dict)

        # Test case 7: Rank is 2 and maxdim is not 2
        rank = 3

        with self.assertRaises(ValueError):
            CX.hodge_laplacian_matrix(rank, signed, weight, index)

    def test_init_type_exception(self):
        """Test if incorrect datatype raises a TypeError exception."""
        with self.assertRaises(TypeError):
            CellComplex(cells=1)

    def test_maxdim_warning(self):
        """Test if Deprecation Warning is raised when the maxdim method is called."""
        with self.assertWarns(DeprecationWarning):
            CX = CellComplex()
            CX.maxdim

    def test_dim_edgecases(self):
        """Test if dim is testing for edge cases."""
        CX = CellComplex()
        CX.add_node(1)
        assert CX.dim == 0

    def test_skeleton_method_exception(self):
        """Test if the skeleton method raises a TopoNetXError exception."""
        with self.assertRaises(TopoNetXError):
            CX = CellComplex()
            CX.skeleton(rank=4)

        with self.assertRaises(TopoNetXError):
            CX = CellComplex()
            CX.skeleton(rank=3)

    def test_getitem_dunder_method(self):
        """Test if the dunder __getitem__ method returns the appropriate neighbors of the given node."""
        CX = CellComplex()
        CX.add_edges_from([(1, 2), (2, 3), (5, 2), (1, 9), (1, 6)])
        assert sorted(list(CX.__getitem__(1))) == [2, 6, 9]
        assert sorted(list(CX.__getitem__(2))) == [1, 3, 5]
        assert sorted(list(CX.__getitem__(6))) == [1]


if __name__ == "__main__":
    unittest.main()
