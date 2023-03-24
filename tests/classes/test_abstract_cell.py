"""Test abstract cell class."""

import numpy as np
import pytest

from toponetx.classes.abstract_cell import AbstractCell, AbstractCellView


class TestAbstractCell:
    def test_init_empty_abstract_cell(self):
        """Test empty abstract cell."""
        AC = AbstractCell([], rank=0)
        assert len(AC.nodes) == 0
        assert AC.rank == 0

    def test_init_nonempty_abstract_cell(self):
        """Test non-empty abstract cells."""
        AC1 = AbstractCell((1, 2, 3))
        AC2 = AbstractCell(("a", "b", "c", "d"))
        assert len(AC1.nodes) == 3
        assert len(AC2.nodes) == 4


class TestAbstractCellView:
    # def test_init_empty_abstract_cell_view(self):
    #     """Test empty abstract cell."""
    #     ACV = AbstractCellView([])
    #     assert len(ACV.cells) == 0

    # def test_init_abstract_cell_view_with_nodes(self):
    #     """Test abstract cell initiated from nodes."""
    #     acv1 = AbstractCellView([1, 2, 3])
    #     acv2 = AbstractCellView(["a","b","c","d"])
    #     self.assertEqual(len(acv1.cells), 1)
    #     self.assertEqual(len(acv2.cells), 1)
    #     self.assertEqualv(len(acv1.nodes), 3)
    #     self.assertEqual(len(acv2.nodes), 4)
    #     self.assertEqual(set(acv1.nodes), {1, 2, 3})
    #     self.assertEqual(set(acv2.nodes), {"a","b","c","d"})

    def test_add_single_cell(self):
        """Test adding single cell to abstract cell."""
        ACV = AbstractCellView()
        ACV1 = AbstractCellView()
        ACV.add_cell((0, 1, 2, 3, 4), rank=1)
        ACV1.add_cell(AbstractCell((0, 1, 2, 3, 4)), rank=1)
        assert len(ACV.cell_dict[1]) == 1
        assert ACV.shape == [5, 1]

    def test_add_many_cells(self):
        """Test adding many cells to abstract cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=2)
        ACV.add_cell([2, 3, 4, 5], rank=2)
        ACV.add_cell([5, 6, 7, 8], rank=2)
        assert len(ACV.cell_dict[2]) == 3
        assert ACV.shape == [8, 3]

    def test_add_single_node(self):
        """Test adding single node to empty abstract cell."""
        ACV = AbstractCellView()
        ACV.add_node([1])
        self.assertEqual(len(ACV.cells), 1)
        self.assertEqual(len(ACV.nodes), 1)

    def test_add_many_nodes(self):
        """Test adding many nodes to empty abstract cell."""
        ACV = AbstractCellView()
        ACV.add_node([1, 2, 3])
        self.assertEqual(len(ACV.cells), 1)
        self.assertEqual(len(ACV.nodes), 3)

    def test_add_node_to_cell(self):
        """Test add node to abstract cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=1)
        ACV.add_node([1])
        self.assertEqual(len(ACV.cells), 2)
        self.assertEqual(len(ACV.nodes), 4)

    def test_remove_cell(self):
        """Test remove cell from abstract cell."""
        ACV = AbstractCellView()
        cell = [1, 2, 3]
        ACV.add_cell(cell, rank=2)
        ACV.remove_cell(cell)
        self.assertEqual(len(ACV.cells), 0)
        self.assertEqual(len(ACV.nodes), 0)

    def test_remove_many_cells(self):
        """Test remove cells from abstract cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3, 4], rank=2)
        ACV.add_cell([2, 3, 4, 5], rank=2)
        ACV.add_cell([5, 6, 7, 8], rank=2)
        ACV.remove_cell([1, 2, 3, 4])
        ACV.remove_cell([2, 3, 4, 5])
        self.assertEqual(len(ACV.cells), 1)
        self.assertEqual(len(ACV.nodes), 4)

    def test_remove_node(self):
        """Test remove node from abstract cell."""
        ACV = AbstractCellView()
        cell = [1, 2, 3]
        ACV.add_cell(cell, rank=2)
        ACV.remove_node([1])
        self.assertEqual(len(ACV.cells), 1)
        self.assertEqual(len(ACV.nodes), 2)

    def test_add_cells_from_list_cells(self):
        """Test adding cells from a list of cells or cell lists."""
        ACV = AbstractCellView()
        cells = [AbstractCellView((1, 2, 3, 4)), AbstractCellView((2, 3, 4, 5))]
        ACV.add_cells_from(cells)
        self.assertEqual(len(ACV.cells), 2)

    def test_add_cells_from_list_of_cell_lists(self):
        """Test adding cells from a list of cell lists."""
        ACV = AbstractCellView()
        cell_lists = [[1, 2, 3, 4], [2, 3, 4, 5]]
        ACV.add_cells_from(cell_lists, rank=2)
        self.assertEqual(len(CX.cells), 2)

    def test_add_cells_from_list_of_lists(self):
        """Test adding cells from a list of lists."""
        ACV = AbstractCellView()
        ACV.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        self.assertEqual(len(ACV.cells), 2)
        self.assertEqual(len(ACV.nodes), 6)

    def test_incidence_matrix_shape(self):
        """Test the shape of the incidence matrix for the abstract cell."""
        ACV = AbstractCellView((1, 2, 3))
        B = ACV.incidence_matrix(2)
        self.assertEqual(ACV.shape, (10, 3))
        B = ACV.incidence_matrix(1)
        self.assertEqual(B.shape, (8, 10))

    def test_incidence_matrix_empty_abstract_cell(self):
        """Test the incidence matrix for an empty abstract cell."""
        ACV = AbstractCellView()
        B = ACV.incidence_matrix(2)
        np.testing.assert_array_equal(B.toarray(), np.zeros((0, 0)))

    def test_incidence_matrix_abstract_cell_with_one_cell(self):
        """Test the incidence matrix for abstract cell with only one cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=2)
        B = ACV.incidence_matrix(2)
        np.testing.assert_array_equal(B.toarray(), np.array([[1, -1, 1]]).T)

    def test_adjacency_matrix_empty_abstract_cell(self):
        """Test adjacency matrix for an empty abstract cell."""
        ACV = AbstractCellView()
        np.testing.assert_array_equal(ACV.adjacency_matrix(0), np.zeros((0, 0)))

    def test_adjacency_matrix_abstract_cell_with_one_cell(self):
        """Test adjacency matrix for an abstract cell with one cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(ACV.adjacency_matrix(0).todense(), A)

    def test_adjacency_matrix_abstract_cell_with_multiple_cell(self):
        """Test adjacency matrix for an abstract cell with multiple cells."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=2)
        ACV.add_cell([2, 3, 4], rank=2)
        ACV.add_cell([4, 5, 6], rank=2)
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
        np.testing.assert_array_equal(ACV.adjacency_matrix(rank=0).toarray(), A)

    def test_coadjacency_matrix_empty_abstract_cell(self):
        """Test coadjacency matrix for an empty abstract cell."""
        ACV = AbstractCellView()
        np.testing.assert_array_equal(ACV.coadjacency_matrix(0), np.zeros((0, 0)))

    def test_coadjacency_matrix_abstract_cell_with_one_cell(self):
        """Test coadjacency matrix for an abstract cell with one cell."""
        ACV = AbstractCellView()
        ACV.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(ACV.coadjacency_matrix(0).todense(), A)
