"""Test cell complex class."""

import networkx as nx
import numpy as np
import pytest
import scipy
from scipy.sparse import bmat

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex

try:
    import hypernetx as hnx
except ImportError:
    hnx = None


class TestCellComplex:
    """Test cell complex class."""

    def test_init_empty_cell_complex(self):
        """Test empty cell complex."""
        CC = CellComplex()
        assert len(CC.cells) == 0
        assert len(CC.nodes) == 0
        assert len(CC.edges) == 0
        assert CC.dim == 0
        assert CC.is_regular

    def test_init_cell_complex_with_list_of_cells(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 2, 3])
        c2 = Cell([1, 2, 3, 4])
        CC = CellComplex([c1, c2])
        assert c1 in CC.cells
        assert c2 in CC.cells
        assert CC.dim == 2

        c1 = Cell((1, 2, 3))
        c2 = Cell((1, 2, 3, 4))
        CC = CellComplex([c1, c2])
        assert len(CC.cells) == 2
        assert len(CC.nodes) == 4
        assert len(CC.edges) == 5

    def test_nodes_and_edges(self):
        """Test cell complex with cells."""
        c1 = Cell([1, 3, 4])
        c2 = Cell([2, 3, 4])
        CC = CellComplex([c1, c2])
        assert set(CC.nodes) == {1, 2, 3, 4}
        assert set(CC.edges) == {(1, 3), (1, 4), (3, 2), (3, 4), (4, 2)}

    def test_init_networkx_graph(self):
        """Test cell complex with networkx graph as input."""
        gr = nx.Graph()
        gr.add_edge(1, 0)
        gr.add_edge(2, 0)
        gr.add_edge(1, 2)
        CC = CellComplex(gr)
        assert CC.dim == 1
        assert len(CC.cells) == 0
        assert len(CC.nodes) == 3
        assert len(CC.edges) == 3

    def test_clone(self):
        """Test CellComplex.clone()."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2, color="blue")
        CC2 = CC.clone()
        assert CC2 is not CC
        assert CC2.cells[(1, 2, 3, 4)] is not CC.cells[(1, 2, 3, 4)]
        CC2.remove_node(1)
        assert 1 in CC.nodes
        assert [1, 2, 3, 4] in CC.cells

    def test_is_regular(self):
        """Test is_regular property."""
        # Test non-regular cell complex
        # allows for constructions of non-regular cells
        CC = CellComplex(regular=False)
        # the "is_regular" method checks if any non-regular cells are added
        assert CC.is_regular
        assert CC.dim == 0

        # test non-regular cell complex
        CC = CellComplex(regular=False)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5, 2, 3, 4, 5], rank=2)  # non-regular 2-cell
        c1 = Cell((1, 2, 3, 4, 5, 1, 2, 3, 4, 5), regular=False)
        CC.add_cell(c1)
        CC.add_cell([5, 6, 7, 8], rank=2)

        assert CC.is_regular is False

    def test_add_cell(self):
        """Test adding cells to a cell complex."""
        # Test adding a single cell
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        assert len(CC.cells) == 1

        # Test adding multiple cells
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        assert len(CC.cells) == 3

        # Test adding cells to CellComplex
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        assert len(CC.cells) == 3
        assert len(CC.nodes) == 8
        assert len(CC.edges) == 10

    def test_add_cells_from(self):
        """Test adding cells from a list of cells or cell lists."""
        # Test adding cells from a list of cells
        CC = CellComplex()
        cells = [Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))]
        CC.add_cells_from(cells)
        assert len(CC.cells) == 2

        # Test adding cells from a list of cell lists
        CC = CellComplex()
        cell_lists = [[1, 2, 3, 4], [2, 3, 4, 5]]
        CC.add_cells_from(cell_lists, rank=2)
        assert len(CC.cells) == 2

        # Test adding cells from a list of lists
        CC = CellComplex()
        CC.add_cells_from([[1, 2, 4], [1, 2, 7]], rank=2)
        assert len(CC.cells) == 2
        assert len(CC.nodes) == 4
        assert len(CC.edges) == 5

        # Test adding multiple cells to an empty cell complex
        CC = CellComplex()
        CC.add_cells_from([[1, 2, 3], [2, 3, 4]], rank=2)

        # Test adding multiple cells with duplicate vertices to a cell complex
        CC.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5]], rank=2)
        assert len(CC.cells) == 4

        # Test adding multiple cells with vertices that do not exist in the cell complex
        CC.add_cells_from([[4, 5, 6], [7, 8, 9]], rank=2)

        assert 6 in CC.nodes
        assert 9 in CC.nodes
        assert 8 in CC.nodes

    def test_add_cell_and_remove_cell(self):
        """Test removing one cell and several cells from a cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.remove_cell([1, 2, 3, 4])
        assert len(CC.cells) == 0

        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        CC.remove_cell([1, 2, 3, 4])
        CC.remove_cell([2, 3, 4, 5])
        assert len(CC.cells) == 1

        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        CC.remove_cell([2, 3, 4, 5])
        assert len(CC.cells) == 2
        assert len(CC.nodes) == 8
        assert len(CC.edges) == 10

    def test_incidence_matrix_shape(self):
        """Test the shape of the incidence matrix for the cell complex."""
        CC = CellComplex()
        CC.add_cells_from([[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]], rank=2)
        row_index, col_index, B = CC.incidence_matrix(2, index=True)
        assert B.shape == (10, 3)
        assert len(row_index) == 10
        assert len(col_index) == 3
        row_index, col_index, B = CC.incidence_matrix(1, index=True)
        assert B.shape == (8, 10)
        assert len(row_index) == 8
        assert len(col_index) == 10
        row_index, col_index, B = CC.incidence_matrix(0, index=True)
        assert B.shape == (0, 8)
        assert len(row_index) == 0
        assert len(col_index) == 8

    def test_incidence_matrix_empty_cell_complex(self):
        """Test the incidence matrix for an empty cell complex."""
        CC = CellComplex()
        np.testing.assert_array_equal(
            CC.incidence_matrix(2).toarray(), np.zeros((0, 0))
        )

    def test_incidence_matrix_cell_complex_with_one_cell(self):
        """Test the incidence matrix for a cell complex with only one cell."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        np.testing.assert_array_equal(
            CC.incidence_matrix(2).toarray(), np.array([[1, -1, 1]]).T
        )

    def test_incidence_matrix_cell_complex_with_multiple_cells(self):
        """Test the incidence matrix for a cell complex with multiple cells."""
        CC = CellComplex()
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([1, 3, 4], rank=2)
        np.testing.assert_array_equal(
            CC.incidence_matrix(rank=2).toarray(),
            np.array([[0, 0, 1, -1, 1], [1, -1, 0, 0, 1]]).T,
        )

        # Test non-regular cell complex
        CC = CellComplex(regular=False)
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        np.testing.assert_array_equal(
            CC.incidence_matrix(rank=2).toarray(),
            np.array([[1, -1, 1, 0, 0], [0, 0, 1, -1, 1]]).T,
        )

    def test_incidence_matrix_unsigned_and_signed(self):
        """Test incidence matrix for the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)

        # Test the incidence matrix for the full cell complex
        B2 = CC.incidence_matrix(rank=2, signed=False)
        assert B2.shape == (7, 3)
        assert (B2[:, 0].T.toarray()[0] == np.array([1, 1, 1, 0, 0, 0, 0])).all()
        assert (B2[:, 1].T.toarray()[0] == np.array([0, 0, 1, 1, 1, 0, 0])).all()
        assert (B2[:, 2].T.toarray()[0] == np.array([0, 0, 0, 0, 1, 1, 1])).all()

        # Test the incidence matrix for the full cell complex
        B1 = CC.incidence_matrix(rank=1, signed=False)
        assert B1.shape == (5, 7)
        assert (B1[:, 0].T.toarray()[0] == np.array([1, 1, 0, 0, 0])).all()
        assert (B1[:, 1].T.toarray()[0] == np.array([1, 0, 1, 0, 0])).all()
        assert (B1[:, 2].T.toarray()[0] == np.array([0, 1, 1, 0, 0])).all()

        B1 = CC.incidence_matrix(rank=1, signed=False, index=False)
        assert isinstance(B1, scipy.sparse._csr.csr_matrix)

        B2_signed = CC.incidence_matrix(rank=2, signed=True)
        B1_signed = CC.incidence_matrix(rank=1, signed=True)

        # B1 * B2 == 0
        assert np.sum(B1_signed.dot(B2_signed).toarray()) == 0.0

        CC = CellComplex()
        B1 = CC.incidence_matrix(rank=1)
        assert B1.shape == (0, 0)

        CC = CellComplex()
        B2 = CC.incidence_matrix(rank=2)
        assert B2.shape == (0, 0)

        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        B0 = CC.incidence_matrix(rank=0, signed=False, index=True)
        assert B0[2].shape == (0, 5)
        B0 = CC.incidence_matrix(rank=0, signed=False, index=False)
        assert B0.shape == (0, 5)

    def test_node_to_all_cell_incidence_matrix(self):
        """Test node_to_all_cell_incidence_matrix."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)

        # Test the function without index
        result = CC.node_to_all_cell_incidence_matrix(weight=False, index=False)
        expected_result = scipy.sparse.csc_matrix(
            np.array(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            )
        )
        assert np.allclose(result.toarray(), expected_result.toarray())

        # Test the function with index
        node_index, cell_index, _ = CC.node_to_all_cell_incidence_matrix(
            weight=False, index=True
        )
        expected_node_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        expected_cell_index = {
            (1, 2): 0,
            (1, 4): 1,
            (2, 3): 2,
            (3, 4): 3,
            (3, 5): 4,
            (4, 5): 5,
            (1, 2, 3, 4): 6,
            (3, 4, 5): 7,
        }
        expected_result = scipy.sparse.csc_matrix(
            np.array(
                [
                    [0.0, 2.0, 1.0, 2.0, 0.0],
                    [2.0, 0.0, 2.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0, 3.0, 2.0],
                    [2.0, 1.0, 3.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0, 2.0, 0.0],
                ]
            )
        )
        assert isinstance(node_index, dict)
        assert isinstance(cell_index, dict)
        assert node_index == expected_node_index
        assert cell_index == expected_cell_index

    def test_node_to_all_cell_adjacnecy_matrix(self):
        """Test node_to_all_cell_adjacnecy_matrix."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)

        # Test the function without index
        result = CC.node_to_all_cell_adjacnecy_matrix(s=2)
        expected_result = np.array(
            [
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 3.0, 2.0],
                [2.0, 0.0, 3.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 2.0, 0.0],
            ]
        )
        assert np.allclose(result.toarray(), expected_result)

        # Test the function with index
        result = CC.node_to_all_cell_adjacnecy_matrix(index=True)
        expected_node_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        expected_result = scipy.sparse.csc_matrix(
            np.array(
                [
                    [0.0, 2.0, 1.0, 2.0, 0.0],
                    [2.0, 0.0, 2.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0, 3.0, 2.0],
                    [2.0, 1.0, 3.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0, 2.0, 0.0],
                ]
            )
        )
        assert result[0] == expected_node_index
        assert np.allclose(result[1].toarray(), expected_result.toarray())

    def test_all_cell_to_node_coadjacnecy_matrix(self):
        """Test all cell to node coadjacnecy matrix."""
        CC = CellComplex()  # Initialize your class object

        # Add some cells to the complex
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)

        # Test the function without index
        result = CC.all_cell_to_node_coadjacency_matrix()
        expected_result = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        assert np.allclose(result.toarray(), expected_result)

        # Test the function with index
        result = CC.all_cell_to_node_coadjacency_matrix(index=True)

        expected_cell_index = {
            (1, 2): 0,
            (1, 4): 1,
            (2, 3): 2,
            (3, 4): 3,
            (3, 5): 4,
            (4, 5): 5,
            (1, 2, 3, 4): 6,
            (3, 4, 5): 7,
        }
        expected_result = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0] == expected_cell_index
        assert np.allclose(result[1].toarray(), expected_result)

    def test_from_networkx_graph(self):
        """Test for the from_networkx_graph method."""
        CC = CellComplex()  # Initialize your class object

        # Create a NetworkX graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2)])

        # Test the function
        CC.from_networkx_graph(G)
        result = list(CC.edges)
        expected_result = [(0, 1), (0, 2), (1, 2)]
        assert result == expected_result

    def test_euler_characteristic(self):
        """Test euler_characteristic."""
        CC = CellComplex()
        CC.add_cells_from([[1, 2, 3], [1, 2, 3]], rank=2)
        assert CC.euler_characterisitics() == 2

    def test_clear(self):
        """Test the clear method of the cell complex."""
        CC = CellComplex()
        CC.add_cells_from([[1, 2, 3, 4], [5, 6, 7, 8]], rank=2)
        CC.clear()
        assert len(CC.cells) == 0

    def test_add_cell_with_color_feature(self):
        """Test adding a cell with a color feature."""
        CC = CellComplex()
        c1 = Cell((2, 3, 4), color="black")
        CC.add_cell(c1, weight=3)
        CC.add_cell([1, 2, 3, 4], rank=2, color="red")
        CC.add_cell([2, 3, 4, 5], rank=2, color="blue")
        CC.add_cell([5, 6, 7, 8], rank=2, color="green")

        assert CC.cells[(1, 2, 3, 4)]["color"] == "red"
        assert CC.cells[(2, 3, 4, 5)]["color"] == "blue"
        assert CC.cells[(5, 6, 7, 8)]["color"] == "green"

    def test_adjacency_matrix_empty_cell_complex(self):
        """Test adjacency matrix for an empty cell complex."""
        CC = CellComplex()
        np.testing.assert_array_equal(CC.adjacency_matrix(0), np.zeros((0, 0)))

    def test_dirac_operator_matrix(self):
        """Test dirac operator."""
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        m = CC.dirac_operator_matrix()
        size = len(CC.nodes) + len(CC.edges) + len(CC.cells)
        assert m.shape == (size, size)

        L = m.dot(m)

        check_L = bmat(
            [
                [CC.hodge_laplacian_matrix(0), None, None],
                [None, CC.hodge_laplacian_matrix(1), None],
                [None, None, CC.hodge_laplacian_matrix(2)],
            ]
        )

        assert np.linalg.norm((check_L - L).todense()) == 0

        index, m = CC.dirac_operator_matrix(index=True)

        assert 1 in index
        assert 2 in index
        assert 3 in index
        assert 4 in index
        assert len(index) == size

        index, m = CC.dirac_operator_matrix(index=True, signed=False)

        assert np.prod(m.todense() >= 0) == 1

        m = CC.dirac_operator_matrix(signed=False)

        assert np.prod(m.todense() >= 0) == 1

    def test_adjacency_matrix_cell_complex_with_one_cell(self):
        """Test adjacency matrix for a cell complex with one cell."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(CC.adjacency_matrix(0).todense(), A)

    def test_adjacency_matrix_cell_complex_with_multiple_cell(self):
        """Test adjacency matrix for a cell complex with multiple cells."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([4, 5, 6], rank=2)
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
        np.testing.assert_array_equal(CC.adjacency_matrix(rank=0).toarray(), A)

    def test_up_laplacian_matrix_empty_cell_complex(self):
        """Test up laplacian matrix for an empty cell complex."""
        CC = CellComplex()
        np.testing.assert_array_equal(CC.up_laplacian_matrix(rank=0), np.zeros((0, 0)))

    def test_up_laplacian_matrix_and_incidence_matrix(self):
        """Test up laplacian matrix and incidence matrix for a cell complex."""
        CC = CellComplex()
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([1, 3, 4], rank=2)

        B1 = CC.incidence_matrix(rank=1)
        L0_up = CC.up_laplacian_matrix(rank=0)
        expected = B1.dot(B1.T)
        np.testing.assert_array_equal(L0_up.toarray(), expected.toarray())

        B2 = CC.incidence_matrix(rank=2)
        L1_up = CC.up_laplacian_matrix(rank=1)
        expected = B2.dot(B2.T)
        np.testing.assert_array_equal(L1_up.toarray(), expected.toarray())

    def test_down_laplacian_matrix_and_incidence_matrix(self):
        """Test down laplacian matrix and incidence matrix for a cell complex."""
        CC = CellComplex()
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([1, 3, 4], rank=2)

        B1 = CC.incidence_matrix(rank=1)
        L1_down = CC.down_laplacian_matrix(rank=1)
        expected = (B1.T).dot(B1)
        np.testing.assert_array_equal(L1_down.toarray(), expected.toarray())

        B2 = CC.incidence_matrix(rank=2)
        L2_down = CC.down_laplacian_matrix(rank=2)
        expected = (B2.T).dot(B2)
        np.testing.assert_array_equal(L2_down.toarray(), expected.toarray())
        CC = CellComplex()
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([1, 3, 4], rank=2)
        with pytest.raises(ValueError):
            CC.down_laplacian_matrix(rank=2, weight="Hello")
        with pytest.raises(ValueError):
            CC.down_laplacian_matrix(rank=3)
        assert np.any(
            CC.down_laplacian_matrix(rank=2, signed=False).todense()
            == np.array([[3.0, 1.0], [1.0, 3.0]])
        )
        row, matrix = CC.down_laplacian_matrix(rank=2, index=True)
        assert row == {(1, 3): 0, (1, 4): 1, (2, 3): 2, (2, 4): 3, (3, 4): 4}
        assert np.any(matrix == np.array([[3.0, 1.0], [1.0, 3.0]]))
        with pytest.raises(ValueError):
            CC.up_laplacian_matrix(rank=2, weight="Hello")
        with pytest.raises(ValueError):
            CC.up_laplacian_matrix(rank=3)
        assert np.any(
            CC.up_laplacian_matrix(rank=1, signed=False).todense()
            == np.array(
                [
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0],
                ]
            )
        )
        row, matrix = CC.up_laplacian_matrix(rank=1, index=True)
        assert row == {(1, 3): 0, (1, 4): 1, (2, 3): 2, (2, 4): 3, (3, 4): 4}
        assert np.any(
            matrix
            == np.array(
                [
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0],
                ]
            )
        )
        CC = CellComplex(regular=False)
        CC.add_cell([1, 2, 3, 2], rank=2)
        CC.add_cell([3, 4, 5, 3, 4, 5], rank=2)
        assert CC.incidence_matrix(
            rank=0, index=False, signed=True
        ).todense().shape == (0, 5)
        with pytest.raises(ValueError):
            CC.incidence_matrix(rank=3)
        CC = CellComplex(regular=False)
        CC.add_cell([1, 2, 3, 2], rank=2)
        CC.add_cell([3, 4, 5, 3, 4, 5], rank=2)
        assert np.any(
            CC.incidence_matrix(rank=2).todense()
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, -2.0], [0.0, 2.0]])
        )

    def test_hodge_laplacian_and_up_down_laplacians(self):
        """Test hodge laplacian matrix and up and down laplacians for a cell complex."""
        CC = CellComplex()
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([1, 3, 4], rank=2)

        L1_up = CC.up_laplacian_matrix(rank=1)
        L1_down = CC.down_laplacian_matrix(rank=1)
        L1_hodge = CC.hodge_laplacian_matrix(rank=1)
        expected = L1_up + L1_down
        np.testing.assert_array_equal(L1_hodge.toarray(), expected.toarray())
        CC = CellComplex()
        CC.add_cell([2, 3], rank=1)
        CC.add_cell([1, 3], rank=1)
        assert np.any(
            CC.hodge_laplacian_matrix(rank=1, signed=True).todense()
            == np.array([[2.0, 1.0], [1.0, 2.0]])
        )

    def test_init_empty_abstract_cell(self):
        """Test empty cell complex."""
        cc = CellComplex([], rank=0)
        assert len(cc.nodes) == 0

    def test_init_nonempty_abstract_cell(self):
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

        with pytest.raises(RuntimeError):
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
        CC = CellComplex()
        assert repr(CC) == "CellComplex()"

    def test_len(self):
        """Test the length of the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        assert len(CC) == 3

    def test_iter(self):
        """Test the iterator of the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 4], rank=1)
        assert set(CC) == {
            (1,),
            (2,),
            (3,),
            (4,),
            (2, 1),
            (2, 3),
            (3, 1),
            (2, 4),
            (1, 2, 3),
        }

    def test_cell_equivalence_class(self):
        """Test the cell equivalence class method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 1], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 6], rank=2)
        CC.add_cell([3, 4, 1, 2], rank=2)
        CC.add_cell([4, 3, 2, 1], rank=2)
        CC.add_cell([1, 2, 7, 3], rank=2)
        c1 = Cell((1, 2, 3, 4, 5))
        CC.add_cell(c1, rank=2)

        equivalence_classes = CC._cell_equivalence_class()

        assert len(equivalence_classes) == 4

    def test_get_cell_attributes2(self):
        """Test the remove equivalent cells method."""
        import networkx as nx

        G = nx.path_graph(3)

        d = {
            ((1, 2, 3, 4), 0): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
        }
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CC.add_cell([3, 4, 8], rank=2)
        CC.set_cell_attributes(d, rank=2)
        cell_color = CC.get_cell_attributes("color", 2)
        assert cell_color == {((1, 2, 3, 4), 0): "red", (1, 2, 4): "blue"}

    def test_set_cell_attributes(self):
        """Test the set cell attributes method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 4], rank=2)
        CC.add_cell([3, 4, 8], rank=2)
        CC.add_cell([2, 6], rank=2)
        d = {(1, 2, 3, 4): "red", (1, 2, 4): "blue", (2, 6): "green"}
        CC.set_cell_attributes(d, rank=2, name="color")
        assert CC.cells[(2, 6)]["color"] == "green"
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CC.add_cell([3, 4, 8], rank=2)
        CC.add_cell([4, 5], rank=2)
        d = {
            (1, 2, 3, 4): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
            (4, 5): {"color": "green", "attr2": 4},
        }
        CC.set_cell_attributes(d, rank=2)
        assert CC.cells[(1, 2, 3, 4)][0]["color"] == "red"
        assert CC.cells[(4, 5)]["color"] == "green"
        d = {
            (1, 2, 3, 4): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
            (4, 5): {"hello": "world"},
        }
        CC.set_cell_attributes(d, rank=2)

    def test_remove_cells(self):
        """Test remove cells."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CC.add_cell([3, 4, 8], rank=2)
        CC.remove_cells([(1, 2, 3, 4), (1, 2, 4)])

        assert len(CC.cells) == 1

    def test_degree(self):
        """Test the degree of the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)

        assert CC.degree(1) == 2
        assert CC.degree(2) == 3
        assert CC.degree(3) == 2
        assert CC.degree(4) == 3
        assert CC.degree(5) == 4
        assert CC.degree(6) == 2
        assert CC.degree(7) == 2
        assert CC.degree(8) == 2

        with pytest.raises(NotImplementedError):
            CC.degree(node=1, rank=2)

    def test_cell_neighbors(self):
        """Test the cell neighbours of cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)

        with pytest.raises(NotImplementedError):
            CC.cell_neighbors(1)

    def test_size(self):
        """Test the size of the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)

        c = Cell([1, 2, 3, 4])
        assert CC.size((1, 2, 3, 4)) == 4
        assert CC.size((2, 3, 4, 5)) == 4
        assert CC.size((5, 6, 7, 8)) == 4

        assert CC.size((1, 2, 3, 4), node_set=[1, 2, 3]) == 3
        assert CC.size((2, 3, 4, 5), node_set=[3, 4, 5]) == 3
        assert CC.size((5, 6, 7, 8), node_set=[6, 7, 8]) == 3
        assert CC.size(c, node_set=[1, 2, 3]) == 3
        with pytest.raises(KeyError):
            CC.size((1, 2, 3, 4, 5, 6))

    def test_insert_cell(self):
        """Test inserting a cell into the cell complex."""
        CC = CellComplex()
        CC._insert_cell([1, 2, 3, 4], rank=2)
        CC._insert_cell([2, 3, 4, 5], rank=2)
        CC._insert_cell([5, 6, 7, 8], rank=2)

        assert len(CC.cells) == 3
        assert [cell.elements for cell in CC.cells] == [
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            (5, 6, 7, 8),
        ]

    def test_delete_cell(self):
        """Test deleting a cell from the cell complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)

        CC._delete_cell((1, 2, 3, 4))
        assert len(CC.cells) == 2
        assert [cell.elements for cell in CC.cells] == [(2, 3, 4, 5), (5, 6, 7, 8)]

        CC._delete_cell((2, 3, 4, 5))
        assert len(CC.cells) == 1
        assert [cell.elements for cell in CC.cells] == [(5, 6, 7, 8)]

    def test_shape(self):
        """Test the shape of the cell complex."""
        CC = CellComplex()
        assert CC.shape == (0, 0, 0)

        CC.add_node(1)
        assert CC.shape == (1, 0, 0)

        CC.add_cell([1, 2], rank=1)
        assert CC.shape == (2, 1, 0)

        CC.add_cell([1, 2, 3, 4], rank=2)
        assert CC.shape == (4, 4, 1)

    def test_skeleton(self):
        """Test the skeleton of the cell complex."""
        CC = CellComplex()
        CC.add_node(1)
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 2, 3, 4], rank=2)

        nodes = CC.skeleton(0)
        assert len(nodes) == 4

        edges = CC.skeleton(1)
        assert len(edges) == 4

        cells = CC.skeleton(2)
        assert len(cells) == 1

    def test_filtration(self):
        """Test the filtration of the cell complex."""
        CC = CellComplex([[1, 2, 3], [4, 5]])
        test_filtration = {3: 2, 1: 4, (2, 3): 3.4, (1, 2, 3): 7.6}
        CC.set_filtration(test_filtration, "test")
        assert CC.get_filtration("test") == test_filtration

        with pytest.raises(ValueError):
            test_filtration = {"dummy": "test"}
            CC.set_filtration(test_filtration, "test")

    def test_set_node_attributes(self):
        """Test setting node attributes."""
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        d = {1: {"color": "red", "attr2": 1}, 2: {"color": "blue", "attr2": 3}}
        CC.set_node_attributes(d)

        assert CC.nodes[1]["color"] == "red"
        assert CC.nodes[2]["color"] == "blue"
        assert CC.nodes[1]["attr2"] == 1

    def test_get_node_attributes(self):
        """Test getting node attributes."""
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        d = {1: {"color": "red", "attr2": 1}, 2: {"color": "blue", "attr2": 3}}
        CC.set_node_attributes(d)

        assert CC.get_node_attributes("color") == {1: "red", 2: "blue"}

    def test_set_edge_attributes(self):
        """Test setting edge attributes."""
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        d = {
            (1, 2): {"color": "red", "attr2": 1},
            (2, 3): {"color": "blue", "attr2": 3},
        }
        CC.set_edge_attributes(d)

        assert CC.edges[(1, 2)]["color"] == "red"
        assert CC.edges[(2, 3)]["color"] == "blue"
        assert CC.edges[(1, 2)]["attr2"] == 1

    def test_get_edge_attributes(self):
        """Test getting edge attributes."""
        G = nx.path_graph(3)
        CC = CellComplex(G)
        CC.add_cell([1, 2, 3, 4], rank=2)
        d = {
            (1, 2): {"color": "red", "attr2": 1},
            (2, 3): {"color": "blue", "attr2": 3},
        }
        CC.set_edge_attributes(d)

        assert CC.get_edge_attributes("color") == {(1, 2): "red", (2, 3): "blue"}

    @pytest.mark.skipif(
        hnx is None, reason="Optional dependency 'hypernetx' not installed."
    )
    def test_to_hypergraph(self):
        """Test the conversion of a cell complex to a hypergraph."""
        CC = CellComplex([[1, 2, 3], [4, 5]])
        hx = CC.to_hypergraph()
        assert set(hx.nodes) == set(CC.nodes)
        assert len(hx.edges) == len(CC.edges) + len(CC.cells)

    def test_to_combinatorial_complex(self):
        """Test the conversion of a cell complex to a combinatorial complex."""
        CC = CellComplex([[1, 2, 3], [4, 5]])
        ccc = CC.to_combinatorial_complex()
        assert len(set(ccc.nodes)) == len(set(CC.nodes))
        assert len(ccc.cells) - len(ccc.nodes) == len(CC.edges) + len(CC.cells)

    def test_to_colored_hypergraph(self):
        """Test the conversion of a cell complex to a colored hypergraph."""
        CC = CellComplex([[1, 2, 3], [4, 5]])
        chg = CC.to_colored_hypergraph()
        assert len(set(chg.nodes)) == len(set(CC.nodes))
        assert len(chg.cells) == len(CC.edges) + len(CC.cells)

    def test_restrict_to_nodes(self):
        """Test restricting a cell complex to a subset of nodes."""
        CC = CellComplex([[1, 2, 3], [3, 4, 5], [1, 4]])
        CC.set_filtration({(1, 2, 3): 1, (3, 4, 5): 2, (1, 4): 0, 3: 1}, "test")
        restricted = CC.restrict_to_nodes({1, 2, 3, 4})
        assert (1, 2, 3) in restricted.cells
        assert (1, 4) in restricted.edges
        assert (3, 4) in restricted.edges
        assert (3, 4, 5) not in restricted.cells
        assert (4, 5) not in restricted.edges
        assert restricted.get_filtration("test") == {(1, 2, 3): 1, (1, 4): 0, 3: 1}

    def test_get_cell_attributes(self):
        """Unit test for the get_cell_attributes method."""
        CC = CellComplex()
        d = {
            ((1, 2, 3, 4), 0): {"color": "red", "attr2": 1},
            (1, 2, 4): {"color": "blue", "attr2": 3},
        }
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell(
            [1, 2, 4],
            rank=2,
        )
        CC.add_cell([3, 4, 8], rank=2)
        CC.set_cell_attributes(d, 2)

        attributes = CC.get_cell_attributes("color", 2)

        with pytest.raises(ValueError):
            CC.set_cell_attributes(d, 4)

        # should not throw error
        CC.set_cell_attributes({(5, 6, 7): {"color": "red"}}, 2)

        attributes = CC.get_cell_attributes("color", 2)

        with pytest.raises(ValueError):
            attributes = CC.get_cell_attributes("color", 4)

        assert attributes == {((1, 2, 3, 4), 0): "red", (1, 2, 4): "blue"}

    def test_remove_equivalent_cells(self):
        """Unit test for the remove_equivalent_cells method."""
        CC = CellComplex()

        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 1], rank=2)
        CC.add_cell([1, 2, 4], rank=2)
        CC.add_cell([3, 4, 8], rank=2)
        assert len(CC.cells) == 5
        CC.remove_equivalent_cells()
        assert len(CC.cells) == 3

    def test_is_insertable_cycle(self):
        """Unit test for the is_insertable_cycle method."""
        CC = CellComplex()

        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)

        assert CC.is_insertable_cycle([1, 2, 3, 4])
        assert not CC.is_insertable_cycle([1, 2, 3, 4, 1, 2])
        assert not CC.is_insertable_cycle([1, 2, 1])

        with pytest.warns():
            CC.is_insertable_cycle([6, 7, 8], warnings_dis=True)

    def test_incidence_matrix(self):
        """Unit test for the incidence_matrix method."""
        CC = CellComplex()

        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        B1 = CC.incidence_matrix(1)
        B2 = CC.incidence_matrix(2)
        assert B1.dot(B2).todense().tolist() == [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]

    def test_restrict_to_cells(self):
        """Test restricting a cell complex to a subset of cells and edges."""
        CC = CellComplex([[1, 2, 3, 4], [3, 4, 5], [1, 2, 3, 6]])
        CC.add_cell((1, 2, 3, 4), rank=2)
        CC.set_filtration(
            {(1, 2, 3, 4): 1, (1, 2): 2, (3, 5): 3, (4, 5): 4, (3, 4, 5): 5}, "test"
        )
        restricted = CC.restrict_to_cells(
            {(1, 2, 3, 4), (3, 5), CC.cells.raw((1, 2, 3, 6))}
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
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4, 5], rank=2)
        CC.add_cell([1, 2, 7, 8], rank=2)
        assert CC.is_connected() is True

        # disconnected cell complex
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4, 5], rank=2)
        CC.add_cell([7, 8, 9, 10], rank=2)
        assert CC.is_connected() is False

    def test_singletons(self):
        """Test singletons."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        CC.add_node(0)
        CC.add_node(10)
        assert CC.singletons() == [0, 10]

    def test_remove_singletons(self):
        """Test remove singletons."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)
        CC.add_cell([5, 6, 7, 8], rank=2)
        CC.add_node(0)
        CC.add_node(10)
        CC.remove_singletons()
        assert CC.singletons() == []

    def test_contains(self):
        """Test __contains__."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)

        assert 1 in CC
        assert 2 in CC
        assert 3 in CC
        assert 4 in CC
        assert 5 in CC
        assert 10 not in CC
        assert (1, 2) in CC
        assert (10, 12) not in CC
        assert (1, 2, 3, 4) in CC
        assert (1, 2, 3, 4, 5) not in CC

    def test_neighbors(self):
        """Test neighbors of nodes."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 3, 4, 5], rank=2)

        assert list(CC.neighbors(1)) == [2, 4]

    def test_insert_cell_2(self):
        """Test for the _insert_cell method."""
        CC = CellComplex()
        CC._insert_cell([1, 2, 3], color="red")
        assert len(CC._cells._cells) == 1
        CC._insert_cell((1, 2, 3), attr1=5)
        assert len(CC._cells._cells) == 1
        CC._insert_cell(Cell([1, 2, 3], name="Cell 1"))
        assert len(CC._cells._cells) == 1
        CC._insert_cell([1, 2, 4])
        assert len(CC._cells._cells) == 2

    def test_insert_cell_invalid_input(self):
        """Test _insert_cell method with invalid input."""
        CC = CellComplex()
        with pytest.raises(TypeError):
            CC._insert_cell("invalid_input")
        with pytest.raises(TypeError):
            CC._insert_cell(123)
        with pytest.raises(TypeError):
            CC._insert_cell({"elements": [1, 2, 3]})

    def test_delete_cell_2(self):
        """Test for the _delete_cell method."""
        CC = CellComplex()
        CC._insert_cell([1, 2, 3], color="red")
        CC._insert_cell([4, 5, 6], color="blue")
        CC._delete_cell((1, 2, 3))
        assert len(CC._cells._cells) == 1
        assert len(CC.cells) == 1
        CC._delete_cell((4, 5, 6))
        assert len(CC._cells._cells) == 0
        assert len(CC.cells) == 0

    def test_delete_cell_invalid_input(self):
        """Test for the _delete_cell method with invalid input."""
        CC = CellComplex()
        with pytest.raises(KeyError):
            CC._delete_cell((1, 2, 3))

    def test_number_of_nodes(self):
        """Test for the number_of_nodes method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_node(4)
        CC.add_node(5)
        assert CC.number_of_nodes() == 5
        assert CC.number_of_nodes([1, 2, 3]) == 3
        assert CC.number_of_nodes([1, 6]) == 1

    def test_number_of_edges(self):
        """Test for the number_of_edges method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_node(4)
        CC.add_node(5)
        assert CC.number_of_edges() == 3
        assert CC.number_of_edges([(1, 2), (2, 3)]) == 2
        assert CC.number_of_edges([(4, 5), (6, 7)]) == 0

    def test_number_of_cells(self):
        """Test for the number_of_cells method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_node(4)
        CC.add_node(5)
        assert CC.number_of_cells() == 1
        assert CC.number_of_cells([(4, 5), (6, 7)]) == 0

    def test_order(self):
        """Test for the order method."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_node(4)
        CC.add_node(5)
        assert CC.order() == 5

    def test_neighbors_error(self):
        """Test to check for ValueError in neighbors method."""
        CC = CellComplex()
        with pytest.raises(KeyError):
            CC.neighbors(10)

    def test_get_cell_data(self):
        """Test the get_cell_data method of CellComplex."""
        CC = CellComplex()

        CC.add_node("A", attribute_name="Value A")
        CC.add_node("B", attribute_name="Value B")

        CC.add_edge("A", "B", attribute_name="Value AB")

        CC.add_cell(["A", "B", "C"], rank=2, attribute_name="Value C")

        data = CC.get_cell_data("A", 0, "attribute_name")
        assert data == "Value A"

        data = CC.get_cell_data(("A", "B"), 1, "attribute_name")
        assert data == "Value AB"

        data = CC.get_cell_data("B", 0)
        assert data == {"attribute_name": "Value B"}

        with pytest.raises(KeyError):
            CC.get_cell_data("D", 1, "attribute_name")

        with pytest.raises(KeyError):
            CC.get_cell_data("A", 0, "invalid_attribute")

        with pytest.raises(ValueError):
            CC.get_cell_data(["C"], 2, "invalid_attribute")

        with pytest.raises(KeyError):
            CC.get_cell_data(["D", "F"], 2, "invalid_attribute")

        with pytest.raises(KeyError):
            CC.get_cell_data("C", 2, "invalid_attribute")

        with pytest.raises(KeyError):
            CC.get_cell_data("E", 0)

        with pytest.raises(ValueError):
            CC.get_cell_data("A", 3, "attribute_name")

    def test_set_cell_data(self):
        """Test the set_cell_data method of CellComplex."""
        CC = CellComplex()

        CC.add_node("A")
        CC.add_node("B")

        CC.add_edge("A", "B")

        CC.add_cell(["A", "B", "C"], rank=2)

        CC.set_cell_data("A", 0, "attribute_name", "Value A")
        assert CC.nodes["A"]["attribute_name"] == "Value A"

        CC.set_cell_data(("A", "B"), 1, "attribute_name", "Value AB")
        assert CC.edges[("A", "B")]["attribute_name"] == "Value AB"

        CC.set_cell_data(["A", "B", "C"], 2, "attribute_name", "Value C")
        assert CC.cells[("A", "B", "C")]["attribute_name"] == "Value C"

        with pytest.raises(KeyError):
            CC.set_cell_data("D", 0, "attribute_name", "Value D")

        with pytest.raises(KeyError):
            CC.set_cell_data("D", 1, "attribute_name", "Value D")

        with pytest.raises(KeyError):
            CC.set_cell_data(["D", "E"], 2, "attribute_name", "Value D")

        with pytest.raises(ValueError):
            CC.set_cell_data("A", 3, "attribute_name", "Value A")

    def test_get_cell_data_after_set(self):
        """Test the get_cell_data method after setting cell data."""
        CC = CellComplex()

        CC.add_node("A")

        CC.set_cell_data("A", 0, "attribute_name", "Value A")

        data = CC.get_cell_data("A", 0, "attribute_name")
        assert data == "Value A"

    def test_get_cell_data_without_attr(self):
        """Test the get_cell_data method without specifying an attribute name."""
        CC = CellComplex()

        CC.add_node("A")
        CC.add_node("B", attribute_name="Value B")

        data = CC.get_cell_data("A", 0)
        assert len(data) == 0

    def test_hodge_laplacian_matrix(self):
        """Test the hodge_laplacian_matrix method of CellComplex."""
        CC = CellComplex()

        # Add cells to the complex for testing
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        CC.add_cell([4, 5, 6], rank=2)

        # Test case 1: Rank is 0
        rank = 0
        signed = True
        weight = None
        index = False

        result = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        assert result.shape == (6, 6)

        assert result.shape == (6, 6)

        # Test case 2: Rank is 0 with index=True
        index = True

        result, index_list = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        assert len(result) == 6

        assert len(result) == 6

        # Test case 3: Rank is 1 and maxdim is 2
        rank = 1

        result = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        # Test case 4: Rank is 1 and maxdim is 2 with index=True
        index = True

        index_list, result = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        # Assert the index_list is of type list
        assert isinstance(index_list, dict)

        # Test case 5: Rank is 2 and maxdim is 2
        rank = 2

        result = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        # Test case 6: Rank is 2 and maxdim is 2 with index=True
        index = True

        index_list, result = CC.hodge_laplacian_matrix(rank, signed, weight, index)

        # Assert the index_list is of type list
        assert isinstance(index_list, dict)

        # Test case 7: Rank is 2 and maxdim is not 2
        rank = 3

        with pytest.raises(ValueError):
            CC.hodge_laplacian_matrix(rank, signed, weight, index)

    def test_to_hasse_graph(self):
        """Test to hasse graph function."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        G = CC.to_hasse_graph()
        assert len(G.nodes) == 9
        assert len(G.edges) == 12
        assert (1,) in G.nodes
        assert (2,) in G.nodes
        assert (3,) in G.nodes
        assert (4,) in G.nodes
        assert (1, 2, 3, 4) in G.nodes

    def test_init_type_exception(self):
        """Test if incorrect datatype raises a TypeError exception."""
        with pytest.raises(TypeError):
            CellComplex(cells=1)

    def test_maxdim_warning(self):
        """Test if Deprecation Warning is raised when the maxdim method is called."""
        with pytest.deprecated_call():
            CC = CellComplex()
            _ = CC.maxdim

    def test_dim_edgecases(self):
        """Test if dim is testing for edge cases."""
        CC = CellComplex()
        CC.add_node(1)
        assert CC.dim == 0

    def test_skeleton_method_exception(self):
        """Test if the skeleton method raises an exception."""
        with pytest.raises(ValueError):
            CC = CellComplex()
            CC.skeleton(rank=4)

        with pytest.raises(ValueError):
            CC = CellComplex()
            CC.skeleton(rank=3)

    def test_getitem_dunder_method(self):
        """Test if the dunder __getitem__ method returns the appropriate neighbors of the given node."""
        CC = CellComplex()
        CC.add_node(1, color="red")
        CC.add_edge(1, 2, weight=10)
        CC.add_cell((2, 3, 4), rank=2, weight=5)

        assert CC[1]["color"] == "red"
        assert CC[(1,)]["color"] == "red"
        assert CC[(1, 2)]["weight"] == 10
        assert CC[(2, 3, 4)]["weight"] == 5

        with pytest.raises(KeyError):
            _ = CC[5]

        CC[(1, 2)]["new"] = 1
        assert CC[(1, 2)]["new"] == 1

    def test_remove_nodes(self):
        """Test remove nodes method of the class Cell Complex."""
        cc = CellComplex()
        cell = [1, 2, 3]
        cc.add_cell(cell, rank=2)
        cc.remove_nodes([1, 2, 3])
        assert len(cc.cells) == 0
        assert len(cc.edges) == 0
        assert len(cc.nodes) == 0

    def test_delete_cell_with_key(self):
        """Test delete cell method with key."""
        CC = CellComplex()
        CC._insert_cell((1, 2, 3, 4))
        CC._insert_cell((1, 2, 3, 4))
        CC._insert_cell((1, 2, 3, 4))
        CC._insert_cell((2, 3, 4, 5))

        assert len(CC.cells) == 4
        assert len(CC._cells._cells.keys()) == 2
        assert sorted(CC._cells._cells[(1, 2, 3, 4)].keys()) == [0, 1, 2]

        CC._delete_cell((1, 2, 3, 4), key=2)
        assert len(CC.cells) == 3
        assert len(CC._cells._cells.keys()) == 2
        assert sorted(CC._cells._cells[(1, 2, 3, 4)].keys()) == [0, 1]

        with pytest.raises(KeyError):
            CC._delete_cell((1, 2, 3, 4), key=10)
            CC._delete_cell((1, 2, 3, 4), key=100)

    def test_add_cell_regularity_conditions(self):
        """Test regularity conditions for add cell method."""
        CC = CellComplex()
        with pytest.raises(RuntimeError) as exp_exception:
            CC.add_cell((1, 1, 2, 3), rank=0)

        assert (
            str(exp_exception.value)
            == "Use `add_node` to insert nodes or zero ranked cells."
        )

        with pytest.raises(ValueError) as exp_exception:
            CC.add_cell((1, 1, 2, 3), rank=1)

        assert (
            str(exp_exception.value)
            == "rank 1 cells (edges) must have exactly two nodes"
        )

        with pytest.raises(ValueError) as exp_exception:
            CC.add_cell((1, 1), rank=1)

        assert (
            str(exp_exception.value)
            == " invalid insertion : self-loops are not allowed."
        )

        with pytest.raises(ValueError) as exp_exception:
            CC.add_cell((1, 1, 2, 3), rank=2)

        assert (
            str(exp_exception.value)
            == "Invalid cycle condition for cell [1, 1, 2, 3]. This input cell is not inserted, check if cell is irregular or if boundary edges of the cell are in the complex."
        )

        with pytest.raises(ValueError) as exp_exception:
            CC.add_cell(1, rank=2)

        assert str(exp_exception.value) == "invalid input, input 1 must be iterable"

        with pytest.raises(ValueError) as exp_exception:
            CC.add_cell({1, 5, 2, 3}, rank=4)

        assert (
            str(exp_exception.value)
            == "Add cell only supports adding cells of dimensions 0,1 or 2-- got 4"
        )

    def test_get_linegraph(self):
        """Test for the get_linegraph function of Cell Complex."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        CC.add_cell([2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        with pytest.raises(ValueError):
            CC.get_linegraph((1, 2, 3, 4, 5, 6))
        assert len(list(CC.get_linegraph(1).adjacency())) == 10
        with pytest.raises(TypeError):
            CC.get_linegraph(cells=2)

    def test_coadjacencies_edge_cases(self):
        """Test coadjacency and adjacency function edge cases."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        ind, incidence = CC.adjacency_matrix(rank=1, index=True)
        assert ind == {(1, 2): 0, (1, 4): 1, (2, 3): 2, (3, 4): 3}
        assert np.any(
            incidence.todense()
            == np.array(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                ]
            )
        )
        ind, incidence = CC.coadjacency_matrix(1, index=True)
        assert ind == {(1, 2): 0, (1, 4): 1, (2, 3): 2, (3, 4): 3}
        assert np.any(
            incidence.todense()
            == np.array(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                ]
            )
        )

    def test_hodge_laplacian_edgecases(self):
        """Test hodge laplacian edge cases."""
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        ind, laplacian_matrix = CC.hodge_laplacian_matrix(0, index=True, signed=False)
        assert ind == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        assert np.any(
            laplacian_matrix.todense()
            == np.array(
                [
                    [2.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 2.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 3.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 3.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 2.0],
                ]
            )
        )
        ind = CC.hodge_laplacian_matrix(0, index=False, signed=False)
        assert np.any(
            ind.todense()
            == np.array(
                [
                    [2.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 2.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 3.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 3.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 2.0],
                ]
            )
        )
        CC = CellComplex()
        CC.add_cell([6, 8], rank=1)
        CC.add_cell([1, 2, 3], rank=2)
        assert CC.hodge_laplacian_matrix(2).todense() == np.array([[3]])
        CC = CellComplex()
        CC.add_node(1)
        CC.add_edge(1, 2)
        with pytest.raises(ValueError):
            CC.hodge_laplacian_matrix(rank=2)
        CC = CellComplex()
        CC.add_cell([1, 2, 3], rank=2)
        assert np.any(
            CC.hodge_laplacian_matrix(rank=2, signed=False).todense() == np.array([[3]])
        )
        CC = CellComplex()
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([3, 4, 5], rank=2)
        np.any(
            CC.hodge_laplacian_matrix(1, signed=False).todense()
            == np.array(
                [
                    [3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 3.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 3.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 3.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
                ]
            )
        )
