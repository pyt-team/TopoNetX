"""Unit tests for the combinatorial complex class."""

import networkx as nx
import pytest

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.exception import TopoNetXError


class TestCombinatorialComplex:
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

    def test_incidence_matrix_to_rank_none(self):
        """Test generating an incidence matrix without setting the to_rank parameter."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        B = CC.incidence_matrix(0)
        assert B.shape == (6, 3)
        assert (
            B.todense()
            == [[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]
        ).all()

    def test_adjacency_matrix(self):
        """Test generating an adjacency matrix."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        A02 = CC.adjacency_matrix(0, 2)
        assert A02.shape == (6, 6)
        assert (
            A02.todense()
            == [
                [0, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1],
                [1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
            ]
        ).all()

    def test_coadjacency_matrix(self):
        """Test generating a coadjacency matrix."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        CA10 = CC.coadjacency_matrix(1, 0)
        assert CA10.shape == (3, 3)
        assert (CA10.todense() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]).all()

    def test_clone(self):
        """Test the clone method of CombinatorialComplex."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CC2 = CC.clone()
        assert len(CC2.cells) == 6
        assert (1, 2, 3) in CC2.cells
        assert (2, 3, 4) in CC2.cells
        CC2.remove_cell([1, 2, 3])
        assert len(CC.cells) == 6

    def test_combinatorial_complex_init(self):
        """Test the init method of CombinatorialComplex class."""
        with pytest.raises(TypeError):
            CombinatorialComplex(cells=1)

    def test_incidence_matrix_to_rank_down(self):
        """Test generating an incidence matrix by setting the down_rank parameter."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        B = CC.incidence_matrix(2, 0, incidence_type="down")
        assert B.shape == (6, 2)
        assert (B.todense() == [[1, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1]]).all()

    def test_incidence_matrix_to_rank_down_without_rank(self):
        """Test generating an incidence matrix by setting the down_rank parameter without mentioning rank."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        B = CC.incidence_matrix(2, incidence_type="down")
        assert B.shape == (3, 2)
        assert (B.todense() == [[1, 0], [1, 0], [0, 0]]).all()

    def test_incidence_matrix_to_rank_with_wrong_incidence_type(self):
        """Test generating an incidence matrix by mentioning wrong rank."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        with pytest.raises(TopoNetXError):
            CC.incidence_matrix(2, incidence_type="wrong")

    def test_incidence_matrix_with_equal_rank(self):
        """Test generating an incidence matrix by having equal rank."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        with pytest.raises(ValueError):
            CC.incidence_matrix(1, 1)

    def test_incidence_dict(self):
        """Test generating an incidence dictionary."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        assert CC.incidence_dict == {
            1: {
                frozenset({1, 2}): {"weight": 1},
                frozenset({1, 3}): {"weight": 1},
                frozenset({2, 5}): {"weight": 1},
            },
            0: {
                frozenset({1}): {"weight": 1},
                frozenset({2}): {"weight": 1},
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
                frozenset({5}): {"weight": 1},
                frozenset({6}): {"weight": 1},
            },
            2: {
                frozenset({1, 2, 3, 4}): {"weight": 1},
                frozenset({2, 4, 6}): {"weight": 1},
            },
        }

    def test_dim(self):
        """
        Test for the dimensionality of the CombinatorialComplex object.

        Gets the highest rank of the cells in the CombinatorialComplex object.
        """
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=3)
        assert CC.dim == 3

    def test_repr(self):
        """Test the represntation function of the CombinatorialComplex object by mentioning a name."""
        CC = CombinatorialComplex(name="sampleobject")
        assert repr(CC) == "CombinatorialComplex(name='sampleobject')"

    def test_contains(self):
        """Test whether the contains method works correctly."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        assert [(1)] in CC.nodes
        assert [2] in CC.nodes
        assert [3] in CC.nodes
        assert [4] in CC.nodes
        assert [5] in CC.nodes
        assert [6] in CC.nodes

    def test_set_item(self):
        """Test for set_item method of the CombinatorialComplex object."""
        CC = CombinatorialComplex()
        CC.add_cell([1], rank=3)
        CC.add_cell([1, 2], rank=3)
        # Updating an node attribute present in the CombinatorialComplex object.
        CC.__setitem__([1], {"weights": 1})
        # Setting a cell attribute present in the CombinatorialComplex object.
        CC.__setitem__([1, 2], {"weights": 1})
        assert CC._complex_set.hyperedge_dict[3][frozenset([1, 2])] == {"weights": 1}
        assert CC.nodes[1]["weights"] == 1

    def test_degree(self):
        """Test for the degree function."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        CC.add_cell([1, 2], rank=2)
        assert CC.degree(1) == 2
        with pytest.raises(TopoNetXError):
            assert CC.degree(1, -1) == TopoNetXError("Rank must be positive")
        assert CC.degree(2, 2) == 3
        with pytest.raises(KeyError):
            node = 7
            assert CC.degree(node, 2) == KeyError(
                f"Node {node} not in Combinatorial Complex."
            )

    def test_size(self):
        """Test for the size function."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        CC.add_cell([1, 2], rank=2)
        assert CC.size(1) == 1
        with pytest.raises(TopoNetXError):
            CC.size(frozenset([1, 2, 3])) == TopoNetXError(
                "Input cell is not in cells of the CC"
            )

    def test_num_nodes_and_cells(self):
        """Test for number of nodes and number of cells."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        CC.add_cell([1, 2], rank=2)
        y1 = HyperEdge([1, 2], rank=1)
        y2 = HyperEdge([1, 3], rank=1)
        assert CC.number_of_nodes() == 6
        assert CC.number_of_nodes([1, 2]) == 2
        assert CC.number_of_cells() == 12
        assert CC.number_of_cells([y1, y2]) == 2

    def test_order(self):
        """Test for the order function."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2, 3, 4, 5, 6, 7, 8], rank=1)
        assert CC.order() == 8
