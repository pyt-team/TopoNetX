"""Unit tests for the colored hypergraph class."""

import networkx as nx
import pytest

from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.combinatorial_complex import (
    CombinatorialComplex2 as CombinatorialComplex,
)
from toponetx.classes.hyperedge import HyperEdge
from toponetx.exception import TopoNetXError


class TestCombinatorialComplex:
    """Test CombinatorialComplex class."""

    def test_init_empty_chg(self):
        """Test creation of an empty CHG."""
        CHG = ColoredHyperGraph()
        assert len(CHG) == 0

    def test_init_from_lists(self):
        """Test creation of a CHG from a list of cells."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert len(CHG.cells) == 6
        assert (1, 2, 3) in CHG.cells
        assert (2, 3, 4) in CHG.cells

    def test_init_from_abstract_cells(self):
        """Test creation of a CHG from abstract cells."""
        y1 = HyperEdge(elements=[1, 2], rank=1)
        y2 = HyperEdge(elements=[2, 4], rank=1)
        y3 = HyperEdge(elements=[3, 5], rank=1)
        y4 = HyperEdge(elements=[4, 5], rank=1)
        y5 = HyperEdge(elements=[5, 7], rank=1)

        CHG = ColoredHyperGraph(cells=[y1, y2, y3, y4, y5])

        assert y1 in CHG.cells
        assert y2 in CHG.cells
        assert y3 in CHG.cells
        assert y4 in CHG.cells
        assert y5 in CHG.cells

    def test_init_from_networkx_graph(self):
        """Test creation of a CHG from a networkx graph."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 3)
        G.add_edge(0, 4)
        G.add_edge(1, 4)

        CHG = ColoredHyperGraph(cells=G)

        assert (0, 1) in CHG.cells
        assert (0, 3) in CHG.cells
        assert (0, 4) in CHG.cells
        assert (1, 4) in CHG.cells
        assert (0, 5) not in CHG.cells

        G = nx.Graph()
        G.add_edge(5, 7)
        G.add_edge("a", "b")

        CHG.from_networkx_graph(G)

        assert "a" in CHG.cells

    def test_add_cell(self):
        """Test adding a cell to a CHG."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2, 3], rank=2)
        assert (1, 2, 3) in CHG.cells

    def test_add_cells_from(self):
        """Test adding multiple cells to a CHG."""
        CHG = ColoredHyperGraph()
        CHG.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=2)

        assert (2, 3, 4) in CHG.cells
        assert (3, 4, 5) in CHG.cells

    def test_remove_cell(self):
        """Test removing a cell from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.remove_cell([1, 2, 3])

        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) in CHG.cells

    def test_remove_cells(self):
        """Test removing multiple cells from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.remove_cells([[1, 2, 3], [2, 3, 4]])

        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) not in CHG.cells

    def test_incidence_matrix(self):
        """Test generating an incidence matrix."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)

        B = CHG.incidence_matrix(rank=0, to_rank=2)
        assert B.shape == (4, 2)
        assert (B.T[0].todense() == [1, 1, 1, 0]).all()
        assert (B.T[1].todense() == [0, 1, 1, 1]).all()
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 2, 3, 4], rank=2)
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3, 2], rank=1)
        CHG.add_cell([1, 2, 3, 4], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        B, row, col = CHG.incidence_matrix(1, index=True)
        assert B[(frozenset({1, 2}))] == 0
        assert B[(frozenset({1, 2, 3}))] == 1
        assert B[(frozenset({2, 5}))] == 2

    def test_incidence_matrix_to_rank_none(self):
        """Test generating an incidence matrix without setting the to_rank parameter."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        B = CHG.incidence_matrix(0)
        assert B.shape == (6, 3)
        assert (
            B.todense()
            == [[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]
        ).all()

    def test_adjacency_matrix(self):
        """Test generating an adjacency matrix."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        A02 = CHG.adjacency_matrix(0, 2)
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
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        CA10 = CHG.coadjacency_matrix(1, 0)
        assert CA10.shape == (3, 3)
        assert (CA10.todense() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]).all()

    def test_clone(self):
        """Test the clone method of ColoredHyperGraph."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG2 = CHG.clone()
        assert len(CHG2.cells) == 6
        assert (1, 2, 3) in CHG2.cells
        assert (2, 3, 4) in CHG2.cells
        CHG2.remove_cell([1, 2, 3])
        assert len(CHG.cells) == 6

    def test_colored_hypergraph_init(self):
        """Test the init method of ColoredHyperGraph class."""
        with pytest.raises(TypeError):
            ColoredHyperGraph(cells=1)

    def test_incidence_matrix_to_rank_down(self):
        """Test generating an incidence matrix by setting the down_rank parameter."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        B = CHG.incidence_matrix(2, 0, incidence_type="down")
        assert B.shape == (6, 2)
        assert (B.todense() == [[1, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1]]).all()

    def test_incidence_matrix_to_rank_down_without_rank(self):
        """Test generating an incidence matrix by setting the down_rank parameter without mentioning rank."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        B = CHG.incidence_matrix(2, incidence_type="down")
        assert B.shape == (3, 2)
        assert (B.todense() == [[1, 0], [1, 0], [0, 0]]).all()

    def test_incidence_matrix_to_rank_with_wrong_incidence_type(self):
        """Test generating an incidence matrix by mentioning wrong rank."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        with pytest.raises(TopoNetXError) as exp:
            CHG.incidence_matrix(2, incidence_type="wrong")
        assert str(exp.value) == "incidence_type must be 'up' or 'down' "

    def test_adjacency_incidence_structure_dict(self):
        """Test for the incidence and adjacency structure dictionaries."""
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3], rank=1)
        CC.add_cell([1, 2, 4, 3], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        dict1 = CC.get_all_incidence_structure_dict()
        assert list(dict1["B_0_1"].keys()) == [0, 1, 2]
        assert list(dict1["B_0_1"].values()) == [
            [0, 1],
            [0, 2],
            [1, 4],
        ]
        assert list(dict1["B_0_2"].keys()) == [0, 1]
        assert list(dict1["B_0_2"].values()) == [[0, 1, 2, 3], [1, 3, 5]]

    def test_cell_node_adjacency_matrix(self):
        """Test for the cells adjacency matrix method."""
        CC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        B = CC.incidence_matrix(rank=0, to_rank=2)
        assert B.shape == (4, 2)
        assert (B.T[0].todense() == [1, 1, 1, 0]).all()
        assert (B.T[1].todense() == [0, 1, 1, 1]).all()
        CC = CombinatorialComplex()
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([1, 2], rank=1)
        CC.add_cell([1, 3, 2], rank=1)
        CC.add_cell([1, 2, 3, 4], rank=2)
        CC.add_cell([2, 5], rank=1)
        CC.add_cell([2, 6, 4], rank=2)
        B, row, col = CC.incidence_matrix(1, index=True)
        assert B[(frozenset({1, 2}))] == 0
        assert B[(frozenset({1, 2, 3}))] == 1
        assert B[(frozenset({2, 5}))] == 2
        assert (
            CC.cell_adjacency_matrix().todense()
            == [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ).all()
        (
            CC.node_adjacency_matrix().todense() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        ).all()
        assert CC.diameter() == 1
        with pytest.raises(TopoNetXError) as exp:
            CC.diameter(s=2)
        assert str(exp.value) == "CC is not s-connected. s=2"


#: TODO add tests for CHG not covered by CC tests
