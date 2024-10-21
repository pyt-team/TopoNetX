"""Unit tests for the colored hypergraph class."""

import networkx as nx
import pytest
from scipy.sparse import csr_array

from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.hyperedge import HyperEdge


class TestColoredHyperGraph:
    """Test ColoredHyperGraph class."""

    def test_init_empty_chg(self):
        """Test creation of an empty CHG."""
        CHG = ColoredHyperGraph()
        assert len(CHG) == 0

    def test_init_chg(self):
        """Test creation of CHG."""
        CHG = ColoredHyperGraph(cells=[[1, 2, 3], [2, 3, 4]], ranks=[1, 2])
        assert len(CHG.cells) == 2
        assert (1, 2, 3) in CHG.cells
        assert (2, 3, 4) in CHG.cells

    def test_init_from_lists(self):
        """Test creation of a CHG from a list of cells."""
        CHG = ColoredHyperGraph([[1, 2, 3]])
        assert len(CHG.cells) == 1
        assert (1, 2, 3) in CHG.cells

        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert len(CHG.cells) == 2
        assert (1, 2, 3) in CHG.cells
        assert (2, 3, 4) in CHG.cells

        with pytest.raises(ValueError) as excinfo:
            ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=[3])
            assert "cells and ranks must have equal number of elements" in str(
                excinfo.value
            )

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

    def test_chg_shortstr(self):
        """Test CHG short string representation."""
        CHG = ColoredHyperGraph()
        assert CHG.__shortstr__ == "CHG"

    def test_chg_str(self):
        """Test CHG string representation."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert (
            str(CHG)
            == f"Colored Hypergraph with {len(CHG.nodes)} nodes and hyperedges with colors {CHG.ranks[1:]} and sizes {CHG.shape[1:]} "
        )

    def test_chg_repr(self):
        """Test CHG repr representation."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert repr(CHG) == "ColoredHyperGraph()"

    def test_chg_iter(self):
        """Test CHG iter."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert set(CHG) == {
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({1, 2, 3}),
            frozenset({2, 3, 4}),
        }

    def test_chg_contains(self):
        """Test chg contains property."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)

        assert 1 in CHG
        assert 2 in CHG
        assert 3 in CHG
        assert 4 in CHG
        assert 5 not in CHG

        assert (1, 2) not in CHG
        assert (1, 2, 3) in CHG

    def test_chg_getitem(self):
        """Test chg get node properties."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.add_cell([5, 6], capacity=10)
        CHG.add_cell([5, 6], key=1, capacity=15)
        CHG.add_cell([5, 6, 7], capacity=5)

        assert CHG[1] == {"weight": 1}
        assert CHG[(5, 6)]["capacity"] == 10
        assert CHG[HyperEdge([5, 6])]["capacity"] == 10
        assert CHG[(5, 6, 7)]["capacity"] == 5
        assert CHG[HyperEdge([5, 6, 7])]["capacity"] == 5

        assert CHG[((5, 6), 0)]["capacity"] == 10
        assert CHG[(HyperEdge([5, 6]), 1)]["capacity"] == 15

        # non-existing hyperedges
        with pytest.raises(KeyError):
            _ = CHG[(0, 1, 2)]
        with pytest.raises(KeyError):
            _ = CHG[((5, 6), 2)]

        # invalid inputs should raise `KeyError`s as well
        with pytest.raises(KeyError):
            _ = CHG[()]
        with pytest.raises(KeyError):
            _ = CHG[((), 0)]

    def test_add_cell(self):
        """Test adding a cell to a CHG."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2, 3], rank=2)
        assert (1, 2, 3) in CHG.cells

        CHG.add_cell([3, 4, 5])
        assert (3, 4, 5) in CHG.cells

        with pytest.raises(ValueError):
            CHG.add_cell([1, 2], rank=-1)

        with pytest.raises(ValueError):
            CHG.add_cell(cell="[1,2]", rank=1)

        with pytest.raises(ValueError):
            CHG.add_cell(cell=1, rank=1)

        with pytest.raises(ValueError):
            CHG.add_cell(cell=[1, [2, 3]], rank=1)

        with pytest.raises(ValueError):
            CHG.add_cell(cell=[1, 2], rank=0)

        with pytest.raises(ValueError):
            CHG.add_cell(cell=None, rank=1)

    def test_add_cells_from(self):
        """Test adding multiple cells to a CHG."""
        CHG = ColoredHyperGraph()
        CHG.add_cells_from([[1, 2, 3], [3, 4, 5]])
        assert (1, 2, 3) in CHG.cells
        assert (3, 4, 5) in CHG.cells

        CHG = ColoredHyperGraph()
        HE = HyperEdge(elements=[6, 7, 8], rank=2)
        CHG.add_cells_from([HE])
        assert (6, 7, 8) in CHG.cells

        CHG = ColoredHyperGraph()
        CHG.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=2)

        assert (2, 3, 4) in CHG.cells
        assert (3, 4, 5) in CHG.cells

        CHG = ColoredHyperGraph()
        CHG.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=[1, 2])
        assert (2, 3, 4) in CHG.cells
        assert (3, 4, 5) in CHG.cells

        with pytest.raises(ValueError):
            CHG.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=[1, 2, 3])

    def test_remove_cell(self):
        """Test removing a cell from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.remove_cell([1, 2, 3])

        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) in CHG.cells

        with pytest.raises(KeyError):
            CHG.remove_cell([1, 2, 3])

        CHG.remove_cell(frozenset({1}))
        with pytest.raises(KeyError):
            CHG._complex_set.hyperedge_dict[0][frozenset({1})]

        CHG = ColoredHyperGraph([(1, 2), (2, 3), (3, 4)])
        CHG.remove_cell(HyperEdge(elements=[1, 2], rank=1))

        assert (1, 2) not in CHG.cells

    def test_remove_cells(self):
        """Test removing multiple cells from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.remove_cells([[1, 2, 3], [2, 3, 4]])

        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) not in CHG.cells

    def test_add_nodes(self):
        """Test adding nodes to a CHG."""
        CHG = ColoredHyperGraph()
        CHG.add_node([1])
        assert 1 in CHG.nodes

        CHG.add_node(1, color="red")
        assert 1 in CHG.nodes
        assert CHG[1] == {"color": "red", "weight": 1}

    def test_remove_node(self):
        """Test removing a node from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        HE = HyperEdge(elements=[1, 2, 3], rank=2)
        with pytest.raises(KeyError):
            CHG.remove_node(HE)

        CHG.remove_node(1)
        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) in CHG.cells

    def test_remove_nodes(self):
        """Test removing multiple nodes from a CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.remove_nodes([1, 2, 3])
        assert (1, 2, 3) not in CHG.cells
        assert (2, 3, 4) not in CHG.cells

        with pytest.raises(TypeError):
            CHG.remove_nodes([[1, 2]])

    def test_chg_degree(self):
        """Test CHG degree property."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        with pytest.raises(RuntimeError):
            CHG.degree(2, 1)

    def test_chg_get_cell_attr(self):
        """Test CHG get_attr method."""
        G = nx.path_graph(3)
        CHG = ColoredHyperGraph(G)
        d = {
            ((1, 2), 0): {"color": "red", "attr2": 1},
            ((0, 1), 0): {"color": "blue", "attr2": 3},
        }
        CHG.set_cell_attributes(d)
        cell_color = CHG.get_cell_attributes("color")
        assert cell_color[(frozenset({0, 1}), 0)] == "blue"
        assert cell_color[(frozenset({1, 2}), 0)] == "red"
        attr2 = CHG.get_cell_attributes("attr2")
        assert attr2[(frozenset({0, 1}), 0)] == 3
        assert attr2[(frozenset({1, 2}), 0)] == 1

    def test_chg_shape(self):
        """Test CHG shape property."""
        y1 = HyperEdge(elements=[1, 2], rank=1)
        y2 = HyperEdge(elements=[2, 4], rank=1)
        y3 = HyperEdge(elements=[3, 5], rank=1)
        y4 = HyperEdge(elements=[4, 5], rank=1)
        y5 = HyperEdge(elements=[5, 7], rank=1)

        CHG = ColoredHyperGraph(cells=[y1, y2, y3, y4, y5])
        assert CHG.shape == (6, 5)

    def test_chg_ranks(self):
        """Test CHG ranks property."""
        y1 = HyperEdge(elements=[1, 2], rank=1)
        y2 = HyperEdge(elements=[2, 4], rank=2)
        y3 = HyperEdge(elements=[3, 5], rank=1)
        y4 = HyperEdge(elements=[4, 5], rank=1)
        y5 = HyperEdge(elements=[5, 7], rank=5)

        CHG = ColoredHyperGraph(cells=[y1, y2, y3, y4, y5])
        assert CHG.ranks == [0, 1, 2, 5]

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
        B, row, col = CHG.incidence_matrix(1, 2, index=True)
        assert B[(frozenset({1, 2}), 0)] == 0
        assert B[(frozenset({1, 2, 3}), 0)] == 2

        with pytest.raises(ValueError):
            CHG.incidence_matrix(2, 2)

    def test_incidence_matrix_to_rank_none(self):
        """Test generating an incidence matrix without setting the to_rank parameter."""
        CHG = ColoredHyperGraph()
        CHG.add_cell([1, 2], rank=1)
        CHG.add_cell([1, 3], rank=1)
        CHG.add_cell([1, 2, 4, 3], rank=2)
        CHG.add_cell([2, 5], rank=1)
        CHG.add_cell([2, 6, 4], rank=2)
        B = CHG.incidence_matrix(0, 1)
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

        # test with index = True
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        _, adj_mat = CHG.adjacency_matrix(0, 2, index=True)
        assert (
            adj_mat.todense()
            == [[0, 1, 1, 0], [1, 0, 2, 1], [1, 2, 0, 1], [0, 1, 1, 0]]
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

        _, CA10 = CHG.coadjacency_matrix(1, 0, index=True)
        assert (CA10.todense() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]).all()

    def test_clone(self):
        """Test the clone method of ColoredHyperGraph."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG2 = CHG.clone()
        assert len(CHG2.cells) == 2
        assert (1, 2, 3) in CHG2.cells
        assert (2, 3, 4) in CHG2.cells
        CHG2.remove_cell([1, 2, 3])
        assert len(CHG2.cells) == 1

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
        B = CHG.incidence_matrix(0, 2)
        assert B.shape == (6, 2)
        assert (B.todense() == [[1, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1]]).all()

    def test_new_hyperedge_key(self):
        """Test adding hyperedge new key."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert CHG.new_hyperedge_key(frozenset({1, 2, 3}), 2) == 1

        assert CHG.new_hyperedge_key(frozenset({1, 2, 3}), 1) == 0

    def test_get_incidence_structure_dict(self):
        """Test get_incidence_structure_dict method of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        res_dict = CHG.get_incidence_structure_dict(0, 2)
        assert res_dict == {0: [0, 1, 2], 1: [1, 2, 3]}

    def test_get_adjacency_structure_dict(self):
        """Test get_adjacency_structure_dict method of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        res_dict = CHG.get_adjacency_structure_dict(0, 2)
        assert res_dict == {1: [0, 2, 3], 2: [0, 1, 3], 0: [1, 2], 3: [1, 2]}

    def test_degree_matrix(self):
        """Test degree matrix method of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        with pytest.raises(ValueError):
            CHG.degree_matrix(0)
        assert (CHG.degree_matrix(1) == [0, 0, 0, 0]).all()
        assert (CHG.degree_matrix(2) == [1, 2, 2, 1]).all()

        row, D = CHG.degree_matrix(1, index=True)
        res, _, _ = CHG.incidence_matrix(0, 1, index=True)
        assert row == res

    def test_laplacian_matrix(self):
        """Test laplacian matrix method of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        with pytest.raises(ValueError):
            CHG.laplacian_matrix(0)

        assert (
            CHG.laplacian_matrix(1)
            == [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ).all()
        assert (
            CHG.laplacian_matrix(2)
            == [[1, -1, -1, 0], [-1, 2, -2, -1], [-1, -2, 2, -1], [0, -1, -1, 1]]
        ).all()

        row, L = CHG.laplacian_matrix(1, sparse=True, index=True)
        res, _ = CHG.adjacency_matrix(0, 1, index=True)
        assert row == res
        assert isinstance(L, csr_array)

        # check cases where the rank is over the maximum rank of the CHG
        CHG = ColoredHyperGraph()
        with pytest.raises(ValueError):
            _ = CHG.laplacian_matrix(1)

    def test_singletons(self):
        """Test singletons of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.add_cell([9])
        assert CHG.singletons() == [(frozenset({9}), 0)]

    def test_restrict_to_cells(self):
        """Test restrict_to_cells of CHG."""
        CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
        CHG.add_cell([9])
        assert len(CHG.remove_singletons().cells) == 2

    def test_from_trimesh(self):
        """Test from_trimesh method of CHG."""
        with pytest.raises(NotImplementedError):
            CHG = ColoredHyperGraph([[1, 2, 3], [2, 3, 4]], ranks=2)
            CHG.from_trimesh(1)
