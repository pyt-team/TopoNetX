"""Unit tests for the combinatorial complex class."""

import networkx as nx
import numpy as np
import pytest

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge


class TestCombinatorialComplex:
    """Test CombinatorialComplex class."""

    def test_init_empty_ccc(self):
        """Test creation of an empty CCC."""
        CCC = CombinatorialComplex()
        assert len(CCC) == 0

    def test_init_from_lists(self):
        """Test creation of a CCC from a list of cells."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        assert len(CCC.cells) == 6
        assert (1, 2, 3) in CCC.cells
        assert (2, 3, 4) in CCC.cells

        with pytest.raises(ValueError):
            _ = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=None)

        with pytest.raises(ValueError):
            _ = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=[1])

    def test_init_from_abstract_cells(self):
        """Test creation of a CCC from abstract cells."""
        y1 = HyperEdge(elements=[1, 2], rank=1)
        y2 = HyperEdge(elements=[2, 4], rank=1)
        y3 = HyperEdge(elements=[3, 5], rank=1)
        y4 = HyperEdge(elements=[4, 5], rank=1)
        y5 = HyperEdge(elements=[5, 7], rank=1)
        y6 = HyperEdge(elements=[9, 8], rank=None)

        CCC = CombinatorialComplex(cells=[y1, y2, y3, y4, y5])

        assert y1 in CCC.cells
        assert y2 in CCC.cells
        assert y3 in CCC.cells
        assert y4 in CCC.cells
        assert y5 in CCC.cells

        with pytest.raises(ValueError):
            CCC = CombinatorialComplex(cells=[y1, y2, y3, y4, y5, y6])

    def test_init_from_networkx_graph(self):
        """Test creation of a CCC from a networkx graph."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 3)
        G.add_edge(0, 4)
        G.add_edge(1, 4)

        CCC = CombinatorialComplex(cells=G)

        assert (0, 1) in CCC.cells
        assert (0, 3) in CCC.cells
        assert (0, 4) in CCC.cells
        assert (1, 4) in CCC.cells
        assert (0, 5) not in CCC.cells

        G = nx.Graph()
        G.add_edge(5, 7)
        G.add_edge("a", "b")

        CCC.from_networkx_graph(G)

        assert "a" in CCC.cells

    def test_node_membership(self):
        """Test creation of _node_membership dictionary."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2, 3], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)

        assert 1 in CCC._node_membership
        assert 2 in CCC._node_membership
        assert 3 in CCC._node_membership
        assert 4 in CCC._node_membership
        assert frozenset({1, 2, 3, 4}) in CCC._node_membership[1]
        assert frozenset({1, 2, 3}) in CCC._node_membership[1]
        assert frozenset({1, 2, 3, 4}) in CCC._node_membership[2]
        assert frozenset({1, 2, 3}) in CCC._node_membership[2]
        assert frozenset({1, 2, 3, 4}) in CCC._node_membership[3]
        assert frozenset({1, 2, 3}) in CCC._node_membership[3]
        assert frozenset({1, 2, 3, 4}) in CCC._node_membership[4]

    def test_add_cell(self):
        """Test adding a cell to a CCC."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2, 3], rank=2)
        assert (1, 2, 3) in CCC.cells
        with pytest.raises(ValueError) as ex:
            CCC.add_cell(HyperEdge([1]), 3)
        assert len(str(ex.value)) > 0

        CCC_1 = CombinatorialComplex(graph_based=True)
        with pytest.raises(ValueError):
            CCC_1.add_cell(cell=[1, 2, 3], rank=1)
        with pytest.raises(TypeError):
            CCC_1.add_cell(cell=1, rank=1)

        with pytest.raises(ValueError):
            CCC.add_cell([1, 2], rank=-1)

        with pytest.raises(ValueError):
            CCC.add_cell(cell="[1,2]", rank=1)

        with pytest.raises(ValueError):
            CCC.add_cell(cell=1, rank=1)

        with pytest.raises(ValueError):
            CCC.add_cell(cell=[1, [2, 3]], rank=1)

        with pytest.raises(ValueError):
            CCC.add_cell(cell=[1, 2], rank=0)

    def test_add_cells_from(self):
        """Test adding multiple cells to a CCC."""
        CCC = CombinatorialComplex()
        CCC.add_cells_from([[2, 3, 4], [3, 4, 5]], ranks=2)

        assert (2, 3, 4) in CCC.cells
        assert (3, 4, 5) in CCC.cells

    def test_remove_cell(self):
        """Test removing a cell from a CCC."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CCC.remove_cell([1, 2, 3])

        assert (1, 2, 3) not in CCC.cells
        assert (2, 3, 4) in CCC.cells

        with pytest.raises(KeyError):
            CCC.remove_cell([1, 2, 3])

        CCC.remove_cell(frozenset({1}))
        with pytest.raises(KeyError):
            CCC._complex_set.hyperedge_dict[0][frozenset({1})]

        CCC = CombinatorialComplex([(1, 2), (2, 3), (3, 4)], ranks=2)
        CCC.remove_cell(HyperEdge(elements=[1, 2], rank=1))

        assert (1, 2) not in CCC.cells

    def test_remove_cells(self):
        """Test removing multiple cells from a CCC."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CCC.remove_cells([[1, 2, 3], [2, 3, 4]])

        assert (1, 2, 3) not in CCC.cells
        assert (2, 3, 4) not in CCC.cells

    def test_incidence_matrix(self):
        """Test generating an incidence matrix."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)

        B = CCC.incidence_matrix(rank=0, to_rank=2)
        assert B.shape == (4, 2)
        assert (B.T[0].todense() == [1, 1, 1, 0]).all()
        assert (B.T[1].todense() == [0, 1, 1, 1]).all()
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        B, row, col = CCC.incidence_matrix(1, index=True)
        assert B[(frozenset({1, 2}))] == 0
        assert B[(frozenset({1, 2, 3}))] == 1
        assert B[(frozenset({2, 5}))] == 2

        with pytest.raises(ValueError):
            CCC.incidence_matrix(rank=2, to_rank=0)

    def test_incidence_matrix_to_rank_none(self):
        """Test generating an incidence matrix without setting the to_rank parameter."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        B = CCC.incidence_matrix(0)
        assert B.shape == (6, 5)
        assert (
            B.todense()
            == [
                [1, 1, 0, 1, 0],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ).all()

    def test_adjacency_matrix(self):
        """Test generating an adjacency matrix."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        A02 = CCC.adjacency_matrix(0, 2)
        assert A02.shape == (6, 6)

        with pytest.raises(ValueError):
            CCC.adjacency_matrix(2, 0)

    def test_coadjacency_matrix(self):
        """Test generating a coadjacency matrix."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        CA10 = CCC.coadjacency_matrix(1, 0)
        assert CA10.shape == (3, 3)
        assert (CA10.todense() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]).all()

        with pytest.raises(ValueError):
            CCC.coadjacency_matrix(0, 1)

    def test_dirac_operator_matrix(self):
        """Test dirac operator matrix."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([2, 3], rank=1)
        CCC.add_cell([1, 4], rank=1)
        CCC.add_cell([3, 4, 8], rank=2)
        m = CCC.dirac_operator_matrix()
        size = sum(CCC.shape)
        assert m.shape == (size, size)
        index, m = CCC.dirac_operator_matrix(index=True)
        assert frozenset({1, 2}) in index
        assert len(index) == size
        assert np.prod(m.todense() >= 0) == 1

    def test_clone(self):
        """Test the clone method of CombinatorialComplex."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CCC2 = CCC.clone()
        assert len(CCC2.cells) == 6
        assert (1, 2, 3) in CCC2.cells
        assert (2, 3, 4) in CCC2.cells
        CCC2.remove_cell([1, 2, 3])
        assert len(CCC.cells) == 6

    def test_combinatorial_complex_init(self):
        """Test the init method of CombinatorialComplex class."""
        with pytest.raises(TypeError):
            CombinatorialComplex(cells=1)

    def test_incidence_matrix_to_rank_down_without_rank(self):
        """Test generating an incidence matrix by setting the down_rank parameter without mentioning rank."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        B = CCC.incidence_matrix(2, incidence_type="down")
        assert B.shape == (9, 2)
        assert (
            B.todense()
            == [[1, 0], [1, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 0], [0, 0]]
        ).all()

    def test_incidence_matrix_to_rank_with_wrong_incidence_type(self):
        """Test generating an incidence matrix by mentioning wrong rank."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        with pytest.raises(ValueError) as exp:
            CCC.incidence_matrix(2, incidence_type="wrong")
        assert (
            str(exp.value) == "Invalid value for incidence_type. Must be 'up' or 'down'"
        )

    def test_incidence_matrix_with_equal_rank(self):
        """Test generating an incidence matrix by having equal rank."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        with pytest.raises(ValueError) as exp:
            CCC.incidence_matrix(1, 1)
        assert (
            str(exp.value)
            == "incidence matrix can be computed for k!=r, got equal r and k."
        )

    def test_incidence_dict(self):
        """Test generating an incidence dictionary."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        assert CCC.incidence_dict == {
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
        """Test for the dimensionality of the CombinatorialComplex object."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=3)
        assert CCC.dim == 3

    def test_repr(self):
        """Test the representation function of the CombinatorialComplex object by mentioning a name."""
        CCC = CombinatorialComplex()
        assert repr(CCC) == "CombinatorialComplex()"

    def test_contains(self):
        """Test whether the contains method works correctly."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)

        assert 1 in CCC
        assert 3 in CCC
        assert 7 not in CCC

        assert (1, 2) in CCC
        assert (1, 4) not in CCC

        assert [(1)] in CCC.nodes
        assert [2] in CCC.nodes
        assert [3] in CCC.nodes
        assert [4] in CCC.nodes
        assert [5] in CCC.nodes
        assert [6] in CCC.nodes

    def test_set_item(self):
        """Test for set_item method of the CombinatorialComplex object."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1], rank=0)
        CCC.add_cell([1, 2], rank=3)
        # Updating an node attribute present in the CombinatorialComplex object.
        CCC.__setitem__([1], {"weights": 1})
        # Setting a cell attribute present in the CombinatorialComplex object.
        CCC.__setitem__([1, 2], {"weights": 1})
        assert CCC._complex_set.hyperedge_dict[3][frozenset([1, 2])] == {"weights": 1}
        assert CCC.nodes[1]["weights"] == 1
        # Setting a cell attribute not present in the CombinatorialComplex object.
        with pytest.raises(KeyError):
            CCC.__setitem__([1, 2, 3], {"weights": 1})

    def test_degree(self):
        """Test for the degree function."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        CCC.add_cell([1, 2], rank=2)
        assert CCC.degree(1) == 1
        with pytest.raises(ValueError) as exp:
            CCC.degree(1, -1)
        assert str(exp.value) == "Rank must be positive"
        assert CCC.degree(2, 2) == 3
        with pytest.raises(KeyError) as exp:
            node = 7
            assert CCC.degree(node, 2)
        assert str(exp.value) == f"'Node 7 not in {CCC.__shortstr__}.'"
        assert CCC.degree(1, rank=None) == 3

    def test_size(self):
        """Test for the size function."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        CCC.add_cell([1, 2], rank=2)
        assert CCC.size(1) == 1
        with pytest.raises(ValueError) as exp:
            CCC.size(frozenset([1, 2, 3]))
        assert str(exp.value) == f"Input cell is not in cells of the {CCC.__shortstr__}"

    def test_num_nodes_and_cells(self):
        """Test for number of nodes and number of cells."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3], rank=1)
        CCC.add_cell([1, 2, 4, 3], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        CCC.add_cell([1, 2], rank=2)
        y1 = HyperEdge([1, 2], rank=1)
        y2 = HyperEdge([1, 3], rank=1)
        assert CCC.number_of_nodes() == 6
        assert CCC.number_of_nodes([1, 2]) == 2
        assert CCC.number_of_cells() == 11
        assert CCC.number_of_cells([y1, y2]) == 2

    def test_order(self):
        """Test for the order function."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2, 3, 4, 5, 6, 7, 8], rank=1)
        assert CCC.order() == 8

    def test_remove_nodes(self):
        """Test for the remove_node and remove_nodes methods."""
        example = CombinatorialComplex()
        example.add_cell([1, 2], rank=1)
        example.add_cell([1, 3, 2], rank=1)
        example.add_cell([1, 2, 4, 3], rank=2)
        example.add_cell([2, 5], rank=1)
        example.add_cell([2, 6, 4], rank=2)
        example.remove_node(1)

        assert example._complex_set.hyperedge_dict == {
            1: {frozenset({2, 5}): {"weight": 1}},
            0: {
                frozenset({2}): {"weight": 1},
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
                frozenset({5}): {"weight": 1},
                frozenset({6}): {"weight": 1},
            },
            2: {frozenset({2, 4, 6}): {"weight": 1}},
        }
        with pytest.raises(KeyError) as exp:
            example.remove_nodes([1])
        assert str(exp.value) == f"'node 1 not in {example.__shortstr__}'"
        example.remove_nodes([2, 5])
        assert example._complex_set.hyperedge_dict == {
            0: {
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
                frozenset({6}): {"weight": 1},
            }
        }

        example.remove_nodes(HyperEdge([3]))
        assert example._complex_set.hyperedge_dict == {
            0: {frozenset({4}): {"weight": 1}, frozenset({6}): {"weight": 1}}
        }
        node = {4: 3}
        with pytest.raises(TypeError) as exp:
            example.remove_node(node)
        assert str(exp.value) == "node must be a HyperEdge or a hashable object"
        example = CombinatorialComplex()
        example.add_cell([1, 2], rank=1)
        example.add_cell([1, 3, 2], rank=1)
        example.add_cell([1, 2, 4, 3], rank=2)
        example.add_cell([2, 5], rank=1)
        example.add_cell([2, 6, 4], rank=2)
        example.remove_nodes([HyperEdge([1]), HyperEdge([2])])
        assert example._complex_set.hyperedge_dict == {
            0: {
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
                frozenset({5}): {"weight": 1},
                frozenset({6}): {"weight": 1},
            }
        }

    def test_set_cell_attributes(self):
        """Test for the set_cell_attributes method."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        d = {(1, 2, 3, 4): "red", (1, 2, 3): "blue"}
        CCC.set_cell_attributes(d, name="color")
        assert CCC.cells[(1, 2, 3, 4)]["color"] == "red"
        d = {(1, 2): {"attr1": "blue", "size": "large"}}
        CCC.set_cell_attributes(d)
        assert CCC.cells[(1, 2)]["attr1"] == "blue"

    def test_get_cell_attributes(self):
        """Test for get_cell_attributes method."""
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        d = {(1, 2, 3, 4): "red", (1, 2, 3): "blue"}
        CCC.set_cell_attributes(d, name="color")
        assert CCC.get_cell_attributes("color") == {
            frozenset({1, 2, 3}): "blue",
            frozenset({1, 2, 3, 4}): "red",
        }

    def test_set_node_attributes(self):
        """Test for the set and get nodes attributes method."""
        example1 = CombinatorialComplex()
        example1.add_cell([4, 6], rank=2)
        node = 4
        d = {node: {"color": "red", "attr2": 1}}
        example1.set_node_attributes(d)
        assert example1.nodes.nodes[frozenset({node})]["color"] == "red"
        assert example1.get_node_attributes("color") == {4: "red"}
        node = 6
        d = {node: "red"}
        example1.set_node_attributes(d, "color")
        # assert example1.nodes.
        assert example1.get_node_attributes("color") == {
            4: "red",
            6: "red",
        }

    def test_add_cells(self):
        """Test for the add_cells method."""
        CCC = CombinatorialComplex()
        CCC.add_cells_from([[1, 2], [1, 3, 2], [1, 2, 3, 4]], ranks=[1, 1, 2])
        assert CCC._complex_set.hyperedge_dict == {
            1: {frozenset({1, 2}): {"weight": 1}, frozenset({1, 2, 3}): {"weight": 1}},
            0: {
                frozenset({1}): {"weight": 1},
                frozenset({2}): {"weight": 1},
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
            },
            2: {frozenset({1, 2, 3, 4}): {"weight": 1}},
        }
        with pytest.raises(ValueError) as exp:
            CCC.add_cells_from([[1, 4]], ranks=[1, 1])
        assert str(exp.value) == "zip() argument 2 is longer than argument 1"
        CCC = CombinatorialComplex()
        CCC.add_cells_from(
            [
                HyperEdge([1, 2], rank=1),
                HyperEdge([1, 3, 2], rank=1),
                HyperEdge([1, 2, 3, 4], rank=2),
            ]
        )
        assert CCC._complex_set.hyperedge_dict == {
            1: {frozenset({1, 2}): {"weight": 1}, frozenset({1, 2, 3}): {"weight": 1}},
            0: {
                frozenset({1}): {"weight": 1},
                frozenset({2}): {"weight": 1},
                frozenset({3}): {"weight": 1},
                frozenset({4}): {"weight": 1},
            },
            2: {frozenset({1, 2, 3, 4}): {"weight": 1}},
        }
        CCC = CombinatorialComplex()
        with pytest.raises(ValueError) as exp:
            CCC.add_cells_from(
                [[1, 2], HyperEdge([1, 3, 2], rank=1), HyperEdge([1, 2, 3, 4], rank=2)]
            )
        assert (
            str(exp.value)
            == "input must be an HyperEdge [1, 2] object when rank is None"
        )
        with pytest.raises(ValueError) as exp:
            CCC.add_cells_from([HyperEdge([1, 3, 2], rank=1), HyperEdge([1, 2, 3, 4])])
        assert (
            str(exp.value)
            == "input HyperEdge Nodes set: (1, 2, 3, 4), attrs: {} has None rank"
        )

    def test_cell_node_adjacency_matrix(self):
        """Test for the cells adjacency matrix method."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        B = CCC.incidence_matrix(rank=0, to_rank=2)
        assert B.shape == (4, 2)
        assert (B.T[0].todense() == [1, 1, 1, 0]).all()
        assert (B.T[1].todense() == [0, 1, 1, 1]).all()
        CCC = CombinatorialComplex()
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([1, 2], rank=1)
        CCC.add_cell([1, 3, 2], rank=1)
        CCC.add_cell([1, 2, 3, 4], rank=2)
        CCC.add_cell([2, 5], rank=1)
        CCC.add_cell([2, 6, 4], rank=2)
        B, row, col = CCC.incidence_matrix(1, index=True)
        assert B[(frozenset({1, 2}))] == 0
        assert B[(frozenset({1, 2, 3}))] == 1
        assert B[(frozenset({2, 5}))] == 2
        assert (
            CCC.node_to_all_cell_adjacnecy_matrix().todense()
            == [
                [0, 3, 2, 1, 0, 0],
                [3, 0, 2, 2, 1, 1],
                [2, 2, 0, 1, 0, 0],
                [1, 2, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
            ]
        ).all()
        (
            CCC.all_cell_to_node_coadjacency_matrix().todense()
            == [
                [0, 2, 1, 2, 1],
                [2, 0, 1, 3, 1],
                [1, 1, 0, 1, 1],
                [2, 3, 1, 0, 2],
                [1, 1, 1, 2, 0],
            ]
        ).all()

    def test_singletons(self):
        """Test singletons of CCC."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CCC.add_cell([9], rank=0)
        assert CCC.singletons() == [frozenset({9})]

    def test_remove_singletons(self):
        """Test remove_singletons of CCC."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)
        CCC.add_cell([9], rank=0)
        CCC = CCC.remove_singletons()
        assert 9 not in CCC.nodes

    def test_CCC_condition(self):
        """Test the _CCC_condition method of the CombinatorialComplex class."""
        CCC = CombinatorialComplex([[1, 2, 3], [2, 3, 4]], ranks=2)

        CCC._CCC_condition({1, 2}, rank=1)

        with pytest.raises(ValueError):
            CCC._CCC_condition({1, 2}, rank=3)
