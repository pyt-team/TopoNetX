"""Test path complex class."""

import networkx as nx
import numpy as np
import pytest

from toponetx.classes.path import Path
from toponetx.classes.path_complex import PathComplex


class TestPathComplex:
    """Test path complex class."""

    def test_init_empty_path_complex(self):
        """Test empty path complex."""
        PC = PathComplex()
        assert len(PC.paths) == 0
        assert len(PC.nodes) == 0
        assert len(PC.edges) == 0
        assert PC.dim == -1

    def test_iterable_paths(self):
        """Test TypeError when paths is not iterable."""
        with pytest.raises(TypeError):
            _ = PathComplex(paths=1)

    def test_shape_property(self):
        """Test shape property."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        assert PC.shape == (4, 3, 2)

        PC = PathComplex()
        assert PC.shape == ()

    def test_dim_property(self):
        """Test dim property."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        assert PC.dim == 2

        PC = PathComplex()
        assert PC.dim == -1  # empty path complex

    def test_nodes_property(self):
        """Test nodes property."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        nodes = PC.nodes
        assert len(nodes) == 4
        assert 1 in nodes
        assert 2 in nodes
        assert 6 not in nodes

    def test_edges_property(self):
        """Test edges property."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        edges = PC.edges
        assert len(edges) == 3
        assert (1, 2) in edges
        assert (2, 3) in edges
        assert (2, 4) in edges
        assert (1, 3) not in edges

    def test_path_property(self):
        """Test path property."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        assert [1, 2, 3] in PC.paths
        assert [1, 2, 4] in PC.paths
        assert [1, 2] in PC.paths
        assert 4 in PC.paths

    def test_add_path(self):
        """Test add path."""
        PC = PathComplex()
        PC.add_path([1, 2, 3])
        assert [1, 2, 3] in PC.paths
        assert [1, 2] in PC.paths
        assert 3 in PC.paths

        with pytest.raises(ValueError):
            PC.add_path([3, 2])

        with pytest.raises(ValueError):
            PC.add_path([3, 2, 2])

        with pytest.raises(ValueError):
            PC.add_path([3, 2, 3])

        PC = PathComplex(reserve_sequence_order=True)
        PC.add_path([3, 2, 1])
        assert [3, 2, 1] in PC.paths

        PC = PathComplex()
        PC.add_path([1, 2])
        PC.add_path([2, 3])
        assert (
            [1, 2, 3] not in PC.paths
        )  # because manually adding subpaths does not add superpath. Future features?
        assert [1, 2] in PC.paths
        assert [2, 3] in PC.paths

    def test_contains(self):
        """Test the __contains__ method."""
        PC = PathComplex([[1, 2], [3], [4]])

        assert 1 in PC
        assert 3 in PC
        assert 5 not in PC

        assert (1, 2) in PC
        assert (1, 3) not in PC

    def test_constructor_using_graph(self):
        """Test constructor using graph."""
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        PC = PathComplex(G)

        assert [1, 2, 3] in PC.paths  # because constructing from graph
        assert [1, 2] in PC.paths
        assert [2, 3] in PC.paths
        assert 1 in PC.paths
        assert 2 in PC.paths
        assert 3 in PC.paths

        PC = PathComplex(G, allowed_paths=[[1], [2], [3], [1, 2]])
        assert [2, 3] in PC.paths
        assert PC._allowed_paths == {(1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3)}

        PC = PathComplex(G, allowed_paths=[])
        assert [
            1,
            2,
            3,
        ] in PC.paths  # because constructing from graph and allowed_paths is empty
        assert [1, 2] in PC.paths
        assert [2, 3] in PC.paths
        assert 1 in PC.paths
        assert 2 in PC.paths
        assert 3 in PC.paths

        G = nx.Graph()
        G.add_edge(3, 2)
        PC = PathComplex(G)
        assert [2, 3] in PC.paths
        assert [3, 2] not in PC.paths

    def test_clone(self):
        """Test clone."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        PX_clone = PC.clone()
        assert set(PX_clone.paths) == set(PC.paths)
        assert PX_clone is not PC

    def test_skeleton(self):
        """Test skeleton."""
        PC = PathComplex(
            [
                [0, 1],
                [1, 2, 3],
                [1, 3, 2],
                [2, 1, 3],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 2, 3],
                [0, 1, 3, 2],
            ],
            max_rank=2,
        )
        assert PC.skeleton(0) == [(0,), (1,), (2,), (3,)]
        assert PC.skeleton(1) == [(0, 1), (1, 2), (1, 3), (2, 3)]
        assert PC.skeleton(2) == [(0, 1, 2), (0, 1, 3), (1, 2, 3), (1, 3, 2), (2, 1, 3)]
        with pytest.raises(ValueError):
            PC.skeleton(3)

    def test_skeleton_raise_errors(self):
        """Test skeleton raise errors."""
        PC = PathComplex([[1, 2, 3], [1, 3]])
        with pytest.raises(ValueError):
            PC.skeleton(-1)

        with pytest.raises(TypeError):
            PC.skeleton(1.5)

        with pytest.raises(TypeError):
            PC.skeleton(1, 2)

        with pytest.raises(ValueError):
            PC.skeleton(3)

    def test_str(self):
        """Test str."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]], name="PC")
        assert str(PC) == "Path Complex with shape (4, 3, 2) and dimension 2"

    def test_add_str_nodes(self):
        """Test add str nodes."""
        PC = PathComplex([[1, 2, 3], [2, 4, "a"]])
        assert "a" in PC.paths
        assert "a" in PC.nodes
        assert [4, "a"] in PC.edges

        with pytest.raises(ValueError):
            PC.add_path(["a", 1, 2])

    def test_repr_str(self):
        """Test repr str."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        assert (repr(PC)) == "PathComplex()"

    def test_getittem_set_attributes(self):
        """Test getitem and set_attributes methods."""
        PC = PathComplex([[1, 2, 3], [1, 2, 4]])
        assert PC[[1, 2, 3]] == {}
        assert PC[1] == {}

        with pytest.raises(KeyError):
            PC[5]

        PC.add_path([1, 2, 5], heat=55)
        assert PC[[1, 2, 5]] == {"heat": 55}

        PC.set_path_attributes({(1, 2, 5): 66}, "heat")
        assert PC[[1, 2, 5]] == {"heat": 66}

        PC.set_path_attributes({(1, 2, 5): {"heat": 77, "color": "red"}})
        assert PC[[1, 2, 5]] == {"heat": 77, "color": "red"}

        PC.set_path_attributes({1: 88, (1, 2, 5): 99}, "heat")
        assert PC[1] == {"heat": 88}
        assert PC.nodes[1] == {"heat": 88}
        assert PC[[1, 2, 5]] == {"heat": 99, "color": "red"}

        PC.set_path_attributes({(1, 2): {"heat": 100, "color": "red"}})
        assert PC[[1, 2]] == {"heat": 100, "color": "red"}
        assert PC.edges[[1, 2]] == {"heat": 100, "color": "red"}

        PC.set_path_attributes({(1,): {"heat": 200, "color": "red"}})
        assert PC[[1]] == {"heat": 200, "color": "red"}
        assert PC.nodes[1] == {"heat": 200, "color": "red"}
        assert PC[1] == {"heat": 200, "color": "red"}

        PC.set_path_attributes({1: 90, 4: 23}, "heat")
        assert PC[1] == {"heat": 90, "color": "red"}
        assert PC.nodes[1] == {"heat": 90, "color": "red"}
        assert PC[4] == {"heat": 23}

        PC.set_path_attributes({(1, 2): 882, (1, 2, 5): 939}, "heat")
        assert PC[[1, 2]] == {"heat": 882, "color": "red"}
        assert PC.edges[[1, 2]] == {"heat": 882, "color": "red"}
        assert PC[[1, 2, 5]] == {"heat": 939, "color": "red"}

        PC.add_path(6, heat=77)
        assert PC[6] == {"heat": 77}
        assert PC.nodes[6] == {"heat": 77}

        PC.set_node_attributes({6: 88, 5: 22}, "heat")
        assert PC[6] == {"heat": 88}
        assert PC.nodes[6] == {"heat": 88}
        assert PC[5] == {"heat": 22}
        assert PC.nodes[5] == {"heat": 22}

        PC.set_node_attributes({6: 99, (5,): 33}, "heat")
        assert PC[6] == {"heat": 99}
        assert PC.nodes[6] == {"heat": 99}
        assert PC[5] == {"heat": 33}
        assert PC.nodes[5] == {"heat": 33}

        PC.set_node_attributes({6: {"heat": 100, "color": "red"}, (5,): {"heat": 34}})
        assert PC[6] == {"heat": 100, "color": "red"}
        assert PC.nodes[6] == {"heat": 100, "color": "red"}
        assert PC[5] == {"heat": 34}
        assert PC.nodes[5] == {"heat": 34}

        PC.add_path([4, 5], heat=77)
        assert PC[[4, 5]] == {"heat": 77}
        assert PC.edges[(4, 5)] == {"heat": 77}

        PC.set_edge_attributes({(4, 5): 88}, "heat")
        assert PC[[4, 5]] == {"heat": 88}
        assert PC.edges[(4, 5)] == {"heat": 88}

        PC.set_edge_attributes({(4, 5): {"heat": 99, "color": "red"}})
        assert PC[[4, 5]] == {"heat": 99, "color": "red"}
        assert PC.edges[(4, 5)] == {"heat": 99, "color": "red"}

        with pytest.raises(ValueError):
            PC.set_node_attributes({(4, 5): 55}, name="heat")
        with pytest.raises(ValueError):
            PC.set_node_attributes({(4, 5): {"heat": 55}})
        with pytest.raises(TypeError):
            PC.set_node_attributes({(4,): [5]})
        with pytest.raises(ValueError):
            PC.set_edge_attributes({(4, 5, 6): 55}, name="heat")
        with pytest.raises(ValueError):
            PC.set_edge_attributes({(4, 5, 6): {"heat": 55}})
        with pytest.raises(TypeError):
            PC.set_edge_attributes({(4, 5): [5]})
        with pytest.raises(TypeError):
            PC.set_path_attributes({(4, 5): [5]})

    def test_get_len_(self):
        """Test get size of the path complex."""
        PC = PathComplex()
        PC.add_paths_from([[1, 2, 3], [1, 2, 4]])
        assert len(PC) == 9

    def test_add_paths_from(self):
        """Test add paths from graph."""
        PC = PathComplex()
        PC.add_paths_from([[1, 2, 3], [1, 2, 4]])
        assert [1, 2, 3] in PC.paths
        assert [1, 2] in PC.paths
        assert [2, 3] in PC.paths
        assert 1 in PC.paths
        assert 2 in PC.paths
        assert 3 in PC.paths
        assert 1 in PC.nodes
        assert 2 in PC.nodes
        assert 3 in PC.nodes
        assert 4 in PC.nodes
        assert [1, 2] in PC.edges
        assert [2, 4] in PC.edges

    def test_add_node(self):
        """Test add node."""
        PC = PathComplex()
        PC.add_node(1)
        assert 1 in PC.paths
        assert [1] in PC.paths
        assert 1 in PC.nodes

        with pytest.raises(TypeError):
            PC.add_node([1, 2])

        PC.add_node(Path(2))
        assert 2 in PC.paths
        assert [2] in PC.paths
        assert 2 in PC.nodes

        with pytest.raises(ValueError):
            PC.add_node(Path([2, 3]))

    def test_remove_nodes(self):
        """Test remove_nodes method."""
        PC = PathComplex([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        assert 1 in PC.nodes
        PC.remove_nodes([1])
        assert 1 not in PC.paths
        assert [1, 2] not in PC.paths
        assert [1, 3] not in PC.paths
        assert 1 not in PC.nodes
        assert [1, 2] not in PC.edges
        assert [1, 3] not in PC.edges

    def test_incidence_matrix(self):
        """Test incidence matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )
        assert np.all(PC.incidence_matrix(0).todense == np.zeros((0, len(PC.nodes))))
        assert np.all(
            PC.incidence_matrix(1, signed=False).todense()
            == np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
        )
        assert np.all(
            PC.incidence_matrix(2, signed=False).todense()
            == np.array(
                [[1, 1, 0, 0, 0], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]
            )
        )

        with pytest.raises(ValueError):
            PC.incidence_matrix(-1)

        with pytest.raises(ValueError):
            PC.incidence_matrix(3)

        row, col, B = PC.incidence_matrix(0, index=True)
        assert row == {}
        assert col == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        assert np.all(B.todense() == np.zeros((0, len(PC.nodes))))
        row, col, B = PC.incidence_matrix(0, index=True, signed=False)
        assert row == {}
        assert col == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        assert np.all(B.todense() == np.zeros((0, len(PC.nodes))))

        row, col, B = PC.incidence_matrix(1, index=True)
        assert row == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        assert col == {(0, 1): 0, (1, 2): 1, (1, 3): 2, (2, 3): 3}
        assert np.all(
            B.todense()
            == np.array([[-1, 0, 0, 0], [1, -1, -1, 0], [0, 1, 0, -1], [0, 0, 1, 1]])
        )
        row, col, B = PC.incidence_matrix(1, index=True, signed=False)
        assert row == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}
        assert col == {(0, 1): 0, (1, 2): 1, (1, 3): 2, (2, 3): 3}
        assert np.all(
            B.todense()
            == np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
        )

        row, col, B = PC.incidence_matrix(2, index=True)
        assert row == {(0, 1): 0, (1, 2): 1, (1, 3): 2, (2, 3): 3}
        assert col == {
            (0, 1, 2): 0,
            (0, 1, 3): 1,
            (1, 2, 3): 2,
            (1, 3, 2): 3,
            (2, 1, 3): 4,
        }
        assert np.all(
            B.todense()
            == np.array(
                [[1, 1, 0, 0, 0], [1, 0, 1, -1, 1], [0, 1, -1, 1, 1], [0, 0, 1, 1, -1]]
            )
        )
        row, col, B = PC.incidence_matrix(2, index=True, signed=False)
        assert row == {(0, 1): 0, (1, 2): 1, (1, 3): 2, (2, 3): 3}
        assert col == {
            (0, 1, 2): 0,
            (0, 1, 3): 1,
            (1, 2, 3): 2,
            (1, 3, 2): 3,
            (2, 1, 3): 4,
        }
        assert np.all(
            B.todense()
            == np.array(
                [[1, 1, 0, 0, 0], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]
            )
        )

    def test_up_laplacian_matrix(self):
        """Test up laplacian matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )
        with pytest.raises(ValueError):
            PC.up_laplacian_matrix(-1)
        with pytest.raises(ValueError):
            PC.up_laplacian_matrix(2)
        with pytest.raises(ValueError):
            PC.up_laplacian_matrix(1, weight="weight")

        row, L_up = PC.up_laplacian_matrix(0, index=True)
        row_B, col_B, B = PC.incidence_matrix(1, index=True)
        L_up_tmp = B.todense() @ B.todense().T

        assert np.all(L_up.todense() == L_up_tmp)
        assert row == row_B

        L_up = PC.up_laplacian_matrix(1)
        B = PC.incidence_matrix(2)
        L_up_tmp = B.todense() @ B.todense().T

        assert np.all(L_up.todense() == L_up_tmp)

        L_up = PC.up_laplacian_matrix(1, signed=True)
        B = PC.incidence_matrix(2, signed=True)
        L_up_tmp = B.todense() @ B.todense().T

        assert np.all(L_up.todense() == L_up_tmp)

    def test_down_laplacian_matrix(self):
        """Test down laplacian matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )
        with pytest.raises(ValueError):
            PC.down_laplacian_matrix(0)
        with pytest.raises(ValueError):
            PC.down_laplacian_matrix(3)
        with pytest.raises(ValueError):
            PC.down_laplacian_matrix(1, weight="weight")

        row, L_down = PC.down_laplacian_matrix(1, index=True)
        row_B, col_B, B = PC.incidence_matrix(1, index=True)
        L_down_tmp = B.todense().T @ B.todense()

        assert np.all(L_down.todense() == L_down_tmp)
        assert row == row_B

        L_down = PC.down_laplacian_matrix(2)
        B = PC.incidence_matrix(2)
        L_down_tmp = B.todense().T @ B.todense()

        assert np.all(L_down.todense() == L_down_tmp)

        L_down = PC.down_laplacian_matrix(2, signed=True)
        B = PC.incidence_matrix(2, signed=True)
        L_down_tmp = B.todense().T @ B.todense()

        assert np.all(L_down.todense() == L_down_tmp)

    def test_hodge_laplacian_matrix(self):
        """Test hodge laplacian matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )

        assert np.all(
            PC.hodge_laplacian_matrix(0).todense()
            == PC.up_laplacian_matrix(0).todense()
        )
        assert np.all(
            PC.hodge_laplacian_matrix(2).todense()
            == PC.down_laplacian_matrix(2).todense()
        )
        assert np.all(
            PC.hodge_laplacian_matrix(1).todense()
            == (PC.up_laplacian_matrix(1) + PC.down_laplacian_matrix(1)).todense()
        )

        assert np.all(
            PC.hodge_laplacian_matrix(0, signed=False).todense()
            == PC.up_laplacian_matrix(0, signed=False).todense()
        )
        assert np.all(
            PC.hodge_laplacian_matrix(2, signed=False).todense()
            == PC.down_laplacian_matrix(2, signed=False).todense()
        )
        assert np.all(
            PC.hodge_laplacian_matrix(1, signed=False).todense()
            == abs(
                PC.up_laplacian_matrix(1, signed=True)
                + PC.down_laplacian_matrix(1, signed=True)
            ).todense()
        )

        row, L_hodge = PC.hodge_laplacian_matrix(0, index=True)
        assert row == PC.incidence_matrix(0, index=True)[1]

        row, L_hodge = PC.hodge_laplacian_matrix(1, index=True)
        assert row == PC.incidence_matrix(1, index=True)[1]

        row, L_hodge = PC.hodge_laplacian_matrix(2, index=True)
        assert row == PC.incidence_matrix(2, index=True)[1]

        with pytest.raises(ValueError):
            PC.hodge_laplacian_matrix(-1)

        with pytest.raises(ValueError):
            PC.hodge_laplacian_matrix(3)

    def test_adjacency_matrix(self):
        """Test adjacency matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )
        with pytest.raises(ValueError):
            PC.adjacency_matrix(-1)
        with pytest.raises(ValueError):
            PC.adjacency_matrix(2)

        adj_0 = PC.adjacency_matrix(0, signed=False).todense()
        assert np.all(
            adj_0 == np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
        )

        adj_1 = PC.adjacency_matrix(1, signed=False).todense()

        assert np.all(
            adj_1 == np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
        )

        row, adj = PC.adjacency_matrix(0, index=True)
        assert row == PC.incidence_matrix(0, index=True)[1]

        row, adj = PC.adjacency_matrix(1, index=True)
        assert row == PC.incidence_matrix(1, index=True)[1]

    def test_coadjacency_matrix(self):
        """Test coadjacency matrix."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]]
        )
        with pytest.raises(ValueError):
            PC.coadjacency_matrix(0)
        with pytest.raises(ValueError):
            PC.coadjacency_matrix(3)

        coadj_1 = PC.coadjacency_matrix(1, signed=False).todense()
        assert np.all(
            coadj_1
            == np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
        )

        coadj_2 = PC.coadjacency_matrix(2, signed=False).todense()
        assert np.all(
            coadj_2
            == np.array(
                [
                    [0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0],
                ]
            )
        )

        row, adj = PC.coadjacency_matrix(1, index=True)
        assert row == PC.incidence_matrix(0, index=True)[1]
        assert row == PC.incidence_matrix(1, index=True)[0]

        row, adj = PC.coadjacency_matrix(2, index=True)
        assert row == PC.incidence_matrix(1, index=True)[1]
        assert row == PC.incidence_matrix(2, index=True)[0]

    def test_restrict_to_nodes(self):
        """Test restrict_to_nodes."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]], max_rank=3
        )
        res = PC.restrict_to_nodes([0, 1, 2])
        assert len(res.paths) == 6
        assert len(res.nodes) == 3
        assert len(res.edges) == 2

        res = PC.restrict_to_nodes([1, 2, 3])
        assert len(res.paths) == 9
        assert len(res.nodes) == 3
        assert len(res.edges) == 3

        with pytest.raises(ValueError):
            PC.restrict_to_nodes([])

    def test_restrict_to_paths(self):
        """Test restrict_to_paths."""
        PC = PathComplex(
            [[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [0, 1, 2], [0, 1, 3]], max_rank=3
        )

        res = PC.restrict_to_paths([[0, 1], [1, 2, 3]])
        assert len(res.paths) == 8
        assert len(res.nodes) == 4
        assert len(res.edges) == 3

        with pytest.raises(ValueError):
            PC.restrict_to_paths([])

    def test_get_node_attributes(self):
        """Test get_node_attributes."""
        PC = PathComplex()
        PC.add_node(0)
        PC.add_node(1, heat=55)
        PC.add_node(2, heat=66)
        PC.add_node(3, color="red")
        PC.add_node(2, color="blue")
        PC.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])

        assert PC.get_node_attributes("heat") == {(1,): 55, (2,): 66}
        assert PC.get_node_attributes("color") == {(2,): "blue", (3,): "red"}

    def test_get_edge_attributes(self):
        """Test get_edge_attributes."""
        PC = PathComplex()
        PC.add_paths_from([[0, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3]])
        PC.add_path([0, 1], weight=32)
        PC.add_path([1, 2], weight=98)
        PC.add_path([1, 3], color="red")
        PC.add_path([2, 3], color="blue")

        assert PC.get_edge_attributes("weight") == {(0, 1): 32, (1, 2): 98}
        assert PC.get_edge_attributes("color") == {(1, 3): "red", (2, 3): "blue"}

    def test_get_path_attributes(self):
        """Test get_path_attributes."""
        PC = PathComplex()
        PC.add_paths_from([[0, 1]])
        PC.add_path([0, 1, 2], weight=43)
        PC.add_path([0, 1, 3], weight=98)
        PC.add_path([1, 2, 3], color="red")
        PC.add_path([1, 3, 2], color="blue")
        PC.add_path([2, 1, 3], color="green")

        assert PC.get_path_attributes("weight") == {(0, 1, 2): 43, (0, 1, 3): 98}
        assert PC.get_path_attributes("color") == {
            (1, 2, 3): "red",
            (1, 3, 2): "blue",
            (2, 1, 3): "green",
        }
