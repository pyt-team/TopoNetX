"""Test simplicial complex class."""

from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import bmat

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.datasets.mesh import stanford_bunny

try:
    import hypernetx as hnx
except ImportError:
    hnx = None

try:
    import spharapy.trimesh as tm
except ImportError:
    tm = None


class TestSimplicialComplex:
    """Test SimplicialComplex class."""

    def test_iterable_simplices(self):
        """Test TypeError for simplices not iterable."""
        with pytest.raises(TypeError):
            SimplicialComplex(simplices=1)

    def test_shape_property(self):
        """Test shape property."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        assert sc.shape == (5, 6, 2)

        sc = SimplicialComplex()
        assert sc.shape == ()

        # make sure that empty dimensions are not part of the shape after removal
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        sc.remove_nodes([2, 3])
        assert sc.shape == (3, 1)

    def test_dim_property(self):
        """Test dim property."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        assert sc.dim == 2

        sc.remove_nodes([2, 3])
        assert sc.dim == 1

    def test_maxdim(self) -> None:
        """Test deprecated maxdim property for deprecation warning."""
        SC = SimplicialComplex()
        with pytest.deprecated_call():
            assert SC.maxdim == -1

        SC.add_simplex([1, 2, 3])
        with pytest.deprecated_call():
            assert SC.maxdim == 2

    def test_nodes_property(self):
        """Test nodes property."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        nodes = sc.nodes
        assert len(nodes) == 5
        assert [1] in nodes
        assert [8] not in nodes

    def test_simplices_property(self):
        """Test simplices property."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        assert len(sc.simplices) == 13
        assert [1, 2, 3] in sc.simplices
        assert [2, 3, 4] in sc.simplices
        assert [0, 1] in sc.simplices

    def test_is_maximal(self):
        """Test is_maximal method."""
        sc = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        assert sc.is_maximal([1, 2, 3])
        assert not sc.is_maximal([1, 2])
        assert not sc.is_maximal([3])

        with pytest.raises(ValueError):
            sc.is_maximal([1, 2, 3, 4])

    def test_get_maximal_simplices_of_simplex(self) -> None:
        """Test the get_maximal_simplices_of_simplex method."""
        SC = SimplicialComplex([(1, 2, 3), (2, 3, 4), (0, 1), (5,)])

        assert SC.get_maximal_simplices_of_simplex((1, 2, 3)) == {Simplex((1, 2, 3))}
        assert SC.get_maximal_simplices_of_simplex((1, 2)) == {Simplex((1, 2, 3))}
        assert SC.get_maximal_simplices_of_simplex((2, 3)) == {
            Simplex((1, 2, 3)),
            Simplex((2, 3, 4)),
        }
        assert SC.get_maximal_simplices_of_simplex((0,)) == {Simplex((0, 1))}
        assert SC.get_maximal_simplices_of_simplex((5,)) == {Simplex((5,))}

    def test_constructor_using_graph(self):
        """Test input a networkx graph in the constructor."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G)

        assert (0, 1) in SC.simplices
        assert (2, 5) in SC.simplices
        assert (5, 4) in SC.simplices
        assert SC.simplices[(5, 4)]["weight"] == 5

    def test_skeleton_raise_errors(self):
        """Test skeleton raises."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G)

        with pytest.raises(ValueError):
            SC.skeleton(-2)

        with pytest.raises(ValueError):
            SC.skeleton(2)

    def test_str(self):
        """Test str method."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        assert (
            str(SC)
        ) == f"Simplicial Complex with shape {SC.shape} and dimension {SC.dim}"

    def test_rep_str(self):
        """Test repr string."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G)
        assert (repr(SC)) == "SimplicialComplex()"

    def test_iter(self):
        """Test iter method."""
        SC = SimplicialComplex([[1, 2, 3], [2, 4], [5]])
        simplices = set(SC)
        assert simplices == {
            Simplex((1,)),
            Simplex((2,)),
            Simplex((3,)),
            Simplex((4,)),
            Simplex((5,)),
            Simplex((1, 2)),
            Simplex((1, 3)),
            Simplex((2, 3)),
            Simplex((2, 4)),
            Simplex((1, 2, 3)),
        }

    def test_getittem__(self):
        """Test __getitem__ methods."""
        SC = SimplicialComplex()
        SC.add_simplex((0, 1), weight=5)
        SC.add_simplex((1, 2, 3), heat=5)

        assert SC[(0, 1)]["weight"] == 5
        assert SC[(1, 2, 3)]["heat"] == 5
        with pytest.raises(KeyError):
            _ = SC[(1, 2, 3, 4, 5)]["heat"]

        SC[(0, 1)]["new"] = 10
        assert SC[(0, 1)]["new"] == 10

    def test_setting_simplex_attributes(self):
        """Test setting simplex attributes through a `SimplicialComplex` object."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 5)
        G.add_edge(5, 4, weight=5)
        SC = SimplicialComplex(G, name="graph complex")
        SC.add_simplex((1, 2, 3), heat=5)

        SC[(1, 2, 3)]["heat"] = 6
        assert SC[(1, 2, 3)]["heat"] == 6

        SC[(2, 5)]["heat"] = 1
        assert SC[(2, 5)]["heat"] == 1

        s = Simplex((1, 2, 3, 4), heat=1)
        SC.add_simplex(s)
        assert SC[(1, 2, 3, 4)]["heat"] == 1

        s = Simplex(("A",), heat=1)
        SC.add_simplex(s)
        assert SC["A"]["heat"] == 1

    def test_add_simplices_from(self):
        """Test add simplices from."""
        with pytest.raises(TypeError):
            SC = SimplicialComplex()
            SC.add_simplices_from(4)

    def test_add_node(self):
        """Test add node."""
        SC = SimplicialComplex()
        SC.add_node(9)
        assert 9 in SC
        with pytest.raises(TypeError):
            SC.add_node({1, 2})

        with pytest.raises(ValueError):
            s = Simplex((1, 2, 3, 4))
            SC.add_node(s)

        s = Simplex({1})  # singleton simplex
        SC.add_node(s)
        assert s in SC

        SC = SimplicialComplex()
        assert SC.dim == -1
        SC.add_node(9)
        assert SC.dim == 0

        SC = SimplicialComplex()
        assert SC.dim == -1
        SC.add_node(9)
        assert SC.dim == 0

    def test_add_simplex(self):
        """Test add_simplex method."""
        # create a SimplicialComplex object with no simplices
        SC = SimplicialComplex()

        # add a simplex using the add_simplex() method
        SC.add_simplex([1, 2, 3])

        # assert that the simplex was added correctly
        assert (1, 2, 3) in SC.simplices
        assert [1, 2, 3] in SC.simplices
        assert (1, 2) in SC.simplices
        assert (1, 3) in SC.simplices
        assert (2, 3) in SC.simplices
        assert (1,) in SC.simplices
        assert (2,) in SC.simplices
        assert (3,) in SC.simplices

        SC.add_simplex((4, 5, 6), heat=5)
        assert (4, 5, 6) in SC.simplices
        assert [4, 5, 6] in SC.simplices
        assert (4, 5) in SC.simplices
        assert (4, 6) in SC.simplices
        assert (5, 6) in SC.simplices
        assert (4,) in SC.simplices
        assert (5,) in SC.simplices
        assert (6,) in SC.simplices

        # check that provided attributes are stored correctly
        SC.add_simplex((1, 2), edge_flow=10)
        assert SC[(1, 2)]["edge_flow"] == 10
        SC.add_simplex(Simplex((1, 2), edge_flow=20))
        assert SC[(1, 2)]["edge_flow"] == 20
        SC.add_simplex(Simplex((2, 3), a=1, b=2), b=5)
        assert SC[(2, 3)]["a"] == 1
        assert SC[(2, 3)]["b"] == 5

        with pytest.raises(TypeError):
            SC.add_simplex(iter([1, 2]))

        # simplex cannot contain unhashable elements
        with pytest.raises(TypeError):
            SC.add_simplex([[1, 2], [2, 3]])

        # simplex cannot contain duplicate nodes
        with pytest.raises(ValueError):
            SC.add_simplex((1, 2, 2))

        # use of reserved attributes is not allowed
        with pytest.raises(ValueError):
            SC.add_simplex((1, 2, 3), is_maximal=True)
        with pytest.raises(ValueError):
            SC.add_simplex((1, 2, 3), membership={})

        # add hashable, non iterable node to SC
        SC.add_simplex(11)
        assert 11 in SC.simplices

        # add random string to SC
        SC.add_simplex("test")
        assert ("test",) in SC.simplices

    def test_contains(self):
        """Test the __contains__ method."""
        SC = SimplicialComplex([[1, 2], [3], [4]])

        assert 1 in SC
        assert 3 in SC
        assert 5 not in SC

        assert (1, 2) in SC
        assert (1, 3) not in SC

    def test_remove_maximal_simplex(self):
        """Test remove_maximal_simplex method."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1], [10]])
        SC.remove_maximal_simplex([1, 2, 3])
        SC.remove_maximal_simplex(10)

        assert (1, 2, 3) not in SC.simplices
        assert (10,) not in SC.simplices

        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4]])
        SC.remove_maximal_simplex([2, 3, 4])

        assert [2, 3, 4] not in SC.simplices

        # check after the add_simplex method
        SC = SimplicialComplex()
        SC.add_simplex((1, 2, 3, 4), weight=1)
        c1 = Simplex((1, 2, 3, 4, 5))
        SC.add_simplex(c1)
        SC.remove_maximal_simplex((1, 2, 3, 4, 5))
        assert (1, 2, 3, 4, 5) not in SC.simplices

        # check removal with Simplex
        SC = SimplicialComplex()
        SC.add_simplex((1, 2, 3, 4), weight=1)
        c1 = Simplex((1, 2, 3, 4, 5))
        SC.add_simplex(c1)
        SC.remove_maximal_simplex(c1)
        assert (1, 2, 3, 4, 5) not in SC.simplices

        # check error when simplex not in complex
        with pytest.raises(ValueError):
            SC = SimplicialComplex()
            SC.add_simplex((1, 2, 3, 4), weight=1)
            SC.remove_maximal_simplex([5, 6, 7])

        # only maximal simplices can be removed
        with pytest.raises(ValueError):
            SC = SimplicialComplex()
            SC.add_simplex((1, 2, 3, 4), weight=1)
            SC.remove_maximal_simplex((1, 2, 3))

    def test_remove_nodes(self) -> None:
        """Test remove_nodes method."""
        SC = SimplicialComplex([[0, 1], [1, 2, 3], [2, 3, 4], [4, 5]])
        SC.remove_nodes([2, 5])

        assert [0, 1] in SC.simplices
        assert [1, 3] in SC.simplices
        assert [3, 4] in SC.simplices
        assert [4] in SC.simplices
        assert [2, 3] not in SC.simplices
        assert [2, 4] not in SC.simplices

        assert SC.is_maximal([0, 1])
        assert SC.is_maximal([1, 3])
        assert SC.is_maximal([3, 4])
        assert not SC.is_maximal([1])

    def test_skeleton_and_cliques(self):
        """Test skeleton and cliques methods."""
        G = nx.karate_club_graph()
        cliques = list(nx.enumerate_all_cliques(G))

        nodes = [i for i in cliques if len(i) == 1]
        edges = [i for i in cliques if len(i) == 2]
        faces = [i for i in cliques if len(i) == 3]
        threefaces = [i for i in cliques if len(i) == 4]
        fourfaces = [i for i in cliques if len(i) == 5]

        SC = SimplicialComplex(cliques)
        assert len(SC.skeleton(rank=0)) == len(nodes)
        assert len(SC.skeleton(rank=1)) == len(edges)
        assert len(SC.skeleton(rank=2)) == len(faces)
        assert len(SC.skeleton(rank=3)) == len(threefaces)
        assert len(SC.skeleton(rank=4)) == len(fourfaces)

    def test_incidence_matrix_1(self):
        """Test incidence_matrix shape and values."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        B2 = SC.incidence_matrix(rank=2)
        assert B2.shape == (6, 2)
        np.testing.assert_array_equal(
            B2.toarray(),
            np.array([[0, 1, -1, 1, 0, 0], [0, 0, 0, 1, -1, 1]]).T,
        )

        # repeat the same test, but with signed=False
        B2 = SC.incidence_matrix(rank=2, signed=False)
        assert B2.shape == (6, 2)
        np.testing.assert_array_equal(
            B2.toarray(),
            np.array([[0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1]]).T,
        )

    def test_hodge_laplacian_matrix(self):
        """Test hodge_laplacian_matrix shape and values."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        L_hodge = SC.hodge_laplacian_matrix(rank=0)

        assert L_hodge.shape == (5, 5)

        D = np.diag([1, 3, 3, 3, 2])
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(L_hodge.toarray(), D - A)

    def test_hodge_laplacian_matrix_index(self):
        """Test unsigned hodge_laplacian_matrix method with index."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        row, L_hodge = SC.hodge_laplacian_matrix(rank=0, signed=False, index=True)

        assert L_hodge.shape == (5, 5)

        D = np.diag([1, 3, 3, 3, 2])
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(L_hodge.toarray(), np.abs(D - A))
        expected_row = {(0,): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4}

        assert row == expected_row

    def test_hodge_laplacian_matrix_rank_2(self):
        """Test unsigned hodge_laplacian_matrix method with index for different ranks."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        column, L_hodge = SC.hodge_laplacian_matrix(rank=2, signed=False, index=True)
        expected_col = {(1, 2, 3): 0, (2, 3, 4): 1}
        assert column == expected_col
        np.testing.assert_array_equal(L_hodge.toarray(), np.array([[3, 1], [1, 3]]))

    def test_hodge_laplacian_matrix_rank_1(self):
        """Test unsigned hodge_laplacian_matrix method with index for different ranks."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])
        column, L_hodge = SC.hodge_laplacian_matrix(rank=1, signed=False, index=True)
        expected_col = {
            (0, 1): 0,
            (1, 2): 1,
            (1, 3): 2,
            (2, 3): 3,
            (2, 4): 4,
            (3, 4): 5,
        }
        assert column == expected_col
        np.testing.assert_array_equal(
            L_hodge.toarray(),
            np.array(
                [
                    [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 3.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 3.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 3.0],
                ]
            ),
        )

        with pytest.raises(ValueError):
            SC.hodge_laplacian_matrix(rank=3)

    def test_adjacency_matrix(self):
        """Test adjacency_matrix shape and values."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

        A = SC.adjacency_matrix(rank=0)

        assert A.shape == (5, 5)

        np.testing.assert_array_equal(
            A.toarray(),
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ),
        )

        ind, A = SC.adjacency_matrix(rank=0, index=True)
        expected_ind = {(0,): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4}
        assert ind == expected_ind

        with pytest.raises(ValueError):
            A = SC.adjacency_matrix(rank=0, weight=1)

    def test_get_boundaries(self):
        """Test the get_boundaries method."""
        simplices = [(1, 2, 3), (2, 3, 4), (0, 1)]
        with pytest.raises(TypeError):
            SimplicialComplex.get_boundaries(1)

        boundaries = SimplicialComplex.get_boundaries(simplices)
        assert frozenset((1, 2)) in boundaries
        assert frozenset((1, 3)) in boundaries
        assert frozenset((2, 3)) in boundaries

        # test for min dim/max dim combinations
        boundaries = SimplicialComplex.get_boundaries(simplices, min_dim=2)
        assert boundaries == {frozenset({1, 2, 3}), frozenset({2, 3, 4})}

        boundaries = SimplicialComplex.get_boundaries(simplices, min_dim=1, max_dim=1)
        assert boundaries == {
            frozenset({3, 4}),
            frozenset({2, 3}),
            frozenset({1, 2}),
            frozenset({0, 1}),
            frozenset({2, 4}),
            frozenset({1, 3}),
        }

        boundaries = SimplicialComplex.get_boundaries(simplices, max_dim=0)
        assert boundaries == {
            frozenset({2}),
            frozenset({3}),
            frozenset({1}),
            frozenset({4}),
            frozenset({0}),
        }

        # test for contradicting min dim/max dim combination
        boundaries = SimplicialComplex.get_boundaries(simplices, min_dim=2, max_dim=1)
        assert boundaries == set()

    def test_get_cofaces(self):
        """Test the get_cofaces method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])

        cofaces = SC.get_cofaces([1, 2, 4], codimension=1)
        cofaces = [simplex.elements for simplex in cofaces]
        assert (1, 2, 3, 4) in cofaces
        assert (3, 4, 8) not in cofaces
        # ... add more assertions based on the expected cofaces

    def test_get_star(self):
        """Test the get_star method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])

        star = SC.get_star([1, 2, 4])
        star = [simplex.elements for simplex in star]

        assert (1, 2, 4) in star
        assert (1, 2, 3, 4) in star
        # ... add more assertions based on the expected star

    def test_set_simplex_attributes(self):
        """Test the set_simplex_attributes method."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        d = {(1, 2, 3): "red", (1, 2, 4): "blue"}
        SC.set_simplex_attributes(d, name="color")
        assert SC[(1, 2, 3)]["color"] == "red"

        # test for non-existing simplex
        d = {(3, 4, 5): "Nope"}
        SC.set_simplex_attributes(d, name="color")  # should not raise an error
        SC.set_simplex_attributes(d)  # should not raise an error

    def test_get_edges_from_matrix(self):
        """Test the get_edges_from_matrix method."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1)]

        edges = SimplicialComplex().get_edges_from_matrix(matrix)

        assert set(edges) == set(expected_edges)

    def test_to_hasse_graph(self):
        """Test to hasse graph function."""
        SC = SimplicialComplex()
        SC.add_simplex([0, 1, 2])
        G = SC.to_hasse_graph()
        assert len(G.nodes) == 7
        assert len(G.edges) == 9

    def test_incidence_matrix(self):
        """Test incidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 5])
        with pytest.raises(ValueError):
            SC.incidence_matrix(10)
        with pytest.raises(ValueError):
            SC.incidence_matrix(-1)

        B1 = SC.incidence_matrix(1)
        B2 = SC.incidence_matrix(2)
        assert B1.shape == tuple(SC.shape[:2])
        assert B2.shape == tuple(SC.shape[1:3])
        # B1 : 1->0
        # B2 : 2->1
        assert np.sum(abs(B1.dot(B2))) == 0  # boundary of boundary = 0
        # for the last to be true, order of bases must be canonical in both matrices

        row, col, B1 = SC.incidence_matrix(1, index=True)

        assert (len(row), len(col)) == B1.shape

        B0 = SC.incidence_matrix(0)
        assert B0.shape[-1] == len(SC.skeleton(0))

        assert np.sum(B0.dot(B1)) == 0  # boundary of boundary = 0

        # check incidence matrix for signed=False
        row, col, B1 = SC.incidence_matrix(1, index=True, signed=False)
        assert (len(row), len(col)) == B1.shape

        np.testing.assert_array_equal(
            B1.toarray(),
            np.array(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                ]
            ),
        )

    def test_coincidence_matrix_2(self):
        """Test coincidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        B1T = SC.coincidence_matrix(1)
        B2T = SC.coincidence_matrix(2)
        assert B1T.shape == tuple(SC.shape[:2][::-1])
        assert B2T.shape == tuple(SC.shape[1:3][::-1])
        # B1T : 0->1
        # B2T : 1->2
        assert np.sum(abs(B2T.dot(B1T))) == 0  # coboundary of coboundary = 0
        # for the last to be true, order of bases must be canonical in both matrices

        row, col, B1T = SC.coincidence_matrix(1, index=True)

        assert (len(col), len(row)) == B1T.shape

        B0 = SC.coincidence_matrix(0)
        assert B0.shape[0] == len(SC.skeleton(0))

    def test_is_triangular_mesh(self):
        """Test is_triangular_mesh."""
        SC = stanford_bunny("simplicial")
        print("dim", SC.dim)
        print(list(SC.get_all_maximal_simplices()))
        assert SC.is_triangular_mesh()

        # test for non triangular mesh
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        assert not SC.is_triangular_mesh()

    def test_to_trimesh(self):
        """Test to_trimesh."""
        SC = stanford_bunny("simplicial")
        trimesh_obj = SC.to_trimesh()
        assert len(trimesh_obj.vertices) == len(SC.skeleton(0))

        # test for non triangular mesh
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3, 4])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        with pytest.raises(RuntimeError):
            SC.to_trimesh()

    @pytest.mark.skipif(
        tm is None, reason="Optional dependency 'spharapy' not installed."
    )
    def test_laplace_beltrami_operator(self):
        """Test laplace_beltrami_operator."""
        SC = stanford_bunny("simplicial")

        laplacian_matrix = SC.laplace_beltrami_operator()

        assert isinstance(laplacian_matrix, np.ndarray)

    @pytest.mark.skipif(
        tm is not None, reason="Optional dependency 'spharapy' installed."
    )
    def test_laplace_beltrami_operator_missing_dependency(self):
        """Test laplace_beltrami_operator for when `spharapy` is missing."""
        SC = stanford_bunny("simplicial")
        with pytest.raises(RuntimeError):
            SC.laplace_beltrami_operator()

    def test_from_nx_graph(self):
        """Test from networkx graph."""
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(3, 4, weight=4)
        SC = SimplicialComplex.from_nx(G)
        assert SC[(1, 2)]["weight"] == 2

    def test_is_connected(self):
        """Test is connected."""
        SC = stanford_bunny("simplicial")
        assert SC.is_connected()

    @pytest.mark.skipif(
        hnx is None, reason="Optional dependency 'hypernetx' not installed."
    )
    def test_simplicial_closure_of_hypergraph(self):
        """Test simplicial_closure_of_hypergraph."""
        hg = hnx.Hypergraph([[1, 2, 3, 4], [1, 2, 3]], static=True)
        sc = SimplicialComplex.simplicial_closure_of_hypergraph(hg)
        expected_simplices = [
            (1,),
            (2,),
            (3,),
            (4,),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (1, 2, 3, 4),
        ]
        assert len(sc.simplices) == len(expected_simplices)

    @pytest.mark.skipif(
        hnx is None, reason="Optional dependency 'hypernetx' not installed."
    )
    def test_to_hypergraph(self):
        """Convert a SimplicialComplex to a hypergraph and compare the number of edges."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        expected_result = hnx.Hypergraph(
            {
                "e0": [1, 2],
                "e1": [1, 3],
                "e2": [1, 4],
                "e3": [2, 3],
                "e4": [2, 4],
                "e5": [2, 5],
                "e6": [1, 2, 3],
                "e7": [1, 2, 4],
            },
            name="",
        )
        result = SC.to_hypergraph()
        assert len(result.edges) == len(expected_result.edges)

    def test_to_cell_complex(self):
        """Test to convert SimplicialComplex to Cell Complex."""
        SC = SimplicialComplex([(1, 2, 3), (1, 2, 4), (2, 5)])
        CC = SC.to_cell_complex()
        assert set(CC.nodes) == {1, 2, 3, 4, 5}
        assert set(CC.edges) == {(2, 1), (3, 1), (2, 3), (2, 4), (1, 4), (2, 5)}

    def test_to_combinatorial_complex(self):
        """Convert a SimplicialComplex to a CombinatorialComplex and compare the number of cells and nodes."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        expected_result = CombinatorialComplex()
        expected_result.add_cell((1, 2, 3), rank=2)
        expected_result.add_cell((1, 2, 4), rank=2)
        expected_result.add_cell((2, 5), rank=1)
        expected_result.add_cell((1, 2), rank=1)
        expected_result.add_cell((1, 3), rank=1)
        expected_result.add_cell((1, 4), rank=1)
        expected_result.add_cell((2, 3), rank=1)
        expected_result.add_cell((2, 4), rank=1)
        expected_result.add_cell((2, 5), rank=1)
        result = SC.to_combinatorial_complex()
        assert len(result.cells) == len(expected_result.cells)
        assert len(result.nodes) == len(expected_result.nodes)

    def test_from_gudhi(self):
        """Create a SimplicialComplex from a Gudhi SimplexTree and compare the number of simplices."""
        gudhi_simplices = [
            ([1, 2, 3, 5], 0.0),
            ([1, 2, 3], 0.0),
            ([1, 2, 5], 0.0),
            ([1, 2], 0.0),
            ([1, 3, 5], 0.0),
            ([1, 3], 0.0),
            ([1, 5], 0.0),
            ([1], 0.0),
            ([2, 3, 5], 0.0),
            ([2, 3], 0.0),
            ([2, 5], 0.0),
            ([2], 0.0),
            ([3, 5], 0.0),
            ([3], 0.0),
            ([5], 0.0),
        ]
        tree = Mock(["get_skeleton", "dimension"])
        tree.get_skeleton.side_effect = lambda i: (
            s for s in gudhi_simplices if len(s[0]) <= i + 1
        )
        tree.dimension.return_value = 3

        expected_result = SimplicialComplex()
        expected_result.add_simplex((1, 2, 3, 5))
        result = SimplicialComplex.from_gudhi(tree)
        assert len(result.simplices) == len(expected_result.simplices)

    def test_add_elements_from_nx_graph(self):
        """Add elements from a networkx graph to a SimplicialComplex and compare the number of simplices."""
        c1 = Simplex((1, 2, 3))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c3])
        G = nx.Graph()
        G.add_edge(4, 5)
        expected_result = SimplicialComplex([c1, c3, Simplex((4, 5))])
        SC.add_elements_from_nx_graph(G)
        assert len(SC.simplices) == len(expected_result.simplices)

    def test_restrict_to_nodes(self):
        """Restrict a SimplicialComplex to the specified nodes and compare the result with the expected SimplicialComplex."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        node_set = [1, 2, 3, 4]
        expected_result = SimplicialComplex(
            [
                Simplex((1,)),
                Simplex((2,)),
                Simplex((3,)),
                Simplex((4,)),
                Simplex((1, 2)),
                Simplex((1, 3)),
                Simplex((1, 4)),
                Simplex((2, 3)),
                Simplex((2, 4)),
                Simplex((1, 2, 3)),
                Simplex((1, 2, 4)),
            ]
        )
        result = SC.restrict_to_nodes(node_set)
        assert len(result.simplices) == len(expected_result.simplices)

    def test_get_all_maximal_simplices(self):
        """Retrieve all maximal simplices from a SimplicialComplex and compare the number of simplices."""
        SC = SimplicialComplex([(1, 2)])
        assert {simplex.elements for simplex in SC.get_all_maximal_simplices()} == {
            (1, 2)
        }

        SC = SimplicialComplex([(1, 2, 3), (1, 2, 4), (1, 2, 5)])
        result = list(SC.get_all_maximal_simplices())
        assert len(result) == 3

    def test_coincidence_matrix(self):
        """Test for coincidence matrix."""
        SC = SimplicialComplex()
        SC.add_simplex([0, 1, 2])

        row, col, B1 = SC.coincidence_matrix(1, index=True)

        assert B1.shape == (3, 3)
        assert np.allclose(
            B1.toarray(),
            np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]]),
        )

        B2 = SC.coincidence_matrix(2)

        assert B2.shape == (1, 3)
        assert np.allclose(B2.toarray(), np.array([[1.0, -1.0, 1.0]]))

    def test_down_laplacian_matrix(self):
        """Test the down_laplacian_matrix method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])

        # Test case 1: Rank is within valid range
        result = SC.down_laplacian_matrix(rank=1, signed=True, weight=None, index=False)
        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        with pytest.raises(ValueError):
            SC.down_laplacian_matrix(rank=0, signed=True, weight=None, index=False)

        # Test case 3: Rank is above the valid range
        with pytest.raises(ValueError):
            SC.down_laplacian_matrix(rank=5, signed=True, weight=None, index=False)

    def test_adjacency_matrix2(self):
        """Test the adjacency_matrix method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])

        # Test case 1: Rank is within valid range
        result = SC.adjacency_matrix(rank=1, signed=True, weight=None, index=False)

        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        with pytest.raises(ValueError):
            SC.adjacency_matrix(rank=-1, signed=False, weight=None, index=False)

        # Test case 3: Rank is above the valid range
        with pytest.raises(ValueError):
            SC.adjacency_matrix(rank=5, signed=False, weight=None, index=False)

    def test_dirac_operator_matrix(self):
        """Test dirac operator."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([1, 2, 4])
        SC.add_simplex([3, 4, 8])
        m = SC.dirac_operator_matrix()
        size = sum(SC.shape)
        assert m.shape == (size, size)

        index, m = SC.dirac_operator_matrix(index=True)

        L = m.dot(m)

        check_L = bmat(
            [
                [SC.hodge_laplacian_matrix(0), None, None],
                [None, SC.hodge_laplacian_matrix(1), None],
                [None, None, SC.hodge_laplacian_matrix(2)],
            ]
        )

        assert np.linalg.norm((check_L - L).todense()) == 0

        assert (1,) in index
        assert (2,) in index
        assert (3,) in index
        assert (4,) in index
        assert len(index) == size

        index, m = SC.dirac_operator_matrix(index=True, signed=False)

        assert np.prod(m.todense() >= 0) == 1

        m = SC.dirac_operator_matrix(signed=False)

        assert np.prod(m.todense() >= 0) == 1

    def test_coadjacency_matrix(self):
        """Test the coadjacency_matrix method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3])
        SC.add_simplex([4, 5, 6])
        # Test case 1: Rank is within valid range
        rank = 1
        signed = False
        weight = None
        index = False

        result = SC.coadjacency_matrix(rank, signed, weight, index)

        # Assert the result is of type scipy.sparse.csr.csr_matrix
        assert result.shape == (6, 6)

        # Test case 2: Rank is below the valid range
        rank = 0

        with pytest.raises(ValueError):
            SC.coadjacency_matrix(rank, signed, weight, index)

        # Test case 3: Rank is above the valid range
        rank = 5

        with pytest.raises(ValueError):
            SC.coadjacency_matrix(rank, signed, weight, index)

        # Test case 4: index is True
        # Test case 1: Rank is within valid range
        rank = 1
        signed = False
        weight = None
        index = True

        ind, result = SC.coadjacency_matrix(rank, signed, weight, index)

        # Assert the result is of type scipy.sparse.csr.csr_matrix
        assert result.shape == (6, 6)

    def test_restrict_to_simplices(self):
        """Test the restrict_to_simplices method of SimplicialComplex."""
        c1 = Simplex((1, 2, 3))
        c2 = Simplex((1, 2, 4))
        c3 = Simplex((1, 2, 5))
        SC = SimplicialComplex([c1, c2, c3])
        SC1 = SC.restrict_to_simplices([c1, (2, 4)])
        assert len(SC1.simplices) == 9
        assert Simplex((1, 2, 3)) in SC1.simplices
        assert Simplex((2, 4)) in SC1.simplices
        assert Simplex((1, 2)) in SC1.simplices
        assert Simplex((1, 3)) in SC1.simplices
        assert Simplex((2, 3)) in SC1.simplices
        assert Simplex((1,)) in SC1.simplices
        assert Simplex((2,)) in SC1.simplices
        assert Simplex((3,)) in SC1.simplices
        assert Simplex((4,)) in SC1.simplices
        assert Simplex((1, 2, 3, 4)) not in SC1.simplices
        assert Simplex((1, 2, 5)) not in SC1.simplices

    def test_clone(self):
        """Test the clone method of SimplicialComplex."""
        SC = SimplicialComplex()
        SC.add_simplex([1, 2, 3], color="red")
        SC2 = SC.clone()
        assert SC2 is not SC
        assert SC2[(1, 2, 3)] is not SC[(1, 2, 3)]
        SC2.remove_maximal_simplex([1, 2, 3])
        assert 1 in SC
        assert (1, 2, 3) in SC.simplices

    def test_normalized_laplacian_matrix(self):
        """Test the normalized_laplacian_matrix method of SimplicialComplex."""
        SC = SimplicialComplex([[1, 2, 3], [2, 3, 5], [0, 1]])
        L = SC.normalized_laplacian_matrix(rank=1)
        assert np.allclose(
            L.toarray(),
            np.array(
                [
                    [0.5, -0.2236068, -0.2236068, 0.0, 0.0, 0.0],
                    [-0.2236068, 0.59999996, 0.0, 0.0, -0.2236068, 0.0],
                    [-0.2236068, 0.0, 0.59999996, 0.0, 0.0, -0.2236068],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, -0.2236068, 0.0, 0.0, 0.75, 0.0],
                    [0.0, 0.0, -0.2236068, 0.0, 0.0, 0.75],
                ]
            ),
        )

    @pytest.mark.skipif(
        tm is None, reason="Optional dependency 'spharapy' not installed."
    )
    def test_from_spharapy(self):
        """Test the from_spharapy method of SimplicialComplex (support for spharapy trimesh)."""
        mesh = tm.TriMesh(
            [[0, 1, 2]], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        SC = SimplicialComplex.from_spharapy(mesh)
        simplices = SC.simplices
        assert len(simplices) == 7
        assert [0, 1, 2] in simplices
        assert [0, 1] in simplices
        assert [0, 2] in simplices
        assert [1, 2] in simplices
        assert [0] in simplices
        assert [1] in simplices
        assert [2] in simplices

    @pytest.mark.skipif(
        tm is None, reason="Optional dependency 'spharapy' not installed."
    )
    def test_from_spharpy(self):
        """Test the deprecated from_spharpy method of SimplicialComplex (support for spharapy trimesh)."""
        mesh = tm.TriMesh(
            [[0, 1, 2]], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        with pytest.deprecated_call():
            SC = SimplicialComplex.from_spharpy(mesh)
        simplices = SC.simplices
        assert len(simplices) == 7
        assert [0, 1, 2] in simplices
        assert [0, 1] in simplices
        assert [0, 2] in simplices
        assert [1, 2] in simplices
        assert [0] in simplices
        assert [1] in simplices
        assert [2] in simplices

    def test_graph_skeleton(self):
        """Test the graph_skeleton method of SimplicialComplex."""
        SC = SimplicialComplex(
            [
                (2, 6),
                (4, 5),
                (4, 7),
                (5, 6),
                (5, 7),
                (1, 2, 3),
                (2, 3, 4),
            ]
        )
        SC[1]["some_data"] = 1
        SC[(2, 6)]["some_data"] = 42

        G = SC.graph_skeleton()
        assert G.number_of_nodes() == 7
        assert G.number_of_edges() == 10
        assert G.nodes[1]["some_data"] == 1
        assert G.edges[(2, 6)]["some_data"] == 42
