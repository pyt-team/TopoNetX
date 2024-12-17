"""Tests for the `simplex_trie` module."""

import pytest

from toponetx.classes.simplex import Simplex
from toponetx.classes.simplex_trie import SimplexNode, SimplexTrie


class TestSimplexNode:
    """Tests for the `SimplexNode` class."""

    def test_invalid_root(self) -> None:
        """Test that invalid root nodes are rejected."""
        with pytest.raises(ValueError):
            _ = SimplexNode(label="invalid", parent=None)

    def test_node_len(self) -> None:
        """Test the length of the node."""
        root_node: SimplexNode[int] = SimplexNode(label=None)
        assert len(root_node) == 0

        child_node: SimplexNode[int] = SimplexNode(label=1, parent=root_node)
        assert len(child_node) == 1

        grandchild_node: SimplexNode[int] = SimplexNode(label=2, parent=child_node)
        assert len(grandchild_node) == 2

    def test_node_repr(self) -> None:
        """Test the string representation of the node."""
        root_node = SimplexNode(label=None)
        assert repr(root_node) == "SimplexNode(None, None)"

        child_node = SimplexNode(label=1, parent=root_node)
        assert repr(child_node) == "SimplexNode(1, SimplexNode(None, None))"

    def test_simplex_property(self) -> None:
        """Test the `simplex` property of a node."""
        root_node = SimplexNode(label=None)
        assert root_node.simplex is None

        child_node = SimplexNode(label=1, parent=root_node)
        assert child_node.simplex == Simplex((1,))

        grandchild_node = SimplexNode(label=2, parent=child_node)
        assert grandchild_node.simplex == Simplex((1, 2))


class TestSimplexTrie:
    """Tests for the `SimplexTree` class."""

    def test_insert(self) -> None:
        """Test that the internal data structures of the simplex trie are correct after insertion."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))

        assert trie.shape == [3, 3, 1]

        assert set(trie.root.children.keys()) == {1, 2, 3}
        assert set(trie.root.children[1].children.keys()) == {2, 3}
        assert set(trie.root.children[1].children[2].children.keys()) == {3}
        assert set(trie.root.children[2].children.keys()) == {3}

        # the label list should contain the nodes of each depth according to their label
        label_to_simplex = {
            1: {1: [(1,)], 2: [(2,)], 3: [(3,)]},
            2: {2: [(1, 2)], 3: [(1, 3), (2, 3)]},
            3: {3: [(1, 2, 3)]},
        }

        assert len(trie.label_lists) == len(label_to_simplex)
        for depth, label_list in trie.label_lists.items():
            assert depth in label_to_simplex
            assert len(label_list) == len(label_to_simplex[depth])
            for label, nodes in label_list.items():
                assert len(nodes) == len(label_to_simplex[depth][label])
                for node, expected in zip(
                    nodes, label_to_simplex[depth][label], strict=True
                ):
                    assert node.simplex.elements == expected

    def test_len(self) -> None:
        """Test the `__len__` method of a simplex trie."""
        trie = SimplexTrie()
        assert len(trie) == 0

        trie.insert((1, 2, 3))
        assert len(trie) == 7

    def test_getitem(self) -> None:
        """Test the `__getitem__` method of a simplex trie."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))

        assert trie[(1,)].simplex == Simplex((1,))
        assert trie[(1, 2)].simplex == Simplex((1, 2))
        assert trie[(1, 2, 3)].simplex == Simplex((1, 2, 3))

        with pytest.raises(KeyError):
            _ = trie[(0,)]

    def test_iter(self) -> None:
        """Test the iteration of the trie."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((2, 3, 4))
        trie.insert((0, 1))

        # We guarantee a specific ordering of the simplices when iterating. Hence, we explicitly compare lists here.
        assert [node.simplex.elements for node in trie] == [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (1, 2, 3),
            (2, 3, 4),
        ]

    def test_cofaces(self) -> None:
        """Test the cofaces method."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))

        # no ordering is guaranteed for the cofaces method
        assert {node.simplex.elements for node in trie.cofaces((1,))} == {
            (1,),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 2, 3),
            (1, 2, 4),
        }
        assert {node.simplex.elements for node in trie.cofaces((2,))} == {
            (2,),
            (1, 2),
            (2, 3),
            (2, 4),
            (1, 2, 3),
            (1, 2, 4),
        }

    def test_is_maximal(self) -> None:
        """Test the `is_maximal` method."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))

        assert trie.is_maximal((1, 2, 3))
        assert trie.is_maximal((1, 2, 4))
        assert not trie.is_maximal((1, 2))
        assert not trie.is_maximal((1, 3))
        assert not trie.is_maximal((1, 4))
        assert not trie.is_maximal((2, 3))

        with pytest.raises(ValueError):
            trie.is_maximal((5,))

    def test_skeleton(self) -> None:
        """Test the skeleton method."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))

        # no ordering is guaranteed for the skeleton method
        assert {node.simplex.elements for node in trie.skeleton(0)} == {
            (1,),
            (2,),
            (3,),
            (4,),
        }
        assert {node.simplex.elements for node in trie.skeleton(1)} == {
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
        }
        assert {node.simplex.elements for node in trie.skeleton(2)} == {
            (1, 2, 3),
            (1, 2, 4),
        }

        with pytest.raises(ValueError):
            _ = next(trie.skeleton(-1))
        with pytest.raises(ValueError):
            _ = next(trie.skeleton(3))

    def test_remove_simplex(self) -> None:
        """Test the removal of simplices from the trie."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))

        trie.remove_simplex((1, 2))
        assert len(trie) == 5
