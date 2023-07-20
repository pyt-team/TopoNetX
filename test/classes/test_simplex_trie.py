"""Tests for the `simplex_trie` module."""

import pytest

from toponetx.classes.simplex_trie import SimplexTrie


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
                for node, expected in zip(nodes, label_to_simplex[depth][label]):
                    assert node.simplex.elements == expected

    def test_iter(self) -> None:
        """Test the iteration of the trie."""
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((2, 3, 4))
        trie.insert((0, 1))

        # We guarantee a specific ordering of the simplices when iterating. Hence, we explicitly compare lists here.
        assert list(map(lambda node: node.simplex.elements, trie)) == [
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
        assert set(map(lambda node: node.simplex.elements, trie.cofaces((1,)))) == {
            (1,),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 2, 3),
            (1, 2, 4),
        }
        assert set(map(lambda node: node.simplex.elements, trie.cofaces((2,)))) == {
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
        assert set(map(lambda node: node.simplex.elements, trie.skeleton(0))) == {
            (1,),
            (2,),
            (3,),
            (4,),
        }
        assert set(map(lambda node: node.simplex.elements, trie.skeleton(1))) == {
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
        }
        assert set(map(lambda node: node.simplex.elements, trie.skeleton(2))) == {
            (1, 2, 3),
            (1, 2, 4),
        }

        with pytest.raises(ValueError):
            _ = next(trie.skeleton(-1))
        with pytest.raises(ValueError):
            _ = next(trie.skeleton(3))
