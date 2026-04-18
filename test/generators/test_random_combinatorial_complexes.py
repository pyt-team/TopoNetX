"""Test the random combinatorial complex generators."""

from itertools import combinations

import pytest

from toponetx.generators.random_combinatorial_complexes import (
    random_combinatorial_complex,
    uniform_random_combinatorial_complex,
)


def _cell_set_by_rank(CC, max_rank=10):
    """Return cells grouped by rank as sets of frozensets."""
    cells_by_rank = {}

    for rank in range(max_rank + 1):
        try:
            cells = CC.skeleton(rank)
        except Exception:
            continue

        cells = {frozenset(cell) for cell in cells}
        if cells:
            cells_by_rank[rank] = cells

    return cells_by_rank


def _all_cells_with_ranks(CC, max_rank=10):
    """Return a dictionary mapping each cell to its rank."""
    result = {}

    for rank, cells in _cell_set_by_rank(CC, max_rank=max_rank).items():
        for cell in cells:
            result[cell] = rank

    return result


class TestUniformRandomCombinatorialComplex:
    """Test the `uniform_random_combinatorial_complex` function."""

    def test_zero_probability(self):
        """Test that zero probability only keeps rank-0 cells."""
        CC = uniform_random_combinatorial_complex(
            n=5,
            p=0.0,
            max_rank=3,
            max_cell_size=5,
            seed=0,
        )

        cells_by_rank = _cell_set_by_rank(CC)
        assert 0 in cells_by_rank
        assert len(cells_by_rank[0]) == 5
        assert all(len(cell) == 1 for cell in cells_by_rank[0])

        for rank in cells_by_rank:
            if rank > 0:
                assert len(cells_by_rank[rank]) == 0

    def test_seed_reproducibility(self):
        """Test that the same seed gives the same complex."""
        CC1 = uniform_random_combinatorial_complex(
            n=6,
            p=0.4,
            max_rank=3,
            max_cell_size=4,
            seed=123,
        )
        CC2 = uniform_random_combinatorial_complex(
            n=6,
            p=0.4,
            max_rank=3,
            max_cell_size=4,
            seed=123,
        )

        assert _all_cells_with_ranks(CC1) == _all_cells_with_ranks(CC2)

    def test_invalid_probability(self):
        """Test invalid probability values."""
        with pytest.raises(ValueError):
            uniform_random_combinatorial_complex(n=5, p=-0.1)

        with pytest.raises(ValueError):
            uniform_random_combinatorial_complex(n=5, p=1.1)


class TestRandomCombinatorialComplex:
    """Test the `random_combinatorial_complex` function."""

    def test_rank_zero_cells_always_present(self):
        """Test that all vertices are present as rank-0 cells."""
        n = 7
        CC = random_combinatorial_complex(
            n=n,
            probs=[0.2, 0.2, 0.2],
            max_rank=3,
            max_cell_size=5,
            seed=0,
        )

        cells_by_rank = _cell_set_by_rank(CC)
        assert 0 in cells_by_rank
        assert cells_by_rank[0] == {frozenset([i]) for i in range(n)}

    def test_zero_probabilities(self):
        """Test that zero probabilities produce only rank-0 cells."""
        CC = random_combinatorial_complex(
            n=6,
            probs=[0.0, 0.0, 0.0],
            max_rank=3,
            max_cell_size=6,
            seed=0,
        )

        cells_by_rank = _cell_set_by_rank(CC)
        assert cells_by_rank[0] == {frozenset([i]) for i in range(6)}

        for rank in cells_by_rank:
            if rank > 0:
                assert len(cells_by_rank[rank]) == 0

    def test_max_rank_one_with_probability_one(self):
        """Test that rank-1 with probability one adds all subsets of size >= 2."""
        n = 5
        CC = random_combinatorial_complex(
            n=n,
            probs=[1.0],
            max_rank=1,
            max_cell_size=n,
            seed=0,
        )

        cells_by_rank = _cell_set_by_rank(CC)

        expected_rank_0 = {frozenset([i]) for i in range(n)}
        expected_rank_1 = {
            frozenset(cell)
            for size in range(2, n + 1)
            for cell in combinations(range(n), size)
        }

        assert cells_by_rank[0] == expected_rank_0
        assert cells_by_rank[1] == expected_rank_1

    def test_rank_monotonicity_under_inclusion(self):
        """Test the combinatorial complex monotonicity condition.

        For any cells x, y with x proper subset of y, rank(x) <= rank(y).
        """
        CC = random_combinatorial_complex(
            n=7,
            probs=[0.8, 0.5, 0.3],
            max_rank=3,
            max_cell_size=5,
            seed=1,
        )

        cell_rank = _all_cells_with_ranks(CC)

        for x, rx in cell_rank.items():
            for y, ry in cell_rank.items():
                if x < y:
                    assert rx <= ry

    def test_invalid_probs_sequence_length(self):
        """Test that too-short probability sequences raise an error."""
        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.5],
                max_rank=3,
                max_cell_size=5,
                seed=0,
            )

    def test_invalid_probability_values(self):
        """Test invalid probability values in sequences."""
        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.2, -0.1, 0.3],
                max_rank=3,
                max_cell_size=5,
                seed=0,
            )

        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.2, 1.2, 0.3],
                max_rank=3,
                max_cell_size=5,
                seed=0,
            )

    def test_invalid_n(self):
        """Test invalid number of vertices."""
        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=0,
                probs=[0.5],
                max_rank=1,
                max_cell_size=1,
                seed=0,
            )

    def test_invalid_max_rank(self):
        """Test invalid maximum rank."""
        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.5],
                max_rank=0,
                max_cell_size=5,
                seed=0,
            )

    def test_invalid_max_cell_size(self):
        """Test invalid maximum cell size."""
        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.5],
                max_rank=1,
                max_cell_size=0,
                seed=0,
            )

        with pytest.raises(ValueError):
            random_combinatorial_complex(
                n=5,
                probs=[0.5],
                max_rank=1,
                max_cell_size=6,
                seed=0,
            )
