"""Generators for random combinatorial complexes."""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import combinations

import networkx as nx

from toponetx.classes import CombinatorialComplex

__all__ = [
    "random_combinatorial_complex",
    "uniform_random_combinatorial_complex",
]


def _normalize_probabilities(
    probs: float | Sequence[float],
) -> list[float]:
    """Normalize probability input.

    Parameters
    ----------
    probs : float or sequence of float
        If a float, the same probability is used for every positive rank.
        If a sequence, entry ``probs[r - 1]`` is used for rank ``r``.

    Returns
    -------
    list[float]
        Probabilities indexed by positive rank minus one.
    """
    if isinstance(probs, float):
        if not 0.0 <= probs <= 1.0:
            raise ValueError("Probability must lie in [0, 1].")
        return [probs]

    probs = list(probs)
    if len(probs) == 0:
        raise ValueError("`probs` must be a float or a non-empty sequence.")
    if any(p < 0.0 or p > 1.0 for p in probs):
        raise ValueError("All probabilities must lie in [0, 1].")
    return probs


def _is_compatible(
    cell: frozenset[int],
    rank: int,
    existing_cells: dict[frozenset[int], int],
) -> bool:
    """Check whether adding ``cell`` at ``rank`` preserves CC monotonicity.

    Since cells are added in increasing rank order, the only possible violation is:
    an already existing lower-rank cell strictly contains the candidate cell.

    Parameters
    ----------
    cell : frozenset[int]
        Candidate cell.
    rank : int
        Proposed rank.
    existing_cells : dict[frozenset[int], int]
        Mapping from cells to their assigned ranks.

    Returns
    -------
    bool
        True if the cell can be added without violating
        ``x subset y => rk(x) <= rk(y)``.
    """
    if cell in existing_cells:
        return False

    for other_cell, other_rank in existing_cells.items():
        if cell < other_cell and rank > other_rank:
            return False

    return True


def random_combinatorial_complex(
    n: int,
    probs: float | Sequence[float],
    max_rank: int = 3,
    max_cell_size: int | None = None,
    seed: int | random.Random | None = None,
) -> CombinatorialComplex:
    """Generate a random combinatorial complex.

    The construction is intentionally more general than the simplicial case.
    It starts with rank-0 cells, then samples higher-rank cells from subsets
    of the vertex set while enforcing the combinatorial-complex monotonicity rule:
    if ``x`` is contained in ``y``, then ``rank(x) <= rank(y)``.

    Parameters
    ----------
    n : int
        Number of vertices.
    probs : float or sequence of float
        Sampling probabilities for positive ranks.
        If a float ``p``, the same probability is used for every rank
        ``1, ..., max_rank``.
        If a sequence, ``probs[r - 1]`` is used for rank ``r``.
    max_rank : int, default=3
        Maximum positive rank to generate.
    max_cell_size : int, optional
        Maximum size of sampled cells. If ``None``, use ``n``.
    seed : int, random.Random, or None, optional
        Random seed/state.

    Returns
    -------
    CombinatorialComplex
        A random combinatorial complex.

    Notes
    -----
    This is an exhaustive generator over all subsets up to ``max_cell_size``,
    so it is intended for small ``n``. For larger problems, a sparse sampler
    would be better.

    Examples
    --------
    >>> cc = random_combinatorial_complex(
    ...     n=6,
    ...     probs=[0.5, 0.25, 0.1],
    ...     max_rank=3,
    ...     max_cell_size=4,
    ...     seed=0,
    ... )
    """
    if n <= 0:
        raise ValueError("`n` must be positive.")
    if max_rank < 1:
        raise ValueError("`max_rank` must be at least 1.")

    prob_list = _normalize_probabilities(probs)
    if len(prob_list) == 1:
        prob_list = prob_list * max_rank
    elif len(prob_list) < max_rank:
        raise ValueError(
            "If `probs` is a sequence, it must have length at least `max_rank`."
        )

    if max_cell_size is None:
        max_cell_size = n
    if max_cell_size < 1 or max_cell_size > n:
        raise ValueError("`max_cell_size` must satisfy 1 <= max_cell_size <= n.")

    rng = nx.utils.create_py_random_state(seed)

    cc = CombinatorialComplex()
    existing_cells: dict[frozenset[int], int] = {}

    # Add rank-0 cells explicitly.
    for v in range(n):
        cell = frozenset([v])
        cc.add_cell([v], rank=0)
        existing_cells[cell] = 0

    vertices = list(range(n))

    # Add higher-rank cells in increasing rank order.
    for rank in range(1, max_rank + 1):
        p = prob_list[rank - 1]

        for size in range(2, max_cell_size + 1):
            for cell_tuple in combinations(vertices, size):
                cell = frozenset(cell_tuple)

                if not _is_compatible(cell, rank, existing_cells):
                    continue

                if rng.random() < p:
                    cc.add_cell(cell_tuple, rank=rank)
                    existing_cells[cell] = rank

    return cc


def uniform_random_combinatorial_complex(
    n: int,
    p: float,
    max_rank: int = 3,
    max_cell_size: int | None = None,
    seed: int | random.Random | None = None,
) -> CombinatorialComplex:
    """Generate a random combinatorial complex with one probability for all ranks.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Sampling probability for every positive rank.
    max_rank : int, default=3
        Maximum positive rank to generate.
    max_cell_size : int, optional
        Maximum sampled cell size. If ``None``, use ``n``.
    seed : int, random.Random, or None, optional
        Random seed/state.

    Returns
    -------
    CombinatorialComplex
        A random combinatorial complex.
    """
    return random_combinatorial_complex(
        n=n,
        probs=p,
        max_rank=max_rank,
        max_cell_size=max_cell_size,
        seed=seed,
    )
