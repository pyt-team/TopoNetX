"""Module with iterable-related utility functions."""

from collections.abc import Sequence
from typing import TypeVar

__all__ = ["is_ordered_subset"]

T = TypeVar("T")


def is_ordered_subset(one: Sequence[T], other: Sequence[T]) -> bool:
    """Return True if the first iterable is a subset of the second iterable.

    This method is specifically optimized for ordered iterables to use return as early as possible for non-subsets.

    Parameters
    ----------
    one : Sequence
        The first iterable.
    other : Sequence
        The second iterable.

    Returns
    -------
    bool
        True if the first iterable is a subset of the second iterable, False otherwise.

    Examples
    --------
    >>> is_ordered_subset((2,), (1, 2))
    True
    >>> is_ordered_subset((1, 2), (1, 2, 3))
    True
    >>> is_ordered_subset((1, 2, 3), (1, 2, 3))
    True
    >>> is_ordered_subset((1, 2, 3), (1, 2))
    False
    >>> is_ordered_subset((1, 2, 3), (1, 2, 4))
    False
    """
    index = 0
    for item in one:
        while (
            index < len(other)
            and isinstance(item, type(other[index]))
            and other[index] < item
        ):
            index += 1
        if index >= len(other) or other[index] != item:
            return False
    return True
