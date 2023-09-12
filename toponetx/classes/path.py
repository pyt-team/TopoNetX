"""Path class."""

from collections.abc import Collection, Hashable
from typing import Any, List, Tuple, Union

from toponetx.classes.complex import Atom

__all__ = ["Path"]


class Path(Atom):
    """
    A class representing a p-path in a path complex, which is the building block for a path complex.

    Unlike the original paper (https://arxiv.org/pdf/1207.2834.pdf) where p-paths span the regular space of boundary-invariant paths,
    our p-paths span the space of simple paths with length p.

    Parameters
    ----------
    elements: Collection
        The nodes in the p-path.
    name : str, optional
        A name for the p-path.
    construct_boundaries : bool, default=False
        If True, construct the entire boundary of the p-path.
    reserve_sequence_order : bool, default=False
        If True, reserve the order of the sub-sequence of nodes in the p-path.
        Else, the sub-sequence of nodes in the p-path will be reversed if the first index is larger than the last index.
    allowed_paths : List[Tuple], optional
        A list of allowed boundaries. If None, all possible boundaries are constructed (similarly to simplex).
    attr: keyword arguments, optional
        Additional attributes to be associated with the p-path.
    """

    def __init__(
        self,
        elements: Union[List, Tuple],
        name: str = "",
        construct_boundaries: bool = False,
        reserve_sequence_order: bool = False,
        allowed_paths: List[Tuple] = None,
        **attr,
    ) -> None:
        self.__check_inputs(elements, reserve_sequence_order)
        super().__init__(tuple(elements), name, **attr)
        if len(set(elements)) != len(self.elements):
            raise ValueError("A p-path cannot contain duplicate nodes.")

        self.construct_boundaries = construct_boundaries
        self.reserve_sequence_order = reserve_sequence_order
        if construct_boundaries:
            self._boundaries = self.construct_path_boundaries(
                elements,
                reserve_sequence_order=reserve_sequence_order,
                allowed_paths=allowed_paths,
            )
        else:
            self._boundaries = list()

    @staticmethod
    def construct_path_boundaries(  # when allowed_paths is not specified, all possible boundaries are constructed similarly to cliques.
        elements: Collection,
        reserve_sequence_order: bool = False,
        allowed_paths: List[Tuple] = None,
    ) -> List[Tuple]:
        """Return list of p-path objects representing the boundaries."""
        boundaries = list()
        for i in range(len(elements)):
            boundary = list(elements[0:i] + elements[(i + 1) :])
            if (
                not reserve_sequence_order
                and len(boundary) > 1
                and boundary[0] > boundary[-1]
            ):
                boundary.reverse()
            if allowed_paths is None or tuple(boundary) in allowed_paths:
                boundaries.append(tuple(boundary))
        return boundaries

    @property
    def boundary(self) -> List[Tuple]:
        """Return list of p-path objects representing the boundaries."""
        return self._boundaries

    def clone(self) -> "Path":
        """Return a copy of the p-path."""
        return Path(
            self.elements,
            name=self.name,
            construct_boundaries=self.construct_boundaries,
            **self._properties,
        )

    def __check_inputs(self, elements: Any, reserve_sequence_order: bool):
        """Sanity check for inputs, as sequence order matters."""
        for i in elements:
            if not isinstance(i, Hashable):
                raise ValueError(f"All nodes of a p-path must be hashable, got {i}")
        if not isinstance(elements, List) and not isinstance(elements, Tuple):
            raise ValueError(
                f"Elements of a p-path must be a list or tuple, got {type(elements)}"
            )
        if (
            not reserve_sequence_order
            and len(elements) > 1
            and elements[0] > elements[-1]
        ):
            raise ValueError(
                "A p-path must have the first index smaller than the last index, got {}".format(
                    elements
                )
            )

    def __repr__(self) -> str:
        """Return string representation of p-paths."""
        return f"Path({self.elements})"

    def __str__(self) -> str:
        """Return string representation of p-paths."""
        return f"Node set: {self.elements}, Boundaries: {self.boundary}, Attributes: {self._properties}"
