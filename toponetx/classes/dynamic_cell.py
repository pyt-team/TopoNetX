"""DynamicCell class."""

from toponetx.classes.ranked_entity import RankedEntity
from toponetx.exception import TopoNetXError

__all__ = ["DynamicCell"]


class DynamicCell(RankedEntity):
    """Class for Dynamic Cell.

    Base class for objects used in building ranked cell objects including cell complexes,
    simplicial complexes and combinatorial complexes.

    Parameters
    ----------
    elements : list or dict, optional, default: None
        a list of entities with identifiers different than uid and/or
        hashables different than uid, see `Honor System`_
    rankedentity : RankedEntity
        a RankedEntity object to be cloned into a new RankedEntity with uid. If the uid is the same as
        RankedEntity.uid then the entities will not be distinguishable and error will be raised.
        The `elements` in the signature will be added to the cloned Rankedentity.
        The rank of the new ranked entity will be the max of the cloned ranked entity and the input rank.
    safe_insert : bool, check for the combinatorial complex condition when constructing the cell
    weight : float, optional, default : 1
    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----

    This class inherets from RankedEntity
    the same properties but it does autmatically assigns a uid to the initiated cell.

    A dynamic cell is completely determined by its elements, a list of iterables
    or other dynamic cells.
    A dynamic cell may contain other cells, in a nested fashion.

    Examples
    --------
    >>> x1 = Node(1)
    >>> x2 = Node(2)
    >>> x3 = Node(3)
    >>> x4 = Node(4)
    >>> x5 = Node(5)
    >>> y1 = DynamicCell([x1,x2], rank = 1 )
    >>> y2 = DynamicCell([x2,x3], rank = 1)
    >>> y3 = DynamicCell([x3,x4], rank = 1)
    >>> y4 = DynamicCell([x4,x1], rank = 1)
    >>> y5 = DynamicCell([x4,x5], rank = 1)
    >>> y6 = DynamicCell([x4,x5], rank = 1)
    >>> z = DynamicCell([y6,x2,x3,x4],rank = 2)
    """

    def __init__(
        self,
        elements=[],
        rank=None,
        rankedentity=None,
        weight=1.0,
        safe_insert=True,
        **props,
    ):
        if not isinstance(rank, int):
            raise TopoNetXError(f"rank's type must be integer, got f{type(rank)} ")
        if rank <= 0:
            raise TopoNetXError(
                f"ObjectCell must have rank larger than or equal to 1, got {rank}"
            )
        super().__init__(
            uid=None,
            elements=elements,
            rank=rank,
            rankedentity=rankedentity,
            safe_insert=safe_insert,
            weight=weight,
            **props,
        )

    def __repr__(self):
        """Returns a string resembling the constructor for ranked entity"""
        return f"DynamicCell({self._uid},elements={list(self.uidset)},rank={self.rank}, {self.properties})"
