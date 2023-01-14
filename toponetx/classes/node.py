"""Node and NodeView classes."""

try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Hashable, Iterable

from toponetx import TopoNetXError
from toponetx.classes.ranked_entity import RankedEntity

__all__ = ["Node", "NodeView"]


class Node(RankedEntity):
    def __init__(self, elements, weight=1.0, **props):
        if not isinstance(elements, Hashable):
            raise TopoNetXError(
                f"node's elements must be hashable, got {type(elements)}"
            )
        super().__init__(uid=elements, elements=[], rank=0, weight=weight, **props)

    def __repr__(self):
        """Returns a string resembling the constructor for ranked entity"""
        return f"Node({self._uid},rank={self.rank}, {self.properties})"


class NodeView:

    """
    >>> CC = AbstractCellView()
    >>> CC.add_cell( (1,2,3,4),rank=1 )
    >>> NV = NodeView(CC.cell_dict)

    """

    def __init__(self, objectdict, cell_type, name=None):
        if name is None:
            self.name = "_"
        else:
            self.name = name
        if len(objectdict) != 0:
            self.nodes = objectdict[0]
        else:
            self.nodes = {}

        if cell_type is None:
            raise ValueError("cell_type cannot be None")

        self.cell_type = cell_type

    def __repr__(self):
        """C
        String representation of nodes
        Returns
        -------
        str
        """

        all_nodes = [tuple(j) for j in self.nodes.keys()]

        return f"NodeView({all_nodes})"

    def __getitem__(self, cell):
        """
        Parameters
        ----------
        cell : tuple list or AbstractCell or Simplex
            DESCRIPTION.
        Returns
        -------
        TYPE : dict or list or dicts
            return dict of properties associated with that cells
        """

        if isinstance(cell, self.cell_type):
            if cell.nodes in self.nodes:
                return self.nodes[cell.nodes]
        elif isinstance(cell, Iterable):
            cell = frozenset(cell)
            if cell in self.nodes:
                return self.nodes[cell]
            else:
                raise KeyError(f"input {cell} is not in the node set of the complex")

        elif isinstance(cell, Hashable):

            if cell in self:

                return self.nodes[frozenset({cell})]

    def __setitem__(self, cell, **attr):
        if cell in self:
            if isinstance(cell, self.cell_type):
                if cell.nodes in self.nodes:
                    self.nodes.update(attr)
            elif isinstance(cell, Iterable):
                cell = frozenset(cell)
                if cell in self.nodes:
                    self.nodes.update(attr)
                else:
                    raise KeyError(f"node {cell} is not in complex")
            elif isinstance(cell, Hashable):
                if frozenset({cell}) in self:
                    self.nodes.update(attr)
        else:
            raise KeyError(f"node  {cell} is not in the complex")

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, e):

        if isinstance(e, Hashable) and not isinstance(e, self.cell_type):
            return frozenset({e}) in self.nodes

        elif isinstance(e, self.cell_type):
            return e.nodes in self.nodes

        elif isinstance(e, Iterable):
            if len(e) == 1:
                return frozenset(e) in self.nodes
        else:
            return False
