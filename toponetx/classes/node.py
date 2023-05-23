"""NodeView class."""

try:
    from collections.abc import Hashable, Iterable
except ImportError:
    from collections import Hashable, Iterable

from toponetx import TopoNetXError

__all__ = ["NodeView"]


class NodeView:
    """Node view class."""

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
        """Return string representation of nodes.

        Returns
        -------
        str
        """
        all_nodes = [tuple(j) for j in self.nodes.keys()]

        return f"NodeView({all_nodes})"

    def __getitem__(self, cell):
        """Get item.

        Parameters
        ----------
        cell : tuple list or AbstractCell or Simplex
            A cell.

        Returns
        -------
        dict or list
            Dict of properties associated with that cells.
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
        """Set the attribute of the node."""
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
        """Compute number of nodes."""
        return len(self.nodes)

    def __contains__(self, e):
        """Check if e is in the nodes."""
        if isinstance(e, Hashable) and not isinstance(e, self.cell_type):
            return frozenset({e}) in self.nodes

        elif isinstance(e, self.cell_type):
            return e.nodes in self.nodes

        elif isinstance(e, Iterable):
            if len(e) == 1:
                return frozenset(e) in self.nodes
        else:
            return False
