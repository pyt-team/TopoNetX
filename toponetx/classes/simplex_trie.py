"""
Implementation of a simplex trie datastructure for simplicial complexes as presented in [1]_.

This module is intended for internal use by the `SimplicialComplex` class only. Any direct interactions with this
module or its classes may break at any time. In particular, this also means that the `SimplicialComplex` class must not
leak any object references to the trie or its nodes.

Some implementation details:
- Inside this module, simplices are represented as ordered sequences with unique elements. It is expected that all
  inputs from outside are already pre-processed and ordered accordingly. This is not checked and the behavior is
  undefined if this is not the case.

References
----------
.. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient Data Structure for General Simplicial
       Complexes. Algorithmica, pages 1-22, 2014
"""

from collections.abc import Generator, Hashable, Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar

from toponetx.classes.simplex import Simplex
from toponetx.utils.iterable import is_ordered_subset

__all__ = ["SimplexNode", "SimplexTrie"]

ElementType = TypeVar("ElementType", bound=Hashable)


class SimplexNode(Generic[ElementType]):
    """Node in a simplex tree.

    Parameters
    ----------
    label : ElementType or None
        The label of the node. May only be `None` for the root node.
    parent : SimplexNode, optional
        The parent node of this node. If `None`, this node is the root node.
    """

    label: ElementType | None
    elements: tuple[ElementType, ...]
    attributes: dict[Hashable, Any]

    depth: int
    parent: "SimplexNode | None"
    children: dict[ElementType, "SimplexNode[ElementType]"]

    def __init__(
        self,
        label: ElementType | None,
        parent: "SimplexNode[ElementType] | None" = None,
    ) -> None:
        """Node in a simplex tree.

        Parameters
        ----------
        label : ElementType or None
            The label of the node. May only be `None` for the root node.
        parent : SimplexNode, optional
            The parent node of this node. If `None`, this node is the root node.
        """
        self.label = label
        self.attributes = {}

        self.children = {}

        self.parent = parent
        if parent is not None:
            parent.children[label] = self
            self.elements = (*parent.elements, label)
            self.depth = parent.depth + 1
        else:
            if label is not None:
                raise ValueError("Root node must have label `None`.")
            self.elements = ()
            self.depth = 0

    def __len__(self) -> int:
        """Return the number of elements in this trie node.

        Returns
        -------
        int
            Number of elements in this trie node.
        """
        return len(self.elements)

    def __repr__(self) -> str:
        """Return a string representation of this trie node.

        Returns
        -------
        str
            A string representation of this trie node.
        """
        return f"SimplexNode({self.label}, {self.parent!r})"

    @property
    def simplex(self) -> Simplex[ElementType] | None:
        """Return a `Simplex` object representing this node.

        Returns
        -------
        Simplex or None
            A `Simplex` object representing this node, or `None` if this node is the root node.
        """
        if self.label is None:
            return None
        simplex = Simplex(self.elements, construct_tree=False)
        simplex._attributes = self.attributes
        return simplex

    def iter_all(self) -> Generator["SimplexNode[ElementType]", None, None]:
        """Iterate over all nodes in the subtree rooted at this node.

        Ordering is according to breadth-first search, i.e., simplices are yielded in increasing order of dimension and
        increasing order of labels within each dimension.

        Yields
        ------
        SimplexNode
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.label is not None:
                # skip root node
                yield node
            queue += [node.children[label] for label in sorted(node.children.keys())]


class SimplexTrie(Generic[ElementType]):
    """
    Simplex tree data structure as presented in [1]_.

    This class is intended for internal use by the `SimplicialComplex` class only. Any
    direct interactions with this class may break at any time.

    References
    ----------
    .. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient
           Data Structure for General Simplicial Complexes. Algorithmica, pages 1-22, 2014
    """

    root: SimplexNode[ElementType]
    label_lists: dict[int, dict[ElementType, list[SimplexNode[ElementType]]]]
    shape: list[int]

    def __init__(self) -> None:
        """Simplex tree data structure as presented in [1]_.

        This class is intended for internal use by the `SimplicialComplex` class only.
        Any direct interactions with this class may break at any time.

        References
        ----------
        .. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient
               Data Structure for General Simplicial Complexes. Algorithmica, pages 1-22, 2014
        """
        self.root = SimplexNode(None)
        self.label_lists = {}
        self.shape = []

    def __len__(self) -> int:
        """Return the number of simplices in the trie.

        Returns
        -------
        int
            Number of simplices in the trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> len(trie)
        7
        """
        return sum(self.shape)

    def __contains__(self, item: Iterable[ElementType]) -> bool:
        """Check if a simplex is contained in this trie.

        Parameters
        ----------
        item : Iterable of ElementType
            The simplex to check for. Must be ordered and contain unique elements.

        Returns
        -------
        bool
            Whether the given simplex is contained in this trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> (1, 2, 3) in trie
        True
        >>> (1, 2, 4) in trie
        False
        """
        return self.find(item) is not None

    def __getitem__(self, item: Iterable[ElementType]) -> SimplexNode[ElementType]:
        """Return the simplex node for a given simplex.

        Parameters
        ----------
        item : Iterable of ElementType
            The simplex to return the node for. Must be ordered and contain only unique
            elements.

        Returns
        -------
        SimplexNode
            The trie node that represents the given simplex.

        Raises
        ------
        KeyError
            If the given simplex does not exist in this trie.
        """
        node = self.find(item)
        if node is None:
            raise KeyError(f"Simplex {item} not found in trie.")
        return node

    def __iter__(self) -> Iterator[SimplexNode[ElementType]]:
        """Iterate over all simplices in the trie.

        Simplices are ordered by increasing dimension and increasing order of labels within each dimension.

        Yields
        ------
        tuple of ElementType
        """
        yield from self.root.iter_all()

    def insert(self, item: Sequence[ElementType], **kwargs) -> None:
        """Insert a simplex into the trie.

        Any lower-dimensional simplices that do not exist in the trie are also inserted
        to fulfill the simplex property. If the simplex already exists, its properties
        are updated.

        Parameters
        ----------
        item : Sequence of ElementType
            The simplex to insert. Must be ordered and contain only unique elements.
        **kwargs : keyword arguments, optional
            Properties associated with the given simplex.
        """
        self._insert_helper(self.root, item)
        self.find(item).attributes.update(kwargs)

    # def insert_raw(self, simplex: Sequence[ElementType], **kwargs) -> None:
    #     """Insert a simplex into the trie without guaranteeing the simplex property.

    #     Sub-simplices are not guaranteed to be inserted, which may break the simplex
    #     property. This allows for faster insertion of a sequence of simplices without
    #     considering their order. In case not all sub-simplices are inserted manually,
    #     the simplex property must be restored by calling `restore_simplex_property`.

    #     Parameters
    #     ----------
    #     simplex : Sequence of ElementType
    #         The simplex to insert. Must be ordered and contain only unique elements.
    #     **kwargs : keyword arguments, optional
    #         Properties associated with the simplex.

    #     See Also
    #     --------
    #     restore_simplex_property
    #         Function to restore the simplex property after using `insert_raw`.
    #     """
    #     current_node = self.root
    #     for label in simplex:
    #         if label in current_node.children:
    #             current_node = current_node.children[label]
    #         else:
    #             current_node = self._insert_child(current_node, label)
    #     current_node.attributes.update(kwargs)

    def _insert_helper(
        self, subtree: SimplexNode[ElementType], items: Sequence[ElementType]
    ) -> None:
        """Insert a simplex node under the given subtree.

        Parameters
        ----------
        subtree : SimplexNode
            The subtree to insert the simplex node under.
        items : Sequence
            The (partial) simplex to insert under the subtree.
        """
        for i, label in enumerate(items):
            if label not in subtree.children:
                self._insert_child(subtree, label)
            self._insert_helper(subtree.children[label], items[i + 1 :])

    def _insert_child(
        self, parent: SimplexNode[ElementType], label: ElementType
    ) -> SimplexNode[ElementType]:
        """Insert a child node with a given label.

        Parameters
        ----------
        parent : SimplexNode
            The parent node.
        label : ElementType
            The label of the child node.

        Returns
        -------
        SimplexNode
            The new child node.
        """
        node = SimplexNode(label, parent=parent)

        # update label lists
        if node.depth not in self.label_lists:
            self.label_lists[node.depth] = {}
        if label in self.label_lists[node.depth]:
            self.label_lists[node.depth][label].append(node)
        else:
            self.label_lists[node.depth][label] = [node]

        # update shape property
        if node.depth > len(self.shape):
            self.shape += [0]
        self.shape[node.depth - 1] += 1

        return node

    # def restore_simplex_property(self) -> None:
    #     """Restore the simplex property after using `insert_raw`."""
    #     all_simplices = set()
    #     for node in self.root.iter_all():
    #         if len(node.children) == 0:
    #             for r in range(1, len(node.elements)):
    #                 all_simplices.update(combinations(node.elements, r))

    #     for simplex in all_simplices:
    #         self.insert_raw(simplex)

    def find(self, search: Iterable[ElementType]) -> SimplexNode[ElementType] | None:
        """Find the node in the trie that matches the search.

        Parameters
        ----------
        search : Iterable of ElementType
            The simplex to search for. Must be ordered and contain only unique elements.

        Returns
        -------
        SimplexNode or None
            The node that matches the search, or `None` if no such node exists.
        """
        node = self.root
        for item in search:
            if item not in node.children:
                return None
            node = node.children[item]
        return node

    def cofaces(
        self, simplex: Sequence[ElementType]
    ) -> Generator[SimplexNode[ElementType], None, None]:
        """Return the cofaces of the given simplex.

        No ordering is guaranteed by this method.

        Parameters
        ----------
        simplex : Sequence of ElementType
            The simplex to find the cofaces of. Must be ordered and contain only unique elements.

        Yields
        ------
        SimplexNode
            The cofaces of the given simplex, including the simplex itself.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> trie.insert((1, 2, 4))
        >>> sorted(map(lambda node: node.elements, trie.cofaces((1,))))
        [(1,), (1, 2), (1, 2, 3), (1, 2, 4), (1, 3), (1, 4)]
        """
        # Find simplices of the form [*, l_1, *, l_2, ..., *, l_n], i.e. simplices that contain all elements of the
        # given simplex plus some additional elements, but sharing the same largest element. This can be done by the
        # label lists.
        simplex_nodes = self._coface_roots(simplex)

        # Found all simplices of the form [*, l_1, *, l_2, ..., *, l_n] in the simplex trie. All nodes in the subtrees
        # rooted at these nodes are cofaces of the given simplex.
        for simplex_node in simplex_nodes:
            yield from simplex_node.iter_all()

    def _coface_roots(
        self, simplex: Sequence[ElementType]
    ) -> Generator[SimplexNode[ElementType], None, None]:
        """Return the roots of the coface subtrees.

        A coface subtree is a subtree of the trie whose simplices are all cofaces of a
        given simplex.

        Parameters
        ----------
        simplex : Sequence of ElementType
            The simplex to find the cofaces of. Must be ordered and contain only unique
            elements.

        Yields
        ------
        SimplexNode
            The trie nodes that are roots of the coface subtrees.
        """
        for depth in range(len(simplex), len(self.shape) + 1):
            if simplex[-1] not in self.label_lists[depth]:
                continue
            for simplex_node in self.label_lists[depth][simplex[-1]]:
                if is_ordered_subset(simplex, simplex_node.elements):
                    yield simplex_node

    def is_maximal(self, simplex: tuple[ElementType, ...]) -> bool:
        """Return True if the given simplex is maximal.

        A simplex is maximal if it has no cofaces.

        Parameters
        ----------
        simplex : tuple of ElementType or SimplexNode
            The simplex to check. Must be ordered and contain only unique elements.

        Returns
        -------
        bool
            True if the given simplex is maximal, False otherwise.

        Raises
        ------
        ValueError
            If the simplex does not exist in the trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> trie.is_maximal((1, 2, 3))
        True
        >>> trie.is_maximal((1, 2))
        False
        """
        if simplex not in self:
            raise ValueError(f"Simplex {simplex} does not exist.")

        gen = self._coface_roots(simplex)
        try:
            first_next = next(gen)
            # sometimes the first root is the simplex itself
            if first_next.elements == simplex:
                if len(first_next.children) > 0:
                    return False
                next(gen)
            return False
        except StopIteration:
            return True

    def skeleton(self, rank: int) -> Generator[SimplexNode, None, None]:
        """Return the simplices of the given rank.

        No particular ordering is guaranteed and is dependent on insertion order.

        Parameters
        ----------
        rank : int
            The rank of the simplices to return.

        Yields
        ------
        SimplexNode
            The simplices of the given rank.

        Raises
        ------
        ValueError
            If the given rank is negative or exceeds the maximum rank of the trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> sorted(map(lambda node: node.elements, trie.skeleton(0)))
        [(1,), (2,), (3,)]
        >>> sorted(map(lambda node: node.elements, trie.skeleton(1)))
        [(1, 2), (1, 3), (2, 3)]
        >>> sorted(map(lambda node: node.elements, trie.skeleton(2)))
        [(1, 2, 3)]
        """
        if rank < 0:
            raise ValueError(f"`rank` must be a positive integer, got {rank}.")
        if rank >= len(self.shape):
            raise ValueError(f"`rank` {rank} exceeds maximum rank {len(self.shape)}.")

        for nodes in self.label_lists[rank + 1].values():
            yield from nodes

    def remove_simplex(self, simplex: Sequence[ElementType]) -> None:
        """Remove the given simplex and all its cofaces from the trie.

        This method ensures that the simplicial property is maintained by removing any simplex that is no longer valid
        after removing the given simplex.

        Parameters
        ----------
        simplex : Sequence of ElementType
            The simplex to remove. Must be ordered and contain only unique elements.

        Raises
        ------
        ValueError
            If the simplex does not exist in the trie.
        """
        # Locate all roots of subtrees containing cofaces of this simplex. They are invalid after removal of the given
        # simplex and thus need to be removed as well.
        coface_roots = self._coface_roots(simplex)
        for subtree_root in coface_roots:
            for node in subtree_root.iter_all():
                self.shape[node.depth - 1] -= 1
                self.label_lists[node.depth][node.label].remove(node)

            # Detach the subtree from the trie. This effectively destroys the subtree as no reference exists anymore,
            # and garbage collection will take care of the rest.
            subtree_root.parent.children.pop(subtree_root.label)

        # After removal of some simplices, the maximal dimension may have decreased. Make sure that there are no empty
        # shape entries at the end of the shape list.
        while len(self.shape) > 0 and self.shape[-1] == 0:
            self.shape.pop()
