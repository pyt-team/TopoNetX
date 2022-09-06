"""
Simplex and SimplexView Classes

"""
from collections import Counter, Hashable, OrderedDict, defaultdict, deque
from collections.abc import Iterable
from itertools import combinations
from warnings import warn

try:
    from gudhi import SimplexTree
except ImportError:
    warn(
        "gudhi library is not installed."
        + " Default computing protocol will be set for 'normal'.\n"
        + " gudhi can be installed using: 'pip install gudhi'",
        stacklevel=2,
    )


__all__ = ["Simplex", "SimplexView", "NodeView"]


class Simplex:
    """A simplex class.
    Parameters
    ==========

    elements: any iterable of hashables.

    name : str

    Examples
        >>> simplex1 = Simplex ( (1,2,3) )
        >>> simplex12 = Simplex ( (1,2,4,5),weight = 1 )
        >>> simplex13 = Simplex ( ("a","b","c") )
    """

    def __init__(self, elements, name=None, construct_tree=True, **attr):

        if name is None:
            self.name = ""
        else:
            self.name = name
        self.construct_tree = construct_tree
        self.nodes = frozenset(elements)
        if len(self.nodes) != len(elements):
            raise ValueError("a simplex cannot contain duplicate nodes")

        if construct_tree:
            self._faces = self.construct_simplex_tree(elements)
        else:
            self._faces = frozenset()
        self.properties = dict()
        self.properties.update(attr)

    @staticmethod
    def construct_simplex_tree(elements):
        faceset = set()
        numnodes = len(elements)
        for r in range(numnodes, 0, -1):
            for face in combinations(elements, r):
                faceset.add(
                    Simplex(elements=sorted(face), construct_tree=False)
                )  # any face is always ordered
        return frozenset(faceset)

    @property
    def boundary(self):
        if self.construct_tree:
            return frozenset(i for i in self._faces if len(i) == len(self) - 1)
        else:
            faces = Simplex.construct_simplex_tree(self.nodes)
            return frozenset(i for i in faces if len(i) == len(self) - 1)

    @property
    def faces(self):
        if self.construct_tree:
            return self._faces
        else:
            return Simplex.construct_simplex_tree(self.nodes)

    def __getitem__(self, item):
        if item not in self.properties:
            raise KeyError(f"attr {item} is not an attr in the cell {self.name}")
        else:
            return self.properties[item]

    def __setitem__(self, key, item):
        self.properties[key] = item

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, e):
        return e in self.nodes

    def __repr__(self):
        """
        String representation of Simplex
        Returns
        -------
        str
        """
        return f"Simplex{tuple(self.nodes)}"

    def __str__(self):
        """
        String representation of a simplex
        Returns
        -------
        str
        """
        return f"Nodes set:{tuple(self.nodes)}, attrs:{self.properties}"


class SimplexView:
    """A SimplexView class for cells of a simplicial_complex
    Parameters
    ----------
    name : str

    Examples
    --------
         >>> SC = SimplexView()
         >>> SC.insert_simplex ( (1,2,3,4),weight=1 )
         >>> SC.insert_simplex ( (2,3,4,1) )
         >>> SC.insert_simplex ( (1,2,3,4) )
         >>> SC.insert_simplex ( (1,2,3,6) )
         >>> c1=Simplex((1,2,3,4,5))
         >>> SC.insert_simplex(c1)
         #>>> c1 in CV

    """

    def __init__(self, name=None):

        if name is None:
            self.name = "_"
        else:
            self.name = name

        self.max_dim = -1
        self.faces_dict = []

    def __getitem__(self, simplex):
        """
        Parameters
        ----------
        cell : tuple list or Simplex
            DESCRIPTION.
        Returns
        -------
        TYPE : dict or ilst or dicts
            return dict of properties associated with that cells

        """
        if isinstance(simplex, Simplex):
            if simplex.nodes in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex.nodes]
        elif isinstance(simplex, Iterable):
            simplex = frozenset(simplex)
            if simplex in self.faces_dict[len(simplex) - 1]:
                return self.faces_dict[len(simplex) - 1][simplex]
            else:
                raise KeyError(f"cell {simplex} is not in the simplex dictionary")

        elif isinstance(simplex, Hashable):

            if frozenset({simplex}) in self:

                return self.faces_dict[0][frozenset({simplex})]

    def __setitem__(self, simplex, **attr):
        if simplex in self:

            if isinstance(simplex, Simplex):
                if simplex.nodes in self.faces_dict[len(simplex) - 1]:
                    self.faces_dict[len(simplex) - 1].update(attr)
            elif isinstance(simplex, Iterable):
                simplex = frozenset(simplex)
                if simplex in self.faces_dict[len(simplex) - 1]:
                    self.faces_dict[len(simplex) - 1].update(attr)
                else:
                    raise KeyError(f"cell {simplex} is not in the simplex dictionary")

            elif isinstance(simplex, Hashable):
                if frozenset({simplex}) in self:
                    self.faces_dict[0].update(attr)
        else:
            raise KeyError(
                "simplex is not in the complex, add simplex using add_simplex."
            )

    # Set methods
    def shape(self):

        if len(self.faces_dict) == 0:
            print("Complex is empty.")
        else:
            return [len(self.faces_dict[i]) for i in range(len(self.faces_dict))]

    def __len__(self):
        import numpy as np

        return np.sum(self.shape())

    def __iter__(self):
        return iter

    def __contains__(self, e):
        if isinstance(e, Iterable):
            return frozenset(e) in self.faces_dict[len(e) - 1]

        elif isinstance(e, Simplex):
            return e.nodes in self.faces_dict[len(e) - 1]

        elif isinstance(e, Hashable):
            return frozenset({e}) in self.faces_dict[0]

        else:
            return False

    def __repr__(self):
        """C
        String representation of simplices
        Returns
        -------
        str
        """
        all_simplices = []
        for i in range(len(self.faces_dict)):
            all_simplices = all_simplices + [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"

    def __str__(self):
        """
        String representation of simplices

        Returns
        -------
        str
        """
        all_simplices = []
        for i in range(len(self.faces_dict)):
            all_simplices = all_simplices + [tuple(j) for j in self.faces_dict[i]]

        return f"SimplexView({all_simplices})"

    @staticmethod
    def _build_faces_dict_from_gudhi_tree(self, simplex_tree):
        """
        extract skeletons from gudhi simples tree

        Remark
            faces_dict[i] = X^i where X^i is the ith skeleton of the input SC X.

        """

        if self.faces_dict != []:
            raise ValueError(
                "self.faces_dict is not empty, this method should "
                + "only called with empty faces_dict"
            )

        self.faces_dict = [dict() for _ in range(simplex_tree.dimension() + 1)]

        for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
            if self.max_dim < len(simplex) - 1:
                self.max_dim = len(simplex) - 1
            k = len(simplex)
            if len(simplex_tree.get_cofaces(simplex)) != 0:
                cofaces = [s[0] for s in simplex_tree.cofaces(simplex)]
                max_simplex_length = len(max(cofaces, key=len))
                max_simplices = set(
                    [frozenset(s) for s in cofaces if len(s) == max_simplex_length]
                )
                self.faces_dict[k - 1][frozenset(sorted(simplex))] = {
                    "id": len(self.faces_dict[k - 1]),
                    "is_maximal": False,
                    "membership": max_simplices,
                }
            else:
                self.faces_dict[k - 1][frozenset(sorted(simplex))] = {
                    "id": len(self.faces_dict[k - 1]),
                    "is_maximal": True,
                    "membership": set(),
                }

    def add_simplices_from(self, simplices):
        if isinstance(simplices, Iterable):
            for s in simplices:
                self.insert_simplex(s)
        else:
            raise ValueError("input simplices must be an iterable of simplex objects")

    def _update_faces_dict_length(self, simplex):

        if len(simplex) > len(self.faces_dict):
            diff = len(simplex) - len(self.faces_dict)
            for _ in range(diff):
                self.faces_dict.append(dict())

    def _update_faces_dict_entry(self, face, simplex_, **attr):

        """
        Parameters:
        ==========
        face :  an iterable, typically a list, tuple, set or a Simplex
        simplex : an iterable, typically a list, tuple, set or a Simplex
        **attr : attrs assocaited with the input simplex

        Note:
        =====
        the input 'face' is a face of the input 'simplex'.
        """

        k = len(face)
        if frozenset(sorted(face)) not in self.faces_dict[k - 1]:
            if len(face) == len(simplex_):
                self.faces_dict[k - 1][frozenset(sorted(face))] = {
                    "id": len(self.faces_dict[k - 1]),
                    "is_maximal": True,
                    "membership": set(),
                }
            else:
                self.faces_dict[k - 1][frozenset(sorted(face))] = {
                    "id": len(self.faces_dict[k - 1]),
                    "is_maximal": False,
                    "membership": set({simplex_}),
                }
        else:
            if len(face) != len(simplex_):
                self.faces_dict[k - 1][frozenset(sorted(face))]["is_maximal"] = False
                self.faces_dict[k - 1][frozenset(sorted(face))]["membership"].add(
                    simplex_
                )
            else:
                self.faces_dict[k - 1][simplex_].update(attr)

    def insert_simplex(self, simplex, **attr):

        if isinstance(simplex, Hashable) and not isinstance(simplex, Iterable):
            simplex = [simplex]
        if isinstance(simplex, Iterable) or isinstance(simplex, Simplex):
            if not isinstance(simplex, Simplex):
                simplex_ = frozenset(
                    sorted(simplex)
                )  # put the simplex in cananical order
                if len(simplex_) != len(simplex):
                    raise ValueError("a simplex cannot contain duplicate nodes")
            else:
                simplex_ = simplex.nodes
            self._update_faces_dict_length(simplex_)

            if simplex_ in self.faces_dict[len(simplex_) - 1]:
                self.faces_dict[len(simplex_) - 1][simplex_].update(attr)
                return
            if self.max_dim < len(simplex) - 1:
                self.max_dim = len(simplex) - 1

            numnodes = len(simplex_)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex_, r):
                    self._update_faces_dict_entry(face, simplex_, **attr)
            if isinstance(simplex, Simplex):

                self.faces_dict[len(simplex_) - 1][simplex_].update(simplex.properties)

        else:
            raise TypeError("input type must be iterable, or Simplex")

    def remove_maximal_simplex(self, simplex):
        """
        Note
        -----
        Only maximal simplices are allowed to be deleted. Otherwise raise ValueError

        Examples
        --------
             >>> SC = SimplexView()
             >>> SC.insert_simplex ( (1,2,3,4),weight=1 )
             >>> c1=Simplex((1,2,3,4,5))
             >>> SC.insert_simplex(c1)
             >>> SC.remove_maximal_simplex((1,2,3,4,5))

        """

        if (
            isinstance(simplex, tuple)
            or isinstance(simplex, list)
            or isinstance(simplex, Simplex)
        ):
            if not isinstance(simplex, Simplex):
                simplex_ = frozenset(
                    sorted(simplex)
                )  # put the simplex in cananical order
            else:
                simplex_ = simplex.nodes
        if simplex_ in self.faces_dict[len(simplex_) - 1]:
            if self.__getitem__(simplex)["is_maximal"]:
                del self.faces_dict[len(simplex_) - 1][simplex_]
                faces = Simplex(simplex_).faces
                for s in faces:
                    if len(s) == len(simplex_):
                        continue
                    else:
                        s = s.nodes
                        self.faces_dict[len(s) - 1][s]["membership"].remove(simplex_)
                        if (
                            len(self.faces_dict[len(s) - 1][s]["membership"]) == 0
                            and len(s) == len(simplex) - 1
                        ):
                            self.faces_dict[len(s) - 1][s]["is_maximal"] = True

                if (
                    len(self.faces_dict[len(simplex_) - 1]) == 0
                    and len(simplex_) - 1 == self.max_dim
                ):
                    del self.faces_dict[len(simplex_) - 1]
                    self.max_dim = len(self.faces_dict) - 1

            else:
                raise ValueError(
                    "only maximal simplices can be deleted, input simplex is not maximal"
                )
        else:
            raise KeyError("simplex is not a part of the simplicial comples")


class NodeView(SimplexView):

    """
    >>> SC = SimplexView()
    >>> SC.insert_simplex ( (1,2,3,4),weight=1 )
    >>> c1=Simplex((1,2,3,4,5))
    >>> NV = NodeView(SC)

    """

    def __init__(self, simplexview, name=None):
        if name is None:
            self.name = "_"
        else:
            self.name = name
        if simplexview.faces_dict != []:
            self.nodes = simplexview.faces_dict[0]
        else:
            self.nodes = []

    def __repr__(self):
        """C
        String representation of nodes
        Returns
        -------
        str
        """

        all_nodes = [tuple(j) for j in self.nodes.keys()]

        return f"NodeView({all_nodes})"

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, e):

        if isinstance(e, Hashable):
            return frozenset({e}) in self.nodes
        elif isinstance(e, Iterable):
            if len(e) == 1:
                return frozenset(e) in self.nodes
        else:
            return False
