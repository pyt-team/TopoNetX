# -*- coding: utf-8 -*-
"""

@author: Mustafa Hajij
"""


from collections import OrderedDict, defaultdict

import numpy as np

# from collections.abc import Iterable
from hypernetx.classes.entity import Entity

from toponetx import TopoNetXError

__all__ = ["RankedEntity", "RankedEntitySet"]


class RankedEntity(Entity):
    """
     Base class for objects used in building ranked cell objects including cell complexes,
     simplicial complexes and combinatorial complexes.


    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or dict, optional, default: None
        a list of entities with identifiers different than uid and/or
        hashables different than uid, see `Honor System`_

    rankedentity : RankedEntity
        a RankedEntity object to be cloned into a new RankedEntity with uid. If the uid is the same as
        RankedEntity.uid then the entities will not be distinguishable and error will be raised.
        The `elements` in the signature will be added to the cloned Rankedentity.
        The rank of the new ranked entity will be the max of the cloned ranked entity and the input rank.

    weight : float, optional, default : 1
    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        Both key and value must be hashable.

    Notes
    -----

    A RankedEntity is a container-like object, which has a unique identifier and
    may contain elements and have properties.
    A RankedEntity must have a rank, a non-negative integer that describes the rank of the entity.
    A RankedEntity may contain other ranked entities such that ranks are consistent with set inclusions:
        if entity1 <= entity2 then rank(entity1) <= rank(entity2).

    The RankedEntity class was created as a generic object providing structure for
    CombinatorialComplex nodes and edges.
    The RankedEntity class is built on the top of the hypernetx.classes.entity class

    - A RankedEntity is distinguished by its identifier (sortable,hashable) :func:`RankedEntity.uid`
    - A RankedEntity is a container for other ranked entities but may not contain itself, :func:`RankedEntity.elements`
    - A RankedEntity may contain other ranked entities such that ranks are consistent with set inclusions:
           if entity1 <= entity2 then rank(entity1) <= rank(entity2).
    - A RankedEntity has properties :func:`RankedEntity.properties`
    - A RankedEntity has memberships to other entities, :func:`RankedEntity.memberships`.
    - A RankedEntity has children, :func:`RankedEntity.children`, which are the elements of its elements.
    - :func:`RankedEntity.children` are registered in the :func:`RankedEntity.registry`.
    - All descendents of RankedEntity are registered in :func:`RankedEntity.fullregistry()`.

     Examples
     --------


         >>> x1 = RankedEntity('x1',rank = 0)
         >>> x2 = RankedEntity('x2',rank = 0)
         >>> x3 = RankedEntity('x3',rank = 0)
         >>> y1 = RankedEntity('y1',[x1], rank = 1)
         >>> y2 = RankedEntity('y2',[y1,x2], rank = 2)
         >>> y3 = RankedEntity('y3',[x2,x3], rank = 1)
         >>> z = RankedEntity('z',[x1,x2,y2,y3],rank = 3)
         #>>> EE=RankedEntitySet("",z.incidence_dict)
         >>> z
         RankedEntity(z,['y1', 'y3', 'x1', 'x2'],2,{'weight': 1.0})


         >>> x1 = RankedEntity('x1',rank = 0)
         >>> x2 = RankedEntity('x2',rank = 0)
         >>> x3 = RankedEntity('x3',rank = 0)
         >>> x4 = RankedEntity('x4',rank = 0)
         >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
         >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
         >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
         >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
         >>> z = RankedEntity('z',[y1,y2,y3,y4],rank = 2)
         >>> w = RankedEntity('w',[z],rank = 3)
         # d={'0': ([1, 2, 3], 2), '1': ([2, 4], 1)}

         >>> d= {'y1': {'elements': [1, 2], 'rank': 1},
          'y2': {'elements': [1, 2, 3], 'rank': 1}}
         >>> z = RankedEntity('z',d,rank = 3)


         >>> d = {}
         >>> d['x1'] = RankedEntity('x1',rank = 0)
         >>> d['x2'] = RankedEntity('x2',rank = 0)
         >>> d['x3'] = RankedEntity('x3',rank = 0)
         >>> d['y1'] = RankedEntity('y1',[x1], rank = 1)
         >>> d['y2'] = RankedEntity('y2',[y1,x2], rank = 2)
         >>> d['y3'] = RankedEntity('y3',[x2,x3], rank = 1)
         >>> z = RankedEntity('z',d,rank = 3)

     See Also
     --------
     RankedEntitySet
    """

    def __init__(
        self,
        uid,
        elements=[],
        rank=None,
        rankedentity=None,
        weight=1.0,
        safe_insert=True,
        **props,
    ):
        super().__init__(uid, None, rankedentity, weight, **props)
        isinstance(safe_insert, bool)  # safe_insert must be boolean
        self._all_ranks = set()
        if isinstance(rankedentity, RankedEntity):
            if rank == None:
                self._rank = rankedentity._rank
            else:
                self._rank = max(rankedentity._rank, rank)
            self._all_ranks = self._all_ranks.union(rankedentity._all_ranks)
        else:
            if rankedentity is not None:
                raise TopoNetXError(
                    "the input entity be be either an instance of RankedEntity or None"
                )
            else:  # entity is None
                self._rank = rank
        self._safe_insert = safe_insert
        self._registry = dict()

        if isinstance(elements, dict):  # often used with RankedEntitySet
            # ex: elements = {'y1': {'elements': [1, 2], 'rank': 1},
            #         'y2': {'elements': [1, 2, 3], 'rank': 1}}
            ordered_keys = self._order_ranked_entity_dictionary(elements)
            for k in ordered_keys:
                if isinstance(elements[k], RankedEntity):
                    self.add_element(
                        elements[k], safe_insert=safe_insert, check_CC_condition=True
                    )
                    self._all_ranks = self._all_ranks.union(elements[k]._all_ranks)
                elif "elements" in elements[k] and "rank" in elements[k]:
                    self.add_element(
                        (k, elements[k]),
                        safe_insert=safe_insert,
                        check_CC_condition=False,
                    )
                    self._all_ranks.add(elements[k]["rank"])
                elif isinstance(elements[k], tuple) or isinstance(elements[k], list):
                    assert (
                        len(elements[k]) == 2
                    )  # elements[k] must of the form (cell,rank)
                    assert isinstance(
                        elements[k][1], int
                    )  # assert second element,rank is an integer
                    # if elements[k][1] == 0 and len(elements[k][1] ):
                    ent = RankedEntity(k, elements[k][0], elements[k][1])
                    self.add_element(
                        ent, safe_insert=safe_insert, check_CC_condition=False
                    )
                    self._all_ranks.add(elements[k][1])
                else:
                    raise TopoNetXError(
                        "the element dictionary must have specific shape e.g."
                        "elements = {'y1': {'elements': [1, 2], 'rank': 1}},"
                        "or elements = {'0':[[1,2],1], '2':[[1,2,3],2]}  "
                    )
        elif isinstance(elements, RankedEntity):
            self.add_element(elements, safe_insert=safe_insert)
            self._all_ranks = self._all_ranks.union(elements._all_ranks)
        elif elements is not None:
            self.add(*elements, safe_insert=safe_insert)
        if rank != np.inf:
            self._all_ranks.add(self._rank)
        if safe_insert:
            assert (
                isinstance(self._rank, int) or self._rank == np.inf
            )  # rank must be an integer
            assert self._rank >= 0  # rank must be non-negative integer
            if self.depth() == 0 and not isinstance(
                self, RankedEntitySet
            ):  # depth of the current entity in the poset structure
                assert self._rank == 0  # rank must be zero for primitive entities
            if self._rank == 0:
                assert (
                    len(self) == 0
                )  # zero ranked entity must be not contain any other entity.
            elif not isinstance(self, RankedEntitySet):
                assert (
                    self._rank > 0
                )  # rank must be a positive integer for non-primitive entities
                for k, v in self.elements.items():

                    if v.rank > self._rank:
                        raise TopoNetXError(
                            f"Error: rank of members must be smaller than"
                            f" the rank of the entity that contains them."
                            f" Input entity {v} has rank {v.rank} and this entity has rank {rank}"
                        )

    @property
    def properties(self):
        """Dictionary of properties of ranked entity"""
        temp = self.__dict__.copy()
        del temp["_elements"]
        del temp["_memberships"]
        del temp["_registry"]
        del temp["_uid"]
        del temp["_rank"]
        del temp["_safe_insert"]
        del temp["_all_ranks"]
        return temp

    @property
    def rank(self):
        """integer specified the rank of the cell"""
        return self._rank

    def set_safe_insert(self, value):

        if not isinstance(value, bool):
            raise TopoNetXError(f" value must be of type bool got {type(value)}.")
        self._safe_insert = value

    def add_element(self, item, safe_insert=True, check_CC_condition=False):
        """
        Adds item to entity elements and adds entity to item.memberships.

        Parameters
        ----------
        item : hashable or Ranked Entity
            If hashable, will be replaced with empty RankedEntity using hashable as uid

        Returns
        -------
        self : RankedEntity

        Notes
        -----
        If item is in entity elements, no new element is added but properties
        will be updated.
        If item is in complete_registry(), only the item already known to self will be added.
        This method employs the `Honor System`_ since membership in complete_registry is checked
        using the item's uid. It is assumed that the user will only use the same uid
        for identical instances within the entities registry.
        """

        checkelts = self.complete_registry()
        if isinstance(item, RankedEntity):
            condition = False
            if check_CC_condition:
                condition = self._insert_cell_condition(item)

            # if item is an Ranked Entity, descendents will be compared to avoid collisions
            if item.uid == self.uid:
                raise TopoNetXError(
                    f"Error: Self reference in submitted elements."
                    f" Entity {self.uid} may not contain itself. "
                )

            elif item.uid in checkelts:
                # if item belongs to an element or a descendent of an element
                # then the existing descendent becomes an element
                # and properties are updated.
                checkelts[item.uid]._memberships[self.uid] = self
                checkelts[item.uid].__dict__.update(item.properties)
                self._elements[item.uid] = item

            elif item in self and safe_insert:
                if condition:  # safe insert is already met in _insert_cell_condition
                    checkelts[item.uid].__dict__.update(item.properties)
                    self._elements[item.uid] = item
                else:
                    # perform mild safe insert
                    for k, v in self.elements.items():
                        if v.rank > item.rank:
                            raise TopoNetXError(
                                f"Error: rank of members must be smaller than"
                                f" the rank of the entity that contains them."
                                f" Input entity {item} has rank {item.rank} and this entity has rank {v.rank}"
                            )
                    checkelts[item.uid].__dict__.update(item.properties)
                    self._elements[item.uid] = item

            else:
                # if item's uid doesn't appear in complete_registry
                # then it is added as something new
                item._memberships[self.uid] = self
                self._elements[item.uid] = item
                self._all_ranks.add(item.rank)
                self._all_ranks = self._all_ranks.union(item._all_ranks)
        else:
            # item must be a hashable.
            # if it appears as a uid in checkelts then
            # the corresponding Entity will become an element of ranked entity.
            # Otherwise, at most it will be added as an empty rank zero RankedEntity.
            if self.uid == item:
                raise TopoNetXError(
                    f"Error: Self reference in submitted elements."
                    f" Entity {self.uid} may not contain itself."
                )

            elif isinstance(
                item, tuple
            ):  # e.g. item=(key, {"element":[1,2,3],"rank":3 }  )
                if isinstance(item[1], dict):

                    if self.uid == item[0]:
                        raise TopoNetXError(
                            f"Error: Self reference in submitted elements."
                            f" Entity {self.uid} may not contain itself."
                        )

                    elif item[0] in checkelts:
                        assert "elements" in item[1] and "rank" in item[1]
                        self._elements[item[0]] = checkelts[item[0]]
                        checkelts[item[0]]._memberships[self.uid] = self
                    else:
                        elem = []
                        for i in item[1]["elements"]:
                            if i in checkelts:
                                elem.append(
                                    checkelts[i]
                                )  # element is an existing RankedEntity
                            else:
                                elem.append(i)  # element is a primitive entity
                        ent = RankedEntity(
                            item[0],
                            elements=elem,
                            _memberships={self.uid: self},
                            rank=item[1]["rank"],
                        )

                        if check_CC_condition:
                            self._insert_cell_condition(ent)
                            self._elements[item[0]] = ent
                        elif safe_insert:
                            if check_CC_condition:
                                self._elements[item[0]] = ent
                            else:
                                for k, v in self.elements.items():
                                    if v.rank > ent.rank:
                                        raise TopoNetXError(
                                            f"Error: rank of members must be smaller than"
                                            f" the rank of the entity that contains them."
                                            f" Input entity {item} has rank {item.rank} and this entity has rank {v.rank}"
                                        )

                                self._elements[item[0]] = ent
                                self._all_ranks.add(ent.rank)
                                self._all_ranks = self._all_ranks.union(ent._all_ranks)

                        else:
                            self._elements[item[0]] = ent
                            self._all_ranks.add(ent.rank)
                            self._all_ranks = self._all_ranks.union(ent._all_ranks)
            elif item not in self._elements:

                if item in checkelts:
                    self._elements[item] = checkelts[item]
                    checkelts[item]._memberships[self.uid] = self
                    self._all_ranks = self._all_ranks.union(checkelts[item]._all_ranks)
                else:  # assume primitive input
                    self._elements[item] = RankedEntity(
                        item, _memberships={self.uid: self}, rank=0
                    )
                    self._all_ranks.add(0)

        return self

    def skeleton(self, k, includeself=True):
        out = []
        if includeself:
            if self.rank == k:
                out.append(self)
        d = self.fullregistry()
        out = out + [d[key] for key in d.keys() if d[key].rank == k]
        return RankedEntitySet("X" + str(k), elements=out, safe_insert=False)

    def add(self, *args, safe_insert=True):
        """
        Adds unpacked args to ranked entity elements. Depends on add_element()

        Parameters
        ----------
        args : One or more entities or hashables

        Returns
        -------
        self : RankedEntity

        """
        for item in args:
            self.add_element(item, safe_insert=safe_insert)

        return self

    def __str__(self):
        """Return the entity uid."""
        return f"{self.uid}, elements = {list(self.uidset)}, rank = {self.rank}"

    def __repr__(self):
        """Returns a string resembling the constructor for ranked entity"""
        return f"RankedEntity({self._uid},elements={list(self.uidset)},rank={self.rank}, {self.properties})"

    def __contains__(self, item):
        """
        Defines containment for Entities.

        Parameters
        ----------
        item : hashable or Entity

        Returns
        -------
        Boolean

        Depends on the `Honor System`_ . Allows for uids to be used as shorthand for their entity.
        This is done for performance reasons, but will fail if uids are
        not unique to their entities.
        Is not transitive.
        """
        if isinstance(item, RankedEntity):
            return item.uid in self._elements
        else:
            return item in self._elements

    def __getitem__(self, item):
        """
        Returns Entity element by uid. Use :func:`E[uid]`.

        Parameters
        ----------
        item : hashable or Entity

        Returns
        -------
        Entity or None

        If item not in entity, returns None.
        """
        if isinstance(item, RankedEntity):
            return self._elements.get(item.uid, "")
        else:
            return self._elements.get(item)

    def restrict_to(self, element_subset, name=None):
        """
        Shallow copy of ranked entity removing elements not in element_subset.

        Parameters
        ----------
        element_subset : iterable
            A subset of ranked entities elements

        name: hashable, optional
            If not given, a name is generated to reflect entity uid

        Returns
        -------
        New RankedEntity : RankedEntity
            Could be empty.
        """
        newelements = [self[e] for e in element_subset if e in self]
        name = name or f"{self.uid}_r"
        if len(newelements) > 0:
            return RankedEntity(
                name, elements=newelements, rank=self.rank, **self.properties
            )
        else:
            raise TopoNetXError("Restriction yielded empty ranked entity.")

    def clone(self, newuid):
        """
        Returns shallow copy of entity with newuid. RankedEntity's elements will
        belong to two distinct Entities.

        Parameters
        ----------
        newuid : hashable
            Name of the new entity

        Returns
        -------
        clone : RankedEntity

        """
        return RankedEntity(newuid, entity=self)

    def ranked_complete_registry(self):
        """
        A dictionary of all entities appearing in any level of
        entity

        Returns
        -------
        complete_registry : dict
        """
        results = dict()
        RankedEntity._ranked_complete_registry(self, results)
        return results

    @staticmethod
    def _ranked_complete_registry(entity, results):
        """
        Helper method for complete_registry
        """
        for uid, e in entity.elements.items():
            if uid not in results:
                results[uid] = e
                RankedEntity._ranked_complete_registry(e, results)

    @property
    def incidence_dict(self):
        """
        Dictionary of element.uid:element.uidset for each element in entity

        To return an incidence dictionary of all nested entities in entity
        use nested_incidence_dict
        """
        temp = OrderedDict()

        registry = self.ranked_complete_registry()

        for ent in registry.values():
            temp[ent.uid] = {}
        ordered_keys = sorted(registry.keys(), key=lambda kv: registry[kv].rank)
        temp = OrderedDict(
            [
                (ent, {"elements": registry[ent].uidset, "rank": registry[ent].rank})
                for ent in ordered_keys
            ]
        )

        return temp

    @staticmethod
    def _order_ranked_entity_dictionary(elements):
        structure = 0
        for k in elements:
            if not ("elements" in elements[k] or "rank" in elements[k]):
                if isinstance(elements[k], tuple) or isinstance(elements[k], list):
                    structure = 1
                    break
                elif isinstance(elements[k], RankedEntity):
                    structure = 2
                    break
                else:
                    raise TopoNetXError(
                        "the element dictionary must have specific shape e.g."
                        "elements = {'y1': {'elements': [1, 2], 'rank': 1}},"
                        "or elements = {'0':[[1,2],1], '2':[[1,2,3],2]  }  "
                        "or elements = {'0':RankedEntity-0, '1':RankedEntity-1 }  "
                    )
        if structure == 1:
            ordered_keys = sorted(elements.keys(), key=lambda kv: elements[k][1])
        elif structure == 2:
            ordered_keys = sorted(elements.keys(), key=lambda kv: elements[k].rank)
        else:  # structure = 0
            ordered_keys = sorted(elements.keys(), key=lambda kv: elements[kv]["rank"])
        return ordered_keys

    def _insert_cell_condition(self, item):

        """
        Returns a boolean indicating if item is insertable
        and satisfy the combinatorial complex condition inside an existing RankedEntity
        """
        for k, v in self.elements.items():
            if v.uidset == item.uidset:  # item == v and rank(v)!=rank(item)
                if v.rank != item.rank:
                    raise TopoNetXError(
                        f"Error: entity uidset exists within the entity with different rank."
                        f"inserted entity has rank {item.rank} and existing entity has rank {v.rank} ."
                    )

            elif v.uidset.issubset(item.uidset):  # item => v and rank(item) < rank(v)
                if v.rank > item.rank:
                    raise TopoNetXError(
                        f"Error: Fails the CC condition for Ranked EntitySet."
                    )
            elif item.uidset.issubset(v.uidset):  # item <= v and rank(item) > rank(v)
                if v.rank < item.rank:
                    raise TopoNetXError(
                        f"Error: Fails the CC condition for Ranked EntitySet."
                    )
            return True

    @property
    def all_ranks(self):
        return self._all_ranks

    def remove(self, *args):
        """
        Removes args from entitie's elements if they belong.
        Does nothing with args not in entity.

        Parameters
        ----------
        args : One or more hashables or entities

        Returns
        -------
        self : Entity


        """
        for item in args:
            RankedEntity.remove_element(self, item)
        return self

    def remove_elements_from(self, arg_set):
        """
        Similar to :func:`remove()`. Removes elements in arg_set.

        Parameters
        ----------
        arg_set : Iterable of hashables or entities

        Returns
        -------
        self : Entity

        """
        for item in arg_set:
            RankedEntity.remove_element(self, item)
        return self

    def remove_element(self, item):
        """
        Removes item from ranked entity and reference to ranked entity from
        item.memberships

        Parameters
        ----------
        item : Hashable or RankedEntity

        Returns
        -------
        self : RankedEntity


        """
        if isinstance(item, RankedEntity):
            del item._memberships[self.uid]
            del self._elements[item.uid]
        else:
            del self[item]._memberships[self.uid]
            del self._elements[item]

        return self


class RankedEntitySet(RankedEntity):
    """

    Parameters
    ----------
    uid : hashable
        a unique identifier

    elements : list or dict, optional, default: None
        a list of entities with identifiers different than uid and/or
        hashables different than uid.
    safe_insert : determine if elements inserted into the ranked entity set satisfies the
                  combintorial complex condition : ent1<=ent2 then rank(ent1)<=rank(ent2)

    props : keyword arguments, optional, default: {}
        properties belonging to the entity added as key=value pairs.
        Both key and value must be hashable.

    Examples:
    ---------
        >>> # example1
        >>> a = RankedEntity('a',[1,2,3],1)
        >>> b = RankedEntity('b',[2,3],1)
        >>> c = RankedEntity('c',[1,5],1)
        >>> E = RankedEntitySet('E',[a,b,c] )
        >>> E

        >>> # example2
        >>> elements= {'y1': {'elements': [1, 2], 'rank': 1},
                       'y2': {'elements': [2, 3], 'rank': 1},
                       'y3': {'elements': [1, 2, 3, 4], 'rank': 3},
                       'y4': {'elements': ['y1', 'y2'], 'rank': 3}}
        >>> E = RankedEntitySet('E',elements )
        >>> E.fullregistry()

                {'y1': RankedEntity(y1,[1, 2],1,{'weight': 1.0}),
                 'y2': RankedEntity(y2,[2, 3],1,{'weight': 1.0}),
                 'y3': RankedEntity(y3,[1, 2, 3, 4],3,{'weight': 1.0}),
                 'y4': RankedEntity(y4,['y2', 'y1'],3,{'weight': 1.0}),
                 1: RankedEntity(1,[],0,{'weight': 1.0}),
                 2: RankedEntity(2,[],0,{'weight': 1.0}),
                 3: RankedEntity(3,[],0,{'weight': 1.0}),
                 4: RankedEntity(4,[],0,{'weight': 1.0})}


    """

    def __init__(self, uid, elements=[], safe_insert=True, **props):
        # assert(isinstance(elements,list)) # elements must be a list of Ranked Entities.
        super().__init__(
            uid=uid, elements=elements, rank=np.inf, safe_insert=safe_insert, **props
        )

    def skeleton(self, k, name=None, safe_insert=False, level=None):
        d = self.ranked_complete_registry()

        if name is None and level is None:
            name = "X" + str(k)
        elif name is None and level == "equal":
            name = "X" + str(k)
        elif name is None and level == "upper":
            name = "X>=" + str(k)
        elif name is None and level == "up":
            name = "X>=" + str(k)
        elif name is None and level == "lower":
            name = "X<=" + str(k)
        elif name is None and level == "down":
            name = "X<=" + str(k)
        else:
            assert isinstance(name, str)
        if level is None or level == "equal":
            return RankedEntitySet(
                name,
                elements=[d[key] for key in d.keys() if d[key].rank == k],
                safe_insert=safe_insert,
            )
        elif level == "upper":
            return RankedEntitySet(
                name,
                elements=[d[key] for key in d.keys() if d[key].rank >= k],
                safe_insert=safe_insert,
            )
        elif level == "lower":
            return RankedEntitySet(
                name,
                elements=[d[key] for key in d.keys() if d[key].rank <= k],
                safe_insert=safe_insert,
            )
        else:
            raise TopoNetXError(
                "level must be None, equal, 'upper', 'lower', 'up', or 'down' "
            )

    def _incidence_matrix_helper(self, children, uidset, sparse=True, index=False):

        """

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        Examples
        --------


         helper to create incidence between two sets of RankedEntities children and uidset

         uidset.uidset <= children.uidset

        """

        if sparse:
            from scipy.sparse import csr_matrix

        ndict = dict(zip(children, range(len(children))))
        edict = dict(zip(uidset, range(len(uidset))))

        r_cell_dict = {j: children[j].skeleton(0).uidset for j in children}
        k_cell_dict = {i: uidset[i].skeleton(0).uidset for i in uidset}

        if len(ndict) != 0:

            if index:
                rowdict = {v: k for k, v in ndict.items()}
                coldict = {v: k for k, v in edict.items()}

            if sparse:
                # Create csr sparse matrix
                rows = list()
                cols = list()
                data = list()
                for e in k_cell_dict:
                    for n in r_cell_dict:
                        if r_cell_dict[n] <= k_cell_dict[e]:
                            data.append(1)
                            rows.append(ndict[n])
                            cols.append(edict[e])
                MP = csr_matrix(
                    (data, (rows, cols)), shape=(len(r_cell_dict), len(k_cell_dict))
                )
            else:
                # Create an np.matrix
                MP = np.zeros((len(children), len(uidset)), dtype=int)
                for e in k_cell_dict:
                    for n in r_cell_dict:
                        if r_cell_dict[n] <= k_cell_dict[e]:
                            MP[ndict[n], edict[e]] = 1
            if index:
                return MP, rowdict, coldict
            else:
                return MP
        else:
            if index:
                return np.zeros(1), {}, {}
            else:
                return np.zeros(1)

    def incidence_matrix(
        self, r, k, incidence_type="up", weight=None, sparse=True, index=False
    ):
        """
        An incidence matrix for the RankedEntitySet indexed by r-ranked entities k-ranked entities
        r !=k, when k is None incidence_type will be considered

        Parameters
        ----------

        incidence_type : str, optional, default 'up', other options 'down'

        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying row with item in entityset's children

        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----

        Examples:
        ---------
                # example_1
                >>> a = RankedEntity('a',[1,2,3],1)
                >>> b = RankedEntity('b',[2,3],1)
                >>> c = RankedEntity('c',[1,5],1)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[a,b,c] )
                >>> # check the incidence matrices
                >>> E.incidence_matrix(0,1,sparse=False, index=True)

                    (array([[1, 0, 1],
                            [1, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1]]),
                     {0: 1, 1: 2, 2: 3, 3: 4},
                     {0: 'a', 1: 'b', 2: 'c'})

                # example_2
                >>> x1 = RankedEntity('x1',rank = 0)
                >>> x2 = RankedEntity('x2',rank = 0)
                >>> x3 = RankedEntity('x3',rank = 0)
                >>> x4 = RankedEntity('x4',rank = 0)
                >>> x5 = RankedEntity('x5',rank = 0)
                >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
                >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
                >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
                >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
                >>> y5 = RankedEntity('y5',[x4,x5], rank = 1)
                >>> z = RankedEntity('z',[x1,x2,x3,x4],rank = 2)
                >>> w = RankedEntity('w',[x4,x5,x1],rank = 2)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,z,w] )

                >>> # check the incidence matrices
                >>> E.incidence_matrix(0,1,sparse=False, index=True)
                >>> E.incidence_matrix(1,2,sparse=False, index=True)
                >>> E.incidence_matrix(0,2,sparse=False, index=True)
        """
        weight = None  # weight is not supported in this version
        assert r != k  # r and k must be different
        if k is None:
            if incidence_type == "up":
                children = self.skeleton(r)
                uidset = self.skeleton(r + 1, level="upper")
            elif incidence_type == "down":
                uidset = self.skeleton(r)
                children = self.skeleton(r - 1, level="lower")
            else:
                raise TopoNetXError("incidence_type must be 'up' or 'down' ")
        else:
            assert (
                r != k
            )  # incidence is defined between two skeletons of different ranks
            if (
                r < k
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(r)
                uidset = self.skeleton(k)

            elif (
                r > k
            ):  # up incidence is defined between two skeletons of different ranks
                children = self.skeleton(k)
                uidset = self.skeleton(r)

        return self._incidence_matrix_helper(children, uidset, sparse, index)

    def adjacency_matrix(self, r, k, s=1, weights=False, index=False):
        """
        A adjacency matrix for the RankedEntitySet of the r-ranked considering thier adjacency with respect to k-ranked entities
        r < k

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of uid : row number

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying row with item in entityset's children

        column dictionary : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----

        Examples:
        ---------
                # example_1
                >>> a = RankedEntity('a',[1,2,3],1)
                >>> b = RankedEntity('b',[2,3],1)
                >>> c = RankedEntity('c',[1,5],1)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[a,b,c] )
                >>> # check the incidence matrices
                >>> E.incidence_matrix(0,1,sparse=False, index=True)


                # example_2
                >>> x1 = RankedEntity('x1',rank = 0)
                >>> x2 = RankedEntity('x2',rank = 0)
                >>> x3 = RankedEntity('x3',rank = 0)
                >>> x4 = RankedEntity('x4',rank = 0)
                >>> x5 = RankedEntity('x5',rank = 0)
                >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
                >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
                >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
                >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
                >>> y5 = RankedEntity('y5',[x4,x5], rank = 1)
                >>> z = RankedEntity('z',[x1,x2,x3,x4],rank = 2)
                >>> w = RankedEntity('w',[x4,x5,x1],rank = 2)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,z,w] )

                >>> # check the incidence matrices
                >>> E.adjacency_matrix(0,1)
                >>> E.adjacency_matrix(1,2)
                >>> E.adjacency_matrix(0,2)


        """

        if k is not None:
            assert r < k  # rank k must be smaller than rank r

        if index:

            MP, row, col = self.incidence_matrix(
                r, k, incidence_type="up", sparse=True, index=index
            )
        else:
            MP = self.incidence_matrix(
                r, k, incidence_type="up", sparse=True, index=index
            )

        weights = False  ## currently weighting is not supported
        if weights == False:
            A = MP.dot(MP.transpose())
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, row
        else:
            return A

    def coadjacency_matrix(self, r, k, s=1, weights=False, index=False):
        """
        A coadjacency matrix for the RankedEntitySet of the r-ranked considering thier adjacency with respect to k-ranked entities
        r > k

        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default : False
            If True return will include a dictionary of children uid : row number
            and element uid : column number

        Returns
        -------
        coadjacency_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        column dictionary of the associted incidence matrix : dict
            Dictionary identifying column with item in entityset's uidset

        Notes
        -----

        Example:
        ---------
                # example_1
                >>> a = RankedEntity('a',[1,2,3],1)
                >>> b = RankedEntity('b',[2,3],1)
                >>> c = RankedEntity('c',[1,5],1)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[a,b,c] )
                >>> # check the incidence matrices
                >>> E.incidence_matrix(0,1,sparse=False, index=True)

                    (array([[1, 0, 1],
                            [1, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1]]),
                     {0: 1, 1: 2, 2: 3, 3: 4},
                     {0: 'a', 1: 'b', 2: 'c'})

                # example_2
                >>> x1 = RankedEntity('x1',rank = 0)
                >>> x2 = RankedEntity('x2',rank = 0)
                >>> x3 = RankedEntity('x3',rank = 0)
                >>> x4 = RankedEntity('x4',rank = 0)
                >>> x5 = RankedEntity('x5',rank = 0)
                >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
                >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
                >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
                >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
                >>> y5 = RankedEntity('y5',[x4,x5], rank = 1)
                >>> z = RankedEntity('z',[x1,x2,x3,x4],rank = 2)
                >>> w = RankedEntity('w',[x4,x5,x1],rank = 2)
                >>> x = RankedEntity('x',[x2,x3],rank = 2)
                >>> # define the Ranked Entity Set
                >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,x,z,w] )

                >>> # check the incidence matrices
                >>> E.coadjacency_matrix(0,1)
                >>> E.coadjacency_matrix(1,2)
                >>> E.coadjacency_matrix(0,2)


        """
        # TODO : None case
        assert r < k  # rank k must be larger than rank r
        if index:

            MP, row, col = self.incidence_matrix(
                r, k, incidence_type="down", sparse=True, index=index
            )
        else:
            MP = self.incidence_matrix(
                k, r, incidence_type="down", sparse=True, index=index
            )
        weights = False  ## currently weighting is not supported
        if weights == False:
            A = MP.T.dot(MP)
            A.setdiag(0)
            A = (A >= s) * 1
        if index:
            return A, col
        else:
            return A

    def add(self, *args, safe_insert=True):
        """
        Adds args to entityset's elements, checking to make sure no self references are
        made to element ids.
        Ensures Bipartite Condition of EntitySet.

        Parameters
        ----------
        args : One or more entities or hashables

        Returns
        -------
        self : RankedEntitySet

        """

        for item in args:

            if isinstance(item, RankedEntity):
                self._all_ranks = self._all_ranks.union(item._all_ranks)
                if safe_insert:
                    self.add_element(item, check_CC_condition=True)
                else:
                    # unsafe insertion --used particulary when
                    # extracting skeletons from existing RankedEntitySet
                    self.add_element(item, safe_insert=False, check_CC_condition=False)

            else:
                if not item in self.children:
                    self.add_element(item, check_CC_condition=True)
                else:
                    raise TopoNetXError(
                        f"Error: {item} references a child of an existing Entity in the EntitySet."
                    )
        return self

    def collapse_identical_elements(self, newuid, return_equivalence_classes=False):
        """
        Returns a deduped copy of the entityset, using representatives of equivalence classes as element keys.
        Two elements of an RankedEntitySet are collapsed if they share the same rank zero entities.

        Parameters
        ----------
        newuid : hashable

        return_equivalence_classes : boolean, default=False
            If True, return a dictionary of equivalence classes keyed by new edge names

        Returns
        -------
         : EntitySet
        eq_classes : dict
            if return_equivalence_classes = True

        Notes
        -----
        Treats elements of the entityset as equal if they have the same uidsets. Using this
        as an equivalence relation, the entityset's uidset is partitioned into equivalence classes.
        The equivalent elements are identified using a single entity by using the
        frozenset of uids associated to these elements as the uid for the new element
        and dropping the properties.
        If use_reps is set to True a representative element of the equivalence class is
        used as identifier instead of the frozenset.

        Examples:
            >>> x1 = RankedEntity('x1',rank = 0)
            >>> x2 = RankedEntity('x2',rank = 0)
            >>> x3 = RankedEntity('x3',rank = 0)
            >>> x4 = RankedEntity('x4',rank = 0)
            >>> x5 = RankedEntity('x5',rank = 0)
            >>> y1 = RankedEntity('y1',[x1,x2], rank = 1)
            >>> y2 = RankedEntity('y2',[x2,x3], rank = 1)
            >>> y3 = RankedEntity('y3',[x3,x4], rank = 1)
            >>> y4 = RankedEntity('y4',[x4,x1], rank = 1)
            >>> y5 = RankedEntity('y5',[x4,x5], rank = 1)
            >>> y6 = RankedEntity('y6',[x4,x5], rank = 1)
            >>> z = RankedEntity('z',[y6,x2,x3,x4],rank = 2)
            >>> w = RankedEntity('w',[y4,x5,x1],rank = 2)
            >>> # define the Ranked Entity Set
            >>> E = RankedEntitySet('E',[y1,y2,y3,y4,y5,z,w] )

            >>> E.collapse_identical_elements("")
            # note that y5 is collapsed with y6 on all levels

        """

        shared_children = defaultdict(set)  # rank zero entities
        skeleton = self.skeleton(1, level="upper")
        for e in skeleton.__call__():
            shared_children[frozenset(e.uidset)].add(e.uid)

        uniqe_reps = (
            {}
        )  # fix the reps across, this will be used when returning the RankedEntitySet
        for _, v in shared_children.items():
            e, *_ = v  # pick arbitrary rep of the set v --> e
            for l in v:
                uniqe_reps[l] = e

        new_entity_dict = dict()
        for k, v in shared_children.items():
            e = v.pop()
            new_entity_dict[e] = {}
            lst = []
            for i in k:
                if i in uniqe_reps:
                    lst.append(uniqe_reps[i])
                else:
                    lst.append(i)
            new_entity_dict[e]["elements"] = frozenset(lst)
            new_entity_dict[e]["rank"] = skeleton[e].rank

        if return_equivalence_classes:
            eq_classes = {
                f"{next(iter(v))}:{len(v)}": v for k, v in shared_children.items()
            }
            return RankedEntitySet(newuid, new_entity_dict), dict(eq_classes)
        else:
            return RankedEntitySet(newuid, new_entity_dict)

    def restrict_to(self, element_subset, name=None):
        """
        Shallow copy of entityset removing elements not in element_subset.

        Parameters
        ----------
        element_subset : iterable
            A subset of the entityset's elements

        name: hashable, optional
            If not given, a name is generated to reflect entity uid

        Returns
        -------
        new entityset : RankedEntitySet
            Could be empty.

        See also
        --------
        RankedEntity.restrict_to

        """
        newelements = [self[e] for e in element_subset if e in self]
        name = name or f"{self.uid}_r"
        return RankedEntitySet(name, newelements, **self.properties)

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f"RankedEntitySet({self._uid},{list(self.uidset)},{self.properties})"
