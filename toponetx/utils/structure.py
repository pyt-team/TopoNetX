"""Utilities on structures.

The first function, sparse_array_to_neighborhood_list, takes a sparse array
representing a higher order structure between two sets of
cells, S and T, and converts it to a neighborhood list. This
is a list of tuples such that each tuple has the form (s, t),
where s and t are indices representing cells in S and T.
The function also allows for optional dictionaries that map
 the indices in S and T to other values.

The second function, neighborhood_list_to_neighborhood_dict,
takes a neighborhood list and converts it to a neighborhood dictionary.
This dictionary maps each cell i in S to a list of cells j that are in
the neighborhood of i.

The third function, sparse_array_to_neighborhood_dict,
combines the first two functions and takes a sparse array as
input and returns a neighborhood dictionary. It first converts
the sparse array to a neighborhood list, and then converts the
neighborhood list to a neighborhood dictionary. Like the first
function, it also allows for optional dictionaries that map the
indices in S and T to other values.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from scipy.sparse import csr_matrix


def sparse_array_to_neighborhood_list(
    sparse_array, src_dict=None, dst_dict=None
) -> zip:
    """Convert sparse array to neighborhood list for arbitrary higher order structures.

    Notes
    -----
    neighborhood list is a list of tuples such that each tuple has the form (s,t), s and t
        are indices representing a cell in a higher order structure.
        this structure can be converted to a matrix of size |S|X|T| where |S| is
        the size of the source cells and |T| is the size of the target cells.

    Parameters
    ----------
    ``sparse_array``:  sparse array representing the higher order structure between S and T cells
    """
    src_idx, dst_idx = sparse_array.nonzero()

    if src_dict is None and dst_dict is None:
        return zip(dst_idx, src_idx)
    elif src_dict is not None and dst_dict is not None:
        src_list = [src_dict[i] for i in src_idx]
        dest_list = [dst_dict[i] for i in dst_idx]
        return zip(dest_list, src_list)
    else:
        raise ValueError("src_dict and dst_dict must be either None or both not None")


def neighborhood_list_to_neighborhood_dict(
    n_list: Iterable[tuple[int, int]], src_dict=None, dst_dict=None
) -> dict[int, list[int]]:
    """Convert neighborhood list to neighborhood dictionary for arbitrary higher order structures.

    Notes
    -----
        for every cell i, neighborhood_dict[i] is describe all cells j that are in the neighborhood to j.

    Parameters
    ----------
        ``n_list`` (``list[tuple[int, int]]``): neighborhood list.
    """
    neighborhood_dict = defaultdict(list)
    if src_dict is None and dst_dict is None:
        for src_idx, dst_idx in n_list:
            neighborhood_dict[src_idx].append(dst_idx)
        return neighborhood_dict
    elif src_dict is not None and dst_dict is not None:
        for src_idx, dst_idx in n_list:
            neighborhood_dict[src_dict[src_idx]].append(dst_dict[dst_idx])
        return neighborhood_dict
    else:
        raise ValueError("src_dict and dst_dict must be either None or both not None")


def sparse_array_to_neighborhood_dict(
    sparse_array, src_dict=None, dst_dict=None
) -> dict[int, list[int]]:
    """Convert sparse array to neighborhood dictionary for arbitrary higher order structures.

    Notes
    -----
        neighborhood list is a list of tuples such that each tuple has the form (s,t), s and t
        are indices representing a cell in a higher order structure.
        this structure can be converted to a matrix of size |S|X|T| where |S| is
        the size of the source cells and |T| is the size of the target cells.

    Parameters
    ----------
        ``sparse_array``:  sparse array representing the higher order structure between S and T cells
    """
    return neighborhood_list_to_neighborhood_dict(
        sparse_array_to_neighborhood_list(sparse_array, src_dict, dst_dict)
    )


def incidence_to_adjacency(B, s: Optional[int] = None, signed=False):
    """Get adjacency matrix from boolean incidence matrix for s-metrics.

    Self loops are not supported.
    The adjacency matrix will define an s-linegraph.

    Parameters
    ----------
    B : scipy.sparse.csr.csr_matrix
        incidence matrix of 0's and 1's
    s : int, list, optional, default : 1
        Minimum number of edges shared by neighbors with node.

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
    """
    B = csr_matrix(B)
    if not signed:
        B = abs(B)  # make sure the incidence matrix has only positive entries

    A = B.T @ B
    A.setdiag(0)
    if s is not None:
        A = (A >= s) * 1

    return A
