"""Methods to compute the Delaunay triangulation of a set of points."""

import numpy as np
from scipy.spatial import Delaunay

from toponetx.classes import SimplicialComplex


def delaunay_triangulation(points: np.ndarray) -> SimplicialComplex:
    """
    Compute the Delaunay triangulation of a set of points in the plane.

    The resulting simplicial complex has nodes 0 to (n-1), where n is the number of
    points following the coordinate order in the input array.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 2) containing the coordinates of the points.

    Returns
    -------
    SimplicialComplex
        The Delaunay triangulation as a SimplicialComplex object.
    """
    triangles = Delaunay(points)
    return SimplicialComplex(triangles.simplices)
