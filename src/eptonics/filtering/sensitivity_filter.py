"""
sensitivity_filter.py
========================
Distance-based spatial sensitivity filter (manuscript Eqs. 43-44,
Section 3.8.1, Listing 12), implemented with a sparse ``scipy.spatial.cKDTree``
neighbourhood query (performance mechanism ★3).
"""

import numpy as np
from scipy.spatial import cKDTree


def sensitivity_filter(dc_arr, rho_func, rmin):
    """
    Apply distance-based spatial sensitivity filtering (Eqs. 43-44).

    Parameters
    ----------
    dc_arr : (n_ele,) ndarray
        Elemental sensitivity values.
    rho_func : dolfinx.fem.Function
        Elemental density field on DG0 space.
    rmin : float
        Filter radius.

    Returns
    -------
    (n_ele,) ndarray
        Filtered sensitivity array.
    """
    coords    = rho_func.function_space.tabulate_dof_coordinates()[:, :3]
    tree      = cKDTree(coords)
    nbr_lists = tree.query_ball_point(coords, r=rmin)
    n = len(dc_arr)
    num = np.empty(n); den = np.empty(n)
    for i, nbrs in enumerate(nbr_lists):
        nbrs  = np.asarray(nbrs, dtype=int)
        dists = np.linalg.norm(coords[nbrs] - coords[i], axis=1)
        w     = np.maximum(0.0, rmin - dists)
        num[i] = w @ dc_arr[nbrs]
        den[i] = w.sum()
    return np.where(den > 0, num/den, dc_arr)
