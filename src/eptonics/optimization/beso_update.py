"""
beso_update.py
================
BESO density-update rule with the addition-ratio limiter (manuscript
Eqs. 48-49, Section 2.3.5 and 3.8, Listing 14): a single global
threshold is found by descending sort; if too many void elements
simultaneously qualify for reinstatement (addition ratio exceeding
``c_ar_max``), separate addition/deletion thresholds are computed in a
two-pass procedure to prevent early-iteration density oscillation.

:func:`update_rho_BESO_HUANG` and :func:`update_rho_BESO` apply the
rule to a single, explicit density/sensitivity array (serial use).
:func:`update_rho_BESO_global` is the MPI-safe variant operating on
full global arrays gathered onto rank 0.
"""

import numpy as np


def update_rho_BESO_HUANG(rho, dc, vol_dom, vol_el, vol, c_ar_max):
    """
    BESO update (Huang and Xie, 2010 convention) for a single,
    explicit-array domain, with the addition-ratio limiter (Eq. 49).

    Parameters
    ----------
    rho : (n_ele,) ndarray
        Current elemental density field.
    dc : (n_ele,) ndarray
        Filtered, damped sensitivity numbers.
    vol_dom : float
        Total domain volume.
    vol_el : float
        Volume of a single element (uniform mesh).
    vol : float
        Target volume fraction for this design iteration.
    c_ar_max : float
        Maximum addition ratio before the two-threshold limiter engages.

    Returns
    -------
    (n_ele,) ndarray
        Updated elemental density field (``rho_min = 0.001`` for void).
    """
    rho_s_ind = np.where(rho > 0.5); rho_v_ind = np.where(rho < 0.5)
    dc_sort = -np.sort(-dc)
    n_vol   = int((vol*vol_dom)/vol_el)
    alpha_th = dc_sort[n_vol]
    rho_new = np.zeros(rho.shape)
    rho_new[rho_s_ind] = np.maximum(0., np.sign(dc[rho_s_ind]-alpha_th))
    rho_new[rho_v_ind] = np.maximum(0,  np.sign(dc[rho_v_ind]-alpha_th))
    rho_new[rho_new==0] = 0.001
    vol_current   = np.sum(rho)*vol_el
    recovered_ele = np.where(dc[rho_v_ind]-alpha_th > 0)
    c_ar = (vol_el*recovered_ele[0].shape[0])/vol_current
    if c_ar > c_ar_max:
        n_add        = int(np.ceil(c_ar_max*(vol_current/vol_el)))
        alpha_th_add = -np.sort(-dc[rho_v_ind])[n_add]
        n_del        = int((vol_current/vol_el+n_add)-vol*vol_dom/vol_el)
        alpha_th_del = np.sort(dc[rho_s_ind])[n_del-1]
        rho_new[rho_s_ind] = np.maximum(0., np.sign(dc[rho_s_ind]-alpha_th_del))
        rho_new[rho_v_ind] = np.maximum(0,  np.sign(dc[rho_v_ind]-alpha_th_add))
        rho_new[rho_new==0] = 0.001
    return rho_new


def update_rho_BESO(rho_, dc_, vol_, c_ar_max_):
    """
    BESO update for a single-mesh domain, applied directly to
    ``dolfinx.fem.Function`` density/sensitivity fields.
    ``vol_dom`` and ``vol_el`` (total domain volume and per-element
    volume) are read from the calling module's global scope rather
    than passed as arguments.
    """
    rho_arr = rho_.x.array; dc_arr = dc_.x.array
    rho_s   = np.where(rho_arr > 0.5); rho_v = np.where(rho_arr < 0.5)
    dc_sort = -np.sort(-dc_arr)
    n_vol   = int((vol_ * vol_dom) / vol_el)
    alpha   = dc_sort[n_vol]
    rho_new = np.zeros_like(rho_arr)
    rho_new[rho_s] = np.maximum(0., np.sign(dc_arr[rho_s] - alpha))
    rho_new[rho_v] = np.maximum(0.,  np.sign(dc_arr[rho_v] - alpha))
    rho_new[rho_new == 0] = 0.001
    vol_curr = np.sum(rho_arr) * vol_el
    rec      = np.where(dc_arr[rho_v] - alpha > 0)
    c_ar     = (vol_el * rec[0].shape[0]) / vol_curr
    if c_ar > c_ar_max_:
        n_add = int(np.ceil(c_ar_max_ * (vol_curr / vol_el)))
        a_add = -np.sort(-dc_arr[rho_v])[n_add]
        n_del = int((vol_curr / vol_el + n_add) - vol_ * vol_dom / vol_el)
        a_del = np.sort(dc_arr[rho_s])[n_del - 1]
        rho_new[rho_s] = np.maximum(0., np.sign(dc_arr[rho_s] - a_del))
        rho_new[rho_v] = np.maximum(0.,  np.sign(dc_arr[rho_v] - a_add))
        rho_new[rho_new == 0] = 0.001
    return rho_new


def update_rho_BESO_global(rho_arr, dc_arr, vol_, c_ar_max_, vol_dom, vol_el):
    """MPI-safe BESO update operating on global arrays (executed on rank 0)."""
    rho_s = np.where(rho_arr > 0.5)[0]
    rho_v = np.where(rho_arr <= 0.5)[0]

    dc_sort = -np.sort(-dc_arr)
    n_vol = int((vol_ * vol_dom) / vol_el)
    n_vol = min(n_vol, len(dc_sort) - 1)
    n_vol = max(n_vol, 0)

    alpha = dc_sort[n_vol]

    rho_new = np.zeros_like(rho_arr)
    rho_new[rho_s] = np.maximum(0., np.sign(dc_arr[rho_s] - alpha))
    rho_new[rho_v] = np.maximum(0., np.sign(dc_arr[rho_v] - alpha))
    rho_new[rho_new == 0] = 0.001

    vol_curr = np.sum(rho_arr) * vol_el
    rec = np.where(dc_arr[rho_v] - alpha > 0)[0]
    c_ar = (vol_el * rec.shape[0]) / vol_curr if vol_curr > 0 else 0

    if c_ar > c_ar_max_:
        n_add = int(np.ceil(c_ar_max_ * (vol_curr / vol_el)))
        n_add = min(n_add, len(rho_v))
        if n_add > 0:
            a_add = -np.sort(-dc_arr[rho_v])[n_add]
        else:
            a_add = np.min(dc_arr)

        n_del = int((vol_curr / vol_el + n_add) - vol_ * vol_dom / vol_el)
        n_del = min(n_del, len(rho_s))
        if n_del > 0:
            a_del = np.sort(dc_arr[rho_s])[n_del - 1]
        else:
            a_del = np.max(dc_arr)

        rho_new[rho_s] = np.maximum(0., np.sign(dc_arr[rho_s] - a_del))
        rho_new[rho_v] = np.maximum(0., np.sign(dc_arr[rho_v] - a_add))
        rho_new[rho_new == 0] = 0.001

    return rho_new
