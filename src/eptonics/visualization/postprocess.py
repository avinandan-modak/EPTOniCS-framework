"""
postprocess.py
================
Element-averaged (DG0), void-masked post-processing fields computed
from converged Gauss-point states for XDMF visualization.
Voigt ordering: [xx, yy, zz, yz, xz, xy].
"""

import numpy as np


def compute_vonMises_DG0(Sig, rho_arr):
    """
    Element-averaged von Mises stress, NaN on void elements (rho <= 0.5).

    Parameters
    ----------
    Sig : (n_gp, 6) ndarray
        Converged Cauchy stress at all Gauss points.
    rho_arr : (n_ele,) ndarray
        Density field before the BESO update.

    Returns
    -------
    (n_ele,) ndarray
    """
    s11, s22, s33 = Sig[:, 0], Sig[:, 1], Sig[:, 2]
    s23, s13, s12 = Sig[:, 3], Sig[:, 4], Sig[:, 5]
    vm_gp  = np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2)
                     + 3.0*(s23**2 + s13**2 + s12**2))
    vm_ele = vm_gp.reshape(-1, 8).mean(axis=1)
    vm_ele[rho_arr <= 0.5] = np.nan
    return vm_ele


def compute_PEEQ_DG0(Ep, rho_arr):
    """
    Element-averaged equivalent plastic strain (PEEQ), NaN on void elements (rho <= 0.5).

    Parameters
    ----------
    Ep : (n_gp, 6) ndarray
        Converged plastic strain at all Gauss points.
    rho_arr : (n_ele,) ndarray
        Density field before the BESO update.

    Returns
    -------
    (n_ele,) ndarray
    """
    ep1, ep2, ep3 = Ep[:, 0], Ep[:, 1], Ep[:, 2]
    ep4, ep5, ep6 = Ep[:, 3], Ep[:, 4], Ep[:, 5]
    EP_H    = (ep1 + ep2 + ep3) / 3.0
    peeq_gp = np.sqrt(2.0/3.0) * np.sqrt(
                  (ep1-EP_H)**2 + (ep2-EP_H)**2 + (ep3-EP_H)**2
                  + 2.0*(ep4**2 + ep5**2 + ep6**2))
    peeq_ele = peeq_gp.reshape(-1, 8).mean(axis=1)
    peeq_ele[rho_arr <= 0.5] = np.nan
    return peeq_ele
