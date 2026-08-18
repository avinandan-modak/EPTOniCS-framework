"""
j2_return_map.py
=================
Gauss-point J2 (von Mises) return-mapping algorithm with linear isotropic
hardening, implementing manuscript Section 2.1.2 (constitutive model) and
Algorithm 1 (radial return mapping), evaluated simultaneously over all
Gauss points of the mesh.

Two interchangeable backends are dispatched through the single public
entry point :func:`update_internal_variables`:

FAST path — Numba ``@njit(parallel=True, cache=True)``
    True multi-threaded execution (``prange``) with no GIL
    (manuscript Section 3.4.1, performance mechanism).
    Active when ``numba`` is importable and ``n_gp >= NUMBA_THRESHOLD``.
    Disable via the environment variable ``BESO_NO_NUMBA=1``.

SAFE path — vectorised NumPy
    Portable fallback; used when Numba is unavailable or the problem
    is smaller than :data:`NUMBA_THRESHOLD`. Numerically identical.

Convention (matches manuscript Algorithm 1 and Eqs. 19-26)
------------------------------------------------------------
3D Voigt ordering    : [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]
Engineering shear    : gamma_ij = 2 * epsilon_ij, stored at Voigt indices 3, 4, 5
J2 weight vector     : W = [1, 1, 1, 2, 2, 2]  (Frobenius inner product weight)

Constitutive model   : J2 plasticity with linear isotropic hardening
  yield function      : f = ||s||_F - sqrt(2/3) * (sigma_y + H * eqp)         (Eq. 8, 20)
  flow rule           : Delta(eps_p) = Delta(gamma) * n_hat                    (Eq. 9, 24)
  hardening law       : Delta(eqp)   = sqrt(2/3) * Delta(gamma)                (Eq. 10, 25)
  plastic multiplier  : Delta(gamma) = f_trial / (2*mu + 2*H/3)                (Eq. 22)
  stress update       : sigma_{n+1}  = sigma_trial - 2*mu*Delta(gamma)*n_hat   (Eq. 23)
  consistent tangent  : C_ep = 3*K*P_vol + 2*mu*theta*P_dev
                              - 2*mu*(A-B)*(n_hat (x) n_hat)                    (Eq. 26)

Setting H = 0 recovers perfect (non-hardening) plasticity.
"""

import os
import numpy as np

# Backend selection
_NO_NUMBA = os.environ.get("BESO_NO_NUMBA", "0") == "1"
NUMBA_THRESHOLD = 500   # minimum Gauss-point count to justify JIT dispatch overhead

try:
    if _NO_NUMBA:
        raise ImportError("disabled via BESO_NO_NUMBA=1")
    from numba import njit, prange
    HAS_NUMBA = True
    print("[Plasticity] Numba backend active  (parallel JIT, cache=True).")
except ImportError as _exc:
    HAS_NUMBA = False
    print(f"[Plasticity] NumPy backend active  ({_exc}).")

# Constant projection tensors (Voigt, engineering-shear convention)
# _VOL, _DEV: volumetric / deviatoric fourth-order projections P_vol, P_dev (Eq. 26)
# _W: Frobenius-norm weight [1,1,1,2,2,2] for engineering-shear indices 3-5
_I   = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
_VOL = (1.0/3.0) * np.array(
    [[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0],
     [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]], dtype=float)
_DEV   = _I - _VOL
_W     = np.array([1., 1., 1., 2., 2., 2.])
_sqrt23 = np.sqrt(2.0 / 3.0)          # used in yield & eqp update


# ===========================================================================
# Numba kernel — one Gauss point per thread (Algorithm 1, Sec. 3.4.1)
# ===========================================================================
if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _numba_return_map(E, Ep_old, eqp_old,
                          K_vec, G_vec, sy_vec, H_vec,
                          Sig, Ep, eqp_new, Cep):
        """
        Parallel radial-return map with linear isotropic hardening
        (manuscript Algorithm 1). Each prange iteration processes one
        Gauss point independently — no cross-point data dependency.

        Per-point update (Algorithm 1):
            f_trial       = ||s_trial||_F - sqrt(2/3) * (sigma_y + H * eqp_n)   (line 4)
            Delta(gamma)  = f_trial / (2*mu + 2*H/3)                            (line 9)
            Delta(eqp)    = sqrt(2/3) * Delta(gamma)                            (line 13)

        Elastic tangent (Eq. 6):
            C_ii  = K + 4*mu/3   (normal-normal diagonal)
            C_ij  = K - 2*mu/3   (normal-normal off-diagonal, i != j)
            C_ii  = mu           (shear-shear diagonal)

        Consistent elastoplastic tangent (Eq. 26):
            C_ep = 3*K*P_vol + 2*mu*theta*P_dev - 2*mu*(A-B)*(n_hat (x) n_hat)
            theta = 1 - 2*mu*Delta(gamma) / ||s_trial||
            A - B = 2*mu / (2*mu + 2*H/3) - 2*mu*Delta(gamma) / ||s_trial||
        """
        _sqrt23_nb = 0.816496580927726   # sqrt(2/3)
        n_gp = E.shape[0]

        for gp in prange(n_gp):
            K  = K_vec[gp]
            G  = G_vec[gp]
            sy = sy_vec[gp]
            H  = H_vec[gp]

            # elastic trial strain: eps_e = eps_{n+1} - eps_p_n (Eq. 19)
            ee0 = E[gp,0]-Ep_old[gp,0];  ee1 = E[gp,1]-Ep_old[gp,1]
            ee2 = E[gp,2]-Ep_old[gp,2];  ee3 = E[gp,3]-Ep_old[gp,3]
            ee4 = E[gp,4]-Ep_old[gp,4];  ee5 = E[gp,5]-Ep_old[gp,5]

            p_ee = (ee0+ee1+ee2)/3.0
            lam  = 3.0*K - 2.0*G          # lam*p_ee = lambda_Lame * tr(eps)

            # trial stress: sigma_trial = C^e : (eps_{n+1} - eps_p_n) (Eq. 19)
            s0=lam*p_ee+2.0*G*ee0; s1=lam*p_ee+2.0*G*ee1; s2=lam*p_ee+2.0*G*ee2
            s3=G*ee3;               s4=G*ee4;               s5=G*ee5

            p_tr=(s0+s1+s2)/3.0
            d0=s0-p_tr; d1=s1-p_tr; d2=s2-p_tr
            d3=s3;      d4=s4;      d5=s5

            # Frobenius norm of deviatoric trial stress ||s_trial||_F
            nrm = (d0*d0+d1*d1+d2*d2 + 2.0*(d3*d3+d4*d4+d5*d5))**0.5

            # trial yield function (Eq. 20, Algorithm 1 line 4)
            f_tr = nrm - _sqrt23_nb*(sy + H*eqp_old[gp])

            # elastic tangent constants
            Cnn_d = K + 4.0*G/3.0
            Cnn_o = K - 2.0*G/3.0

            if f_tr <= 1.0e-12:
                # ELASTIC STEP (Algorithm 1, line 6)
                Sig[gp,0]=s0; Sig[gp,1]=s1; Sig[gp,2]=s2
                Sig[gp,3]=s3; Sig[gp,4]=s4; Sig[gp,5]=s5
                for k in range(6):
                    Ep[gp,k] = Ep_old[gp,k]
                eqp_new[gp] = eqp_old[gp]

                # C^ep = C^e
                for ii in range(6):
                    for jj in range(6):
                        Cep[gp,ii,jj] = 0.0
                Cep[gp,0,0]=Cnn_d; Cep[gp,1,1]=Cnn_d; Cep[gp,2,2]=Cnn_d
                Cep[gp,0,1]=Cnn_o; Cep[gp,1,0]=Cnn_o
                Cep[gp,0,2]=Cnn_o; Cep[gp,2,0]=Cnn_o
                Cep[gp,1,2]=Cnn_o; Cep[gp,2,1]=Cnn_o
                Cep[gp,3,3]=G; Cep[gp,4,4]=G; Cep[gp,5,5]=G

            else:
                # PLASTIC STEP (Algorithm 1, lines 8-15)
                # plastic multiplier (Eq. 22, line 9)
                denom   = 2.0*G + 2.0*H/3.0
                dgamma  = f_tr / denom

                inv_nrm = 1.0/nrm
                # unit normal to yield surface: n_hat = s_trial / ||s_trial|| (Eq. 21, line 10)
                n0=d0*inv_nrm; n1=d1*inv_nrm; n2=d2*inv_nrm
                n3=d3*inv_nrm; n4=d4*inv_nrm; n5=d5*inv_nrm

                # radial-return stress update (Eq. 23, line 11)
                c = 2.0*G*dgamma
                Sig[gp,0]=s0-c*n0; Sig[gp,1]=s1-c*n1; Sig[gp,2]=s2-c*n2
                Sig[gp,3]=s3-c*n3; Sig[gp,4]=s4-c*n4; Sig[gp,5]=s5-c*n5

                # plastic strain update (Eq. 24, line 12)
                # engineering-shear indices (3,4,5) carry 2*eps_ij → multiply by 2
                Ep[gp,0]=Ep_old[gp,0]+dgamma*n0
                Ep[gp,1]=Ep_old[gp,1]+dgamma*n1
                Ep[gp,2]=Ep_old[gp,2]+dgamma*n2
                Ep[gp,3]=Ep_old[gp,3]+2.0*dgamma*n3
                Ep[gp,4]=Ep_old[gp,4]+2.0*dgamma*n4
                Ep[gp,5]=Ep_old[gp,5]+2.0*dgamma*n5

                # equivalent plastic strain update (Eq. 25, line 13)
                eqp_new[gp] = eqp_old[gp] + _sqrt23_nb*dgamma

                # consistent elastoplastic tangent (Eq. 26)
                theta     = 1.0 - 2.0*G*dgamma*inv_nrm
                a_minus_b = 2.0*G/denom - 2.0*G*dgamma*inv_nrm
                twoG_amb  = 2.0*G*a_minus_b

                nv = (n0,n1,n2,n3,n4,n5)
                for ii in range(6):
                    for jj in range(6):
                        nn = twoG_amb * nv[ii]*nv[jj]
                        if ii < 3 and jj < 3:
                            if ii == jj:
                                Cep[gp,ii,jj] = K + 4.0*G*theta/3.0 - nn
                            else:
                                Cep[gp,ii,jj] = K - 2.0*G*theta/3.0 - nn
                        elif ii == jj:
                            Cep[gp,ii,jj] = G*theta - nn
                        else:
                            Cep[gp,ii,jj] = -nn


# ===========================================================================
# NumPy backend (vectorised over all Gauss points simultaneously)
# ===========================================================================

def _numpy_return_map(E, Ep_old, eqp_old, Bulk_mat, Shear_mat, Sig_y, Hard_mat):
    """
    Vectorised counterpart of :func:`_numba_return_map`.
    Identical radial return-mapping algorithm using NumPy broadcasting
    over the leading Gauss-point axis (manuscript Section 3.4.1).
    """
    K  = Bulk_mat.ravel()
    G  = Shear_mat.ravel()
    sy = Sig_y.ravel()
    H  = Hard_mat.ravel()
    n_gp = E.shape[0]

    ee   = E - Ep_old
    p_ee = ee[:, :3].sum(axis=1) / 3.0
    lame_diff = 3.0*K - 2.0*G      # lame_diff*p_ee = lambda_Lame * tr(eps)

    sig_tr = np.empty((n_gp, 6))
    sig_tr[:,0] = lame_diff*p_ee + 2.0*G*ee[:,0]
    sig_tr[:,1] = lame_diff*p_ee + 2.0*G*ee[:,1]
    sig_tr[:,2] = lame_diff*p_ee + 2.0*G*ee[:,2]
    sig_tr[:,3] = G*ee[:,3]
    sig_tr[:,4] = G*ee[:,4]
    sig_tr[:,5] = G*ee[:,5]

    p_tr = sig_tr[:,:3].sum(axis=1) / 3.0
    sD   = sig_tr.copy()
    sD[:,:3] -= p_tr[:,None]

    # Frobenius norm of deviatoric trial stress ||s_trial||_F
    norm_sD = np.sqrt((sD**2 * _W).sum(axis=1))

    # trial yield function (Eq. 20)
    f_tr    = norm_sD - _sqrt23*(sy + H*eqp_old)
    plastic = f_tr > 1.0e-12
    elastic = ~plastic

    Sig     = np.empty((n_gp, 6))
    Ep      = Ep_old.copy()
    eqp_new = eqp_old.copy()
    Cep     = np.empty((n_gp, 6, 6))

    if elastic.any():
        Sig[elastic] = sig_tr[elastic]
        Ke=K[elastic]; Ge=G[elastic]
        Cep[elastic] = (3.0*Ke[:,None,None]*_VOL[None]
                      + 2.0*Ge[:,None,None]*_DEV[None])

    if plastic.any():
        Gp=G[plastic]; Kp=K[plastic]; Hp=H[plastic]
        nrm=norm_sD[plastic]; sDp=sD[plastic]

        # plastic multiplier (Eq. 22)
        denom  = 2.0*Gp + 2.0*Hp/3.0
        dgamma = f_tr[plastic] / denom

        n_hat  = sDp/nrm[:,None]
        theta  = 1.0 - (2.0*Gp*dgamma)/nrm

        # A - B = 2*mu/denom - (1 - theta)
        a_minus_b = 2.0*Gp/denom - (1.0 - theta)

        # stress update (Eq. 23)
        Sig[plastic] = sig_tr[plastic] - 2.0*Gp[:,None]*dgamma[:,None]*n_hat

        # plastic strain update (Eq. 24)
        deps = dgamma[:,None]*n_hat
        deps[:,3:] *= 2.0              # engineering-shear correction
        Ep[plastic] = Ep_old[plastic] + deps

        # equivalent plastic strain update (Eq. 25)
        eqp_new[plastic] = eqp_old[plastic] + _sqrt23*dgamma

        # consistent tangent (Eq. 26)
        n_outer = np.einsum('ni,nj->nij', n_hat, n_hat)
        Cep[plastic] = (3.0*Kp[:,None,None]*_VOL[None]
                      + 2.0*Gp[:,None,None]*theta[:,None,None]*_DEV[None]
                      - 2.0*Gp[:,None,None]*a_minus_b[:,None,None]*n_outer)

    return Sig, Ep, eqp_new, Cep


# ===========================================================================
# Public API — auto-selects backend
# ===========================================================================

def update_internal_variables(E, Ep_old, eqp_old, Bulk_mat, Shear_mat, Sig_y, Hard_mat):
    """
    Radial return map for 3D J2 plasticity with linear isotropic hardening,
    evaluated over all Gauss points (manuscript Section 2.1.2, Algorithm 1).
    Set ``Hard_mat = 0`` to recover perfect (non-hardening) plasticity.

    Dispatches to the Numba-parallel kernel when available and
    ``n_gp >= NUMBA_THRESHOLD``; otherwise uses the vectorised NumPy
    fallback. Both backends are numerically equivalent.

    Parameters
    ----------
    E : (n_gp, 6) ndarray
        Total strain at every Gauss point, Voigt notation with
        engineering shear (indices 3-5 carry 2*epsilon_ij).
    Ep_old : (n_gp, 6) ndarray
        Converged plastic strain at the previous load step, t_n.
    eqp_old : (n_gp,) ndarray
        Converged equivalent plastic strain at t_n.
    Bulk_mat : (n_gp, 1) ndarray
        Bulk modulus, K = E / [3(1 - 2*nu)].
    Shear_mat : (n_gp, 1) ndarray
        Shear modulus, G = E / [2(1 + nu)].
    Sig_y : (n_gp, 1) ndarray
        Initial uniaxial yield stress, sigma_y.
    Hard_mat : (n_gp, 1) ndarray
        Linear isotropic hardening modulus, H.

    Returns
    -------
    Sig : (n_gp, 6) ndarray
        Updated Cauchy stress at t_{n+1}.
    Ep : (n_gp, 6) ndarray
        Updated plastic strain at t_{n+1}.
    eqp_new : (n_gp,) ndarray
        Updated equivalent plastic strain at t_{n+1}.
    Cep : (n_gp, 6, 6) ndarray
        Consistent elastoplastic tangent modulus C^ep (Eq. 26).
    """
    n_gp = E.shape[0]

    if HAS_NUMBA and n_gp >= NUMBA_THRESHOLD:
        Sig     = np.empty((n_gp, 6))
        Ep      = np.empty((n_gp, 6))
        eqp_new = np.empty(n_gp)
        Cep     = np.zeros((n_gp, 6, 6))
        _numba_return_map(
            np.ascontiguousarray(E),
            np.ascontiguousarray(Ep_old),
            np.ascontiguousarray(eqp_old),
            np.ascontiguousarray(Bulk_mat.ravel()),
            np.ascontiguousarray(Shear_mat.ravel()),
            np.ascontiguousarray(Sig_y.ravel()),
            np.ascontiguousarray(Hard_mat.ravel()),
            Sig, Ep, eqp_new, Cep)
        return Sig, Ep, eqp_new, Cep

    return _numpy_return_map(E, Ep_old, eqp_old, Bulk_mat, Shear_mat, Sig_y, Hard_mat)
