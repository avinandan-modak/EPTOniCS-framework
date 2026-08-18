"""
kinematics.py
=============
Strain-displacement operator and UFL-symbolic stress invariants
(manuscript Section 3.3, Listing 4).

Pure, stateless UFL-symbolic functions shared verbatim across all
three example drivers.
"""

from ufl import as_vector, sqrt


def epsilon(u):
    """
    Voigt strain-displacement operator (manuscript Eq. 51, Listing 4).

    Returns the symmetric gradient of ``u`` in 6-component Voigt
    notation with engineering shear (``2*epsilon_ij`` at indices 3-5),
    consistent with the Voigt weight ``[1,1,1,2,2,2]`` used in the
    constitutive kernel (:mod:`eptonics.constitutive.j2_return_map`).

    Parameters
    ----------
    u : ufl.core.expr.Expr
        Vector-valued UFL expression on a 3D vector function space.

    Returns
    -------
    ufl.core.expr.Expr
        ``[eps_11, eps_22, eps_33, 2*eps_23, 2*eps_13, 2*eps_12]``.
    """
    return as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2),
                      u[1].dx(2)+u[2].dx(1),
                      u[0].dx(2)+u[2].dx(0),
                      u[0].dx(1)+u[1].dx(0)])


def von_mises_stress(sigma):
    """
    UFL-symbolic von Mises equivalent stress from a 6-component Voigt
    Cauchy stress tensor (manuscript Eq. 7).

    Parameters
    ----------
    sigma : ufl.core.expr.Expr
        6-component Voigt stress: ``[sxx, syy, szz, syz, sxz, sxy]``.

    Returns
    -------
    ufl.core.expr.Expr
        Scalar UFL expression for the von Mises equivalent stress.
    """
    sxx,syy,szz,syz,sxz,sxy = sigma
    return sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)
                +3.0*(syz**2+sxz**2+sxy**2))


def PEEQ_stress(ep):
    """
    UFL-symbolic equivalent plastic strain (PEEQ) from a 6-component
    Voigt plastic strain tensor.

    Parameters
    ----------
    ep : ufl.core.expr.Expr
        6-component Voigt plastic strain (engineering shear convention,
        indices 3-5 carry ``2*eps_p_ij``).

    Returns
    -------
    ufl.core.expr.Expr
        Scalar UFL expression for the equivalent plastic strain.
    """
    EP_H = (ep[0]+ep[1]+ep[2])/3.0
    return sqrt(2./3.)*sqrt((ep[0]-EP_H)**2+(ep[1]-EP_H)**2+(ep[2]-EP_H)**2
                            +2.*(ep[3]**2+ep[4]**2+ep[5]**2))
