"""
EPTONiCS — Elastoplastic Topology Optimization in FEniCSx
===========================================================

Library components for the EPTONiCS framework described in

    Modak, A., Chowdhury, R., Gangwar, T.
    "EPTONiCS: An Efficient FEniCSx Implementation for Three-Dimensional
    Topology Optimization of Elastoplastic Structures."
    Structural and Multidisciplinary Optimization (SMO).

EPTONiCS couples bi-directional evolutionary structural optimization
(BESO) with J2 elastoplasticity and a path-dependent adjoint
sensitivity analysis, implemented on the FEniCSx finite-element platform.

Package layout
--------------
``eptonics.constitutive``
    Gauss-point J2 radial return-mapping algorithm with linear isotropic
    hardening (manuscript Section 2.1.2 / Algorithm 1).
``eptonics.fem``
    Stateless finite-element helpers: the Voigt strain-displacement
    operator (``kinematics``) and a Function-array reset utility
    (``utils``).
``eptonics.solvers``
    Persistent-PETSc-KSP construction and assemble-and-solve plumbing
    (manuscript Section 3.5.1, performance mechanism ★2).
``eptonics.visualization``
    Element-averaged (DG0), void-masked von Mises stress and PEEQ fields
    for XDMF output.
``eptonics.optimization``
    BESO density-update routines with the addition-ratio limiter
    (manuscript Eqs. 48-49, Section 3.8).
``eptonics.filtering``
    Distance-based spatial sensitivity filter (manuscript Eqs. 43-44,
    Section 3.8.1), sparse cKDTree implementation (performance mechanism ★3).
``eptonics.mpi``
    Distributed k-th-value selection via MPI bisection (Section 3.9),
    replacing a gather-to-rank-0 / serial-sort / scatter bottleneck.
``eptonics.sensitivities``, ``eptonics.utils``, ``eptonics.io``
    Reserved; intentionally empty in this release.
"""

__all__ = [
    "constitutive",
    "fem",
    "solvers",
    "visualization",
    "optimization",
    "filtering",
    "mpi",
    "sensitivities",
    "utils",
    "io",
]
__version__ = "1.1.0"

# Subpackages are not eagerly imported: several require a full DOLFINx/PETSc
# installation while eptonics.constitutive only needs NumPy (and optionally
# Numba). Import the specific subpackage you need, e.g.:
#   from eptonics.constitutive import j2_return_map
#   from eptonics.fem.kinematics import epsilon
