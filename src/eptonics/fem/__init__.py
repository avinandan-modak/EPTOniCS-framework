"""
eptonics.fem
=============

Finite-element helper functions (manuscript Section 3.3).

Modules
-------
``kinematics``
    Voigt strain-displacement operator and UFL-symbolic stress invariants.
``utils``
    State reset utilities for ``dolfinx.fem.Function`` arrays.
"""

from eptonics.fem import kinematics, utils

__all__ = ["kinematics", "utils"]
