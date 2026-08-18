"""
eptonics.optimization
========================

BESO density-update routines with the addition-ratio limiter
(manuscript Section 2.3.5, Eqs. 48-49; Section 3.8, Listing 14).

:func:`~beso_update.update_rho_BESO_global` — MPI-safe variant
    Operates on global arrays, for use inside a rank-0 gather block.
:func:`~beso_update.update_rho_BESO_HUANG` and
:func:`~beso_update.update_rho_BESO` — single-array variants
    The same update rule for serial / single-array use.
"""

from eptonics.optimization.beso_update import (
    update_rho_BESO_HUANG,
    update_rho_BESO,
    update_rho_BESO_global,
)

__all__ = ["update_rho_BESO_HUANG", "update_rho_BESO", "update_rho_BESO_global"]
