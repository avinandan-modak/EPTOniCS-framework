"""
eptonics.constitutive
======================

Gauss-point constitutive state-update kernels.

Currently provides the J2 radial return-mapping algorithm with linear
isotropic hardening (manuscript Section 2.1.2 and Algorithm 1),
implemented in :mod:`eptonics.constitutive.j2_return_map`.

The module is designed so that alternative constitutive models
(e.g. kinematic/mixed hardening, finite-strain plasticity) can be
added as additional sibling modules exposing the same
``update_internal_variables(...)`` call signature, without touching
the surrounding finite-element or optimization code (manuscript
Section 5, "Summary, conclusions, and outlook").
"""

from eptonics.constitutive import j2_return_map

__all__ = ["j2_return_map"]
