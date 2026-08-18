"""
eptonics.sensitivities
=========================

Reserved for a shared path-dependent adjoint sensitivity assembly
routine (manuscript Eq. 41-42, Section 3.7). Intentionally empty in
this release.

Each example driver defines its own ``sensitivity_func()`` with a
signature suited to its own geometry: the cantilever version takes
``V0`` and ``dx_q`` as explicit arguments, while the L-bracket and
portal versions read them from the enclosing module's scope instead.
This subpackage is kept as the natural home for a unified version if
a future contribution adds one.
"""
