"""
eptonics.solvers
==================

Persistent PETSc KSP solver construction and assemble-and-solve wrappers
(manuscript Section 3.5.1, performance mechanism ★2).
"""

from eptonics.solvers.petsc_backend import _build_ksp, _petsc_solve

__all__ = ["_build_ksp", "_petsc_solve"]
