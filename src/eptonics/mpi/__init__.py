"""
eptonics.mpi
==============

Distributed order-statistic selection via MPI bisection (manuscript
Section 3.9). Replaces a gather-to-rank-0 / serial-sort / scatter
pattern with O(60) small ``MPI_Allreduce`` calls per threshold search,
with no bulk data movement.

See :mod:`eptonics.mpi.parallel_order_statistics` for
:func:`_parallel_kth_largest` and :func:`_parallel_kth_smallest`.
"""

from eptonics.mpi.parallel_order_statistics import (
    _parallel_kth_largest,
    _parallel_kth_smallest,
)

__all__ = ["_parallel_kth_largest", "_parallel_kth_smallest"]
