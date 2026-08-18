"""
parallel_order_statistics.py
===============================
Distributed order-statistic selection across MPI ranks via bisection over
``MPI_Allreduce`` calls (manuscript Section 3.9).

Replaces a centralized sort by resolving the required global quantile
iteratively across all ranks with O(60) small Allreduce calls.
"""

import numpy as np
from mpi4py import MPI


def _parallel_kth_largest(comm_, local_vals, k):
    """Return the k-th largest value across all ranks (0-indexed)."""
    if local_vals.size == 0:
        lo_loc, hi_loc = np.inf, -np.inf
    else:
        lo_loc, hi_loc = float(local_vals.min()), float(local_vals.max())
    lo = comm_.allreduce(lo_loc, op=MPI.MIN)
    hi = comm_.allreduce(hi_loc, op=MPI.MAX)
    if lo >= hi:
        return float(lo)
    for _ in range(60):
        mid   = 0.5 * (lo + hi)
        count = int(comm_.allreduce(int(np.sum(local_vals > mid)), op=MPI.SUM))
        if count > k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _parallel_kth_smallest(comm_, local_vals, k):
    """Return the k-th smallest value across all ranks (0-indexed)."""
    if local_vals.size == 0:
        lo_loc, hi_loc = np.inf, -np.inf
    else:
        lo_loc, hi_loc = float(local_vals.min()), float(local_vals.max())
    lo = comm_.allreduce(lo_loc, op=MPI.MIN)
    hi = comm_.allreduce(hi_loc, op=MPI.MAX)
    if lo >= hi:
        return float(lo)
    for _ in range(60):
        mid   = 0.5 * (lo + hi)
        count = int(comm_.allreduce(int(np.sum(local_vals < mid)), op=MPI.SUM))
        if count > k:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
