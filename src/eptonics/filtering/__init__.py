"""
eptonics.filtering
=====================

Distance-based spatial sensitivity filter (manuscript Eqs. 43-44,
Section 3.8.1, Listing 12), using ``scipy.spatial.cKDTree`` to query
only elements within the filter radius (performance mechanism ★3).
This single-array form is for serial / single-rank use; the example
drivers apply the same weighting inline on MPI-gathered global arrays.
"""

from eptonics.filtering.sensitivity_filter import sensitivity_filter

__all__ = ["sensitivity_filter"]
