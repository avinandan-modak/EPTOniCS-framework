"""
eptonics.visualization
========================

Post-processing helpers converting converged Gauss-point state (Cauchy stress,
plastic strain) into element-averaged DG0 fields for XDMF output, with void
elements masked to NaN.
"""

from eptonics.visualization.postprocess import compute_vonMises_DG0, compute_PEEQ_DG0

__all__ = ["compute_vonMises_DG0", "compute_PEEQ_DG0"]
