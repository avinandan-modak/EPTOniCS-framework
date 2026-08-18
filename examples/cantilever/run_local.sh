#!/bin/bash
# run_local.sh - Example 1: Cantilever beam (manuscript Section 4.1)
# Optimized for a Threadripper PRO 5995WX (64 physical cores) reference
# workstation, as used for the wall-time figures reported in Section 4.1.

# ==============================================================================
# 1. CRITICAL: Prevent Thread Oversubscription
# ==============================================================================
# We force all math/Numba libraries to use EXACTLY 1 thread per MPI rank.
# If we don't do this, PETSc/Numba will see 128 logical cores and spawn
# 128 threads per rank, resulting in 64x128 = 8,192 threads and crashing performance.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# ==============================================================================
# 2. Activate Conda / Python Environment
# ==============================================================================
# Adjust the path below to match your actual miniconda/anaconda installation.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fenicsx-hpc
#
# The eptonics package (src/eptonics) must be installed beforehand, e.g.:
#   pip install -e /path/to/EPTONICS
# See ../../docs/installation.md for the full environment setup.
echo "Using python: $(which python)"

# ==============================================================================
# 3. Run the Code
# ==============================================================================
echo "========================================================"
echo " Starting BESO Cantilever Beam on 64 Physical Cores (Pure MPI)"
echo "========================================================"

# Launch with 64 MPI ranks
mpirun -np 64 python -u run_cantilever.py

echo "Job finished at $(date)"
