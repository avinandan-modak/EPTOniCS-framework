# Implementation details

This page summarizes how the EPTONiCS implementation realizes the
theory of `docs/theory.md`, and documents the five targeted performance
mechanisms described in manuscript Section 3.

## Repository / code architecture

Each of the three numerical examples (manuscript Section 4) is a
**self-contained driver script** under `examples/<name>/`, matching the
reproducibility package described in manuscript Appendix A: one main
driver plus whatever shared library functions it actually calls. This
keeps every example runnable in isolation with a single `mpirun`
invocation — there is no hidden cross-example shared mutable state.

```
EPTONICS/
├── src/eptonics/
│   ├── __init__.py
│   ├── constitutive/
│   │   └── j2_return_map.py      # Algorithm 1 (Section 2.1.2 / 3.4)
│   ├── fem/
│   │   ├── kinematics.py          # epsilon(), von_mises_stress(), PEEQ_stress() (Sec. 3.3)
│   │   └── utils.py               # zero()
│   ├── solvers/
│   │   └── petsc_backend.py       # _build_ksp(), _petsc_solve() (Sec. 3.5.1)
│   ├── visualization/
│   │   └── postprocess.py         # compute_vonMises_DG0(), compute_PEEQ_DG0()
│   ├── optimization/
│   │   └── beso_update.py         # update_rho_BESO*() (Eq. 48-49)
│   ├── filtering/
│   │   └── sensitivity_filter.py  # sensitivity_filter() (Eq. 43-44)
│   └── mpi/, sensitivities/, utils/, io/   # reserved, intentionally empty (see below)
├── examples/
│   ├── cantilever/                # Section 4.1
│   ├── l_bracket/                 # Section 4.2
│   └── portal_frame/              # Section 4.3
├── docs/
├── tests/
├── scripts/                       (reserved for future batch/HPC tooling)
└── data/                          (reserved for mesh/output artifacts)
```

`src/eptonics/` holds every component that is common to all three
examples:

1. **The constitutive kernel** (`eptonics.constitutive.j2_return_map`)
   — the Gauss-point J2 return-mapping algorithm, identical for every
   benchmark since the material model does not depend on geometry.
2. **Shared FEM, solver, post-processing, optimization, and filtering
   helpers** — the Voigt strain operator and UFL stress invariants
   (`eptonics.fem.kinematics`), a Function-array reset helper
   (`eptonics.fem.utils`), the persistent-PETSc-KSP plumbing
   (`eptonics.solvers.petsc_backend`), the DG0 post-processing fields
   (`eptonics.visualization.postprocess`), the BESO density-update
   rule (`eptonics.optimization.beso_update`), and the sensitivity
   filter (`eptonics.filtering.sensitivity_filter`). None of these
   depend on a particular example's geometry or boundary conditions.
3. **The distributed BESO threshold search**
   (`eptonics.mpi.parallel_order_statistics`): a
   bisection-over-`MPI_Allreduce` order-statistic selection used by
   every example's density update, described in detail below.

**What stays in each driver, and why.** Mesh/submesh construction,
boundary conditions, the path-dependent adjoint sensitivity assembly
(`sensitivity_func`), the rank-0-only print helper (`pout`), the
Newton-Raphson corrector/predictor wrappers, and the full three-level
BESO/load/Newton loop (including the MPI gather → update → scatter
sequence) are specific to each example and remain local to its driver:

- `pout()` is defined inline in each driver, immediately after the
  block that sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
  `OPENBLAS_NUM_THREADS` / `NUMBA_NUM_THREADS` and before the first
  import of any PETSc/DOLFINx/Numba module (see "Thread-oversubscription
  guard" below). This ordering is deliberate: `mpi4py` must be
  importable at this point without disturbing the thread-environment
  variables that have to be set *before* any PETSc/Numba module loads
  — see `src/eptonics/mpi/__init__.py` for the full reasoning behind
  keeping this helper close to that guard rather than importing it
  from a shared module.
- `sensitivity_func()` differs genuinely between examples, not just
  cosmetically: the cantilever driver's version takes `V0`/`dx_q` as
  explicit arguments, while the L-bracket/portal drivers' versions
  read them from the enclosing module's scope instead — see
  `src/eptonics/sensitivities/__init__.py`.

The constitutive kernel is the primary example of the intended
extension point: alternative constitutive models can be added as
sibling modules under `eptonics.constitutive` in the future without
touching any example driver (manuscript Section 5, "Summary,
conclusions, and outlook").


## Mesh, function spaces, and quadrature storage (Section 3.1)

Each example discretizes its domain with first-order hexahedral (Q1)
elements. Displacement lives in a vector-valued CG1 space; the design
density lives in a scalar DG0 space (one value per element). The
constitutive history variables (stress, plastic strain, equivalent
plastic strain, consistent tangent) are stored directly at quadrature
points via `basix.ufl.quadrature_element` (degree 2, giving 8 Gauss
points per hexahedron) — never projected to a nodal space, which would
introduce a smoothing error that degrades element-wise sensitivities
and can destabilize BESO convergence (Section 3.1).

The L-bracket and portal-frame examples additionally carve their true
non-box geometry out of a full bounding-box mesh via
`dolfinx.mesh.create_submesh`, so that all finite-element analysis runs
exclusively on the physically meaningful mesh — there is no
void-quadrant density suppression or masking anywhere in the analysis.

## Five performance mechanisms (Section 3, ⋆1-⋆5 in the driver comments)

1. **Pre-compiled UFL forms** (Section 3.5.1) — each distinct
   variational form (predictor / corrector / adjoint / residual) is
   compiled via FFCx exactly once at start-up (`fem.form(...)`) and the
   resulting `fem.Form` objects are reused for the entire run.
2. **Persistent PETSc KSP solvers with GAMG** (Section 3.5.1,
   `eptonics.solvers.petsc_backend`) — one KSP object per solve type is
   built once; only matrix *values* are reassembled each Newton step,
   preserving the sparsity pattern and, for GAMG, the algebraic-
   multigrid smoother hierarchy across calls (FGMRES + aggregation
   AMG).
3. **MPI-enabled mesh parallelization** (Section 3.9) — the mesh is
   partitioned across ranks by the built-in graph partitioner; the
   sensitivity filter and BESO threshold search are then coordinated
   via `MPI_Allgather` / `MPI_Allreduce` so every rank updates only its
   own owned elements (see the "Complete BESO workflow" section below).
4. **Sparse cKDTree sensitivity filter** (Section 3.8.1,
   `eptonics.filtering.sensitivity_filter` for the single-array form)
   — uses `scipy.spatial.cKDTree.query_ball_point` to retrieve only
   the neighbours within the filter radius, an O(N_e k) sparse query
   replacing an O(N_e^2) dense distance-matrix construction.
5. **Optimized quadrature-point state storage** (Section 3.1, 3.4) —
   history variables are read/written directly as flat quadrature
   arrays reshaped to `(n_gp, 6)` / `(n_gp, 6, 6)`, minimizing memory
   overhead and redundant data movement through the load-stepping and
   BESO loops.

Additionally, the Gauss-point constitutive update itself is
**Numba JIT-parallelized** (Section 3.4.1): the return-mapping kernel
runs as compiled, thread-parallel (`prange`) native code with no GIL,
avoiding the serialization overhead that process-based parallelism
would introduce for internal-variable arrays. See the module docstring
of `src/eptonics/constitutive/j2_return_map.py` for details.

## Complete BESO workflow and MPI parallel execution (Section 3.9)

Each example implements the three nested loops of Algorithm 2 (design
iterations / load increments / Newton-Raphson iterations) exactly as
described in manuscript Fig. 3:

- **Per-rank analysis**: tangent assembly, the persistent-KSP linear
  solve, and the thread-parallel constitutive update all run
  independently on each rank's owned subdomain.
- **Fully distributed sensitivity post-processing and BESO update**:
  see the dedicated subsection below.
- **Ghost synchronization**: after the density update, `rho.x.array`
  ghost values are refreshed (`scatter_forward()`) before the next
  design iteration begins.

This keeps the two most expensive kernels — the Newton-Raphson
equilibrium solve and the constitutive update — fully distributed and
thread-accelerated, while confining comparatively inexpensive
post-processing to replicated arithmetic and scalar collective
communication.

### Fully distributed BESO threshold search

All three example drivers implement the BESO density-update stage
(manuscript Section 2.3.5, Eqs. 48-49) with no single-rank
gather/sort/scatter bottleneck:

1. A global element-centroid `cKDTree` is built **once**, before the
   BESO loop begins (the mesh is fixed across design iterations), from
   ownership-partitioned coordinates gathered once via `MPI_Allgather`.
2. Each design iteration, every rank computes its own filtered, damped
   sensitivity for its **owned** elements only, querying the shared
   global tree — no per-iteration tree rebuild.
3. The BESO threshold (and, when the addition-ratio limiter activates,
   the separate addition/deletion thresholds) is found by
   `eptonics.mpi.parallel_order_statistics._parallel_kth_largest` /
   `_parallel_kth_smallest`: a bisection search using O(60) small
   `MPI_Allreduce` calls per invocation (each reducing a single float
   or integer), equivalent to gathering all values to one rank,
   sorting, and indexing — without ever performing that gather.
4. Each rank updates the density of only its own owned elements, and
   `rho.x.scatter_forward()` refreshes ghost values before the next
   iteration.

This bisection-based mechanism replaces a simpler but less scalable
pattern — gather all sensitivities to rank 0, sort them there with
NumPy, then scatter the resulting density update back out — with one
that performs no full gather at all, which matters once the element
count is large enough that a rank-0 gather and serial sort become the
bottleneck.

## Thread-oversubscription guard (`run_local.sh`)

When launching with `mpirun -np <N>` on a shared-memory workstation, each rank's Numba thread pool and NumPy/PETSc BLAS backend default to using a single thread unless explicitly overridden:

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
```

### Other Configuration Possibilities
- **Pure MPI (Default):** `OMP_NUM_THREADS=1` with `-np <N_cores>`. Maximizes domain-decomposition scalability.
- **Hybrid MPI + OpenMP:** Override per-run via environment variable (e.g., `OMP_NUM_THREADS=4 ./run_local.sh` with `-np 16` on 64 cores), ensuring $N_{\text{ranks}} \times N_{\text{threads}} \le N_{\text{cores}}$.
- **CLI Parameterization:** Pass rank/thread counts as positional script arguments (`./run_local.sh <ranks> <threads_per_rank>`) with validation against `nproc`.
- **HPC / Slurm Workloads:** Omit script-level hardcoding and let the scheduler manage affinity (`--cpus-per-task=$SLURM_CPUS_PER_TASK`).

