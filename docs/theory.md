# Theoretical framework

This page summarizes the theory implemented by EPTONiCS and points to
the corresponding manuscript sections/equations for full derivations.
It is a navigational aid, not a substitute for the manuscript itself.

## 1. Continuum elastoplastic problem (manuscript Section 2.1)

EPTONiCS solves the quasi-static balance of linear momentum on a
domain with Dirichlet and Neumann boundaries, under prescribed
displacements, tractions, and body force (Eq. 1). Loading is applied
through a sequence of load increments rather than in one shot, because
the stress at any load step depends on the entire preceding
deformation history (Eq. 4).

The material model is small-strain **J2 plasticity with linear
isotropic hardening** (Section 2.1.2): the total strain splits
additively into elastic and plastic parts (Eq. 5), stress follows
linear elasticity applied to the elastic strain (Eq. 6), and the yield
surface is a cylinder in deviatoric stress space that expands
uniformly with the accumulated equivalent plastic strain (Eq. 8). Flow
is associative (Eq. 9) and the consistency parameter satisfies the
standard Karush-Kuhn-Tucker complementarity conditions (Eq. 11).

## 2. Finite-element discretization and incremental solution (Section 2.2)

The domain is discretized with hexahedral elements; a binary design
variable per element scales the local stress contribution between a
soft-void density (`rho_min = 1e-3`) and full solid (`rho = 1`)
(Eq. 12-13). Because the constitutive response is path-dependent,
equilibrium at each load step is resolved via a **radial return
mapping** (elastic predictor / plastic corrector, Eqs. 19-26,
Algorithm 1): an elastic trial stress is computed first; if it violates
the yield condition, a closed-form plastic multiplier increment
restores admissibility along the (purely radial, for J2 plasticity)
return direction. The corresponding **consistent algorithmic tangent**
(Eq. 26) is required for quadratic Newton-Raphson convergence and is
implemented in
[`eptonics.constitutive.j2_return_map`](../src/eptonics/constitutive/j2_return_map.py).

Global equilibrium at each load step is then resolved by
Newton-Raphson linearization (Eq. 28), iterating until the residual
norm falls below a tolerance.

## 3. Topology optimization with BESO (Section 2.3)

The design objective is the total mechanical work absorbed over the
full loading history, approximated by the trapezoidal rule over load
steps (Eq. 29), maximized subject to element-wise equilibrium and a
prescribed total-mass constraint (Eq. 30).

Because the objective is path-dependent, its sensitivity with respect
to each element's density cannot be obtained from a single adjoint
solve at one load step. Section 2.3.2 derives a **path-dependent
adjoint formulation**: Lagrange multipliers are introduced for the
residual at both ends of each trapezoidal interval, and by exploiting
the symmetry of the tangent stiffness matrix (whose Cholesky factor is
already available from the forward Newton solve), the multipliers are
obtained from two adjoint linear systems per load step (Eqs. 38-40) at
negligible extra factorization cost. The resulting sensitivity
(Eq. 41-42) couples the stress and adjoint state at *adjacent* load
steps — a direct consequence of the trapezoidal objective. For linear
elasticity this reduces exactly to the classical BESO sensitivity
number of Huang and Xie (2010).

Raw sensitivities are post-processed in three steps before the density
update (Section 2.3.3):

1. **Spatial filter** (Eq. 43-44) — distance-weighted neighborhood
   averaging within a filter radius `r_min`, which decays linearly
   across design iterations to allow fine detail to emerge as the
   design converges.
2. **Damping** (Eq. 45) — raising the filtered sensitivity to a power
   `beta` in (0, 1] to suppress oscillations introduced by plastic
   effects.
3. **History averaging** (Eq. 46) — blending with the previous design
   iteration's filtered sensitivity.

The design volume is reduced gradually (Eq. 47) via an evolutionary
ratio, and elements are added/removed via a single global threshold
(Eq. 48), with an **addition-ratio limiter** (Eq. 49, Section 2.3.5) to
prevent oscillation when many void elements simultaneously qualify for
reinstatement. Algorithm 2 assembles all of the above into the
complete nested (design / load / Newton) BESO loop, with the
convergence criterion of Eq. 50.

## Where each piece lives in the code

| Manuscript concept                          | Code location                                                          |
|-----------------------------------------------|--------------------------------------------------------------------------|
| Radial return map (Algorithm 1)              | `src/eptonics/constitutive/j2_return_map.py`                            |
| Variational forms (Eq. 52-53)                | `_a_ufl_*`, `_L_ufl_*` in each `examples/*/run_*.py`                    |
| Newton-Raphson loop (Eq. 28, Algorithm 2 L9-19)| predictor / corrector / `assemble_residual()` in each example driver   |
| Path-dependent adjoint sensitivity (Eq. 41-42)| `sensitivity_func()` in each example driver                             |
| Sensitivity filter (Eq. 43-44)                | `sensitivity_filter()` (per-example) / cKDTree-based                    |
| BESO density update (Eq. 48-49)               | `update_rho_BESO*()` and the MPI gather/scatter block in each driver    |
| Complete BESO loop (Algorithm 2)              | The outer `for i in range(max_iter):` loop in each example driver       |

See `docs/implementation.md` for the software-engineering side of
these mappings (performance mechanisms, MPI parallelization strategy).
