# Tests

No automated test suite is included in this release. This directory
is reserved so that test infrastructure has an obvious home when it
is added.

## Recommended coverage for a future addition

- **Constitutive kernel** (`eptonics.constitutive.j2_return_map`):
  - Purely elastic step reproduces the linear-elastic stiffness
    (`C_ep == C_e`) for `f_trial <= 0`.
  - A single-Gauss-point uniaxial stress path exceeding yield matches
    the closed-form radial-return solution for prescribed `sigma_y`,
    `H`, `E`, `nu`.
  - Numba and NumPy backends (`BESO_NO_NUMBA=1`) produce numerically
    identical output on the same input for both elastic and plastic
    branches (this is the property the manuscript's Section 3.4.1
    claims and is the highest-value regression check to add first).
  - `H = 0` recovers the perfect-plasticity consistent tangent.
- **Example drivers** (smoke tests, not full BESO runs):
  - Each example's Newton-Raphson loop converges (`|R| <= eps_tol`)
    within the stated iteration budget for the first load step of a
    coarse, fast-running mesh.
  - The sensitivity filter's weighted average reduces to the raw
    sensitivity when `r_min` is smaller than the element spacing.

Adding these tests is left as a deliberate follow-up rather than being
implemented here.
