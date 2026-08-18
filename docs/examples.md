# Examples

Three benchmarks of increasing scale and geometric complexity are
provided, matching manuscript Section 4. All three use displacement-
controlled loading, J2 plasticity with isotropic hardening, and the
BESO algorithmic parameters of Algorithm 2 (Newton-Raphson tolerance
`eps_tol = 1e-10`, BESO convergence tolerance `tau = 1e-3` over a
9-iteration sliding window, sensitivity damping exponent `beta = 0.5`,
maximum addition ratio `c_ar_max = 0.01`, filter radius decaying
linearly from `20*l_e` to `4*l_e`).

## 1. Cantilever beam — `examples/cantilever/` (Section 4.1)

A 2000 x 1000 x 1000 mm domain, clamped at `x1 = 0`, displacement-loaded
at a mid-span patch on the free end.

| Quantity                | Value                          |
|--------------------------|---------------------------------|
| Mesh                     | 50 x 25 x 25 hexahedral (Q1)    |
| Elements / DoFs / Gauss pts | 31,250 / ~103k / 250,000    |
| Target volume fraction   | 0.15                            |
| Load steps                | 10 x 1 mm                       |
| Material (E, nu, sigma_y, H) | 75 GPa, 0.30, 0.1 GPa, 1.5 GPa |

```bash
cd examples/cantilever
bash run_local.sh
```

Outputs (in `BESO_J2_Outputs/`): `density.xdmf` (design evolution),
`vonMises_solid.xdmf`, `PEEQ_solid.xdmf`, `density_visual.xdmf`
(last-load-step plasticity fields on the pre-update solid domain), and
`optimized_geometry.xdmf` (final solid-only submesh, importable
directly into CAD/visualization tools).

## 2. L-shaped bracket — `examples/l_bracket/` (Section 4.2)

A 100 x 100 x 20 mm bounding box with the upper-right quadrant removed,
introducing a re-entrant corner and its associated stress
concentration. Clamped across the top of the vertical arm; loaded on a
patch at the free end of the horizontal arm.

| Quantity                | Value                          |
|--------------------------|---------------------------------|
| Mesh (full bounding box) | 200 x 200 x 40 hexahedral (Q1)  |
| Active L-shaped elements / DoFs / Gauss pts | ~128,000 / ~415,863 / ~1,024,000 |
| Target volume fraction   | 0.12                            |
| Load steps                | 5 x 0.2 mm                      |

```bash
cd examples/l_bracket
bash run_local.sh
```

## 3. V-notched portal — `examples/portal_frame/` (Section 4.3)

A 120 x 60 x 20 mm bounding box with a V-shaped lower void, forming a
simply-supported arch-like portal frame (pin support at the left base,
roller support at the right base).

| Quantity                | Value                          |
|--------------------------|---------------------------------|
| Mesh (full bounding box) | 120 x 60 x 40 hexahedral (Q1)   |
| Active portal-frame elements / DoFs / Gauss pts | ~201,600 / ~647,103 / ~1,612,800 |
| Target volume fraction   | 0.15                            |
| Load steps                | 5 x 0.05 mm                     |

```bash
cd examples/portal_frame
bash run_local.sh
```

## Interpreting the console log

Each example prints, per BESO design iteration: the objective `fw` and
its percentage change, the current vs. target volume fraction, the
solid/void element counts, the maximum von Mises stress and PEEQ over
the solid domain, and the wall-clock time for that iteration. Per load
step within an iteration, each Newton-Raphson correction prints the
residual norm; convergence to `|R| <= 1e-10` is expected within roughly
2-11 iterations depending on how far the current load step is into the
plastic regime (manuscript Section 4.4, Table 1).

## Visualizing results

All XDMF/H5 output pairs can be opened directly in ParaView or VisIt.
Void elements in the von Mises / PEEQ visualization fields are written
as `NaN` so they render as hidden/transparent automatically.

## A note on wall time

The wall-clock figures reported in manuscript Section 4 (e.g. ~27 min
for the cantilever, ~4.2 h for the L-bracket, ~3.6 h for the portal
frame at the paper's mesh resolutions) were measured on a 64-physical-
core AMD Threadripper PRO 5995WX workstation with 64 MPI ranks. Your
wall time will scale with core count, memory bandwidth, and mesh
resolution; see manuscript Section 4.4 and Figs. 16-17 for the
strong-/weak-scaling characterization.
