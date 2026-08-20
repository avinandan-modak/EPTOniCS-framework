# EPTOniCS

**E**lasto**p**lastic **T**opology **O**ptimizatio**n** **i**n Feni**CS**x

EPTOniCS is an open-source, educational framework for three-dimensional
elastoplastic topology optimization built on the
[FEniCSx](https://fenicsproject.org/) finite-element platform. It
couples a well-established bi-directional evolutionary structural
optimization (BESO) formulation with J2 plasticity (linear isotropic
hardening) and a path-dependent adjoint sensitivity analysis, and
integrates five targeted performance mechanisms — precompiled UFL
forms with persistent PETSc solvers, Numba-accelerated parallel
return-mapping, MPI mesh parallelization, sparse cKDTree sensitivity
filtering, and optimized quadrature-point state storage — to make
large-scale 3D elastoplastic topology optimization (millions of Gauss
points) computationally tractable on modern multi-core workstations
and portable to HPC clusters.

![EPTOniCS](docs/EPTOniCS.png)

This repository accompanies the manuscript:

> A. Modak, R. Chowdhury, T. Gangwar. *EPTONiCS: An Efficient FEniCSx
> Implementation for Three-Dimensional Topology Optimization of
> Elastoplastic Structures.* (Under Review), 2026.

See `CITATION.cff` for citation details.

## Why elastoplastic topology optimization?

Classical topology optimization overwhelmingly assumes linear-elastic
material response. Structures subjected to extreme loading — impact,
blast, progressive collapse — yield locally, redistribute load, and
dissipate energy through irreversible plastic deformation well before
failure. Elastoplastic topology optimization exploits a structure's
full load-carrying capacity beyond first yield, producing lighter,
more damage-tolerant designs than purely elastic approaches — at the
cost of a substantially harder computational problem: path-dependent,
history-dependent nonlinear analysis repeated at every design
iteration. EPTONiCS targets exactly this computational bottleneck for
large-scale, three-dimensional problems.

## Repository layout

```
EPTONICS/
├── README.md                  (this file)
├── LICENSE
├── CITATION.cff
├── environment.yml            conda-forge environment (recommended)
├── requirements.txt           pip-installable subset
├── pyproject.toml
│
├── docs/
│   ├── installation.md        environment setup
│   ├── theory.md               theory -> equation/section map
│   ├── implementation.md      architecture & performance mechanisms
│   └── examples.md             benchmark walkthroughs & expected outputs
│
├── examples/
│   ├── cantilever/             Example 1 (manuscript Section 4.1)
│   ├── l_bracket/               Example 2 (manuscript Section 4.2)
│   └── portal_frame/           Example 3 (manuscript Section 4.3)
│
├── src/eptonics/
│   ├── __init__.py
│   ├── constitutive/            J2 radial return map (Algorithm 1)
│   ├── fem/                     Voigt strain operator, stress invariants, Function utils
│   ├── solvers/                 persistent-PETSc-KSP plumbing
│   ├── visualization/           DG0 post-processing fields
│   ├── optimization/            BESO density-update rule
│   ├── filtering/               sensitivity filter
│   └── mpi/, sensitivities/, utils/, io/   reserved (see docs/implementation.md)
│
├── tests/                       (recommended coverage documented; see tests/README.md)
├── scripts/                     (reserved for future batch/HPC tooling)
└── data/                         (reserved for mesh/output artifacts)
```

`src/eptonics/` holds the pieces of the implementation that are
identical across all three example drivers (the constitutive kernel,
FEM helpers, solver plumbing, post-processing, the BESO update rule,
and the sensitivity filter). Logic that is genuinely specific to one
benchmark's geometry or boundary conditions — mesh/submesh
construction, boundary conditions, the adjoint sensitivity assembly,
and the outer BESO/load/Newton loop — stays in that example's driver
script. See `docs/implementation.md` for the full architecture
rationale.

## Quick start

```bash
conda env create -f environment.yml
conda activate eptonics
pip install -e .

cd examples/cantilever
bash run_local.sh
```

Full installation notes (including a pip-only path for existing
FEniCSx environments) are in `docs/installation.md`.

## Numerical examples

| # | Example         | Manuscript section | Elements | DoFs   | Gauss points |
|---|------------------|---------------------|----------|--------|--------------|
| 1 | Cantilever beam  | 4.1                 | 31,250   | ~103k  | 250,000      |
| 2 | L-shaped bracket | 4.2                 | ~128,000 | ~416k  | ~1,024,000   |
| 3 | V-notched portal | 4.3                 | ~201,600 | ~647k  | ~1,612,800   |

Each example is runnable independently; see `docs/examples.md` for
parameters, run commands, and how to interpret the console log and
XDMF outputs.

## Extending EPTONiCS

The constitutive state-update module is deliberately isolated
(`src/eptonics/constitutive/j2_return_map.py`) so that alternative
constitutive models — kinematic/mixed hardening, finite-strain
plasticity — can be added as sibling modules exposing the same
`update_internal_variables(...)` signature, without modifying the
surrounding finite-element or optimization code. The BESO formulation
itself could similarly be generalized to density-based (SIMP) or
level-set parametrizations, or to alternative objectives (energy
absorption, ductile damage, stress constraints); these are documented
as future directions in the manuscript's Section 5 and are not
implemented here.

## Reproducibility

All algorithms, tolerances, solver configurations, and physical
parameters in this repository match those reported in the manuscript.
Detailed mathematical derivations and section mappings are documented
in `docs/theory.md` and `docs/implementation.md`.

## License

This work is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0). 
Full legal text of this license is available at:
https://creativecommons.org/licenses/by/4.0/legalcode

## Citation

See `CITATION.cff`, or cite:

```
To be updated.
```

## Acknowledgements

This work was supported by the Ministry of Education, Government of
India (A. Modak) and the Faculty Initiation Grant from IIT Roorkee (T.
Gangwar). Computations for the manuscript's benchmarks used the PARAM
Ganga Supercomputing Facility at the Institute Computer Centre, IIT
Roorkee.
