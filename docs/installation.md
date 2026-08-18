# Installation

EPTONiCS builds on the FEniCSx finite-element platform (DOLFINx, UFL,
basix), PETSc, and mpi4py, plus a small pure-Python layer (NumPy, SciPy,
optionally Numba and Matplotlib). The versions below are those reported
in manuscript Table 2 (Appendix A, "Code reproducibility") for the
runs described in the paper; `environment.yml` pins only
`fenics-dolfinx` to its exact version and lets UFL/basix/ffcx resolve
automatically as matching dependencies, with lower-bound pins on the
pure-Python packages — see the troubleshooting note below for why.

| Package             | Version (as reported in the manuscript) | Role |
|---------------------|---------|-----------------------------------------|
| FEniCSx (DOLFINx)    | 0.9.0   | FEM assembly and solver                |
| UFL                  | 2024.2.0| Variational form language              |
| PETSc                | 3.24.4  | Linear algebra backend                 |
| NumPy                | 2.4.3   | Array operations                       |
| SciPy                | 1.17.1  | Sensitivity filtering / bisection      |
| Numba (optional)     | 0.65.0  | JIT-compiled parallel return map       |
| Matplotlib           | 3.10.8  | Post-processing plots                  |

## Recommended: conda-forge environment

FEniCSx and PETSc are compiled, MPI-linked libraries that are most
reliably installed through conda-forge rather than pip:

```bash
conda env create -f environment.yml
conda activate eptonics
pip install -e .
```

The last step installs the `eptonics` package (the shared
`eptonics.constitutive.j2_return_map` kernel used by every example
driver) in editable mode from `src/`.

### Troubleshooting: `PackagesNotFoundError` for `ufl` / `basix`

If `conda env create` fails with something like:

```
PackagesNotFoundError: The following packages are not available from current channels:
  - ufl=2024.2.0*
  - basix
```

this means `environment.yml` tried to pin bare package names `ufl` /
`basix` that do not exist under those names on conda-forge (the actual
package names are `fenics-ufl` and `fenics-basix`, and — more to the
point — `fenics-dolfinx` already pulls in matching, compatible
versions of both automatically as transitive dependencies, so they
should not be listed as separate top-level entries at all). The
`environment.yml` shipped with this repository has already been fixed
to rely on this automatic resolution; if you are still hitting this
error, confirm you are using the current `environment.yml` and not an
older cached copy.

More generally, if `conda env create` reports channels including
`repo.anaconda.com/pkgs/main` or `.../pkgs/r` alongside
`conda.anaconda.org/conda-forge` (visible in the error message's
"Current channels" list — this happens by default when creating an
environment from an Anaconda/Miniconda `base` install rather than
Miniforge), the solve can become slower and more prone to conflicts,
since FEniCSx packages are conda-forge-only. If problems persist,
force conda-forge-only resolution:

```bash
conda env create -f environment.yml --override-channels -c conda-forge
```

or, for a faster and generally more reliable solver, use `mamba`
instead of `conda`:

```bash
mamba env create -f environment.yml
```

## Alternative: existing FEniCSx environment + pip

If you already have a working FEniCSx 0.9.0 installation (an HPC
environment module, a Spack install, or the official FEniCSx Docker
image), you only need the pure-Python remainder:

```bash
pip install -r requirements.txt
pip install -e .
```

## Verifying the install

```bash
python -c "from eptonics.constitutive import j2_return_map; print(j2_return_map.HAS_NUMBA)"
```

This prints `True` if the Numba JIT-parallel backend is active, or
`False` if EPTONiCS is running on the vectorised NumPy fallback (both
are numerically equivalent; see manuscript Section 3.4.1).

## Running an example

```bash
cd examples/cantilever
bash run_local.sh          # launches `mpirun -np 64 python -u run_cantilever.py`
```

Adjust the MPI rank count in `run_local.sh` to match your machine's
physical core count (see the thread-oversubscription note inside that
script — it is not optional). Each example writes its XDMF outputs
into a subdirectory of the current working directory
(`BESO_J2_Outputs/`, `BESO_Lbracket_Outputs/`, or
`BESO_PortalFrame_Outputs/`), viewable in ParaView or VisIt.

See `docs/examples.md` for a walkthrough of each benchmark and expected
outputs.
