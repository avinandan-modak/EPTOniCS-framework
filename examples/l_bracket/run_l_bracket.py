"""
run_l_bracket.py — Example 2: L-shaped bracket (manuscript Section 4.2)
=========================================================================
Three-dimensional elastoplastic BESO topology optimization of an
L-shaped bracket, a re-entrant corner benchmark, using J2 plasticity
with linear isotropic hardening (manuscript Section 2.1.2) and the
path-dependent adjoint sensitivity analysis of Section 2.3.2.

Implements the complete BESO workflow of Algorithm 2 — three nested
loops over design iterations, load increments, and Newton-Raphson
equilibrium exactly as described in manuscript Section 3.9 and Fig. 3.

Geometry (units: mm)
---------------------
The L-shape is carved from a bounding-box mesh via
``dolfinx.mesh.create_submesh``; all analysis runs on the resulting
L-shaped mesh.

    Bounding box    : 100 × 100 × 20
    Vertical arm    : x ∈ [0, 40],   y ∈ [0, 100]
    Horizontal arm  : x ∈ [40, 100], y ∈ [0, 40]
    Removed quadrant: x ∈ (40, 100], y ∈ (40, 100]

Boundary conditions
--------------------
    Support : u = (0, 0, 0) on y = 100 (top of vertical arm)
    Load    : u_y = Δu on the patch x = 100, y ∈ [20, 40], z ∈ [5, 15]

Target volume fraction: 0.12

Dependencies
------------
DOLFINx / UFL / PETSc (FEniCSx), mpi4py, NumPy,
SciPy (cKDTree), optionally Numba.
See ``environment.yml`` / ``requirements.txt`` for pinned versions.

Related manuscript sections
----------------------------
Section 3.4 — J2 constitutive module
Section 3.5 — Variational forms and solver setup
Section 3.6 — Newton-Raphson equilibrium loop
Section 3.7 — Path-dependent adjoint sensitivity
Section 3.8 — Sensitivity filter and BESO density update
Section 3.9 — Complete BESO workflow
Section 4.2 — Numerical example 2: L-shaped bracket

Usage
-----
    mpirun -np <N> python -u run_l_bracket.py

Output XDMF files are written to ``BESO_Lbracket_Outputs/``.
"""

import os   # needed immediately below, before any numerical library is imported

# Set thread counts for NumPy BLAS and Numba before any numerical import (Sec. 3.9, ★5)
# This MUST happen before dolfinx/petsc4py/numba are imported: those libraries read
# these environment variables once, at import time, to size their internal thread pools.
_n_cores = str(os.cpu_count() or 8)                      # physical/logical core count on this machine
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, _n_cores)                  # only set if not already set by the launch script

from mpi4py import MPI as _MPI_early   # early, minimal import: just enough to know our rank number
def pout(*args, **kwargs):
    """Print only on MPI rank 0."""
    if _MPI_early.COMM_WORLD.rank == 0:   # every rank runs this function; only rank 0 actually prints
        print(*args, **kwargs)
pout(f"[BESO] Using {_n_cores} threads for BLAS / Numba.", flush=True)


# ---- Main imports (safe now that the thread-count environment variables are set) ----
import numpy as np                                  # arrays, linear algebra
from mpi4py import MPI                               # full MPI interface (communicators, reductions)
from dolfinx import mesh, fem, io                     # FEniCSx: mesh generation, function spaces, file I/O
import ufl                                            # symbolic variational-form language
from ufl import Measure, TrialFunction, TestFunction, dot, inner   # UFL building blocks used below
from petsc4py.PETSc import ScalarType                 # PETSc's floating-point type (matches build config)
from petsc4py import PETSc                            # PETSc bindings (vectors, matrices, insert modes)
from dolfinx.fem import petsc as fem_petsc            # DOLFINx's PETSc-backed assembly helpers
from dolfinx.fem.petsc import set_bc                  # applies Dirichlet values directly into a PETSc vector
from basix.ufl import quadrature_element              # defines a function space living AT quadrature points
from scipy.spatial import cKDTree                     # nearest-neighbour queries for the sensitivity filter
from eptonics.constitutive import j2_return_map as J2  # J2 return-map kernel (Sec. 3.4)
from eptonics.fem.kinematics import epsilon            # Voigt strain operator (Sec. 3.3)
from eptonics.fem.utils import zero                    # Function-array reset helper
from eptonics.solvers.petsc_backend import _build_ksp, _petsc_solve  # persistent KSP (Sec. 3.5.1, ★2)
from eptonics.visualization.postprocess import compute_vonMises_DG0, compute_PEEQ_DG0  # DG0 post-processing
from eptonics.mpi import _parallel_kth_largest, _parallel_kth_smallest  # distributed threshold (Sec. 3.9)
import time                                           # wall-clock timing for the console log

os.makedirs("BESO_Lbracket_Outputs", exist_ok=True)   # every rank creates this; exist_ok avoids a race error

# ============================================================================
# Geometry parameters
# ============================================================================
Lx, Ly, Lz = 100.0, 100.0, 20.0   # bounding-box dimensions (mm)
arm_x = 40.0   # x-width of vertical arm
arm_y = 40.0   # y-height of horizontal arm
load_y_lo, load_y_hi = 20.0, 40.0   # y-extent of the loaded patch
load_z_lo, load_z_hi =  5.0, 15.0   # z-extent (through-thickness) of the loaded patch

nx, ny, nz = 100, 100, 20   # element counts of the full bounding-box mesh

# ============================================================================
# Step 1 — Create L-shaped submesh
# ============================================================================
_box = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (Lx, Ly, Lz)),
    n=(nx, ny, nz),
    cell_type=mesh.CellType.hexahedron)          # full rectangular bounding-box mesh, before carving the L-shape

_V0_box = fem.functionspace(_box, ("DG", 0))      # one value per element, used only to get element centroids
_centroids = _V0_box.tabulate_dof_coordinates()[:, :3]   # (n_box_elements, 3) centroid coordinates

_L_mask    = ~((_centroids[:, 0] > arm_x) & (_centroids[:, 1] > arm_y))   # True everywhere except the removed quadrant
_L_indices = np.where(_L_mask)[0].astype(np.int32)                        # element indices that form the L-shape
pout(f"[BESO] L-shape cells: {len(_L_indices)} / {len(_L_mask)}", flush=True)

prob_mesh, _cell_map, _vertex_map, _geom_map = mesh.create_submesh(
    _box, _box.topology.dim, _L_indices)          # the actual L-shaped mesh that all analysis runs on

gdim      = prob_mesh.topology.dim
dx_domain = Measure("dx", domain=prob_mesh)        # standard cell-integration measure over the L-shaped mesh
V         = fem.functionspace(prob_mesh, ("CG", 1, (gdim,)))   # vector CG1: nodal displacement field
V0        = fem.functionspace(prob_mesh, ("DG", 0))             # scalar DG0: one density value per element

# ============================================================================
# Step 2 — Quadrature spaces
# ============================================================================
Vcep = fem.functionspace(prob_mesh, quadrature_element(
    prob_mesh.topology.cell_name(), degree=2, value_shape=(6,6)))   # quadrature space for the 6x6 tangent Cep
Veps = fem.functionspace(prob_mesh, quadrature_element(
    prob_mesh.topology.cell_name(), degree=2, value_shape=(6,)))    # quadrature space for 6-component Voigt tensors
dx_q = ufl.Measure("dx", domain=prob_mesh, metadata={"quadrature_degree": 2})   # integration measure tied to the Gauss rule

u          = fem.Function(V); u_old = fem.Function(V); Du = fem.Function(V)   # current / previous / incremental displacement
eps_quad   = fem.Function(Veps)        # total strain, interpolated at quadrature points
Sigma      = fem.Function(Veps)        # Cauchy stress at quadrature points
Epsilon_p  = fem.Function(Veps)        # plastic strain at quadrature points
Cep_gp     = fem.Function(Vcep)        # consistent elastoplastic tangent at quadrature points
lambda_adj = fem.Function(V); mu_adj = fem.Function(V)   # adjoint displacement fields (Eq. 38-40)

n_gp  = Vcep.tabulate_dof_coordinates().shape[0]                     # total Gauss points on this rank
n_ele_local = V0.tabulate_dof_coordinates().shape[0]                  # elements owned by this rank
n_ele_global = prob_mesh.topology.index_map(prob_mesh.topology.dim).size_global   # elements across all ranks

pout(f"[BESO] n_ele = local: {n_ele_local}, global: {n_ele_global} | n_gp = {n_gp}", flush=True)

# ============================================================================
# Step 3 — Material parameters (manuscript Section 2.1.2)
# ============================================================================
E_mod, nu = 75.0, 0.30                    # Young's modulus (GPa), Poisson's ratio
shear = E_mod / (2*(1+nu))                # shear modulus G
bulk  = 2*shear*(1+nu) / (3*(1-2*nu))     # bulk modulus K
sig_y = 0.1                               # initial uniaxial yield stress
hard  = 1.5            # linear isotropic hardening modulus (0 = perfect plasticity)
Bulk_mat  = bulk  * np.ones((n_gp, 1))    # per-Gauss-point material arrays, ready for the constitutive kernel
Shear_mat = shear * np.ones((n_gp, 1))
Sig_y     = sig_y * np.ones((n_gp, 1))
Hard_mat  = hard  * np.ones((n_gp, 1))
Ep_old    = np.zeros((n_gp, 6))           # converged plastic strain at the previous load step
Eqp_old   = np.zeros(n_gp)            # equivalent plastic strain ε̄^p

# ============================================================================
# Step 4 — Density field
# ============================================================================
rho = fem.Function(V0); rho.x.array[:] = 1.0   # design density field, starts fully solid

# ============================================================================
# Step 5 — Boundary conditions on the L-shaped submesh
# ============================================================================
tol = 1e-10   # geometric tolerance for facet-location comparisons

_sup_facets  = mesh.locate_entities_boundary(
    prob_mesh, gdim-1, lambda x: np.isclose(x[1], Ly))   # top face of the vertical arm (y = Ly): clamped support
_load_facets = mesh.locate_entities_boundary(
    prob_mesh, gdim-1,
    lambda x: (np.isclose(x[0], Lx) &                     # free end of the horizontal arm (x = Lx)
               (x[1] >= load_y_lo - tol) & (x[1] <= load_y_hi + tol) &   # within the loaded y-range
               (x[2] >= load_z_lo - tol) & (x[2] <= load_z_hi + tol)))   # within the loaded z-range

support_dofs = fem.locate_dofs_topological(V,        gdim-1, _sup_facets)   # all 3 displacement components, clamped face
bc_load_dofs = fem.locate_dofs_topological(V.sub(1), gdim-1, _load_facets)  # y-displacement DOFs, loaded face

n_f_local = len(bc_load_dofs)
n_f_global = MPI.COMM_WORLD.allreduce(n_f_local, op=MPI.SUM)   # total loaded DOFs, summed across ranks
pout(f"[BESO] n_f (load DOFs) = local: {n_f_local}, global: {n_f_global}", flush=True)
assert n_f_global > 0, "No load DOFs found globally — check load patch geometry / mesh alignment!"

# Restrict to owned DOFs to avoid ghost double-counting in force extraction
owned_flat_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs   # flat size of this rank's owned DOFs
bc_load_dofs_owned = bc_load_dofs[bc_load_dofs < owned_flat_size]          # keep only owned (non-ghost) load DOFs
n_f = len(bc_load_dofs_owned)


def get_bc_incremental(disp_increment):
    """Dirichlet BCs for one Newton predictor step: clamp + prescribed y-displacement increment."""
    bc_sup  = fem.dirichletbc(ScalarType((0., 0., 0.)), support_dofs, V)          # u = (0,0,0) at the clamped face
    bc_load = fem.dirichletbc(ScalarType(disp_increment), bc_load_dofs, V.sub(1))  # this step's incremental u_y
    return [bc_sup, bc_load]

def get_bc_homogeneous():
    """Zero-valued version of the same BCs, used inside Newton corrector iterations."""
    bc_sup  = fem.dirichletbc(ScalarType((0., 0., 0.)), support_dofs, V)
    bc_load = fem.dirichletbc(ScalarType(0.0),          bc_load_dofs, V.sub(1))    # zero increment: correction only
    return [bc_sup, bc_load]

bcs_homog = get_bc_homogeneous()   # zero-valued BCs, reused every Newton correction

# ============================================================================
# Step 6 — Load stepping & BESO parameters
# ============================================================================
load_incr = -0.2 * np.ones(5)   # 5 equal displacement increments of -0.2 mm each (negative: pulling downward)
n_load    = len(load_incr)       # number of load steps per BESO design iteration

max_iter  = 120     # maximum BESO design iterations
er        = 0.02    # evolutionary ratio, fraction of volume removed per iteration (Eq. 47)
volfrac   = 0.12    # target final volume fraction
c_ar_max  = 0.02     # maximum addition ratio before the two-threshold limiter engages (Eq. 49)
dampCoeff = 0.6      # sensitivity damping exponent beta (Eq. 45)

vol_dom_local = fem.assemble_scalar(fem.form(fem.Constant(prob_mesh, ScalarType(1.)) * dx_domain))  # this rank's volume
vol_dom = prob_mesh.comm.allreduce(vol_dom_local, op=MPI.SUM)   # total L-shaped domain volume, all ranks
vol_el  = vol_dom / n_ele_global                                 # volume of one (uniform) element
pout(f"[BESO] L-domain vol = {vol_dom:.1f}  vol_el = {vol_el:.2f}", flush=True)

du_trial = TrialFunction(V); v_test = TestFunction(V)   # trial/test functions for the predictor/corrector forms
dl       = TrialFunction(V); v_adj  = TestFunction(V)   # trial/test functions for the adjoint form

# ============================================================================
# Sensitivity assembly (manuscript Eq. 41-42, Section 3.7)
# ============================================================================
def sensitivity_func(rho_, Sig_data, lambda_data, mu_data):
    """Path-dependent adjoint sensitivity, MPI-safe (owned/ghost via PETSc copy)."""
    w = ufl.TestFunction(V0)         # DG0 test function: one weight per element, gives elemental sensitivities
    dc_form = 0.0                    # accumulate the trapezoidal sum over load steps (Eq. 41-42)
    for ii in range(len(Sig_data)):
        if ii == 0:
            # first load step: only the lambda (forward-adjoint) term contributes
            dc_form += -0.5 * inner(rho_*Sig_data[ii], epsilon(lambda_data[ii]))
        else:
            # later steps: lambda term at step ii plus mu term coupling back to step ii-1 (Eq. 42)
            dc_form += -0.5*(inner(rho_*Sig_data[ii],   epsilon(lambda_data[ii]))
                           + inner(rho_*Sig_data[ii-1], epsilon(mu_data[ii])))
    dc_vec = fem_petsc.assemble_vector(fem.form(dc_form * w * dx_q))   # assemble elemental sensitivity vector
    dc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # sum contributions from ghost DOFs
    dc_fn = fem.Function(V0)
    dc_vec.copy(dc_fn.x.petsc_vec)                                     # copy PETSc vector into a Function
    dc_fn.x.scatter_forward()                                          # broadcast owned values back to ghosts
    return dc_fn


# ============================================================================
# ★1 — Pre-compile UFL forms once (Sec. 3.5.1)
# ============================================================================
pout("[BESO] Pre-compiling UFL forms ...", flush=True)
_t0 = time.perf_counter()

_a_ufl_q   = rho * inner(epsilon(v_test), dot(Cep_gp, epsilon(du_trial))) * dx_q      # predictor bilinear form
_a_ufl_dom = rho * inner(epsilon(v_test), dot(Cep_gp, epsilon(du_trial))) * dx_domain  # corrector bilinear form
_a_ufl_adj = rho * inner(epsilon(v_adj),  dot(Cep_gp, epsilon(dl)))       * dx_q      # adjoint bilinear form (same tangent)
_L_ufl_q   = -rho * inner(Sigma, epsilon(v_test)) * dx_q                              # predictor RHS: -internal force
_L_ufl_dom = -rho * inner(Sigma, epsilon(v_test)) * dx_domain                         # corrector RHS
_L_ufl_adj =  rho * inner(fem.Constant(prob_mesh, ScalarType((0.,0.,0.))), v_adj) * dx_q  # adjoint RHS (BCs carry the load)
_R_ufl_dom = -rho * inner(Sigma, epsilon(v_test)) * dx_domain                         # residual, for convergence check

_af_q   = fem.form(_a_ufl_q)      # compiling each form once (via FFCx) avoids recompiling every Newton iteration
_af_dom = fem.form(_a_ufl_dom)
_af_adj = fem.form(_a_ufl_adj)
_Lf_q   = fem.form(_L_ufl_q)
_Lf_dom = fem.form(_L_ufl_dom)
_Lf_adj = fem.form(_L_ufl_adj)
_Rf_dom = fem.form(_R_ufl_dom)

pout(f"[BESO] Forms compiled in {time.perf_counter()-_t0:.1f} s", flush=True)


# ============================================================================
# ★2 — Persistent KSP solvers (Sec. 3.5.1)
# ============================================================================
pout("[BESO] Building KSP solvers ...", flush=True)
_ksp_q,   _A_q   = _build_ksp(_af_q,   prob_mesh.comm)    # predictor
_ksp_dom, _A_dom = _build_ksp(_af_dom,  prob_mesh.comm)    # corrector
_ksp_adj, _A_adj = _build_ksp(_af_adj,  prob_mesh.comm)    # adjoint
_dU_scratch = fem.Function(V)      # scratch buffer for one corrector's incremental solution
pout("[BESO] KSP solvers ready.", flush=True)


# ============================================================================
# Inner-loop helpers
# ============================================================================
def update_state(u_field):
    """Interpolate strain, run J2 return map, update quadrature fields."""
    eps_quad.interpolate(
        fem.Expression(epsilon(u_field), Veps.element.interpolation_points()))  # evaluate strain at Gauss points
    Sig, Ep, eqp, Cep = J2.update_internal_variables(          # radial return map (Algorithm 1)
        eps_quad.x.array.reshape(-1, 6), Ep_old, Eqp_old,
        Bulk_mat, Shear_mat, Sig_y, Hard_mat)
    Sigma.x.array[:]     = Sig.flatten()          # write updated stress back into the UFL-visible Function
    Epsilon_p.x.array[:] = Ep.flatten()           # updated plastic strain
    Cep_gp.x.array[:]    = Cep.flatten()          # updated consistent tangent, used by the next assembly
    return Sig, Ep, eqp


def _solve_predictor(bcs):
    """Predictor step: result written into Du."""
    return _petsc_solve(_ksp_q, _A_q, _af_q, _Lf_q, bcs, Du)


def _solve_corrector(bcs):
    """Corrector step: incremental solution added to Du."""
    _petsc_solve(_ksp_dom, _A_dom, _af_dom, _Lf_dom, bcs, _dU_scratch)
    Du.x.array[:] += _dU_scratch.x.array          # accumulate the correction onto the running increment


def _solve_adjoint(bcs):
    """Adjoint step: result written into lambda_adj."""
    return _petsc_solve(_ksp_adj, _A_adj, _af_adj, _Lf_adj, bcs, lambda_adj)


def assemble_residual():
    """Assemble residual vector; returns reaction copy and global norm."""
    R_vec = fem_petsc.assemble_vector(_Rf_dom)                                         # assemble -internal force
    R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)        # sum ghost contributions
    R_reaction = R_vec.copy()                                                           # keep an unmodified copy (reactions)
    with R_vec.localForm() as R_local:
        for bc in bcs_homog:
            bc.set(R_local.array_w, None, 0.0)    # zero the residual at constrained DOFs before taking the norm
    return R_reaction, R_vec.norm()


# ============================================================================
# XDMF output files
# ============================================================================
dc_old = fem.Function(V0); dc = fem.Function(V0)   # previous / current raw sensitivity fields
dc_old_local = None                                  # history-averaging state (Eq. 47), set after iteration 1
rho_opt_data=[]; Sig_opt_data=[]; Ep_opt_data=[]      # per-iteration snapshots, kept for potential post-processing
fw_data=[]; vol_data=[]; load_avg_data=[]; disp_avg_data=[]   # objective / volume / load-displacement history

rho_xdmf         = io.XDMFFile(prob_mesh.comm, "BESO_Lbracket_Outputs/density.xdmf",         "w")   # density evolution
vonMises_xdmf    = io.XDMFFile(prob_mesh.comm, "BESO_Lbracket_Outputs/vonMises_solid.xdmf",  "w")
PEEQ_xdmf        = io.XDMFFile(prob_mesh.comm, "BESO_Lbracket_Outputs/PEEQ_solid.xdmf",      "w")
density_vis_xdmf = io.XDMFFile(prob_mesh.comm, "BESO_Lbracket_Outputs/density_visual.xdmf",  "w")
for _xf in (rho_xdmf, vonMises_xdmf, PEEQ_xdmf, density_vis_xdmf):
    _xf.write_mesh(prob_mesh)                          # write the (fixed) mesh once to every output file
vonMises_solid  = fem.Function(V0, name="VonMises_Solid")   # element-averaged von Mises stress (solid only)
PEEQ_solid      = fem.Function(V0, name="PEEQ_Solid")       # element-averaged equivalent plastic strain
density_vis     = fem.Function(V0, name="Density_Visual")   # density snapshot used for the visualization files
rho.name        = "Density"


# ============================================================================
# Precompute global element centroids once (mesh fixed across BESO iterations)
# ============================================================================
comm = prob_mesh.comm
_beso_local_sizes  = comm.allgather(rho.x.index_map.size_local)      # owned-element count on every rank
_beso_owned        = _beso_local_sizes[comm.rank]                    # this rank's own count
_beso_owned_start  = int(sum(_beso_local_sizes[:comm.rank]))          # this rank's offset into the global ordering
_beso_owned_end    = _beso_owned_start + _beso_owned
_coords_local_own  = rho.function_space.tabulate_dof_coordinates()[:_beso_owned, :3]   # this rank's element centroids
coords_global_fixed = np.concatenate(comm.allgather(_coords_local_own))  # (n_ele_global, 3)
global_coord_tree   = cKDTree(coords_global_fixed)   # ★3: built once on all ranks
coords_owned_fixed  = coords_global_fixed[_beso_owned_start:_beso_owned_end]  # this rank's own centroids, same order
pout(f"[BESO] Global coord tree built: {len(coords_global_fixed)} elements.", flush=True)

vol    = 1.0   # current target volume fraction, updated every iteration
change = 1.0   # convergence metric, updated every iteration

# ============================================================================
# BESO outer loop (Algorithm 2, Sec. 3.9)
# ============================================================================
for i in range(max_iter):
    vol  = max(vol*(1-er), volfrac)                      # shrink target volume geometrically toward volfrac (Eq. 47)
    rmin = (Lx/nx) * max(2.0, 6.0 - i*0.15)              # filter radius, linearly decaying with iteration
    vol_data.append(vol)

    zero(u, u_old, Sigma, Epsilon_p, Cep_gp, lambda_adj, mu_adj)   # reset all state fields for the new iteration
    Ep_old[:] = 0.0
    Eqp_old[:] = 0.0      # reset ε̄^p for new design iteration
    fw = 0.0                                    # work objective, accumulated over this iteration's load steps
    Fext_E_old = np.zeros(n_f)                   # previous load step's reaction force, for the trapezoidal rule
    total_app_disp = 0.0                          # running total applied displacement
    load_avg = np.zeros(n_load); disp_avg = np.zeros(n_load)   # per-load-step logging arrays
    Sig_data_iter=[]; lambda_data_iter=[]; mu_data_iter=[]      # per-load-step history, used by sensitivity_func

    pout(f"\n{'='*60}", flush=True)
    pout(f"  BESO Iteration {i+1:3d} | vol = {vol:.4f} | rmin = {rmin:.1f}", flush=True)
    pout(f"{'='*60}", flush=True)

    start1 = time.perf_counter()

    for i_load in range(n_load):
        total_app_disp += load_incr[i_load]                    # running total applied displacement
        pout(f"\n  Load step {i_load+1}/{n_load}"
             f"  (Δu = {load_incr[i_load]:.3f}, u_total = {total_app_disp:.3f})", flush=True)

        bcs_incr = get_bc_incremental(load_incr[i_load])         # this step's Dirichlet BCs

        # Predictor (★1, ★2)
        Du.x.array[:] = 0.0                       # reset the incremental displacement for this load step
        set_bc(Du.x.petsc_vec, bcs_incr)           # write the prescribed displacement directly into Du
        update_state(u_old)                         # elastic-predictor stress, based on the previous step's strain
        _solve_predictor(bcs_incr)                   # linear elastic predictor solve

        # Newton-Raphson corrector (Sec. 3.6)
        for itr in range(1, 12):
            u.x.array[:] = u_old.x.array + Du.x.array   # trial total displacement for this Newton iterate
            Sig, Ep, eqp = update_state(u)               # re-run the return map at the new strain
            R_reaction, nRes = assemble_residual()       # residual norm drives the convergence check
            pout(f"    Newton iter {itr:2d} | |R| = {nRes:.4e}", flush=True)
            if nRes <= 1e-10:
                break                                    # converged: exit the Newton loop early
            _solve_corrector(bcs_homog)                  # otherwise take another corrector step

        u_old.x.array[:] = u.x.array                      # commit the converged displacement as the new "previous"
        Ep_old  = Ep.copy()                                # persist plastic strain across load steps
        Eqp_old = eqp.copy()   # persist ε̄^p across load steps (manuscript Eq. 25)

        # Objective function (external work, Sec. 2.3.1)
        Fint_vec = fem_petsc.assemble_vector(fem.form(
            rho * inner(Sigma, epsilon(v_test)) * dx_domain))         # internal force vector at the converged state
        Fint_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with Fint_vec.localForm() as f_local:
            Fext_E = f_local.array[bc_load_dofs_owned].copy()  # owned DOFs only
        Du_E = Du.x.array.reshape(-1, gdim)[bc_load_dofs_owned // gdim, 1]   # this step's y-displacement increment at the load patch

        fw_local_step = 0.5 * np.matmul((Fext_E + Fext_E_old).T, Du_E)     # trapezoidal work increment (Eq. 29), this rank
        fw_global_step = MPI.COMM_WORLD.allreduce(fw_local_step, op=MPI.SUM)   # sum across ranks
        fw += fw_global_step                                                   # running total work objective

        Fext_E_old = Fext_E                                                    # carry forward for next step's trapezoid
        load_avg[i_load] = np.average(Fext_E) if Fext_E.size > 0 else 0.0
        disp_avg[i_load] = total_app_disp
        fext_norm_sq_local = float(np.dot(Fext_E, Fext_E))
        fext_norm_global = MPI.COMM_WORLD.allreduce(fext_norm_sq_local, op=MPI.SUM) ** 0.5
        pout(f"    fw = {fw:.6e} | ||Fext_E|| = {fext_norm_global:.4e}", flush=True)

        # Adjoint solve (★2)
        adjoint_bcs = [
            fem.dirichletbc(ScalarType((0., 0., 0.)), support_dofs, V),            # zero at the clamp, as in the corrector
            fem.dirichletbc(ScalarType(-load_incr[i_load]), bc_load_dofs, V.sub(1))  # adjoint load term (Eq. 38-40)
        ]
        _solve_adjoint(adjoint_bcs)
        lambda_data_iter.append(lambda_adj.copy())          # store this step's lambda for use in the sensitivity sum

        mu_adj.x.array[:] = (lambda_adj.x.array if i_load == 0
                              else lambda_data_iter[i_load-1].x.array)   # mu at step n reuses lambda from step n-1
        mu_data_iter.append(mu_adj.copy())
        Sig_data_iter.append(Sigma.copy())                    # store this step's converged stress too

    elapsed = time.perf_counter() - start1

    load_avg_data.append(load_avg); disp_avg_data.append(disp_avg)
    rho_opt_data.append(rho.copy()); Sig_opt_data.append(Sigma.copy())
    Ep_opt_data.append(Ep.copy()); fw_data.append(fw)

    # Sensitivity (Eq. 41-42, Sec. 3.7)
    dc = sensitivity_func(rho, Sig_data_iter, lambda_data_iter, mu_data_iter)   # raw, elemental sensitivity field

    # ── Distributed filter + BESO update (Sec. 3.9) ──────────────────────────
    # Step 1: allgather owned sensitivities to every rank
    dc_global_all = np.concatenate(comm.allgather(dc.x.array[:_beso_owned]))   # full sensitivity vector, all ranks
    rho_owned     = rho.x.array[:_beso_owned].copy()                            # this rank's current densities

    # Step 2: shift non-negative (Eq. 43)
    dc_min_global = float(dc_global_all.min()) if dc_global_all.size > 0 else 0.0
    if dc_min_global < 0:
        dc_global_all -= 1.1 * dc_min_global          # shift so every sensitivity is positive before damping/filtering

    # Step 3: damp (Eq. 46)
    dc_global_all = dc_global_all ** dampCoeff        # nonlinear damping suppresses oscillation from plastic effects

    # Step 4: sensitivity filter (Eq. 43-44, ★3)
    nbr_lists = global_coord_tree.query_ball_point(coords_owned_fixed, r=rmin)   # neighbours within rmin, per owned element
    num_f = np.empty(_beso_owned); den_f = np.empty(_beso_owned)
    for ii, nbrs in enumerate(nbr_lists):
        nbrs_arr  = np.asarray(nbrs, dtype=int)
        dists     = np.linalg.norm(coords_global_fixed[nbrs_arr] - coords_owned_fixed[ii], axis=1)  # distance to each neighbour
        w         = np.maximum(0.0, rmin - dists)      # linear distance weight (Eq. 44), zero beyond rmin
        num_f[ii] = w @ dc_global_all[nbrs_arr]         # weighted sum of neighbour sensitivities
        den_f[ii] = w.sum()                              # sum of weights, for normalization
    dc_filtered_owned = np.where(
        den_f > 0,
        num_f / den_f,                                    # weighted average (Eq. 43)
        dc_global_all[_beso_owned_start:_beso_owned_end])  # fallback: unfiltered value if no neighbours found

    # Step 5: history smoothing (Eq. 47)
    if i > 0 and dc_old_local is not None:
        dc_filtered_owned = 0.5 * dc_filtered_owned + 0.5 * dc_old_local   # blend with the previous iteration's filtered value
    dc_old_local = dc_filtered_owned.copy()               # save for next iteration's smoothing

    # Step 6: global threshold via distributed bisection (Sec. 3.9)
    n_vol = int(vol * vol_dom / vol_el)                    # target number of solid elements globally
    n_vol = min(n_vol, n_ele_global - 1)                    # clamp to a valid rank index
    n_vol = max(n_vol, 0)
    alpha = _parallel_kth_largest(comm, dc_filtered_owned, n_vol)   # the n_vol-th largest sensitivity, found without a full gather

    # Step 7: local BESO density update (Eq. 48)
    rho_s = np.where(rho_owned > 0.5)[0]                    # indices of currently-solid owned elements
    rho_v = np.where(rho_owned <= 0.5)[0]                    # indices of currently-void owned elements
    rho_new_local = np.zeros_like(rho_owned)
    rho_new_local[rho_s] = np.maximum(0., np.sign(dc_filtered_owned[rho_s] - alpha))   # keep solid iff sensitivity > alpha
    rho_new_local[rho_v] = np.maximum(0., np.sign(dc_filtered_owned[rho_v] - alpha))   # reinstate void iff sensitivity > alpha
    rho_new_local[rho_new_local == 0] = 0.001                # soft-void density floor, avoids a singular stiffness matrix

    # Step 8: addition-ratio limiter (Eq. 49)
    vol_curr   = comm.allreduce(float(np.sum(rho_owned) * vol_el), op=MPI.SUM)   # current total solid volume
    rec_global = int(comm.allreduce(
        int(np.sum(dc_filtered_owned[rho_v] > alpha)) if rho_v.size > 0 else 0,
        op=MPI.SUM))                                          # count of void elements about to be reinstated, globally
    c_ar = (vol_el * rec_global) / vol_curr if vol_curr > 0 else 0.0   # fraction of current volume being added back

    if c_ar > c_ar_max:
        # too many elements would be reinstated at once: fall back to separate add/delete thresholds
        n_add_total = int(np.ceil(c_ar_max * (vol_curr / vol_el)))         # capped number of elements to add
        n_del_total = max(int((vol_curr / vol_el + n_add_total) - vol * vol_dom / vol_el), 0)   # elements to remove to compensate
        a_add = _parallel_kth_largest(
            comm,
            dc_filtered_owned[rho_v] if rho_v.size > 0 else np.empty(0),
            n_add_total)                                                    # addition threshold, among void elements only
        a_del = _parallel_kth_smallest(
            comm,
            dc_filtered_owned[rho_s] if rho_s.size > 0 else np.empty(0),
            n_del_total)                                                    # deletion threshold, among solid elements only
        rho_new_local[rho_s] = np.maximum(0., np.sign(dc_filtered_owned[rho_s] - a_del))
        rho_new_local[rho_v] = np.maximum(0., np.sign(dc_filtered_owned[rho_v] - a_add))
        rho_new_local[rho_new_local == 0] = 0.001

    # Step 9: write back and propagate ghost DOFs
    rho_pre_local = rho_owned.copy()                          # density before this iteration's update, for visualization
    rho.x.array[:_beso_owned] = rho_new_local                 # commit the new density for owned elements
    rho.x.scatter_forward()                                    # propagate to ghost entries on neighbouring ranks
    rho_xdmf.write_function(rho, i)                             # write this iteration's density field

    rho_pre_update_func = fem.Function(V0)
    rho_pre_update_func.x.array[:_beso_owned] = rho_pre_local
    rho_pre_update_func.x.scatter_forward()
    rho_pre_update = rho_pre_update_func.x.array

    # Visualization: last-load-step plasticity state on pre-update solid domain
    vonMises_solid.x.array[:] = compute_vonMises_DG0(Sig, rho_pre_update)   # element-averaged von Mises, NaN on void
    PEEQ_solid.x.array[:]     = compute_PEEQ_DG0(Ep,  rho_pre_update)      # element-averaged PEEQ, NaN on void
    density_vis.x.array[:]    = rho_pre_update
    vonMises_xdmf.write_function(vonMises_solid, i)
    PEEQ_xdmf.write_function(PEEQ_solid, i)
    density_vis_xdmf.write_function(density_vis, i)

    # Summary & convergence check (owned DOFs only to avoid ghost double-counting)
    rho_owned = rho.x.array[:rho.x.index_map.size_local]
    vol_curr_global = comm.allreduce(rho_owned.sum() * vol_el, op=MPI.SUM)   # updated total solid volume
    solid_global    = comm.allreduce(int(np.sum(rho_owned > 0.5)),  op=MPI.SUM)   # solid element count, all ranks
    void_global     = comm.allreduce(int(np.sum(rho_owned <= 0.5)), op=MPI.SUM)   # void element count, all ranks

    vm_max_local = np.nanmax(vonMises_solid.x.array) if np.any(~np.isnan(vonMises_solid.x.array)) else -np.inf
    vm_max_global = comm.allreduce(vm_max_local, op=MPI.MAX)                       # peak von Mises stress, all ranks

    peeq_max_local = np.nanmax(PEEQ_solid.x.array) if np.any(~np.isnan(PEEQ_solid.x.array)) else -np.inf
    peeq_max_global = comm.allreduce(peeq_max_local, op=MPI.MAX)                   # peak PEEQ, all ranks

    if i > 10:
        # Eq. 50: relative change between two adjacent 5-iteration windows of the objective history
        change = abs(sum(fw_data[i-9:i-5]) - sum(fw_data[i-4:i])) / \
                (abs(sum(fw_data[i-4:i])) + 1e-30)
    prev_fw = fw_data[i-1] if i else fw_data[0]
    pct_chg = 100*abs(fw_data[i]-prev_fw) / (abs(prev_fw)+1e-30)   # single-step percentage change, for the log only

    pout(f" \n  ── Iteration {i+1} Summary ────────────────────────────── ", flush=True)
    pout(f"  Objective fw      : {fw_data[i]:.6e}  (Δ={pct_chg:.1f}%  conv={change:.6f}) ", flush=True)
    pout(f"  Volume fraction   : {vol_curr_global/vol_dom:.4f}  (target={vol:.4f}) ", flush=True)
    pout(f"  Solid/Void : {solid_global}/{void_global}  (total {n_ele_global}) ", flush=True)
    pout(f"  von Mises (solid) max : {vm_max_global:.4e} ", flush=True)
    pout(f"  PEEQ      (solid) max : {peeq_max_global:.4e} ", flush=True)
    pout(f"  Wall time         : {elapsed:.2f} s ", flush=True)
    pout(f"  {'─'*52} ", flush=True)

    if change < 0.001 and abs(vol - volfrac) <= 1e-10:
        pout("\n  ✓ Converged.", flush=True)
        break     # stop once the objective has flattened at the target volume

for _xf in (rho_xdmf, vonMises_xdmf, PEEQ_xdmf, density_vis_xdmf):
    _xf.close()

# ============================================================================
# Post-process: extract optimized solid sub-mesh
# ============================================================================
from dolfinx import mesh as dmesh   # local import: only needed for this final extraction step
solid_mask    = rho.x.array > 0.5                        # final solid/void classification
solid_indices = np.where(solid_mask)[0].astype(np.int32)  # element indices to keep in the exported geometry
pout(f"\nSolid elements : {len(solid_indices)}/{n_ele_global}  "
     f"(vf={len(solid_indices)/n_ele_global:.4f})", flush=True)
opt_mesh, *_ = dmesh.create_submesh(prob_mesh, prob_mesh.topology.dim, solid_indices)   # carve out just the solid elements
with io.XDMFFile(MPI.COMM_WORLD, "BESO_Lbracket_Outputs/optimized_geometry.xdmf", "w") as f:
    f.write_mesh(opt_mesh)                                 # write the final optimized geometry, viewable in ParaView
pout("Optimized geometry → BESO_Lbracket_Outputs/optimized_geometry.xdmf", flush=True)
