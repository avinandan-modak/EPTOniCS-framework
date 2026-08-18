"""
run_cantilever.py — Example 1: Cantilever beam (manuscript Section 4.1)
=========================================================================
Three-dimensional elastoplastic BESO topology optimization of a
cantilever beam under displacement-controlled tip loading, using J2
plasticity with linear isotropic hardening (manuscript Section 2.1.2)
and the path-dependent adjoint sensitivity analysis of Section 2.3.2.

Implements the complete BESO workflow of Algorithm 2 — three nested
loops over design iterations, load increments, and Newton-Raphson
equilibrium — exactly as described in manuscript Section 3.9 and Fig. 3.

Geometry (units: mm)
---------------------
Domain  : 2000 × 1000 × 1000, mesh 50 × 25 × 25 (N_e = 31,250 Q1 elements)
Support : u = 0 on x₁ = 0 (clamped)
Load    : prescribed u_y on a central patch at the free end x₁ = 2000
Target volume fraction: 0.15

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
Section 4.1 — Numerical example 1: Cantilever beam

Usage
-----
    mpirun -np <N> python -u run_cantilever.py

Output XDMF files are written to ``BESO_J2_Outputs/``.
"""

import os  # needed immediately below, before any numerical library is imported

# Set thread counts for NumPy BLAS and Numba before any numerical import (Sec. 3.9)
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
from scipy.spatial import cKDTree                     # Sparse sensitivity filter (Sec. 3.9)
from eptonics.constitutive import j2_return_map as J2  # J2 return-map kernel (Sec. 3.4)
from eptonics.fem.kinematics import epsilon            # Voigt strain operator (Sec. 3.3)
from eptonics.fem.utils import zero                    # Function-array reset helper
from eptonics.solvers.petsc_backend import _build_ksp, _petsc_solve  # persistent KSP (Sec. 3.5.1)
from eptonics.visualization.postprocess import compute_vonMises_DG0, compute_PEEQ_DG0  # DG0 post-processing
from eptonics.mpi import _parallel_kth_largest, _parallel_kth_smallest  # distributed threshold (Sec. 3.9)
import time                                           # wall-clock timing for the console log

os.makedirs("BESO_J2_Outputs", exist_ok=True)   # every rank creates this; exist_ok avoids a race error


# ============================================================================
# Mesh and function spaces
# ============================================================================

def Cantilever3D_Setup(shape, nx, ny, nz):
    """Build the box mesh and the displacement (V) / density (V0) function spaces."""
    prob_mesh = mesh.create_box(
        comm=MPI.COMM_WORLD, points=((0,0,0), shape), n=(nx,ny,nz),
        cell_type=mesh.CellType.hexahedron)             # structured hex mesh, partitioned across MPI ranks
    gdim      = prob_mesh.topology.dim                  # geometric dimension (3 here)
    dx_domain = Measure("dx", domain=prob_mesh)          # standard cell-integration measure over the whole mesh
    V         = fem.functionspace(prob_mesh, ("CG", 1, (gdim,)))  # vector CG1: nodal displacement field
    V0        = fem.functionspace(prob_mesh, ('DG', 0))  # scalar DG0: one density value per element
    return V0, V, prob_mesh, dx_domain


def get_n_f(V, prob_mesh, shape):
    """Locate the loaded-face DOFs and the clamped-face DOFs."""
    gdim        = prob_mesh.topology.dim
    L, H, W     = shape                                  # domain length, height, width
    right_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x:
        np.isclose(x[0], L) &                            # facets on the free end (x = L)
        (x[1] >= 0.5*H - 0.3*H) & (x[1] <= 0.5*H + 0.3*H) &   # within a centred patch in y
        (x[2] >= 0.5*W - 0.3*W) & (x[2] <= 0.5*W + 0.3*W))    # within a centred patch in z
    dofs_right_y  = fem.locate_dofs_topological(V.sub(1), gdim - 1, right_facets)  # y-displacement DOFs on that patch
    support_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x: np.isclose(x[0], 0.0))  # x = 0 face
    support_dofs   = fem.locate_dofs_topological(V, gdim - 1, support_facets)      # all 3 components clamped there
    return len(dofs_right_y), dofs_right_y, support_dofs


def get_bc_incremental(V, prob_mesh, shape, disp_increment):
    """Dirichlet BCs for one Newton predictor step: clamp + prescribed y-displacement increment."""
    gdim        = prob_mesh.topology.dim
    L, H, W     = shape
    support_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x: np.isclose(x[0], 0.0))
    bc_left     = fem.dirichletbc(ScalarType((0.0, 0.0, 0.0)),        # u = (0,0,0) at the clamped end
                                  fem.locate_dofs_topological(V, gdim - 1, support_facets), V)
    right_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x:
        np.isclose(x[0], L) &
        (x[1] >= 0.5*H - 0.3*H) & (x[1] <= 0.5*H + 0.3*H) &
        (x[2] >= 0.5*W - 0.3*W) & (x[2] <= 0.5*W + 0.3*W))
    bc_right    = fem.dirichletbc(ScalarType(disp_increment),          # this step's incremental u_y
                                  fem.locate_dofs_topological(V.sub(1), gdim - 1, right_facets), V.sub(1))
    return [bc_left, bc_right]


def get_bc_homogeneous(V, prob_mesh, shape):
    """Zero-valued version of the same BCs, used inside Newton corrector iterations."""
    gdim        = prob_mesh.topology.dim
    L, H, W     = shape
    support_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x: np.isclose(x[0], 0.0))
    bc_left     = fem.dirichletbc(ScalarType((0.0, 0.0, 0.0)),
                                  fem.locate_dofs_topological(V, gdim - 1, support_facets), V)
    right_facets = mesh.locate_entities_boundary(prob_mesh, gdim - 1, lambda x:
        np.isclose(x[0], L) &
        (x[1] >= 0.5*H - 0.3*H) & (x[1] <= 0.5*H + 0.3*H) &
        (x[2] >= 0.5*W - 0.3*W) & (x[2] <= 0.5*W + 0.3*W))
    bc_right    = fem.dirichletbc(ScalarType(0.0),                     # zero increment: correction only
                                  fem.locate_dofs_topological(V.sub(1), gdim - 1, right_facets), V.sub(1))
    return [bc_left, bc_right]


# ============================================================================
# Sensitivity assembly (manuscript Eq. 41-42, Section 3.7)
# ============================================================================

def sensitivity_func(rho, Sig_data, lambda_data, mu_data, V0, dx_q):
    """Path-dependent adjoint sensitivity, MPI-safe (owned/ghost via PETSc copy)."""
    w = ufl.TestFunction(V0)         # DG0 test function: one weight per element, gives elemental sensitivities
    dc_form = 0.0                    # accumulate the trapezoidal sum over load steps (Eq. 41-42)
    for i in range(len(Sig_data)):
        if i == 0:
            # first load step: only the lambda (forward-adjoint) term contributes
            dc_form += -0.5 * inner(rho*Sig_data[i], epsilon(lambda_data[i]))
        else:
            # later steps: lambda term at step i plus mu term coupling back to step i-1 (Eq. 42)
            dc_form += -0.5*(inner(rho*Sig_data[i],   epsilon(lambda_data[i]))
                           + inner(rho*Sig_data[i-1], epsilon(mu_data[i])))
    dc_vec = fem_petsc.assemble_vector(fem.form(dc_form*w*dx_q))   # assemble elemental sensitivity vector
    dc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # sum contributions from ghost DOFs
    dc_fn = fem.Function(V0)
    dc_vec.copy(dc_fn.x.petsc_vec)                                  # copy PETSc vector into a Function
    dc_fn.x.scatter_forward()                                       # broadcast owned values back to ghosts
    return dc_fn


# ============================================================================
# Problem setup
# ============================================================================

shape       = (2000., 1000., 1000.)   # domain dimensions (mm): length, height, width
x1, y1, z1 = shape                    # unpacked for readability further down
nx, ny      = 50, 25                  # element counts along length / height
nz          = ny                      # same resolution in the width direction as in height

V0, V, prob_mesh, dx_domain = Cantilever3D_Setup(shape, nx, ny, nz)  # build mesh + function spaces
gdim = prob_mesh.topology.dim

Vcep = fem.functionspace(prob_mesh, quadrature_element(
    prob_mesh.topology.cell_name(), degree=2, value_shape=(6,6)))   # quadrature space for the 6x6 tangent Cep
Veps = fem.functionspace(prob_mesh, quadrature_element(
    prob_mesh.topology.cell_name(), degree=2, value_shape=(6,)))    # quadrature space for 6-component Voigt tensors
V0_q = fem.functionspace(prob_mesh, quadrature_element(
    prob_mesh.topology.cell_name(), degree=0))                      # scalar quadrature space (unused directly below)
dx_q = ufl.Measure("dx", domain=prob_mesh, metadata={"quadrature_degree": 2})  # integration measure tied to the Gauss rule

u         = fem.Function(V); u_old  = fem.Function(V); Du = fem.Function(V)  # current / previous / incremental displacement
eps_quad  = fem.Function(Veps)        # total strain, interpolated at quadrature points
Sigma     = fem.Function(Veps)        # Cauchy stress at quadrature points
Epsilon_p = fem.Function(Veps)        # plastic strain at quadrature points
Cep_gp    = fem.Function(Vcep)        # consistent elastoplastic tangent at quadrature points
lambda_adj = fem.Function(V); mu_adj = fem.Function(V)   # adjoint displacement fields (Eq. 38-40)

n_gp = Vcep.tabulate_dof_coordinates().shape[0]                     # total Gauss points on this rank
n_ele_local = V0.tabulate_dof_coordinates().shape[0]                 # elements owned by this rank
n_ele_global = prob_mesh.topology.index_map(prob_mesh.topology.dim).size_global  # elements across all ranks

n_f_raw, bc_load_dofs, support_dofs = get_n_f(V, prob_mesh, shape)   # locate loaded-face / clamped-face DOFs
n_f_local  = len(bc_load_dofs)
n_f_global = MPI.COMM_WORLD.allreduce(n_f_local, op=MPI.SUM)         # total loaded DOFs, summed across ranks
pout(f"[BESO] n_f (load DOFs) = local: {n_f_local}, global: {n_f_global}  n_gp = {n_gp}", flush=True)
assert n_f_global > 0, "No load DOFs found globally — check load patch geometry / mesh alignment!"
# Restrict to owned DOFs to avoid ghost double-counting in force extraction
owned_flat_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs   # flat size of this rank's owned DOFs
bc_load_dofs_owned = bc_load_dofs[bc_load_dofs < owned_flat_size]         # keep only owned (non-ghost) load DOFs
n_f = len(bc_load_dofs_owned)

load_incr = 1*np.ones((10,))         # 10 equal displacement increments of 1 mm each
n_load = load_incr.shape[0]          # number of load steps per BESO design iteration

E_mod, nu = 75.0, 0.30                                             # Young's modulus (GPa), Poisson's ratio
shear = E_mod/(2*(1+nu)); bulk = 2*shear*(1+nu)/(3*(1-2*nu)); sig_y = 0.1   # shear/bulk moduli, yield stress
hard  = 1.5            # linear isotropic hardening modulus (0 = perfect plasticity)
Bulk_mat  = bulk  * np.ones((n_gp,1))   # per-Gauss-point material arrays, ready for the constitutive kernel
Shear_mat = shear * np.ones((n_gp,1))
Sig_y     = sig_y * np.ones((n_gp,1))
Hard_mat  = hard  * np.ones((n_gp,1))
Ep_old    = np.zeros((n_gp,6))        # converged plastic strain at the previous load step
Eqp_old   = np.zeros(n_gp)            # equivalent plastic strain ε̄^p

du_trial = TrialFunction(V); v_test = TestFunction(V)   # trial/test functions for the predictor/corrector forms
dl       = TrialFunction(V); v_adj  = TestFunction(V)   # trial/test functions for the adjoint form
bcs_homog = get_bc_homogeneous(V, prob_mesh, shape)      # zero-valued BCs, reused every Newton correction

max_iter=140; er=0.02; volfrac=0.15; move=0.05; dampCoeff=0.6; c_ar_max=0.02
# max_iter    : maximum BESO design iterations
# er          : evolutionary ratio, fraction of volume removed per iteration (Eq. 47)
# volfrac     : target final volume fraction
# move        : unused move-limit placeholder (kept for parity with the manuscript's parameter list)
# dampCoeff   : sensitivity damping exponent beta (Eq. 45)
# c_ar_max    : maximum addition ratio before the two-threshold limiter engages (Eq. 49)

rho = fem.Function(V0); rho.x.array[:] = 1.0    # design density field, starts fully solid
dc_old = fem.Function(V0); dc = fem.Function(V0)  # previous / current raw sensitivity fields
dc_old_local = None                               # history-averaging state (Eq. 47), set after iteration 1

vol_dom_local = fem.assemble_scalar(fem.form(fem.Constant(prob_mesh,ScalarType(1.))*dx_domain))  # this rank's volume
vol_dom = prob_mesh.comm.allreduce(vol_dom_local, op=MPI.SUM)        # total domain volume, all ranks
vol_el  = vol_dom / n_ele_global                                     # volume of one (uniform) element
pout(f"[BESO] n_ele = local: {n_ele_local}, global: {n_ele_global} | vol_dom = {vol_dom:.1f}  vol_el = {vol_el:.2f}", flush=True)

rho_opt_data=[]; Sig_opt_data=[]; Ep_opt_data=[]      # per-iteration snapshots, kept for potential post-processing
fw_data=[]; vol_data=[]; load_avg_data=[]; disp_avg_data=[]   # objective / volume / load-displacement history

rho_xdmf = io.XDMFFile(prob_mesh.comm, "BESO_J2_Outputs/density.xdmf", "w")   # density-evolution output file
rho_xdmf.write_mesh(prob_mesh); rho.name = "Density"

vonMises_solid   = fem.Function(V0, name="VonMises_Solid")   # element-averaged von Mises stress (solid only)
PEEQ_solid       = fem.Function(V0, name="PEEQ_Solid")       # element-averaged equivalent plastic strain
density_vis      = fem.Function(V0, name="Density_Visual")   # density snapshot used for the visualization files
vonMises_xdmf    = io.XDMFFile(prob_mesh.comm, "BESO_J2_Outputs/vonMises_solid.xdmf", "w")
PEEQ_xdmf        = io.XDMFFile(prob_mesh.comm, "BESO_J2_Outputs/PEEQ_solid.xdmf",     "w")
density_vis_xdmf = io.XDMFFile(prob_mesh.comm, "BESO_J2_Outputs/density_visual.xdmf", "w")
vonMises_xdmf.write_mesh(prob_mesh)
PEEQ_xdmf.write_mesh(prob_mesh)
density_vis_xdmf.write_mesh(prob_mesh)


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
_L_ufl_adj = rho * inner(fem.Constant(prob_mesh, ScalarType((0.,0.,0.))), v_adj) * dx_q  # adjoint RHS (BCs carry the load)
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
_ksp_q,   _A_q   = _build_ksp(_af_q,   prob_mesh.comm)   # predictor
_ksp_dom, _A_dom = _build_ksp(_af_dom,  prob_mesh.comm)   # corrector
_ksp_adj, _A_adj = _build_ksp(_af_adj,  prob_mesh.comm)   # adjoint
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
    R_vec = fem_petsc.assemble_vector(_Rf_dom)                                        # assemble -internal force
    R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)       # sum ghost contributions
    R_reaction = R_vec.copy()                                                          # keep an unmodified copy (reactions)
    with R_vec.localForm() as R_local:
        for bc in bcs_homog:
            bc.set(R_local.array_w, None, 0.0)    # zero the residual at constrained DOFs before taking the norm
    return R_reaction, R_vec.norm()


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

vol = 1.0; change = 1.0   # current target volume fraction; convergence metric, both updated every iteration

# ============================================================================
# BESO outer loop (Algorithm 2, Sec. 3.9)
# ============================================================================
for i in range(max_iter):
    vol  = max(vol*(1-er), volfrac)                      # shrink target volume geometrically toward volfrac (Eq. 47)
    rmin = (x1/nx)*max(2.0, 6.0-i*0.15)                  # filter radius, linearly decaying with iteration
    vol_data.append(vol)

    zero(u, u_old, Sigma, Epsilon_p, Cep_gp, lambda_adj, mu_adj)   # reset all state fields for the new iteration
    Ep_old[:] = 0.0
    Eqp_old[:] = 0.0      # reset ε̄^p for new design iteration
    fw = 0.0; Fext_E_old = np.zeros(n_f); total_app_disp = 0.0    # objective, previous-step force, applied displacement
    load_avg = np.zeros(n_load); disp_avg = np.zeros(n_load)      # per-load-step logging arrays
    Sig_data=[]; lambda_data=[]; mu_data=[]                       # per-load-step history, used by sensitivity_func

    pout(f"\n{'='*60}", flush=True)
    pout(f"  BESO Iteration {i+1:3d} | vol = {vol:.4f} | rmin = {rmin:.1f}", flush=True)
    pout(f"{'='*60}", flush=True)

    start1 = time.perf_counter()

    for i_load in range(n_load):
        total_app_disp += load_incr[i_load]                      # running total applied tip displacement
        pout(f"\n  Load step {i_load+1}/{n_load}"
             f"  (Δu = {load_incr[i_load]}, u_total = {total_app_disp})", flush=True)

        bcs_incr = get_bc_incremental(V, prob_mesh, shape, load_incr[i_load])   # this step's Dirichlet BCs

        # Predictor (★1: pre-compiled form, ★2: persistent KSP)
        Du.x.array[:] = 0.0                       # reset the incremental displacement for this load step
        set_bc(Du.x.petsc_vec, bcs_incr)           # write the prescribed displacement directly into Du
        update_state(u_old)                        # elastic-predictor stress, based on the previous step's strain
        _solve_predictor(bcs_incr)                  # linear elastic predictor solve

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
            rho*inner(Sigma, epsilon(v_test))*dx_domain))            # internal force vector at the converged state
        Fint_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with Fint_vec.localForm() as f_local:
            Fext_E = f_local.array[bc_load_dofs_owned].copy()  # owned DOFs only
        Du_E  = Du.x.array.reshape(-1,gdim)[bc_load_dofs_owned//gdim, 1]   # this step's y-displacement increment at the load patch
        fw_local_step = 0.5 * np.matmul((Fext_E + Fext_E_old).T, Du_E)     # trapezoidal work increment (Eq. 29), this rank
        fw_global_step = MPI.COMM_WORLD.allreduce(fw_local_step, op=MPI.SUM)  # sum across ranks
        fw += fw_global_step                                                # running total work objective
        Fext_E_old       = Fext_E                                            # carry forward for next step's trapezoid
        load_avg[i_load] = np.average(Fext_E) if Fext_E.size > 0 else 0.0
        disp_avg[i_load] = total_app_disp
        fext_norm_sq_local = float(np.dot(Fext_E, Fext_E))
        fext_norm_global = MPI.COMM_WORLD.allreduce(fext_norm_sq_local, op=MPI.SUM) ** 0.5
        pout(f"    fw = {fw:.6e} | ||Fext_E|| = {fext_norm_global:.4e}", flush=True)

        # Adjoint solve (★2: persistent KSP)
        adjoint_bcs = [
            fem.dirichletbc(ScalarType((0.,0.,0.)),    support_dofs,  V),         # zero at the clamp, as in the corrector
            fem.dirichletbc(ScalarType(-load_incr[i_load]), bc_load_dofs, V.sub(1))  # adjoint load term (Eq. 38-40)
        ]
        _solve_adjoint(adjoint_bcs)
        lambda_data.append(lambda_adj.copy())          # store this step's lambda for use in the sensitivity sum

        mu_adj.x.array[:] = (lambda_adj.x.array if i_load==0
                              else lambda_data[i_load-1].x.array)   # mu at step n reuses lambda from step n-1
        mu_data.append(mu_adj.copy())
        Sig_data.append(Sigma.copy())                    # store this step's converged stress too

    elapsed = time.perf_counter()-start1

    load_avg_data.append(load_avg); disp_avg_data.append(disp_avg)
    rho_opt_data.append(rho.copy()); Sig_opt_data.append(Sigma.copy())
    Ep_opt_data.append(Ep.copy()); fw_data.append(fw)

    # Sensitivity (Eq. 41-42, Sec. 3.7)
    dc = sensitivity_func(rho, Sig_data, lambda_data, mu_data, V0, dx_q)   # raw, elemental sensitivity field

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

    # Step 4: sensitivity filter (Eq. 43-44)
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
    rho_new_local[rho_s] = np.maximum(0., np.sign(dc_filtered_owned[rho_s] - alpha))  # keep solid iff sensitivity > alpha
    rho_new_local[rho_v] = np.maximum(0., np.sign(dc_filtered_owned[rho_v] - alpha))  # reinstate void iff sensitivity > alpha
    rho_new_local[rho_new_local == 0] = 0.001               # soft-void density floor, avoids a singular stiffness matrix

    # Step 8: addition-ratio limiter (Eq. 49)
    vol_curr   = comm.allreduce(float(np.sum(rho_owned) * vol_el), op=MPI.SUM)   # current total solid volume
    rec_global = int(comm.allreduce(
        int(np.sum(dc_filtered_owned[rho_v] > alpha)) if rho_v.size > 0 else 0,
        op=MPI.SUM))                                          # count of void elements about to be reinstated, globally
    c_ar = (vol_el * rec_global) / vol_curr if vol_curr > 0 else 0.0   # fraction of current volume being added back

    if c_ar > c_ar_max:
        # too many elements would be reinstated at once: fall back to separate add/delete thresholds
        n_add_total = int(np.ceil(c_ar_max * (vol_curr / vol_el)))         # capped number of elements to add
        n_del_total = max(int((vol_curr / vol_el + n_add_total) - vol * vol_dom / vol_el), 0)  # elements to remove to compensate
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
    PEEQ_solid.x.array[:]     = compute_PEEQ_DG0(Ep,     rho_pre_update)   # element-averaged PEEQ, NaN on void
    density_vis.x.array[:]    = rho_pre_update
    vonMises_xdmf.write_function(vonMises_solid, i)
    PEEQ_xdmf.write_function(PEEQ_solid,         i)
    density_vis_xdmf.write_function(density_vis,  i)

    # Summary & convergence check
    rho_owned = rho.x.array[:rho.x.index_map.size_local]
    vol_curr_global = comm.allreduce(rho_owned.sum() * vol_el, op=MPI.SUM)   # updated total solid volume
    solid_global    = comm.allreduce(int(np.sum(rho_owned > 0.5)),  op=MPI.SUM)  # solid element count, all ranks
    void_global     = comm.allreduce(int(np.sum(rho_owned <= 0.5)), op=MPI.SUM)  # void element count, all ranks

    vm_max_local   = np.nanmax(vonMises_solid.x.array) if np.any(~np.isnan(vonMises_solid.x.array)) else -np.inf
    vm_max_global  = comm.allreduce(vm_max_local, op=MPI.MAX)                  # peak von Mises stress, all ranks
    peeq_max_local  = np.nanmax(PEEQ_solid.x.array) if np.any(~np.isnan(PEEQ_solid.x.array)) else -np.inf
    peeq_max_global = comm.allreduce(peeq_max_local, op=MPI.MAX)               # peak PEEQ, all ranks

    if i > 10:
        # Eq. 50: relative change between two adjacent 5-iteration windows of the objective history
        change = abs(sum(fw_data[i-9:i-5])-sum(fw_data[i-4:i])) / (abs(sum(fw_data[i-4:i])) + 1e-30)
    prev_fw = fw_data[i-1] if i else fw_data[0]
    pct_chg = 100*abs(fw_data[i]-prev_fw) / (abs(prev_fw) + 1e-30)   # single-step percentage change, for the log only

    pout(f"\n  ── Iteration {i+1} Summary ──────────────────────────────", flush=True)
    pout(f"  Objective fw      : {fw:.6e}  (Δ={pct_chg:.1f}%  conv={change:.6f})", flush=True)
    pout(f"  Volume fraction   : {vol_curr_global/vol_dom:.4f}  (target={vol:.4f})", flush=True)
    pout(f"  Solid/Void : {solid_global}/{void_global}  (total {n_ele_global})", flush=True)
    pout(f"  von Mises (solid) max : {vm_max_global:.4e}", flush=True)
    pout(f"  PEEQ      (solid) max : {peeq_max_global:.4e}", flush=True)
    pout(f"  Wall time         : {elapsed:.2f} s", flush=True)
    pout(f"  {'─'*52}", flush=True)

    if change < 0.001 and abs(vol-volfrac) <= 1e-10:
        pout("\n  ✓ Converged.", flush=True); break     # stop once the objective has flattened at the target volume

rho_xdmf.close()
vonMises_xdmf.close()
PEEQ_xdmf.close()
density_vis_xdmf.close()

# ============================================================================
# Post-process: extract optimized solid sub-mesh
# ============================================================================
from dolfinx import mesh as dmesh   # local import: only needed for this final extraction step

solid_mask    = rho.x.array > 0.5                        # final solid/void classification
solid_indices = np.where(solid_mask)[0].astype(np.int32)  # element indices to keep in the exported geometry
pout(f"\nSolid elements : {len(solid_indices)}/{n_ele_global}  "
     f"(vf={len(solid_indices)/n_ele_global:.4f})", flush=True)
optimized_mesh, *_ = dmesh.create_submesh(
    prob_mesh, prob_mesh.topology.dim, solid_indices)      # carve out just the solid elements as a standalone mesh
with io.XDMFFile(MPI.COMM_WORLD, "BESO_J2_Outputs/optimized_geometry.xdmf","w") as f:
    f.write_mesh(optimized_mesh)                            # write the final optimized geometry, viewable in ParaView
pout("Optimized geometry → BESO_J2_Outputs/optimized_geometry.xdmf", flush=True)
