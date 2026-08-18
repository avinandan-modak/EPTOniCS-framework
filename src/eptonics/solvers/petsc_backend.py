"""
petsc_backend.py
=================
Persistent PETSc KSP construction and assemble-and-solve wrapper
(manuscript Section 3.5.1). Implements performance mechanism ★2:
one KSP object per solve type (predictor, corrector, adjoint) is built
once at start-up; only matrix values are reassembled in the Newton loop,
reusing sparsity pattern and algebraic multigrid (GAMG) hierarchies.
"""

from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc


def _build_ksp(a_form, mesh_comm):
    """Build and return a configured PETSc KSP for the given compiled form."""
    A   = fem_petsc.create_matrix(a_form)
    ksp = PETSc.KSP().create(mesh_comm)
    ksp.setOperators(A)

    # AMG + FGMRES (Section 3.5.1)
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setTolerances(rtol=1e-8, max_it=500)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.GAMG)
    pc.setGAMGType("agg")           # aggregation AMG for elasticity

    ksp.setFromOptions()
    return ksp, A


def _petsc_solve(ksp, A, a_form, L_form, bcs, x_func):
    """
    Assemble A and b, apply Dirichlet boundary conditions, solve, and scatter.
    ``x_func.x.array`` is updated in-place and returned.
    """
    A.zeroEntries()
    fem_petsc.assemble_matrix(A, a_form, bcs=bcs)
    A.assemble()
    ksp.setOperators(A)

    b = fem_petsc.create_vector(L_form)
    with b.localForm() as b_loc:
        b_loc.set(0.0)
    fem_petsc.assemble_vector(b, L_form)
    fem_petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    ksp.solve(b, x_func.x.petsc_vec)
    x_func.x.scatter_forward()
    return x_func
