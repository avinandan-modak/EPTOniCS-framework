"""
utils.py
========
Small, generic ``dolfinx.fem.Function`` helpers shared verbatim across
all three example drivers.
"""


def zero(*funcs):
    """
    Reset the underlying array of one or more ``dolfinx.fem.Function``
    objects to zero, in place.

    Used at the start of every BESO design iteration to clear the
    displacement, stress, plastic-strain, and adjoint fields before
    the load-stepping loop begins (manuscript Algorithm 2, "Initialize
    all state fields to zero").

    Parameters
    ----------
    *funcs : dolfinx.fem.Function
        Any number of Function objects whose ``.x.array`` will be set
        to ``0.0`` in place. Nothing is returned; each Function is
        mutated directly.
    """
    for f in funcs:
        f.x.array[:] = 0.0
