# Data

Reserved for mesh files, benchmark output artifacts, or large
auxiliary data that should not live inside `examples/` or `src/`.
Empty in this release — every example generates its mesh
programmatically (`dolfinx.mesh.create_box` + `dolfinx.mesh.create_submesh`)
and writes its own outputs to a per-example directory at runtime (see
`docs/examples.md`), so no static input data currently ships with the
repository.
