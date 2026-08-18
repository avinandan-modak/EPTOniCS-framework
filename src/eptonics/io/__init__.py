"""
eptonics.io
==============

Reserved for shared I/O and file management helpers (e.g. XDMF/H5
output setup, checkpoint/restart utilities).

Each example driver currently opens and writes its own
``dolfinx.io.XDMFFile`` objects directly and inline (see
`docs/examples.md` for the output files each example produces).
This subpackage is kept as the natural home for shared I/O helpers
if a future contribution adds them (e.g. a common checkpoint/restart
mechanism).
"""
