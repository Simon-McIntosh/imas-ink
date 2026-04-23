"""3D coilset and vessel rendering — pyvista primary, vedo for primitives.

All heavy imports (``pyvista``, ``vedo``, ``vtk``, ``shapely``) are
lazy-loaded inside submodules so that ``import imas_ink`` does not pull
VTK.  Consumers should import the specific submodule they need::

    from imas_ink.three_d.coilset import build_coilset
    mesh = build_coilset("imas:hdf5?path=/path/to/iter/")
"""
