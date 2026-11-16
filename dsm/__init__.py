"""Time-domain direct sampling method for moving inclusions.

Modules
-------
mesh        gmsh ball mesh, CFL estimate, Fibonacci sampling
interp      stable tetrahedral interpolation (KD-tree + CSR)
fem_base    reference explicit FEM (lumped mass, central differences)
fem_solver  multi-threaded numba kernels used in production
inversion   receiver sampling and the DSM index function
experiment  CommonParameters / UnifiedExperiment: one interface per case
driver      forward solve and inversion of the four examples
"""

__all__ = ["mesh", "interp", "fem_base", "fem_solver", "inversion",
           "experiment", "driver"]