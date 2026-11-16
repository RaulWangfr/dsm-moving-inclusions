"""Central configuration for the four numerical examples of the manuscript.

Nothing in this file is tuned at run time: every number below is the value
used for the experiments reported in Section 6.  Change a field here (or pass
the corresponding command-line flag) rather than editing the solver.

Reference paper
---------------
Hongjie Li, Fengru Wang, Jun Zou,
"Direct sampling for moving inclusions: construction and conditional
localization", 2026.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Forward solver, incident field and inversion grid."""

    # ---------------------------------------------------------------- domain
    radius: float = 7.0            # ball  Omega_FEM = B(0, radius)
    lc: float = 0.1                # gmsh target element size (0.1 -> ~1e6 tets)
    n_outer: int = 64              # points per direction on the triangulated sphere
    dt_cfl: float = 0.5            # safety factor in dt = dt_cfl * h_min / c_max

    # ------------------------------------------------------- time integration
    t_final: float = 26.0          # recording window [0, t_final]
    record_interval: int = 3       # record every 3rd time step (3 h_t)

    # -------------------------------------------------- incident plane wave
    pulse_frequency: float = 0.5   # f_p: peak frequency of ONE Ricker pulse
    pulse_spacing: float = 1.0     # shift between successive pulses in the train
    pulse_count: int = 26          # m = 0..25
    pulse_shift: float = 2.0       # delay of the first pulse (travel time from y1=-inf)

    # --------------------------------------------------- geometry of motion
    orbit_radius: float = 0.5
    orbit_omega: float = math.pi / 10.0      # xi(t) = R (cos wt, sin wt, 0)
    inclusion_radius: float = 0.3

    # ------------------------------------------------ active source (case MS)
    source_frequency: float = 2.0 * math.pi

    # ----------------------------------------------------------- inversion
    img_window: float = 2.0        # image square [-2,2]^2 in the plane z = 0
    n_img: int = 101               # 101 x 101 image grid
    s_vals: Tuple[float, ...] = tuple(float(s) for s in range(2, 19))  # s = 2..18
    noise_seed: int = 20260923     # fixed seed: the noise realisations are reproducible

    # ------------------------------------------- receiver surfaces (geometry)
    sphere_radius: float = 2.0 * math.sqrt(3.0)   # MS: Fibonacci sphere
    cube_edge: float = 4.0                        # MI1: cube surface
    cloud_edge: float = 4.0                       # MI2: candidate cloud on the cube faces
    cloud_spacing: float = 0.25                   # 0.25 grid -> 17x17 per face
    frame_half_height: float = 0.75               # MI2: closed frame |y3| <= 0.75
    plane_z: float = 2.5                          # MI3: measurement plane (y3)
    plane_half: float = 1.0                       # MI3: square half edge (edge = 2)
    plane_grid: int = 9                           # MI3: 9x9 = 81 receivers


@dataclass(frozen=True)
class CaseSpec:
    """One numerical example.

    kind       : "source"    -> prescribed moving source, no inclusion
                 "inclusion" -> moving material contrast, plane-wave illumination
    receivers  : "sphere" | "cube" | "cloud" | "square"
    aperture   : "full" | "frame" | "square"  (frame = closed rectangle masked
                 out of the candidate cloud)
    """

    name: str
    kind: str
    receivers: str
    n_recv: int
    rho_in: float = 1.0
    mu_in: float = 1.0
    gamma_in: float = 0.0
    sigma_in: float = 0.0
    noise: float = 0.10
    time_decimation: int = 1
    aperture: str = "full"


CASES = {
    # MS: moving source, sparse full spherical aperture, 30% noise
    "MS": CaseSpec("MS", "source", "sphere", 96, noise=0.30),
    # MI1: restoring contrast, cube surface, 10% noise, time decimated by 15
    "MI1": CaseSpec("MI1", "inclusion", "cube", 768, sigma_in=1.0,
                    noise=0.10, time_decimation=15),
    # MI2: damping contrast, closed frame aperture on the cube faces
    "MI2": CaseSpec("MI2", "inclusion", "cloud", 448, gamma_in=1.0,
                    noise=0.10, aperture="frame"),
    # MI3: inertia-stiffness contrast, minimal 9x9 square aperture
    "MI3": CaseSpec("MI3", "inclusion", "square", 81, rho_in=1.2, mu_in=1.2,
                    noise=0.10, aperture="square"),
}

CONFIG = Config()



__all__ = ["Config", "CaseSpec", "CASES", "CONFIG"]