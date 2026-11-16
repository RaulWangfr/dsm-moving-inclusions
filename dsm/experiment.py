"""Unified MS/MI1-MI3 forward/inverse interface."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import numpy as np

from .mesh import get_or_generate_mesh, sample_sphere_fibonacci
from .mesh import estimate_h_min, compute_cfl_dt
from .fem_solver import UnifiedFEMSolver3D, set_num_threads
from .inversion import (invert_surface, receiver_map, sample_snapshots,
                         sphere_normals, surface_radius)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    source_mode: str
    rho_in: float = 1.0
    mu_in: float = 1.0
    gamma: float = 0.0
    sigma: float = 0.0


PROBLEMS = {
    "MS": ProblemSpec("MS", "moving_source"),
    "MI1": ProblemSpec("MI1", "fixed_source", sigma=1.0),
    "MI2": ProblemSpec("MI2", "fixed_source", gamma=1.0),
    "MI3": ProblemSpec("MI3", "fixed_source", rho_in=1.2, mu_in=1.2),
}


@dataclass(frozen=True)
class CommonParameters:
    radius: float = 7.0
    lc: float = 0.1
    n_outer: int = 64
    receiver_radius: float = float(2.0 * np.sqrt(3.0))
    n_recv: int = 768
    t_final: float = 17.0
    dt_cfl: float = 0.5
    record_interval: int = 3
    inclusion_radius: float = 0.3
    # None -> sphere of inclusion_radius.  (ax,ay,az) -> ellipsoid with
    # those three semi-axes (world axes, constant orientation).
    inclusion_axes: tuple | None = None
    orbit_radius: float = 0.5
    orbit_omega: float = float(np.pi / 10.0)
    source_frequency: float = 2.0 * np.pi
    pulse_frequency: float = 0.5
    pulse_shift: float = 2.0
    pulse_spacing: float = 1.0
    pulse_count: int = 11
    n_threads: int = 6


class UnifiedExperiment:
    """One interface for MS/MI1-MI3 forward data and sphere DSM."""

    def __init__(self, code_dir, problem="MS", params=None, mesh_dir=None):
        self.code_dir = os.path.abspath(code_dir)
        self.mesh_dir = os.path.abspath(mesh_dir or self.code_dir)
        self.spec = PROBLEMS[problem] if isinstance(problem, str) else problem
        self.params = params or CommonParameters()
        set_num_threads(self.params.n_threads)
        self.nodes, self.elems, self.bdy_faces, self.bdy_areas = get_or_generate_mesh(
            self.mesh_dir, self.params.radius, self.params.lc, self.params.n_outer)
        h_min = estimate_h_min(self.nodes, self.elems)
        # CFL speed: max over the materials of the physical sound speed
        # c = sqrt(mu/rho).  Background is (rho, mu) = (1, 1).
        c_in = math.sqrt(self.spec.mu_in / self.spec.rho_in)
        c_max = max(c_in, 1.0)
        self.dt = compute_cfl_dt(h_min, c_max, self.params.dt_cfl)
        self.receiver_points = sample_sphere_fibonacci(
            self.params.receiver_radius, self.params.n_recv)
        self.receiver_map = receiver_map(self.nodes, self.elems,
                                          self.receiver_points)

    def orbit(self, t):
        p = self.params
        return np.array([p.orbit_radius * np.cos(p.orbit_omega * t),
                         p.orbit_radius * np.sin(p.orbit_omega * t), 0.0])

    def _source(self, t):
        if self.spec.source_mode == "moving_source":
            return np.sin(self.params.source_frequency * t) * min(1.0, t)
        return 0.0

    def _source_centre(self, t):
        """Centre of the moving source (MS); the plane-wave cases have none."""
        return self.orbit(t) if self.spec.source_mode == "moving_source" else np.zeros(3)

    def _inclusion_centre(self, t):
        """Moving inclusion centre for MI1-MI3; MS has no inclusion."""
        return np.zeros(3) if self.spec.source_mode == "moving_source" else self.orbit(t)

    def _pulse(self, t, pts, derivative=0):
        p = self.params
        x = pts[:, 0]
        a = np.pi ** 2 * p.pulse_frequency ** 2
        out = np.zeros(len(pts))
        for n in range(p.pulse_count):
            tau = t - x - p.pulse_shift - n * p.pulse_spacing
            tau2 = tau * tau
            e = np.exp(-a * tau2)
            if derivative == 0:
                out += (1.0 - 2.0 * a * tau2) * e
            elif derivative == 1:
                out += (-6.0 * a * tau + 4.0 * a * a * tau ** 3) * e
            else:
                out += (-6.0 * a + 24.0 * a * a * tau2
                        - 8.0 * a ** 3 * tau2 * tau2) * e
        return out

    def run_forward(self, mmap_path=None):
        p = self.params
        solver = UnifiedFEMSolver3D(
            self.nodes, self.elems, self.bdy_faces, self.bdy_areas,
            rho_out=1.0, mu_out=1.0,
            rho_in=self.spec.rho_in, mu_in=self.spec.mu_in)
        solver.source_centre_fn = self._source_centre
        is_source = self.spec.source_mode == "moving_source"
        use_contrast = not is_source
        times, snaps = solver.run(
            p.t_final, self.dt, self._source, self._inclusion_centre,
            inclusion_radius=0.0 if is_source else p.inclusion_radius,
            record_interval=p.record_interval,
            use_external_source=True,
            source_bump_radius=p.inclusion_radius,
            sigma_abs=self.spec.sigma, gamma_abs=self.spec.gamma,
            snap_mmap_path=mmap_path,
            use_contrast_source=use_contrast,
            pinc_fn=lambda t, pts: self._pulse(t, pts, 0),
            pinc_dt_fn=lambda t, pts: self._pulse(t, pts, 1),
            pinc_dtt_fn=lambda t, pts: self._pulse(t, pts, 2),
            inclusion_axes=p.inclusion_axes,
            source_bump_axes=p.inclusion_axes)
        recv = sample_snapshots(snaps, self.elems, self.receiver_map)
        return times, recv, snaps

    def invert(self, recv, times, s_vals, window=2.0, n_img=101):
        mask = None
        if self.spec.name == "MI2":
            mask = self.receiver_points[:, 1] <= 0.0
        elif self.spec.name == "MI3":
            mask = self.receiver_points[:, 2] >= 0.0
        pts = np.asarray(self.receiver_points, dtype=np.float64)
        normals = sphere_normals(pts)
        R_surf = surface_radius(pts)
        dS = 4.0 * np.pi * R_surf ** 2 / len(pts)
        return invert_surface(recv, times, pts, normals, R_surf, s_vals,
                              dS, window=window, n_img=n_img,
                              receiver_mask=mask)


__all__ = ["ProblemSpec", "CommonParameters", "PROBLEMS", "UnifiedExperiment"]
