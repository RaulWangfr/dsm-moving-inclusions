"""Forward solve and inversion for the four examples MS, MI1, MI2, MI3.

The driver is a thin layer: it builds the receiver geometry of a case, calls
the FEM solver (``dsm.fem_solver``), stores the recorded scattered field, and
then inverts it with the direct sampling index of the manuscript
(``dsm.inversion.invert_surface``).

Only the configuration reported in the paper is implemented; the many
experimental switches of the original development scripts (superset
apertures, ellipsoidal inclusions, alternative plane heights) are not.
"""
from __future__ import annotations

import gc
import math
import os
import time

import numpy as np

from config import CASES, CONFIG, CaseSpec, Config
from .experiment import CommonParameters, ProblemSpec, UnifiedExperiment
from .inversion import (cube_normals, invert_surface, receiver_map,
                        sphere_normals, surface_radius)
from .mesh import sample_sphere_fibonacci

__all__ = ["make_params", "receiver_geometry", "forward", "indicator",
           "run_case", "load_forward"]


# --------------------------------------------------------------------- setup
def make_params(spec: CaseSpec, cfg: Config = CONFIG, threads: int = 6):
    """CommonParameters of the FEM solver for one case."""
    return CommonParameters(
        radius=cfg.radius,
        lc=cfg.lc,
        n_outer=cfg.n_outer,
        receiver_radius=cfg.sphere_radius,
        n_recv=spec.n_recv,
        t_final=cfg.t_final,
        dt_cfl=cfg.dt_cfl,
        record_interval=cfg.record_interval,
        inclusion_radius=cfg.inclusion_radius,
        orbit_radius=cfg.orbit_radius,
        orbit_omega=cfg.orbit_omega,
        source_frequency=cfg.source_frequency,
        pulse_frequency=cfg.pulse_frequency,
        pulse_shift=cfg.pulse_shift,
        pulse_spacing=cfg.pulse_spacing,
        pulse_count=cfg.pulse_count,
        n_threads=threads,
    )


def problem_spec(spec: CaseSpec) -> ProblemSpec:
    return ProblemSpec(spec.name,
                       "moving_source" if spec.kind == "source" else "fixed_source",
                       rho_in=spec.rho_in, mu_in=spec.mu_in,
                       gamma=spec.gamma_in, sigma=spec.sigma_in)


# ---------------------------------------------------------------- geometries
def cube_geometry(n_recv: int, edge: float) -> np.ndarray:
    """`n_recv` points on the surface of the cube [-edge/2, edge/2]^3.

    Directions come from a Fibonacci sphere; each ray is scaled so that its
    largest coordinate hits the cube face, which makes the points equidistributed
    in surface area.
    """
    unit = sample_sphere_fibonacci(1.0, n_recv)
    r_cube = (0.5 * edge) / np.max(np.abs(unit), axis=1)
    return np.ascontiguousarray(unit * r_cube[:, None])


def cloud_geometry(edge: float, spacing: float) -> np.ndarray:
    """Candidate cloud: the six faces of the cube on a regular grid.

    The closed frame aperture of case MI2 is a sub-region of two of these faces,
    so the full field is stored once and masked at inversion (no second solve).
    """
    g = np.arange(-0.5 * edge, 0.5 * edge + 1e-9, spacing)
    pts = []
    for a in (g[0], g[-1]):
        for u in g:
            for v in g:
                pts.append((a, u, v))
                pts.append((u, a, v))
                pts.append((u, v, a))
    return np.ascontiguousarray(np.unique(np.asarray(pts, np.float64), axis=0))


def square_geometry(center_z: float, half: float, n: int) -> np.ndarray:
    """n x n grid on the plane y3 = center_z, outward normal (0, 0, 1)."""
    x1 = np.linspace(-half, half, n)
    x2 = np.linspace(-half, half, n)
    g1, g2 = np.meshgrid(x1, x2)
    pts = np.stack([g1.ravel(), g2.ravel(), np.full(g1.size, center_z)], axis=1)
    return np.ascontiguousarray(pts)


def receiver_geometry(spec: CaseSpec, cfg: Config = CONFIG):
    """Receiver points, outward normals and surface element of one case."""
    if spec.receivers == "sphere":
        pts = sample_sphere_fibonacci(cfg.sphere_radius, spec.n_recv)
        normals = sphere_normals(pts)
        dS = 4.0 * math.pi * cfg.sphere_radius ** 2 / len(pts)
    elif spec.receivers == "cube":
        pts = cube_geometry(spec.n_recv, cfg.cube_edge)
        normals = cube_normals(pts)
        dS = 6.0 * cfg.cube_edge ** 2 / len(pts)
    elif spec.receivers == "cloud":
        pts = cloud_geometry(cfg.cloud_edge, cfg.cloud_spacing)
        # The whole cloud is stored once; the inversion keeps only the closed
        # frame (see frame_aperture), and dS below is the area per frame point.
        normals = cube_normals(pts)
        dS = cfg.cloud_spacing ** 2
    elif spec.receivers == "square":
        pts = square_geometry(cfg.plane_z, cfg.plane_half, cfg.plane_grid)
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        dS = cfg.cloud_spacing ** 2          # grid spacing 0.25 on the square
    else:
        raise ValueError("unknown receiver geometry: %r" % spec.receivers)
    return np.ascontiguousarray(pts), np.ascontiguousarray(normals), dS


def frame_aperture(pts: np.ndarray, cfg: Config = CONFIG):
    """Closed rectangle frame on the cube faces: |y1| = |y2| = 2, |y3| <= 0.75.

    Returns the mask and the outward normals of the selected points.
    """
    half = 0.5 * cfg.cloud_edge
    on_face = ((np.abs(np.abs(pts[:, 0]) - half) < 1e-9)
               | (np.abs(np.abs(pts[:, 1]) - half) < 1e-9))
    mask = on_face & (np.abs(pts[:, 2]) <= cfg.frame_half_height + 1e-9)
    normals = np.zeros_like(pts)
    m1 = np.abs(np.abs(pts[:, 0]) - half) < 1e-9
    normals[m1, 0] = np.sign(pts[m1, 0])
    m2 = ~m1 & (np.abs(np.abs(pts[:, 1]) - half) < 1e-9)
    normals[m2, 1] = np.sign(pts[m2, 1])
    return mask, normals


# ---------------------------------------------------------------- file names
def forward_path(case: str, data_dir: str) -> str:
    return os.path.join(data_dir, "forward_%s.npz" % case)


def indicator_path(case: str, noise: float, out_dir: str) -> str:
    return os.path.join(out_dir, "ind_%s_n%02d.npz" % (case, int(round(noise * 100))))


def load_forward(case: str, data_dir: str) -> dict:
    path = forward_path(case, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found; run the forward stage first:\n"
            "    python run_case.py --case %s --stage forward" % (path, case))
    with np.load(path, allow_pickle=True) as fwd:
        return {k: np.asarray(fwd[k]) if k != "sampling" else str(fwd[k])
                for k in fwd.files}


# ------------------------------------------------------------------- forward
def forward(case: str, data_dir: str, mesh_dir: str, cfg: Config = CONFIG,
            threads: int = 6) -> str:
    """Solve the forward problem of one case and store the recorded field."""
    spec = CASES[case]
    params = make_params(spec, cfg, threads)
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp = UnifiedExperiment(code_dir, problem=problem_spec(spec),
                            params=params, mesh_dir=mesh_dir)

    pts, normals, dS = receiver_geometry(spec, cfg)
    exp.receiver_points = pts
    exp.receiver_map = receiver_map(exp.nodes, exp.elems, pts)
    print("[%s] dt=%.9f  t_final=%g  receivers=%d  dS=%.6g"
          % (case, exp.dt, cfg.t_final, len(pts), dS), flush=True)

    mmap_path = os.path.join(data_dir, "_%s_snapshots.npy" % case.lower())
    cpu0, wall0 = time.process_time(), time.perf_counter()
    times, recv, snaps = exp.run_forward(mmap_path=mmap_path)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    try:
        if hasattr(snaps, "_mmap"):
            snaps._mmap.close()
        elif hasattr(snaps, "base") and hasattr(snaps.base, "close"):
            snaps.base.close()
    except Exception:
        pass
    del snaps
    gc.collect()
    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    os.makedirs(data_dir, exist_ok=True)
    path = forward_path(case, data_dir)
    np.savez(path,
             times=times, recv_s=recv, recv_pts=pts, normals=normals, dS=dS,
             dt=exp.dt, T_final=cfg.t_final, record_interval=cfg.record_interval,
             problem=case, n_recv=len(pts), R_domain=cfg.radius,
             R_meas=cfg.sphere_radius,
             inclusion_radius=cfg.inclusion_radius,
             rho_in=spec.rho_in, mu_in=spec.mu_in,
             gamma_in=spec.gamma_in, sigma_in=spec.sigma_in,
             pulse_fp=cfg.pulse_frequency, pulse_spacing=cfg.pulse_spacing,
             pulse_count=cfg.pulse_count,
             time_decimation=spec.time_decimation,
             s_vals=np.asarray(cfg.s_vals))
    print("[%s] forward done: wall=%.1fs cpu=%.1fs -> %s"
          % (case, wall, cpu, path), flush=True)
    return path


# ----------------------------------------------------------------- inversion
def indicator(case: str, data_dir: str, out_dir: str, cfg: Config = CONFIG,
              noise: float | None = None, threads: int = 6):
    """Evaluate the index function on the image grid and cache |I| to npz."""
    from numba import set_num_threads
    set_num_threads(threads)

    spec = CASES[case]
    if noise is None:
        noise = spec.noise
    fwd = load_forward(case, data_dir)
    times = np.asarray(fwd["times"], np.float64)
    recv = np.asarray(fwd["recv_s"], np.float64)
    pts = np.asarray(fwd["recv_pts"], np.float64)

    if spec.aperture == "frame":
        mask, all_normals = frame_aperture(pts, cfg)
        idx = np.flatnonzero(mask)
        pts = pts[idx]
        recv = recv[idx]
        normals = all_normals[idx]
        dS = cfg.cloud_spacing ** 2
    else:
        _, normals, dS = receiver_geometry(spec, cfg)
        mask = np.ones(len(pts), dtype=bool)

    inv_t, inv_d = times, recv
    if spec.time_decimation > 1:
        inv_t = times[::spec.time_decimation]
        inv_d = recv[:, ::spec.time_decimation]

    if noise > 0.0:
        rng = np.random.default_rng(cfg.noise_seed)
        eps = rng.uniform(-1.0, 1.0, size=inv_d.shape)
        inv_d = inv_d + noise * eps * float(np.abs(inv_d).max())

    R_surf = surface_radius(pts)
    s_vals = np.asarray(cfg.s_vals, np.float64)
    I, _, _ = invert_surface(inv_d, inv_t, pts, normals, R_surf, s_vals, dS,
                             window=cfg.img_window, n_img=cfg.n_img,
                             receiver_mask=np.ones(len(pts), dtype=bool))

    A = np.abs(np.asarray(I, np.float64)).reshape(len(s_vals), cfg.n_img, cfg.n_img)
    os.makedirs(out_dir, exist_ok=True)
    path = indicator_path(case, noise, out_dir)
    np.savez(path, A=A, s_vals=s_vals, win=cfg.img_window, n_img=cfg.n_img,
             problem=case, noise=noise, n_recv=len(pts), dS=dS)
    print("[%s] index: receivers=%d  n_times=%d  noise=%.2f  dS=%.6g -> %s"
          % (case, len(pts), inv_t.size, noise, dS, path), flush=True)
    return path


# ------------------------------------------------------------------- driver
def run_case(case: str, stage: str = "all", data_dir: str = "data",
             mesh_dir: str = "mesh", out_dir: str = "outputs",
             noise: float | None = None, cfg: Config = CONFIG,
             threads: int = 6):
    case = case.upper()
    if case not in CASES:
        raise SystemExit("unknown case %r (choose from %s)"
                         % (case, ", ".join(sorted(CASES))))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    if stage in ("forward", "all"):
        forward(case, data_dir, mesh_dir, cfg=cfg, threads=threads)
    if stage in ("invert", "all"):
        return indicator(case, data_dir, out_dir, cfg=cfg,
                         noise=noise, threads=threads)
    return None