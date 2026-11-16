"""Common receiver sampling, surface normals, and parallel DSM.

This module implements the DSM index exactly as described in the
manuscript ("A direct sampling approach for moving objects"):

    K(s, z; tau, y) = delta( tau - s - (|y_R - z| - R + |y|)/c0 )
                      * (y . n) / (|y_R - z| |y|^2)

    I(s, z) = int_Gamma K(s, z; tau, y) p^s(tau, y) dS(y)

with

    R    = max_{Gamma} |y|        (radial projection radius)
    y_R  = R y / |y|              (radial projection of y onto the
                                   sphere of radius R)
    n    = outward unit normal of the measurement surface Gamma
    c0   = background wave speed (default 1)

The time integral is eliminated by the Dirac distribution: each
measurement point contributes the recorded field at the retarded time

    tau = s + (|y_R - z| - R + |y|) / c0.

No assumption is made that Gamma is spherical; the formula holds for a
general (possibly non-spherical) measurement surface.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from .interp import build_tet_interp_node_safe


@njit(parallel=True, fastmath=False, cache=False)
def sample_snapshots_kernel(snaps, elems, tet_idx, bary):
    n_times = snaps.shape[0]
    n_recv = tet_idx.shape[0]
    out = np.empty((n_recv, n_times), dtype=np.float64)
    for flat in prange(n_recv * n_times):
        j = flat // n_times
        k = flat - j * n_times
        e = tet_idx[j]
        out[j, k] = (bary[j, 0] * snaps[k, elems[e, 0]]
                     + bary[j, 1] * snaps[k, elems[e, 1]]
                     + bary[j, 2] * snaps[k, elems[e, 2]]
                     + bary[j, 3] * snaps[k, elems[e, 3]])
    return out


@njit(parallel=True, fastmath=False, cache=False)
def dsm_kernel(times, recv, recv_pts, normals, R_surface, image_pts,
               s_vals, dS, receiver_mask, c0, use_weight=True):
    """Evaluate the manuscript indicator
        I(s,z) = (1/4pi) int_Gamma (yhat.n) p^s(s + |y-z|/c0, y) / |y-z| dS(y)
    with the exact distance |y-z| and exact retarded time s + |y-z|/c0.
    Receivers within 0.05 of a sampling point are skipped (singularity cut).
    R_surface is accepted only for call compatibility and is unused.

    Parameters
    ----------
    times        : (n_times,)  recorded snapshot times
    recv         : (n_recv, n_times)  recorded scattered field
    recv_pts     : (n_recv, 3)  measurement points y on Gamma
    normals      : (n_recv, 3)  outward unit normals n of Gamma
    R_surface    : unused (kept for signature compatibility)
    image_pts    : (n_img, 3)  sampling points z
    s_vals       : (n_s,)  sampling times s
    dS           : scalar, quadrature weight for each receiver point
    receiver_mask: (n_recv,) bool, which receivers are used
    c0           : scalar, background wave speed

    Returns
    -------
    out : (n_s, n_img) float64
    """
    n_s = s_vals.shape[0]
    n_img = image_pts.shape[0]
    n_recv = recv.shape[0]
    n_times = times.shape[0]
    dt_rec = times[1] - times[0]
    out = np.zeros((n_s, n_img), dtype=np.float64)
    for flat in prange(n_s * n_img):
        si = flat // n_img
        zi = flat - si * n_img
        dx0 = image_pts[zi, 0]
        dy0 = image_pts[zi, 1]
        dz0 = image_pts[zi, 2]
        acc = 0.0
        for j in range(n_recv):
            if not receiver_mask[j]:
                continue
            yx = recv_pts[j, 0]
            yy = recv_pts[j, 1]
            yz = recv_pts[j, 2]
            dx = yx - dx0
            dy = yy - dy0
            dz = yz - dz0
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 0.05:          # singularity cut, one grid step
                continue
            query = s_vals[si] + dist / c0
            value = 0.0
            if query >= times[0] and query < times[-1]:
                pos = (query - times[0]) / dt_rec
                k = int(pos)
                if k >= n_times - 1:
                    k = n_times - 2
                frac = pos - k
                value = recv[j, k] * (1.0 - frac) + recv[j, k + 1] * frac
            ry = np.sqrt(yx * yx + yy * yy + yz * yz)
            ydotn = (yx * normals[j, 0] + yy * normals[j, 1]
                     + yz * normals[j, 2]) / max(ry, 1e-30)   # yhat . n
            if use_weight:
                acc += dS * ydotn * value / (4.0 * np.pi * dist)
            else:
                acc += dS * value / (4.0 * np.pi * dist)
        out[si, zi] = acc
    return out


def receiver_map(nodes, elems, points):
    tet_idx, bary = build_tet_interp_node_safe(nodes, elems, points)
    return {"points": np.asarray(points), "tet_idx": tet_idx, "bary": bary}


def sample_snapshots(snaps, elems, mapping):
    return sample_snapshots_kernel(np.asarray(snaps), elems,
                                   mapping["tet_idx"], mapping["bary"])


def subset_receiver_data(recv, points, n_points):
    """Select a subset of existing samples; no forward solve is needed."""
    if n_points > len(points):
        raise ValueError("n_points cannot exceed available receivers")
    if n_points == len(points):
        idx = np.arange(len(points), dtype=np.int64)
    else:
        idx = np.rint(np.linspace(0, len(points) - 1, n_points)).astype(np.int64)
        idx = np.unique(idx)
        if len(idx) != n_points:
            idx = np.linspace(0, len(points) - 1, n_points).astype(np.int64)
    return np.asarray(recv)[idx], np.asarray(points)[idx], idx


def image_grid(window=2.0, n_img=101):
    x = np.linspace(-window, window, n_img)
    gx, gy = np.meshgrid(x, x)
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    return gx, gy, pts


def sphere_normals(pts):
    """Outward unit normals for points on a sphere centred at the origin."""
    pts = np.asarray(pts, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(r, 1.0e-30)


def cube_normals(pts):
    """Outward unit normals for points on an axis-aligned cube.

    The normal is the coordinate axis of the face the point lies on.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ax = np.argmax(np.abs(pts), axis=1)
    n = np.zeros_like(pts)
    rows = np.arange(pts.shape[0])
    n[rows, ax] = np.sign(pts[rows, ax])
    return n


def surface_radius(pts):
    """R = max_Gamma |y|: the radial projection radius used by the kernel."""
    pts = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(pts, axis=1).max())


def invert_surface(recv, times, recv_pts, normals, R_surface, s_vals, dS,
                   window=2.0, n_img=101, receiver_mask=None, c0=1.0,
                   use_weight=True):
    """DSM inversion on a general measurement surface Gamma.

    Implements the manuscript indicator
        I(s,z) = (1/4pi) int_Gamma (yhat.n) p^s(s + |y-z|/c0, y) / |y-z| dS(y)
    with exact distances and exact retarded times.

    Parameters
    ----------
    recv          : (n_recv, n_times) recorded scattered field
    times         : (n_times,) snapshot times
    recv_pts      : (n_recv, 3) measurement points y on Gamma
    normals       : (n_recv, 3) outward unit normals of Gamma
    R_surface     : unused (kept for signature compatibility)
    s_vals        : (n_s,) sampling times
    dS            : scalar quadrature weight (true surface area element)
    window, n_img : imaging window half-width and grid size
    receiver_mask : optional bool mask selecting receivers
    c0            : background wave speed
    """
    recv = np.asarray(recv, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    recv_pts = np.asarray(recv_pts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    s_vals = np.asarray(s_vals, dtype=np.float64)
    if normals.shape != recv_pts.shape:
        raise ValueError("normals must have one vector per receiver point")
    if receiver_mask is None:
        receiver_mask = np.ones(recv.shape[0], dtype=np.bool_)
    else:
        receiver_mask = np.asarray(receiver_mask, dtype=np.bool_)
        if receiver_mask.shape != (recv.shape[0],):
            raise ValueError("receiver_mask must have one entry per receiver")
    gx, gy, img = image_grid(window, n_img)
    return dsm_kernel(times, recv, recv_pts, normals, float(R_surface),
                      img, s_vals, float(dS), receiver_mask, float(c0),
                      bool(use_weight)), \
        gx, gy


def invert_sphere(recv, times, recv_pts, radius, s_vals,
                  window=2.0, n_img=101, receiver_mask=None, c0=1.0):
    """DSM inversion when Gamma is the sphere of radius `radius`.

    Backwards-compatible wrapper: the normals are y/|y| and R = radius.
    The quadrature weight is the full-sphere weight 4 pi R^2 / n_recv;
    a mask only removes terms (as in the MI2 frame and MI3
    conical configurations).
    """
    recv = np.asarray(recv, dtype=np.float64)
    recv_pts = np.asarray(recv_pts, dtype=np.float64)
    if receiver_mask is None:
        receiver_mask = np.ones(recv.shape[0], dtype=np.bool_)
    else:
        receiver_mask = np.asarray(receiver_mask, dtype=np.bool_)
    dS = 4.0 * np.pi * float(radius) ** 2 / recv.shape[0]
    return invert_surface(recv, times, recv_pts,
                          sphere_normals(recv_pts), float(radius), s_vals,
                          dS, window=window, n_img=n_img,
                          receiver_mask=receiver_mask, c0=c0)


__all__ = ["receiver_map", "sample_snapshots", "subset_receiver_data",
           "image_grid", "invert_surface", "invert_sphere",
           "sphere_normals", "cube_normals", "surface_radius"]
