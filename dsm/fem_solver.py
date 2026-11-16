"""Unified multi-threaded FEM kernels for MS/MI1-MI3.

This module keeps the public run() contract of FEMSolver3D, while replacing
per-step full-mesh work with race-free Numba kernels. Element-to-node
reductions use one private nodal buffer per Numba worker and a deterministic
serial reduction over worker buffers. Floating-point differences from the
old np.add.at path are round-off level.
"""
from __future__ import annotations

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

from .fem_base import FEMSolver3D


@njit(parallel=True, fastmath=False, cache=False)
def material_mass_kernel(nodes, elems, elem_vol, centre, axes,
                         rho_in, rho_out, mu_in, mu_out, n_nodes):
    n_tets = elems.shape[0]
    ax, ay, az = axes[0], axes[1], axes[2]
    centroids = np.empty((n_tets, 3), dtype=np.float64)
    inside = np.empty(n_tets, dtype=np.bool_)
    mu_elem = np.empty(n_tets, dtype=np.float64)
    rho_elem = np.empty(n_tets, dtype=np.float64)

    for q in prange(n_tets):
        i0 = elems[q, 0]
        i1 = elems[q, 1]
        i2 = elems[q, 2]
        i3 = elems[q, 3]
        cx = 0.25 * (nodes[i0, 0] + nodes[i1, 0] + nodes[i2, 0] + nodes[i3, 0])
        cy = 0.25 * (nodes[i0, 1] + nodes[i1, 1] + nodes[i2, 1] + nodes[i3, 1])
        cz = 0.25 * (nodes[i0, 2] + nodes[i1, 2] + nodes[i2, 2] + nodes[i3, 2])
        centroids[q, 0] = cx
        centroids[q, 1] = cy
        centroids[q, 2] = cz
        dx = cx - centre[0]
        dy = cy - centre[1]
        dz = cz - centre[2]
        u = dx / ax
        v = dy / ay
        w = dz / az
        hit = u * u + v * v + w * w <= 1.0
        inside[q] = hit
        if hit:
            mu_elem[q] = mu_in
            rho_elem[q] = rho_in
        else:
            mu_elem[q] = mu_out
            rho_elem[q] = rho_out

    n_threads = get_num_threads()
    local_mass = np.zeros((n_threads, n_nodes), dtype=np.float64)
    for q in prange(n_tets):
        tid = get_thread_id()
        w = elem_vol[q] * 0.25 * rho_elem[q]
        local_mass[tid, elems[q, 0]] += w
        local_mass[tid, elems[q, 1]] += w
        local_mass[tid, elems[q, 2]] += w
        local_mass[tid, elems[q, 3]] += w

    mass = np.zeros(n_nodes, dtype=np.float64)
    for tid in range(n_threads):
        for i in range(n_nodes):
            mass[i] += local_mass[tid, i]
    return centroids, inside, mu_elem, rho_elem, mass


@njit(parallel=True, fastmath=False, cache=False)
def stiffness_kernel(p, mu, S, elems, n_nodes):
    n_threads = get_num_threads()
    local = np.zeros((n_threads, n_nodes), dtype=np.float64)
    for q in prange(elems.shape[0]):
        tid = get_thread_id()
        i0 = elems[q, 0]
        i1 = elems[q, 1]
        i2 = elems[q, 2]
        i3 = elems[q, 3]
        m = mu[q]
        p0 = p[i0]
        p1 = p[i1]
        p2 = p[i2]
        p3 = p[i3]
        local[tid, i0] += m * (S[q, 0] * p0 + S[q, 1] * p1 + S[q, 2] * p2 + S[q, 3] * p3)
        local[tid, i1] += m * (S[q, 1] * p0 + S[q, 4] * p1 + S[q, 5] * p2 + S[q, 6] * p3)
        local[tid, i2] += m * (S[q, 2] * p0 + S[q, 5] * p1 + S[q, 7] * p2 + S[q, 8] * p3)
        local[tid, i3] += m * (S[q, 3] * p0 + S[q, 6] * p1 + S[q, 8] * p2 + S[q, 9] * p3)
    out = np.zeros(n_nodes, dtype=np.float64)
    for tid in range(n_threads):
        for i in range(n_nodes):
            out[i] += local[tid, i]
    return out


@njit(parallel=True, fastmath=False, cache=False)
def element_source_kernel(centroids, elem_vol, elems, centre, axes,
                          source_value, n_nodes):
    n_threads = get_num_threads()
    local = np.zeros((n_threads, n_nodes), dtype=np.float64)
    ax, ay, az = axes[0], axes[1], axes[2]
    for q in prange(elems.shape[0]):
        dx = centroids[q, 0] - centre[0]
        dy = centroids[q, 1] - centre[1]
        dz = centroids[q, 2] - centre[2]
        u = dx / ax
        v = dy / ay
        w = dz / az
        qn = u * u + v * v + w * w
        g = 0.0
        if qn < 1.0:
            tau = qn
            g = np.exp(-1.0 / (1.0 - tau + 1e-30))
        w = g * elem_vol[q] * 0.25 * source_value
        tid = get_thread_id()
        local[tid, elems[q, 0]] += w
        local[tid, elems[q, 1]] += w
        local[tid, elems[q, 2]] += w
        local[tid, elems[q, 3]] += w
    out = np.zeros(n_nodes, dtype=np.float64)
    for tid in range(n_threads):
        for i in range(n_nodes):
            out[i] += local[tid, i]
    return out


@njit(parallel=True, fastmath=False, cache=False)
def subset_scatter_kernel(elems_subset, weights, n_nodes):
    n_threads = get_num_threads()
    local = np.zeros((n_threads, n_nodes), dtype=np.float64)
    for q in prange(elems_subset.shape[0]):
        tid = get_thread_id()
        w = weights[q]
        local[tid, elems_subset[q, 0]] += w
        local[tid, elems_subset[q, 1]] += w
        local[tid, elems_subset[q, 2]] += w
        local[tid, elems_subset[q, 3]] += w
    out = np.zeros(n_nodes, dtype=np.float64)
    for tid in range(n_threads):
        for i in range(n_nodes):
            out[i] += local[tid, i]
    return out


@njit(parallel=True, fastmath=False, cache=False)
def advance_kernel(p_old, p_cur, r, mass, boundary, damping,
                   absorb, dt, sigma, gamma):
    n = p_cur.shape[0]
    out = np.empty_like(p_cur)
    dt2 = dt * dt
    for i in prange(n):
        if boundary[i]:
            alpha = mass[i] / dt2 + damping[i] / (2.0 * dt)
            beta = 2.0 * mass[i] / dt2
            gam = mass[i] / dt2 - damping[i] / (2.0 * dt)
            out[i] = (beta * p_cur[i] - gam * p_old[i] - r[i]) / alpha
        else:
            value = 2.0 * p_cur[i] - p_old[i] - dt2 / mass[i] * r[i]
            if sigma != 0.0 and absorb[i]:
                value -= dt2 * sigma * p_cur[i]
            if gamma != 0.0 and absorb[i]:
                fac = 0.5 * dt * gamma
                value = (value + fac * p_old[i]) / (1.0 + fac)
            out[i] = value
    return out


class UnifiedFEMSolver3D(FEMSolver3D):
    """Common race-free multi-threaded FEM solver for MS/MI1-MI3."""

    def _update_materials(self, centre, incl_radius):
        axes = getattr(self, "incl_axes", None)
        if axes is None:
            axes = (incl_radius, incl_radius, incl_radius)
        axes = np.asarray(axes, dtype=np.float64)
        centroids, inside, mu_elem, rho_elem, mass = material_mass_kernel(
            self.nodes, self.elems, self.elem_vol,
            np.asarray(centre, dtype=np.float64), axes,
            float(self.rho_in), float(self.rho_out),
            float(self.mu_in), float(self.mu_out), len(self.nodes),
        )
        absorb_nodes = np.zeros(len(self.nodes), dtype=bool)
        if inside.any():
            absorb_nodes[np.unique(self.elems[inside])] = True
        self.centroids = centroids
        self.inside_elems = inside
        self.mu_elem = mu_elem
        self.rho_elem = rho_elem
        self.absorb_nodes = absorb_nodes
        self.M_diag = mass
        self._protect_zero_mass()

    def _apply_stiffness_mu(self, p, mu):
        return stiffness_kernel(
            np.asarray(p, dtype=np.float64), np.asarray(mu, dtype=np.float64),
            self.S_e, self.elems, len(self.nodes),
        )

    def _assemble_source(self, s_val, t=None):
        if hasattr(self, "source_centre_fn"):
            centre = self.source_centre_fn(0.0 if t is None else t)
        elif hasattr(self, "source_centre"):
            centre = self.source_centre
        else:
            raise RuntimeError("source centre not set")
        axes = getattr(self, "source_bump_axes", None)
        if axes is None:
            r = self.source_bump_radius
            axes = (r, r, r)
        return element_source_kernel(
            self.centroids, self.elem_vol, self.elems,
            np.asarray(centre, dtype=np.float64),
            np.asarray(axes, dtype=np.float64), float(s_val), len(self.nodes),
        )

    def _assemble_contrast_source(self, t):
        inside = self.inside_elems
        if inside is None or not inside.any():
            return np.zeros(len(self.nodes), dtype=np.float64)
        idx = np.flatnonzero(inside)
        elem_subset = self.elems[idx]
        ce = self.centroids[idx]
        vol4 = self.elem_vol[idx] * 0.25
        weights = np.zeros(len(idx), dtype=np.float64)
        if self.rho_in != self.rho_out:
            dtt = self.pinc_dtt_fn(t, ce)
            weights += -(self.rho_elem[idx] - self.rho_out) * dtt * vol4
        if getattr(self, "gamma_abs", 0.0) != 0.0:
            weights += -self.gamma_abs * self.pinc_dt_fn(t, ce) * vol4
        if getattr(self, "sigma_abs", 0.0) != 0.0:
            weights += -self.sigma_abs * self.pinc_fn(t, ce) * vol4
        out = subset_scatter_kernel(elem_subset, weights, len(self.nodes))
        if self.mu_in != self.mu_out:
            n_ball = np.unique(elem_subset)
            pn = np.zeros(len(self.nodes), dtype=np.float64)
            pn[n_ball] = self.pinc_fn(t, self.nodes[n_ball])
            out -= self._apply_stiffness_mu(pn, self.mu_elem - self.mu_out)
        return out

    def _step(self, dt, source_val, t=None):
        r = self._apply_stiffness(self.p_cur)
        if self._use_external_source:
            if self._use_contrast_source and self.pinc_fn is not None:
                r -= self._assemble_contrast_source(t)
            else:
                r -= self._assemble_source(source_val, t)
        p_new = advance_kernel(
            self.p_old, self.p_cur, r, self.M_diag, self.is_bdy,
            self.B_diag, self.absorb_nodes, float(dt),
            float(getattr(self, "sigma_abs", 0.0)),
            float(getattr(self, "gamma_abs", 0.0)),
        )
        self.p_old, self.p_cur = self.p_cur, p_new


__all__ = ["UnifiedFEMSolver3D", "material_mass_kernel", "stiffness_kernel",
           "element_source_kernel", "subset_scatter_kernel", "advance_kernel",
           "set_num_threads"]
