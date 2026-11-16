"""
fem_solver_3d.py — Explicit FEM solver for the 3D wave equation with a moving
inclusion/source (ball mesh, tetrahedra, first-order absorbing BC).

PDE:  rho d_tt p  -  div(mu grad p) = f(x,t)
  - absorbing BC: d_n p + (1/c) d_t p = 0
  - external source: f(x,t) = g(x - xc(t)) * s(t), moving C^infty bump

The solver is a direct 3D port of the 2D fem_solver.py:
  - lumped mass matrix (volume/4 to each node of a tet)
  - explicit central difference
  - boundary damping lumped to nodes from face areas
  - element-loop stiffness matvec (vectorized over tets)

Usage:
    solver = FEMSolver3D(nodes, elems, bdy_faces, bdy_areas, ...)
    times, snaps = solver.run(T_final, dt, source_fn, centre_fn, ...)
"""
import numpy as np


class FEMSolver3D:
    """3D wave-equation FEM solver on a tetrahedral ball mesh.

    Parameters
    ----------
    nodes     : (N, 3) float — nodal coordinates
    elems     : (M, 4) int   — tet connectivity (0-based)
    bdy_faces : (F, 3) int   — boundary face node triples (0-based)
    bdy_areas : (F,)  float  — boundary face areas
    rho_out, mu_out : background density / modulus
    rho_in,  mu_in  : inclusion density / modulus
    """

    def __init__(self, nodes, elems, bdy_faces, bdy_areas,
                 rho_out=1.0, mu_out=1.0, rho_in=4.0, mu_in=4.0):
        self.nodes = nodes
        self.elems = elems
        self.bdy_faces = bdy_faces
        self.bdy_areas = bdy_areas

        self.rho_out = rho_out
        self.mu_out = mu_out
        self.rho_in = rho_in
        self.mu_in = mu_in
        self.c_out = np.sqrt(mu_out / rho_out)

        self._precompute_element_data()
        self._precompute_boundary_damping()

        bdy_nodes = np.unique(bdy_faces) if len(bdy_faces) else np.array([], dtype=int)
        self.is_bdy = np.zeros(len(nodes), dtype=bool)
        self.is_bdy[bdy_nodes] = True

        self.p_old = None
        self.p_cur = None

        # contrast-source (scattered-field) mode
        self._use_contrast_source = False
        self.pinc_fn = None      # p_inc(x, t) -> (K,)
        self.pinc_dt_fn = None   # d_t p_inc(x, t) -> (K,)
        self.pinc_dtt_fn = None  # d_tt p_inc(x, t) -> (K,)
        self.centroids = None
        self.inside_elems = None

    # ------------------------------------------------------------------
    def _precompute_element_data(self):
        v0 = self.nodes[self.elems[:, 0]]
        v1 = self.nodes[self.elems[:, 1]]
        v2 = self.nodes[self.elems[:, 2]]
        v3 = self.nodes[self.elems[:, 3]]

        # Jacobian columns (J = [v1-v0, v2-v0, v3-v0]); det = J11(J22J33-J23J32) - ...
        J = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)  # (M,3,3)
        det = (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
               - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
               + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))
        vol = np.abs(det) / 6.0
        self.elem_vol = vol

        # Inverse J^T applied sides: gradient of shape functions.
        # grad phi_i = (J^{-T}) * side_i  where side_0 = -1-1-1 etc.
        # Build inv(det) * adj(J)^T * side vectors.
        a11 =  J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]
        a21 = -J[:, 1, 0] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 0]
        a31 =  J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]
        a12 = -J[:, 0, 1] * J[:, 2, 2] + J[:, 0, 2] * J[:, 2, 1]
        a22 =  J[:, 0, 0] * J[:, 2, 2] - J[:, 0, 2] * J[:, 2, 0]
        a32 = -J[:, 0, 0] * J[:, 2, 1] + J[:, 0, 1] * J[:, 2, 0]
        a13 =  J[:, 0, 1] * J[:, 1, 2] - J[:, 0, 2] * J[:, 1, 1]
        a23 = -J[:, 0, 0] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 0]
        a33 =  J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

        s = np.array([-1.0, -1.0, -1.0])  # side 0
        g1 = np.column_stack([a11, a12, a13])
        g2 = np.column_stack([a21, a22, a23])
        g3 = np.column_stack([a31, a32, a33])
        g0 = -(g1 + g2 + g3)  # phi0 = 1 - phi1 - phi2 - phi3

        invdet = 1.0 / (det + 1e-30)
        g0 *= invdet[:, None]
        g1 *= invdet[:, None]
        g2 *= invdet[:, None]
        g3 *= invdet[:, None]
        self.grad = (g0, g1, g2, g3)

        # stiffness (4x4 symmetric) upper triangle (10 entries) * vol
        V = vol
        G = [g0, g1, g2, g3]
        S = np.empty((len(self.elems), 10))
        k = 0
        for i in range(4):
            for j in range(i, 4):
                S[:, k] = V * np.sum(G[i] * G[j], axis=1)
                k += 1
        self.S_e = S

        # area-weighted lumped mass (volume/4 to each node)
        # masses recomputed each material update; keep raw vol here.

    def _precompute_boundary_damping(self):
        B = np.zeros(len(self.nodes))
        coeff = self.mu_out / self.c_out
        if len(self.bdy_faces):
            A = self.bdy_areas
            f = self.bdy_faces
            for k in range(3):
                np.add.at(B, f[:, k], coeff * A / 3.0)
        self.B_diag = B

    # ------------------------------------------------------------------
    def _update_materials(self, centre, incl_radius):
        centroids = (self.nodes[self.elems[:, 0]]
                     + self.nodes[self.elems[:, 1]]
                     + self.nodes[self.elems[:, 2]]
                     + self.nodes[self.elems[:, 3]]) / 4.0
        d = centroids - centre[None, :]
        dist2 = np.sum(d * d, axis=1)
        inside = dist2 <= incl_radius ** 2
        self.mu_elem = np.where(inside, self.mu_in, self.mu_out)
        self.rho_elem = np.where(inside, self.rho_in, self.rho_out)

        # absorption (zero-order sigma) marks nodes of inclusion elements
        self.absorb_nodes = np.zeros(len(self.nodes), dtype=bool)
        if inside.any():
            self.absorb_nodes[np.unique(self.elems[inside])] = True

        M = np.zeros(len(self.nodes))
        vol4 = self.elem_vol / 4.0
        for k in range(4):
            np.add.at(M, self.elems[:, k], self.rho_elem * vol4)
        self.M_diag = M
        self.centroids = centroids
        self.inside_elems = inside
        self._protect_zero_mass()

    def _apply_stiffness(self, p):
        return self._apply_stiffness_mu(p, self.mu_elem)

    def _apply_stiffness_mu(self, p, mu):
        """Stiffness matvec with an explicit per-element mu array."""
        r = np.zeros(len(self.nodes))
        S = self.S_e
        e = self.elems
        pe = p[e]  # (M,4)
        # S upper-triangular indices
        # 00 01 02 03 11 12 13 22 23 33
        r0 = mu * (S[:, 0] * pe[:, 0] + S[:, 1] * pe[:, 1] + S[:, 2] * pe[:, 2] + S[:, 3] * pe[:, 3])
        r1 = mu * (S[:, 1] * pe[:, 0] + S[:, 4] * pe[:, 1] + S[:, 5] * pe[:, 2] + S[:, 6] * pe[:, 3])
        r2 = mu * (S[:, 2] * pe[:, 0] + S[:, 5] * pe[:, 1] + S[:, 7] * pe[:, 2] + S[:, 8] * pe[:, 3])
        r3 = mu * (S[:, 3] * pe[:, 0] + S[:, 6] * pe[:, 1] + S[:, 8] * pe[:, 2] + S[:, 9] * pe[:, 3])
        np.add.at(r, e[:, 0], r0)
        np.add.at(r, e[:, 1], r1)
        np.add.at(r, e[:, 2], r2)
        np.add.at(r, e[:, 3], r3)
        return r

    def _assemble_contrast_source(self, t):
        """Contrast-source RHS for the scattered-field equation
           rho d_tt p^s - div(mu grad p^s) + gamma d_t p^s + sigma p^s
               = div((mu-mu_o) grad p_inc) - [(rho)-(rho_o)] d_tt p_inc
                 - gamma d_t p_inc - sigma p_inc
           Only nonzero inside the inclusion ball (background operator on
           p_inc vanishes outside).  LHS is the same operator the solver
           already time-steps (M_diag uses rho, stiffness uses mu).
        """
        F = np.zeros(len(self.nodes))
        e = self.elems
        inside = self.inside_elems
        if inside is None or not inside.any():
            return F
        idx = np.flatnonzero(inside)
        ce = self.centroids[idx]
        vol4 = self.elem_vol[idx] / 4.0

        # rho contrast: -[rho - rho_o] d_tt p_inc
        if self.rho_in != self.rho_out:
            dtt = self.pinc_dtt_fn(t, ce)
            w = -(self.rho_elem[idx] - self.rho_out) * dtt * vol4
            for k in range(4):
                np.add.at(F, e[idx, k], w)

        # gamma term: -gamma d_t p_inc  (gamma only inside the ball)
        if getattr(self, 'gamma_abs', 0.0) != 0.0:
            dti = self.pinc_dt_fn(t, ce)
            w = -self.gamma_abs * dti * vol4
            for k in range(4):
                np.add.at(F, e[idx, k], w)

        # sigma term: -sigma p_inc  (sigma only inside the ball)
        if getattr(self, 'sigma_abs', 0.0) != 0.0:
            pv = self.pinc_fn(t, ce)
            w = -self.sigma_abs * pv * vol4
            for k in range(4):
                np.add.at(F, e[idx, k], w)

        # mu contrast: +div((mu-mu_o) grad p_inc)
        #   weak form: -K_contrast * p_inc  (only ball elements contribute)
        if self.mu_in != self.mu_out:
            n_ball = np.unique(e[idx])
            pn = np.zeros(len(self.nodes))
            pn[n_ball] = self.pinc_fn(t, self.nodes[n_ball])
            mu_c = self.mu_elem - self.mu_out
            F -= self._apply_stiffness_mu(pn, mu_c)
        return F

    def _assemble_source(self, s_val, t=None):
        centroids = (self.nodes[self.elems[:, 0]]
                     + self.nodes[self.elems[:, 1]]
                     + self.nodes[self.elems[:, 2]]
                     + self.nodes[self.elems[:, 3]]) / 4.0
        if hasattr(self, 'source_centre_fn'):
            centre = self.source_centre_fn(0.0 if t is None else t)
        elif hasattr(self, 'source_centre'):
            centre = self.source_centre
        else:
            raise RuntimeError('source centre not set: pass source_centre_fn to run() or set solver.source_centre_fn')
            centre = np.zeros(3)
        d = centroids - centre[None, :]
        r2 = np.sum(d * d, axis=1)
        R_src = self.source_bump_radius
        g_elem = np.zeros(len(self.elems))
        inside = r2 < R_src ** 2
        tt = r2[inside] / R_src ** 2
        g_elem[inside] = np.exp(-1.0 / (1.0 - tt + 1e-30))
        F = np.zeros(len(self.nodes))
        ge = g_elem * (self.elem_vol / 4.0)
        for k in range(4):
            np.add.at(F, self.elems[:, k], ge * s_val)
        return F

    # ------------------------------------------------------------------
    def _protect_zero_mass(self):
        zero = self.M_diag == 0
        if zero.any():
            self.M_diag[zero] = 1e-15
            self.is_bdy[zero] = True
            self.B_diag[zero] = np.maximum(self.B_diag[zero], 1e-15)

    def _step(self, dt, source_val, t=None):
        r = self._apply_stiffness(self.p_cur)
        if self._use_external_source:
            if self._use_contrast_source and self.pinc_fn is not None:
                F = self._assemble_contrast_source(t)
            else:
                F = self._assemble_source(source_val, t)
            r -= F
        p_new = np.empty_like(self.p_cur)
        M = self.M_diag
        B = self.B_diag
        dt2 = dt * dt
        interior = ~self.is_bdy
        p_new[interior] = (2.0 * self.p_cur[interior]
                           - self.p_old[interior]
                           - dt2 / M[interior] * r[interior])
        if getattr(self, 'sigma_abs', 0.0) != 0.0:
            ab = self.absorb_nodes & interior
            if ab.any():
                p_new[ab] -= dt2 * self.sigma_abs * self.p_cur[ab]
        # first-order time damping: (d2p/dt2) + gamma*(dp/dt) -> semi-implicit
        # p_{n+1} = [2p_n - p_{n-1} - dt^2/M r + (gamma*dt/2) p_{n-1}] / (1 + gamma*dt/2)
        if getattr(self, 'gamma_abs', 0.0) != 0.0:
            ab = self.absorb_nodes & interior
            if ab.any():
                gfac = 0.5 * dt * self.gamma_abs
                p_new[ab] = (p_new[ab] + gfac * self.p_old[ab]) / (1.0 + gfac)
        bdy = self.is_bdy
        alpha = M[bdy] / dt2 + B[bdy] / (2.0 * dt)
        beta = 2.0 * M[bdy] / dt2
        gamma = M[bdy] / dt2 - B[bdy] / (2.0 * dt)
        p_new[bdy] = (beta * self.p_cur[bdy]
                      - gamma * self.p_old[bdy]
                      - r[bdy]) / alpha
        self.p_old, self.p_cur = self.p_cur, p_new

    # ------------------------------------------------------------------
    def run(self, T_final, dt, source_fn, inclusion_centre_fn,
            inclusion_radius=0.15, record_interval=1,
            use_external_source=True, source_bump_radius=0.15,
            sigma_abs=0.0, gamma_abs=0.0, snap_mmap_path=None,
            use_contrast_source=False, pinc_fn=None, pinc_dt_fn=None,
            pinc_dtt_fn=None, inclusion_axes=None, source_bump_axes=None):
        self._use_external_source = use_external_source
        self._use_contrast_source = use_contrast_source
        self.pinc_fn = pinc_fn
        self.pinc_dt_fn = pinc_dt_fn
        self.pinc_dtt_fn = pinc_dtt_fn
        self.source_bump_radius = source_bump_radius
        self.incl_axes = (tuple(float(v) for v in inclusion_axes)
                          if inclusion_axes is not None else None)
        self.source_bump_axes = (tuple(float(v) for v in source_bump_axes)
                                 if source_bump_axes is not None else None)
        self.sigma_abs = sigma_abs
        self.gamma_abs = gamma_abs

        N_steps = int(np.ceil(T_final / dt))
        n_record = N_steps // record_interval + 1

        self.p_old = np.zeros(len(self.nodes))
        self.p_cur = np.zeros(len(self.nodes))

        self._update_materials(inclusion_centre_fn(0.0), inclusion_radius)

        times = np.empty(n_record)
        if snap_mmap_path is None:
            snaps = np.empty((n_record, len(self.nodes)), dtype=np.float64)
        else:
            snaps = np.lib.format.open_memmap(snap_mmap_path, mode='w+',
                                              dtype=np.float64,
                                              shape=(n_record, len(self.nodes)))
        rec_idx = 0
        times[rec_idx] = 0.0
        snaps[rec_idx] = self.p_cur.copy()
        rec_idx += 1

        import time as _time
        _t0 = _time.time()
        for n in range(1, N_steps + 1):
            t = n * dt
            centre = inclusion_centre_fn(t)
            self._update_materials(centre, inclusion_radius)
            source_val = source_fn(t)
            self._step(dt, source_val, t)
            if n % record_interval == 0:
                times[rec_idx] = t
                snaps[rec_idx] = self.p_cur.copy()
                rec_idx += 1
                if snap_mmap_path is not None:
                    snaps.flush()
            if n % 100 == 0 or n == N_steps:
                el = _time.time() - _t0
                rate = n / max(el, 1e-9)
                print(f"    step {n}/{N_steps}  t={t:.3f}  "
                      f"elapsed={el:.1f}s  rate={rate:.0f} step/s  "
                      f"max|p|={np.abs(self.p_cur).max():.3e}", flush=True)

        self.recorder = (times[:rec_idx], snaps[:rec_idx])
        return self.recorder

