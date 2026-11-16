"""Stable tetrahedral interpolation via node KD-tree and vectorized CSR."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_node_tet_adjacency(elems, n_nodes):
    """Return offsets, incident tets using vectorized sorting."""
    n_tets = len(elems)
    flat_nodes = np.asarray(elems, dtype=np.int32).reshape(-1)
    tet_ids = np.repeat(np.arange(n_tets, dtype=np.int32), 4)
    order = np.argsort(flat_nodes, kind="stable")
    sorted_nodes = flat_nodes[order]
    incident = tet_ids[order]
    counts = np.bincount(sorted_nodes, minlength=n_nodes)
    offsets = np.empty(n_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, incident


def _barycentric(pt, verts):
    """Barycentric coordinates using explicit Cramer determinants.

    This avoids NumPy/LAPACK solve calls, which are unstable in the current
    Python 3.14 environment for some nearly-degenerate candidate tets.
    """
    a = verts[0]
    u = verts[1] - a
    v = verts[2] - a
    w = verts[3] - a
    q = pt - a

    # det([u v w]) and the three Cramer determinants.
    det = (u[0] * (v[1] * w[2] - v[2] * w[1])
           - v[0] * (u[1] * w[2] - u[2] * w[1])
           + w[0] * (u[1] * v[2] - u[2] * v[1]))
    if abs(det) < 1e-24:
        return None
    d1 = (q[0] * (v[1] * w[2] - v[2] * w[1])
          - v[0] * (q[1] * w[2] - q[2] * w[1])
          + w[0] * (q[1] * v[2] - q[2] * v[1]))
    d2 = (u[0] * (q[1] * w[2] - q[2] * w[1])
          - q[0] * (u[1] * w[2] - u[2] * w[1])
          + w[0] * (u[1] * q[2] - u[2] * q[1]))
    d3 = (u[0] * (v[1] * q[2] - v[2] * q[1])
          - v[0] * (u[1] * q[2] - u[2] * q[1])
          + q[0] * (u[1] * v[2] - u[2] * v[1]))
    l1 = d1 / det
    l2 = d2 / det
    l3 = d3 / det
    return np.array([1.0 - l1 - l2 - l3, l1, l2, l3], dtype=np.float64)


def build_tet_interp_node_safe(nodes, elems, pts, k_nodes=16, tol=2e-7):
    tree = cKDTree(nodes)
    offsets, incident = build_node_tet_adjacency(elems, len(nodes))
    tet_idx = np.empty(len(pts), dtype=np.int32)
    bary = np.empty((len(pts), 4), dtype=np.float64)
    node_dist = np.empty(len(pts), dtype=np.float64)
    unresolved = []
    for i, pt in enumerate(np.asarray(pts, dtype=np.float64)):
        distances, near_nodes = tree.query(pt, k=min(k_nodes, len(nodes)))
        candidates = []
        for node in np.atleast_1d(near_nodes):
            lo, hi = offsets[int(node)], offsets[int(node) + 1]
            candidates.extend(incident[lo:hi].tolist())
        if candidates:
            for e in np.unique(np.asarray(candidates, dtype=np.int32)):
                weights = _barycentric(pt, nodes[elems[int(e)]])
                if weights is not None and np.all(weights >= -tol) and np.all(weights <= 1.0 + tol):
                    tet_idx[i] = int(e)
                    bary[i] = weights / np.sum(weights)
                    node_dist[i] = float(np.atleast_1d(distances)[0])
                    break
            else:
                unresolved.append(i)
        else:
            unresolved.append(i)
    if unresolved:
        raise RuntimeError(f"unresolved receiver points: {unresolved[:20]}")
    print(f"[tet interp] points={len(pts)}, node-k={k_nodes}, "
          f"nearest-node distance=[{node_dist.min():.6e}, {node_dist.max():.6e}], "
          f"bary sum error={np.max(np.abs(np.sum(bary, axis=1)-1.0)):.3e}", flush=True)
    return tet_idx, bary
