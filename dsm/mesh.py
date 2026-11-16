"""
gmsh_gen_3d.py — Generate a 3D ball mesh (tetrahedra) with GMSH for the
3D FEM forward solver.  Supports caching to a .npz file so the mesh is
generated only once.

Geometry: a ball of radius R.
  - Outer surface: first-order absorbing BC (triangular faces).
  - Measurement surface: an inner sphere at radius r_inner < R (interpolated).

Returns:
    nodes     (N, 3)  float  — nodal (x,y,z)
    elems     (M, 4)  int    — tetra connectivity (0-based)
    bdy_faces (F, 3)  int    — boundary face node triples (0-based)
    bdy_areas (F,)    float  — boundary face areas
"""
import numpy as np
import gmsh
import os


def generate_mesh(radius=1.0, lc=0.1, n_outer=64, open_gui=False):
    """Create a ball mesh (tetrahedra) via GMSH.

    Parameters
    ----------
    radius  : float — ball radius
    lc      : float — characteristic mesh size
    n_outer : int   — number of seam points on a great circle (ico-ish)
    open_gui: bool

    Returns
    -------
    nodes     : (N, 3) float
    elems     : (M, 4) int   — tet connectivity (0-based)
    bdy_faces : (F, 3) int   — boundary face node triples (0-based)
    bdy_areas : (F,)  float
    """
    gmsh.initialize()
    if open_gui:
        gmsh.fltk.initialize()

    model = gmsh.model
    model.add("ball")

    # Build a ball from a sphere surface (openCASCADE kernel)
    # Volume domain with center at origin, radius R.
    factory = model.occ
    sphere = factory.addSphere(0, 0, 0, radius)
    factory.synchronize()

    # set mesh size on all points (characteristic length)
    tags = model.getEntities(0)
    for t in tags:
        model.mesh.setSize([t], lc)

    # physical volume
    vols = model.getEntities(3)
    model.addPhysicalGroup(3, [v[1] for v in vols], name="Domain")
    # physical surface (outer boundary)
    surfs = model.getEntities(2)
    model.addPhysicalGroup(2, [s[1] for s in surfs], name="Boundary")
    model.geo.synchronize()

    model.mesh.generate(3)

    # ---- extract nodes ----
    node_tags, node_coords, _ = model.mesh.getNodes()
    node_tags = np.array(node_tags, dtype=int)
    order = np.argsort(node_tags)
    nodes = np.column_stack([
        np.array(node_coords[0::3], dtype=float)[order],
        np.array(node_coords[1::3], dtype=float)[order],
        np.array(node_coords[2::3], dtype=float)[order],
    ])

    # ---- extract tets ----
    elem_types, _, elem_conn_lists = model.mesh.getElements(dim=3)
    elems = None
    for et, conn in zip(elem_types, elem_conn_lists):
        name, dim, _, n_nodes, _, _ = model.mesh.getElementProperties(et)
        if "Tetrahedron" in name and n_nodes == 4:
            elems = np.array(conn, dtype=int).reshape(-1, n_nodes) - 1
            break
    if elems is None:
        raise RuntimeError("No 4-node tetrahedra found in mesh.")

    # ---- extract boundary faces (dim=2, triangles on outer surface) ----
    elem_types_2d, _, elem_conn_2d = model.mesh.getElements(dim=2)
    faces = []
    for et, conn in zip(elem_types_2d, elem_conn_2d):
        name, _, _, n_f, _, _ = model.mesh.getElementProperties(et)
        if "Triangle" in name:
            tri = np.array(conn, dtype=int).reshape(-1, n_f) - 1
            # keep only faces whose centroid is near the outer radius
            c = nodes[tri].mean(axis=1)
            r = np.linalg.norm(c, axis=1)
            mask = r >= 0.95 * radius
            faces.append(tri[mask])
    if faces:
        bdy_faces = np.vstack(faces)
    else:
        bdy_faces = np.empty((0, 3), dtype=int)

    # face areas
    if len(bdy_faces) > 0:
        a = nodes[bdy_faces[:, 0]]
        b = nodes[bdy_faces[:, 1]]
        c = nodes[bdy_faces[:, 2]]
        cross = np.cross(b - a, c - a)
        bdy_areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    else:
        bdy_areas = np.array([], dtype=float)

    if not open_gui:
        gmsh.finalize()

    return nodes, elems, bdy_faces, bdy_areas


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def estimate_h_min(nodes, elems):
    """Minimum tetrahedron altitude — for CFL condition."""
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]
    v3 = nodes[elems[:, 3]]
    # volume
    vol = np.abs(np.einsum('ij,ij->i',
                           v1 - v0,
                           np.cross(v2 - v0, v3 - v0))) / 6.0
    # largest face area
    def fa(p, q, r):
        return 0.5 * np.sqrt(np.sum(np.cross(q - p, r - p) ** 2, axis=1))
    A = np.max([fa(v0, v1, v2), fa(v0, v1, v3),
                fa(v0, v2, v3), fa(v1, v2, v3)], axis=0)
    alt = 3.0 * vol / (A + 1e-30)
    return alt.min()


def compute_cfl_dt(h_min, c_max, cfl=0.5):
    return cfl * h_min / c_max


def sample_sphere_fibonacci(r, n_pts):
    """Nearly-uniform points on a sphere of radius r (Fibonacci lattice)."""
    i = np.arange(n_pts)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - 2.0 * (i + 0.5) / n_pts   # [-1, 1)
    rad = np.sqrt(1.0 - y * y)
    th = phi * i
    pts = np.column_stack([rad * np.cos(th), y, rad * np.sin(th)]) * r
    return pts


def mesh_cache_path(code_dir, radius, lc, n_outer):
    return os.path.join(
        code_dir,
        f"mesh_3d_lc{lc}_R{radius}_n{n_outer}.npz")


def get_or_generate_mesh(code_dir, radius, lc, n_outer=64, force=False):
    """Return (nodes, elems, bdy_faces, bdy_areas); cache to npz."""
    path = mesh_cache_path(code_dir, radius, lc, n_outer)
    if os.path.exists(path) and not force:
        d = np.load(path)
        print(f"[mesh] loaded cache {os.path.basename(path)}", flush=True)
        return d["nodes"], d["elems"], d["bdy_faces"], d["bdy_areas"]
    print(f"[mesh] generating 3D mesh lc={lc} R={radius} ...", flush=True)
    nodes, elems, bdy_faces, bdy_areas = generate_mesh(
        radius=radius, lc=lc, n_outer=n_outer)
    np.savez(path,
             nodes=nodes, elems=elems,
             bdy_faces=bdy_faces, bdy_areas=bdy_areas)
    print(f"[mesh] saved cache {os.path.basename(path)}: "
          f"{len(nodes)} nodes, {len(elems)} tets", flush=True)
    return nodes, elems, bdy_faces, bdy_areas


# ------------------------------------------------------------------
#  Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    R = 5.0
    for lc, n_outer in [(0.15, 64), (0.12, 64)]:
        nodes, elems, bf, ba = get_or_generate_mesh(
            code_dir, R, lc, n_outer)
        h_min = estimate_h_min(nodes, elems)
        print(f"\n=== lc={lc} ===  nodes={len(nodes)}  tets={len(elems)}  "
              f"bdy_faces={len(bf)}")
        print(f"  h_min={h_min:.5f}  dt(CFL=.5)={compute_cfl_dt(h_min,1.0):.6f}")
        print(f"  mesh file ~ {os.path.getsize(mesh_cache_path(code_dir,R,lc,n_outer))/1e6:.1f} MB")
