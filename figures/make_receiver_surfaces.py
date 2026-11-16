"""Final sampling-surface figure v2 (per user feedback 2026-08-25 09:30):

1. MS panel: draw a semi-transparent shaded sphere instead of great circles.
2. Four separate subfigure files, no legends/titles inside; LaTeX organizes
   the layout via subfigure.
3. Axis labels: y1, y2, y3 (not x), rendered LARGE (truetype font).
4. MI2 panel: fix the frame outline placement.

Outputs: fig_ps.png, fig_p0.png, fig_p1.png, fig_p2.png in DATA_DIR.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "figures", "out")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)   # dsm/ lives at the repository root

from dsm.mesh import sample_sphere_fibonacci

AZIM = -60.0
ELEV = 25.0
SCALE = 72.0
PANEL = 560
FONT_SIZE = 34


def _load_first(folder, names):
    """Load the first existing file from a list of candidate names."""
    for nm in names:
        p = os.path.join(folder, nm)
        if os.path.exists(p):
            return np.load(p, allow_pickle=True)
    raise FileNotFoundError("none of %s in %s" % (names, folder))


def load_font(size=FONT_SIZE):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "arial.ttf", "Arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def project(pts3):
    az = math.radians(AZIM)
    el = math.radians(ELEV)
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    x1 = pts3[:, 0] * ca + pts3[:, 2] * sa
    y1 = pts3[:, 1]
    z1 = -pts3[:, 0] * sa + pts3[:, 2] * ca
    y2 = y1 * ce - z1 * se
    return np.stack([x1, -y2], axis=1)


def orbit_pts(n=240):
    t = np.linspace(0.0, 20.0, n)
    return np.stack([0.5 * np.cos(math.pi * t / 10.0),
                     0.5 * math.pi / 10.0 * 0 * t + 0.5 * np.sin(math.pi * t / 10.0),
                     np.zeros_like(t)], axis=1)


def draw_points(draw, pts3, cx, cy, scale, color, r=3):
    s = project(pts3)
    px = cx + scale * s[:, 0]
    py = cy + scale * s[:, 1]
    for x, y in zip(px, py):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_line3(draw, seg3, cx, cy, scale, color, width=2):
    s = project(seg3)
    draw.line([(cx + scale * s[0, 0], cy + scale * s[0, 1]),
               (cx + scale * s[1, 0], cy + scale * s[1, 1])],
              fill=color, width=width)


def draw_orbit(draw, cx, cy, scale):
    o = orbit_pts()
    s = project(o)
    pts = [(cx + scale * x, cy + scale * y) for x, y in s]
    draw.line(pts, fill=(255, 140, 0), width=4, joint="curve")


def new_canvas():
    img = Image.new("RGB", (PANEL, PANEL), "white")
    return img, ImageDraw.Draw(img)


def fill_quad(img, quad3, rgba):
    """Fill a 3D quadrilateral (4x3 array) with a translucent color."""
    s = project(quad3)
    poly = [(PANEL / 2 + SCALE * x, PANEL / 2 + SCALE * y) for x, y in s]
    tmp = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.polygon(poly, fill=rgba)
    out = Image.alpha_composite(img.convert("RGBA"), tmp).convert("RGB")
    img.paste(out)


def panel_MS(font):
    R = 2.0 * math.sqrt(3.0)
    img, draw = new_canvas()
    cx = cy = PANEL // 2
    u = sample_sphere_fibonacci(R, 96)
    # back half points dimmer, front half brighter: split by projected depth
    az = math.radians(AZIM)
    el = math.radians(ELEV)
    depth = (u[:, 0] * math.sin(az) * math.cos(el)
             + u[:, 1] * math.sin(el) + u[:, 2] * (-math.cos(az)) * math.cos(el))
    front = depth > 0
    shade = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    sd.ellipse((cx - R * SCALE, cy - R * SCALE,
                cx + R * SCALE, cy + R * SCALE),
               fill=(205, 220, 250, 110))
    img.paste(Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB"),
              (0, 0))
    draw = ImageDraw.Draw(img)
    # back points
    sb = project(u[~front])
    for x, y in zip(cx + SCALE * sb[:, 0], cy + SCALE * sb[:, 1]):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(150, 170, 210))
    draw_orbit(draw, cx, cy, SCALE)
    sf = project(u[front])
    for x, y in zip(cx + SCALE * sf[:, 0], cy + SCALE * sf[:, 1]):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 80, 220))
    return img


def panel_MI1(font):
    img, draw = new_canvas()
    cx = cy = PANEL // 2
    h = 2.0
    fwd = np.load(os.path.join(DATA_DIR, "forward_MI1.npz"))
    pts = np.asarray(fwd["recv_pts"], dtype=np.float64)
    # wireframe: 12 cube edges
    corners = []
    for sx in (-h, h):
        for sy in (-h, h):
            for sz in (-h, h):
                corners.append([sx, sy, sz])
    edges = []
    for i, c1 in enumerate(corners):
        for c2 in corners[i + 1:]:
            diff = sum(1 for a, b in zip(c1, c2) if abs(a - b) > 1e-9)
            if diff == 1:
                edges.append((c1, c2))
    for p1, p2 in edges:
        draw_line3(draw, np.array([p1, p2], dtype=float), cx, cy, SCALE,
                   (100, 110, 130), width=2)
    draw_points(draw, pts, cx, cy, SCALE, (30, 80, 220), r=3)
    draw_orbit(draw, cx, cy, SCALE)
    return img


def panel_MI2(font):
    """MI2: closed rectangular frame on the four side faces of the cube.

    The 448 receiver points are masked from the stored MI2 cube-face cloud;
    this is y1=+/-2, y2=+/-2, with y3 in [-0.75, 0.75].
    """
    img, draw = new_canvas()
    cx = cy = PANEL // 2
    cloud = np.load(os.path.join(DATA_DIR, "forward_MI2.npz"),
                    allow_pickle=True)
    all_pts = np.asarray(cloud["recv_pts"], dtype=np.float64)
    sel = (((np.abs(np.abs(all_pts[:, 0]) - 2.0) < 1e-9)
            | (np.abs(np.abs(all_pts[:, 1]) - 2.0) < 1e-9))
           & (np.abs(all_pts[:, 2]) <= 0.75 + 1e-9))
    pts = all_pts[sel]
    z0, z1 = -0.75, 0.75
    quads = [
        np.array([[-2,-2,z0],[-2, 2,z0],[-2, 2,z1],[-2,-2,z1]],float),
        np.array([[ 2,-2,z0],[ 2, 2,z0],[ 2, 2,z1],[ 2,-2,z1]],float),
        np.array([[-2,-2,z0],[ 2,-2,z0],[ 2,-2,z1],[-2,-2,z1]],float),
        np.array([[-2, 2,z0],[ 2, 2,z0],[ 2, 2,z1],[-2, 2,z1]],float),
    ]
    for q in quads:
        fill_quad(img, q, (120, 170, 235, 65))
    draw = ImageDraw.Draw(img)
    for q in quads:
        for k in range(4):
            draw_line3(draw, np.array([q[k], q[(k+1)%4]]), cx, cy, SCALE,
                       (100, 110, 130), width=2)
    draw_points(draw, pts, cx, cy, SCALE, (30, 80, 220), r=3)
    draw_orbit(draw, cx, cy, SCALE)
    return img


def panel_MI3(font):
    img, draw = new_canvas()
    cx = cy = PANEL // 2
    fwd = _load_first(DATA_DIR, ["forward_MI3.npz"])
    pts = np.asarray(fwd["recv_pts"], dtype=np.float64)
    # MI3 aperture: 9x9 design square (edge 2) masked from the 17x17
    # superset; on the legacy 81-pt file this mask is a no-op.
    sel = ((np.abs(pts[:, 0]) <= 1.0 + 1e-9)
           & (np.abs(pts[:, 1]) <= 1.0 + 1e-9))
    pts = pts[sel]
    quad = np.array([
        [-1, -1, 2.5], [1, -1, 2.5],
        [1, 1, 2.5], [-1, 1, 2.5],
    ])
    fill_quad(img, quad, (120, 170, 235, 70))
    draw = ImageDraw.Draw(img)
    rect = [
        ([-1, -1, 2.5], [1, -1, 2.5]),
        ([1, -1, 2.5], [1, 1, 2.5]),
        ([1, 1, 2.5], [-1, 1, 2.5]),
        ([-1, 1, 2.5], [-1, -1, 2.5]),
    ]
    for p1, p2 in rect:
        draw_line3(draw, np.array([p1, p2]), cx, cy, SCALE,
                   (100, 110, 130), width=2)
    draw_points(draw, pts, cx, cy, SCALE, (30, 80, 220), r=3)
    draw_orbit(draw, cx, cy, SCALE)
    return img


def main(argv=None):
    global DATA_DIR, OUT_DIR
    import argparse
    ap = argparse.ArgumentParser(description="Acquisition-geometry figure (4 panels).")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="folder holding forward_*.npz (receiver coordinates only)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)
    DATA_DIR, OUT_DIR = args.data_dir, args.out_dir
    font = load_font(FONT_SIZE)
    out = {
        "fig_ms.png": panel_MS(font),
        "fig_mi1.png": panel_MI1(font),
        "fig_mi2.png": panel_MI2(font),
        "fig_mi3.png": panel_MI3(font),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, img in out.items():
        path = os.path.join(OUT_DIR, name)
        img.save(path)
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
