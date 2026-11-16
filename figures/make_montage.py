#!/usr/bin/env python
"""Draw the 3 x 3 montage of the index magnitude |I(s,z)| for one case.

Everything is inside the PNG: panel frames, the $s$ labels above each column,
the tick labels, the axis names, the swept-support envelope, the inclusion
(source) cross-section, its centre, and the current-time maximum.  The LaTeX
side therefore only includes the image at \\linewidth and no longer places any
label, so panel frames cannot be cropped away.

Figure rules
  * binary map: green where |I| > THRESH * max|I| of the same panel, white else;
  * orange outline: the boundary of the support swept from s = 0 to the panel
    time (annulus band r = R_orbit +/- r_support, closed by the two caps);
  * black circle: cross-section of the inclusion (in MS: support of the source)
    at the panel time; black cross: its centre;
  * red cross: location of the maximum of |I| at the panel time.

Usage:
    python figures/make_montage.py ind_MS_n30.npz Irad_MS.png [--thresh 0.75]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ORBIT_R = 0.5
ORBIT_W = math.pi / 10.0
R_SUPPORT = 0.3           # source support radius = inclusion radius

PANEL = 431               # px of each image panel
GAP = 16                  # px between panels
M_LEFT, M_BOT, M_TOP, M_RIGHT = 40, 40, 8, 8

WHITE = (255, 255, 255)
RED = (0, 140, 40)          # the map is green
ORANGE = (245, 140, 0)
DEEPBLUE = (215, 25, 25)    # the cross is red
BLACK = (0, 0, 0)
FRAME = (120, 120, 120)
TEXT = (20, 20, 20)

FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/System/Library/Fonts/Supplemental/Arial.ttf",
                   r"C:\Windows\Fonts\arial.ttf",
                   r"C:\Windows\Fonts\segoeui.ttf")


def load_font(size):
    for cand in FONT_CANDIDATES:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    return ImageFont.load_default()


def centre(theta):
    return ORBIT_R * math.cos(theta), ORBIT_R * math.sin(theta)


def envelope(theta):
    """Boundary of the support swept over [0, theta]: two arcs and two caps."""
    n = 240
    tw = np.linspace(0.0, theta, n)                                   # outer arc
    ox, oy = (ORBIT_R + R_SUPPORT) * np.cos(tw), (ORBIT_R + R_SUPPORT) * np.sin(tw)
    ph = np.linspace(theta, theta + math.pi, n)                        # leading cap
    cx, cy = centre(theta)
    lx, ly = cx + R_SUPPORT * np.cos(ph), cy + R_SUPPORT * np.sin(ph)
    ti = np.linspace(theta, 0.0, n)                                    # inner arc
    ix, iy = (ORBIT_R - R_SUPPORT) * np.cos(ti), (ORBIT_R - R_SUPPORT) * np.sin(ti)
    p0 = centre(0.0)
    ps = np.linspace(math.pi, 2 * math.pi, n)                          # trailing cap
    sx, sy = p0[0] + R_SUPPORT * np.cos(ps), p0[1] + R_SUPPORT * np.sin(ps)
    return np.concatenate([np.stack([ox, oy], 1), np.stack([lx, ly], 1),
                           np.stack([ix, iy], 1), np.stack([sx, sy], 1)], axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="indicator npz written by figures/make_indicator.py")
    ap.add_argument("out", help="output PNG")
    ap.add_argument("--thresh", type=float, default=0.75,
                    help="red where |I| > thresh * max|I| of the panel (default 0.75)")
    args = ap.parse_args(argv)

    d = np.load(args.src, allow_pickle=True)
    A = np.asarray(d["A"], np.float64)
    S_VALS = np.asarray(d["s_vals"], np.float64)
    WIN = float(d["win"])
    N_IMG = int(d["n_img"])
    PROBLEM = str(d["problem"])

    S0 = int(S_VALS[0])
    panel_s = [s for s in S_VALS if int(round(s)) % 2 == 0]
    grid = np.linspace(-WIN, WIN, N_IMG)
    peaks = []
    for frame in A:
        iy, ix = np.unravel_index(int(np.argmax(frame)), frame.shape)
        peaks.append((float(grid[ix]), float(grid[iy])))

    W = M_LEFT + 3 * PANEL + 2 * GAP + M_RIGHT
    H = M_TOP + 3 * PANEL + 2 * GAP + M_BOT
    img = Image.new("RGB", (W, H), WHITE)

    f_tick = load_font(26)
    f_s = load_font(28)

    for k, s_now in enumerate(panel_s):
        row, col = divmod(k, 3)
        x0 = M_LEFT + col * (PANEL + GAP)
        y0 = M_TOP + row * (PANEL + GAP)

        frame = A[int(round(s_now)) - S0]
        vmax = float(frame.max()) if frame.max() > 0 else 1.0
        mask = frame > args.thresh * vmax
        rgb = np.where(mask[..., None], np.array(RED, np.uint8),
                       np.array(WHITE, np.uint8)).astype(np.uint8)
        rgb = rgb[::-1]          # data row 0 is y = -WIN, PIL row 0 is the top
        img.paste(Image.fromarray(rgb, "RGB").resize((PANEL, PANEL), Image.NEAREST),
                  (x0, y0))

        layer = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        def px(x, y, _x0=0.0, _y0=0.0):
            """data coordinates -> panel pixels (y up -> y down)"""
            return ((x + WIN) / (2 * WIN) * PANEL,
                    (WIN - y) / (2 * WIN) * PANEL)

        theta = ORBIT_W * s_now
        poly = [px(a, b) for a, b in envelope(theta)]
        poly.append(poly[0])                     # closed annulus band
        draw.line(poly, fill=ORANGE, width=3, joint="curve")

        ccx, ccy = px(*centre(theta))
        rr = R_SUPPORT / (2 * WIN) * PANEL
        draw.ellipse([ccx - rr, ccy - rr, ccx + rr, ccy + rr],
                     outline=BLACK, width=3)

        cx_, cy_ = ccx, ccy
        a = 13.0
        draw.line([cx_ - a, cy_ - a, cx_ + a, cy_ + a], fill=WHITE, width=7)
        draw.line([cx_ - a, cy_ + a, cx_ + a, cy_ - a], fill=WHITE, width=7)
        draw.line([cx_ - a, cy_ - a, cx_ + a, cy_ + a], fill=BLACK, width=4)
        draw.line([cx_ - a, cy_ + a, cx_ + a, cy_ - a], fill=BLACK, width=4)

        gx, gy = px(*peaks[int(round(s_now)) - S0])
        a2 = 18.0
        draw.line([gx - a2, gy, gx + a2, gy], fill=DEEPBLUE, width=5)
        draw.line([gx, gy - a2, gx, gy + a2], fill=DEEPBLUE, width=5)

        img.paste(layer, (x0, y0), layer)
        ImageDraw.Draw(img).rectangle([x0, y0, x0 + PANEL - 1, y0 + PANEL - 1],
                                      outline=FRAME, width=1)


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print("[montage %s] thresh=%.2f peaks=%s -> %s"
          % (PROBLEM, args.thresh, [(round(x, 2), round(y, 2)) for x, y in peaks],
             args.out), flush=True)


if __name__ == "__main__":
    main()