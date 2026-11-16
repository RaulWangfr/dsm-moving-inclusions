#!/usr/bin/env python
"""Run one numerical example (MS, MI1, MI2 or MI3).

Examples
--------
    # everything: mesh, forward solve, index function
    python run_case.py --case MS

    # forward solve only (writes data/forward_PS.npz)
    python run_case.py --case MI1 --stage forward

    # index function only, reusing stored forward data
    python run_case.py --case MI1 --stage invert --noise 0.10

    # smaller mesh for a quick smoke test (not the paper mesh)
    python run_case.py --case MS --lc 0.3 --threads 4
"""
from __future__ import annotations

import argparse
import os
import sys

import config as cfgmod
from dsm.driver import run_case


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=sorted(cfgmod.CASES),
                   help="MS (moving source), MI1/MI2/MI3 (moving inclusions)")
    p.add_argument("--stage", default="all",
                   choices=("all", "forward", "invert"),
                   help="which stage to run (default: all)")
    p.add_argument("--noise", type=float, default=None,
                   help="relative noise level; default is the value of the paper "
                        "(0.30 for MS, 0.10 for MI1-MI3)")
    p.add_argument("--data-dir", default="data",
                   help="where forward data and index files are written/read")
    p.add_argument("--mesh-dir", default="mesh",
                   help="mesh cache directory (created on first run)")
    p.add_argument("--out-dir", default="outputs",
                   help="where the cached index files are written")
    p.add_argument("--threads", type=int, default=6,
                   help="numba threads (default 6)")
    p.add_argument("--lc", type=float, default=None,
                   help="override the mesh size (paper value 0.1)")
    p.add_argument("--t-final", type=float, default=None,
                   help="override the recording window (paper value 26)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = cfgmod.CONFIG
    if args.lc is not None or args.t_final is not None:
        cfg = cfgmod.Config(
            **{**cfg.__dict__,
               **({"lc": args.lc} if args.lc is not None else {}),
               **({"t_final": args.t_final} if args.t_final is not None else {})})
    os.makedirs(args.data_dir, exist_ok=True)
    path = run_case(args.case, stage=args.stage, data_dir=args.data_dir,
                    mesh_dir=args.mesh_dir, out_dir=args.out_dir,
                    noise=args.noise, cfg=cfg, threads=args.threads)
    if path:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())