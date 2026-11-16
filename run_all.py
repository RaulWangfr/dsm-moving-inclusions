#!/usr/bin/env python
"""Run all four examples (MS, MI1, MI2, MI3) sequentially.

Equivalent to four calls of ``run_case.py``.  The forward solves are the
expensive part; the mesh is generated once and cached in ``mesh/``.

    python run_all.py --threads 6
"""
from __future__ import annotations

import argparse
import sys

import config as cfgmod
from dsm.driver import run_case


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=("all", "forward", "invert"))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mesh-dir", default="mesh")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--cases", default="MS,MI1,MI2,MI3",
                   help="comma separated subset, e.g. --cases MS,MI3")
    args = p.parse_args(argv)

    for case in [c.strip().upper() for c in args.cases.split(",") if c.strip()]:
        print("\n=== %s ===" % case, flush=True)
        run_case(case, stage=args.stage, data_dir=args.data_dir,
                 mesh_dir=args.mesh_dir, out_dir=args.out_dir,
                 cfg=cfgmod.CONFIG, threads=args.threads)
    print("\n[all done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())