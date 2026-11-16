#!/usr/bin/env python
"""Stage 1 of the figures: cache |I| on the image grid for one case.

Pure numpy/numba: matplotlib is deliberately not imported, because on some
systems it cannot coexist with the numba parallel backend in one process.

    python figures/make_indicator.py --case MS

writes outputs/ind_PS_n30.npz, which figures/make_montage.py turns into a PNG.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfgmod                      # noqa: E402
from dsm.driver import indicator             # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=sorted(cfgmod.CASES))
    p.add_argument("--noise", type=float, default=None,
                   help="default: paper value (0.30 for MS, 0.10 for MI1-MI3)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args(argv)
    path = indicator(args.case, args.data_dir, args.out_dir,
                     cfg=cfgmod.CONFIG, noise=args.noise, threads=args.threads)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())