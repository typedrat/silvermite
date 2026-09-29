#!/usr/bin/env python3
"""Interleaved A/B timing of the Rust solvers against LEMON.

Runs both binaries alternately (ABAB..., reversing the order every round) so
that slow periods on the machine hit both sides equally, and reports the
median, minimum, and maximum solve time of each.

Usage: tools/abtest.py [--rounds N] [--algo ns|cs] FILE.min...
Set RUST_BIN to time a different build of solve_dimacs.
"""

import argparse
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEMON = os.path.join(ROOT, "tools", "lemon-ref", "mcf_ref")
RUST = os.environ.get("RUST_BIN") or os.path.join(ROOT, "target", "release", "examples", "solve_dimacs")


def solve_ms(exe, algo, path):
    out = subprocess.run([exe, algo, path], capture_output=True, text=True, check=True).stdout.split()
    if out[0] != "OPTIMAL":
        sys.exit(f"{exe} on {path}: {out[0]}")
    return float(out[2])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--algo", default="ns")
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    print(f"{'instance':<28} {'LEMON med':>10} {'min..max':>17} {'Rust med':>10} {'min..max':>17} {'Rust/LEMON':>10}")
    for path in args.files:
        times = {"lemon": [], "rust": []}
        for r in range(args.rounds):
            order = [("lemon", LEMON), ("rust", RUST)]
            if r % 2:
                order.reverse()
            for name, exe in order:
                times[name].append(solve_ms(exe, args.algo, path))
        l, rs = times["lemon"], times["rust"]
        lm, rm = statistics.median(l), statistics.median(rs)
        name = os.path.basename(path)
        print(f"{name:<28} {lm:10.1f} {min(l):8.1f}..{max(l):<8.1f} {rm:10.1f} {min(rs):8.1f}..{max(rs):<8.1f} {rm / lm:10.2f}",
              flush=True)


if __name__ == "__main__":
    main()
