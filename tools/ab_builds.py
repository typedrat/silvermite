#!/usr/bin/env python3
"""Interleaved A/B timing of two builds of examples/solve_dimacs.

Runs both binaries alternately (ABAB..., reversing the order every round) so
that slow periods on the machine hit both sides equally, checks that they
agree on every optimal cost, and reports each side's median solve time.

Wall-clock medians still drift by a few percent between runs on a busy
machine. With --instructions, each binary instead runs once under
`perf stat` and the user-space instruction counts are compared; those are
deterministic, so they expose small code-generation regressions that timing
cannot separate from noise. Parsing is included in the count, so compare
instances rather than reading the ratio as a pure solver figure.

Usage: tools/ab_builds.py [--rounds N] [--instructions] A B ALGO:FILE.min...
  ALGO is ns, cs, cs-push, or cs-augment, e.g. ns:bench-data/x.min
"""

import argparse
import os
import statistics
import subprocess
import sys


def solve(exe, algo, path):
    out = subprocess.run([exe, algo, path], capture_output=True, text=True, check=True).stdout.split()
    if out[0] != "OPTIMAL":
        return out[0], None
    return out[1], float(out[2])


def instructions(exe, algo, path):
    cmd = ["perf", "stat", "-x,", "-e", "instructions:u", exe, algo, path]
    err = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    for line in err.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[2].startswith("instructions"):
            return int(fields[0])
    sys.exit(f"no instruction count from perf for {exe}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--instructions", action="store_true")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("instances", nargs="+", metavar="ALGO:FILE")
    args = p.parse_args()

    unit = "M instr" if args.instructions else "ms"
    print(f"{'instance':<40} {'algo':<10} {'A ' + unit:>12} {'B ' + unit:>12} {'B/A':>7}")
    ratios = []
    for spec in args.instances:
        algo, path = spec.split(":", 1)
        if args.instructions:
            a = instructions(args.a, algo, path) / 1e6
            b = instructions(args.b, algo, path) / 1e6
        else:
            times = {args.a: [], args.b: []}
            for r in range(args.rounds):
                order = [args.a, args.b] if r % 2 == 0 else [args.b, args.a]
                outcomes = set()
                for exe in order:
                    outcome, ms = solve(exe, algo, path)
                    outcomes.add(outcome)
                    if ms is not None:
                        times[exe].append(ms)
                if len(outcomes) != 1:
                    sys.exit(f"{spec}: builds disagree: {outcomes}")
            if not times[args.a]:
                print(f"{os.path.basename(path):<40} {algo:<10} {'not optimal':>12}")
                continue
            a, b = statistics.median(times[args.a]), statistics.median(times[args.b])
        ratios.append(b / a)
        print(f"{os.path.basename(path):<40} {algo:<10} {a:12.1f} {b:12.1f} {b / a:7.3f}", flush=True)
    if ratios:
        print(f"geometric mean B/A: {statistics.geometric_mean(ratios):.3f}")


if __name__ == "__main__":
    main()
