#!/usr/bin/env python3
"""Differential test and benchmark against LEMON.

Generates instances with gen_instances.py, solves each with LEMON's
NetworkSimplex and CostScaling (tools/lemon-ref/mcf_ref) and with this
crate's (examples/solve_dimacs), and checks that every solver reports the
same outcome and optimal cost as LEMON's NetworkSimplex.

Build both sides first:
  make -C tools/lemon-ref
  cargo build --release --example solve_dimacs

Usage:
  tools/compare.py correctness [--count N]   many small and medium instances
  tools/compare.py bench [--repeat N]        timing table on larger instances
  tools/compare.py suite DIR [--csv FILE]    every *.min under DIR, e.g. the
                                             benchmark suite that
                                             tools/benchmarks/fetch.sh builds
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "tools", "gen_instances.py")
LEMON = os.path.join(ROOT, "tools", "lemon-ref", "mcf_ref")
RUST = os.path.join(ROOT, "target", "release", "examples", "solve_dimacs")


def run(cmd, timeout=3600):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", None, None)
    if out.returncode != 0:
        return ("CRASH", None, None)
    parts = out.stdout.split()
    if not parts:
        return ("CRASH", None, None)
    if parts[0] == "OPTIMAL":
        return ("OPTIMAL", int(parts[1]), float(parts[2]))
    return (parts[0], None, None)


def generate(path, family, **params):
    cmd = [sys.executable, GEN, family] + [f"--{k}={v}" for k, v in params.items()]
    with open(path, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)


def correctness(args):
    families = [
        ("netgen", dict(n=200, deg=6)),
        ("netgen", dict(n=500, deg=10, inf=0.2)),
        ("netgen", dict(n=300, deg=8, lower=0.2, maxcap=50)),
        ("netgen", dict(n=300, deg=8, neg=0.2)),
        ("netgen", dict(n=2000, deg=4, sources=50, maxcost=1000000)),
        ("grid", dict(n=400, sources=5)),
        ("grid", dict(n=900, sources=20, inf=0.3)),
        ("transport", dict(n=120, sources=40, supply=5000)),
    ]
    failures = 0
    runs = 0
    statuses = {}
    with tempfile.TemporaryDirectory() as tmp:
        for family, params in families:
            for seed in range(1, args.count + 1):
                path = os.path.join(tmp, "inst.min")
                generate(path, family, seed=seed, **params)
                desc = f"{family} {params} seed={seed}"
                variants = [(), ("--leq",)] if params.get("lower", 0) == 0 else [()]
                for extra in variants:
                    ref = run([LEMON, "ns", *extra, path])
                    statuses[ref[0]] = statuses.get(ref[0], 0) + 1
                    rust = {
                        algo: run([RUST, algo, *extra, path])
                        for algo in ("ns", "cs", "cs-push", "cs-augment")
                    }
                    for algo, res in rust.items():
                        runs += 1
                        if res[:2] != ref[:2]:
                            failures += 1
                            print(f"MISMATCH {algo} {' '.join(extra)} {desc}: rust {res[:2]} lemon-ns {ref[:2]}")
                    if not extra:
                        lemon_cs = run([LEMON, "cs", path])
                        if lemon_cs[:2] != ref[:2]:
                            print(f"note: LEMON's own CostScaling disagrees on {desc}: {lemon_cs[:2]} vs {ref[:2]}")
    print(f"{runs} Rust solves compared against LEMON NetworkSimplex, {failures} mismatches")
    print("reference outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    return 1 if failures else 0


def bench(args):
    suites = [
        ("netgen", dict(n=10000, deg=4, sources=100, supply=100000)),
        ("netgen", dict(n=10000, deg=16, sources=100, supply=100000)),
        ("netgen", dict(n=10000, deg=64, sources=100, supply=100000)),
        ("netgen", dict(n=100000, deg=4, sources=300, supply=1000000)),
        ("netgen", dict(n=100000, deg=16, sources=300, supply=1000000)),
        ("netgen", dict(n=300000, deg=8, sources=1000, supply=3000000)),
        ("grid", dict(n=40000, sources=100, supply=100000)),
        ("grid", dict(n=250000, sources=300, supply=1000000)),
        ("transport", dict(n=600, sources=200, supply=100000)),
        ("transport", dict(n=2000, sources=500, supply=1000000)),
    ]
    header = f"{'instance':<44} {'n':>7} {'m':>9}  {'LEMON NS':>9} {'Rust NS':>9} {'LEMON CS':>9} {'Rust CS':>9}  ok"
    print(header)
    print("-" * len(header))
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for family, params in suites:
            path = os.path.join(tmp, "inst.min")
            generate(path, family, seed=1, **params)
            with open(path) as f:
                for line in f:
                    if line.startswith("p "):
                        _, _, n, m = line.split()
                        break
            rep = ["--repeat", str(args.repeat)]
            results = [
                run([LEMON, "ns", *rep, path]),
                run([RUST, "ns", *rep, path]),
                run([LEMON, "cs", *rep, path]),
                run([RUST, "cs", *rep, path]),
            ]
            ok = all(r[:2] == results[0][:2] for r in results)
            bad += not ok
            name = f"{family} " + " ".join(f"{k}={v}" for k, v in params.items() if k in ("n", "deg", "sources"))
            times = " ".join(f"{r[2]:9.1f}" if r[2] is not None else f"{r[0]:>9}" for r in results)
            print(f"{name:<44} {n:>7} {m:>9}  {times}  {'yes' if ok else 'NO'}")
    print("times in ms (best of %d); ok = all four agree on the optimal cost" % args.repeat)
    return 1 if bad else 0


def suite(args):
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.min"), recursive=True))
    if not paths:
        sys.exit(f"no .min files under {args.dir}")
    solvers = [("lemon_ns", LEMON, "ns"), ("rust_ns", RUST, "ns"), ("lemon_cs", LEMON, "cs"), ("rust_cs", RUST, "cs")]
    header = f"{'instance':<34} {'n':>8} {'m':>9}  {'LEMON NS':>9} {'Rust NS':>9} {'LEMON CS':>9} {'Rust CS':>9}  ok"
    print(header)
    print("-" * len(header))
    rows = []
    bad = 0
    for path in paths:
        with open(path) as f:
            n = m = None
            for line in f:
                if line.startswith("p "):
                    _, _, n, m = line.split()
                    break
        rep = ["--repeat", str(args.repeat)]
        results = {name: run([exe, algo, *rep, path], args.timeout) for name, exe, algo in solvers}
        finished = [r for r in results.values() if r[0] not in ("TIMEOUT", "CRASH")]
        ok = bool(finished) and all(r[:2] == finished[0][:2] for r in finished)
        bad += not ok
        name = os.path.relpath(path, args.dir)
        times = " ".join(f"{r[2]:9.1f}" if r[2] is not None else f"{r[0]:>9}" for r in results.values())
        print(f"{name:<34} {n:>8} {m:>9}  {times}  {'yes' if ok else 'NO'}", flush=True)
        rows.append({
            "instance": name, "n": n, "m": m, "status": finished[0][0] if finished else "NONE",
            **{k: (r[2] if r[2] is not None else r[0]) for k, r in results.items()},
            "agree": ok,
        })
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"times in ms (best of {args.repeat}, timeout {args.timeout}s); ok = all finished solvers agree")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correctness")
    c.add_argument("--count", type=int, default=10, help="seeds per instance family")
    b = sub.add_parser("bench")
    b.add_argument("--repeat", type=int, default=3)
    s = sub.add_parser("suite")
    s.add_argument("dir")
    s.add_argument("--repeat", type=int, default=1)
    s.add_argument("--timeout", type=int, default=300, help="seconds per solve")
    s.add_argument("--csv", help="also write the results to this CSV file")
    args = p.parse_args()
    for exe in (LEMON, RUST):
        if not os.path.exists(exe):
            sys.exit(f"missing {exe}; see the build steps in this script's docstring")
    sys.exit({"correctness": correctness, "bench": bench, "suite": suite}[args.cmd](args))


if __name__ == "__main__":
    main()
