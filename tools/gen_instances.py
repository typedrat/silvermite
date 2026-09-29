#!/usr/bin/env python3
"""Generates DIMACS min-cost-flow instances for differential testing and
benchmarking.

Families:
  netgen     Random sparse network in the spirit of NETGEN: K sources feed K
             sinks through high-capacity skeleton paths, plus random arcs.
  grid       W x W grid with arcs to all four neighbors (road-network-like).
  transport  Complete bipartite graph from sources to sinks (dense).

Infinite capacities are written as "-1" (below the lower bound), which LEMON's
DIMACS reader interprets as infinite.
"""

import argparse
import random
import sys


def arc_line(rng, args, u, v, cap=None, cost=None):
    if cost is None:
        cost = rng.randint(1, args.maxcost)
    if cap is None:
        if rng.random() < args.inf:
            cap = None
        else:
            cap = rng.randint(1, args.maxcap)
    low = 0
    if cap is not None and rng.random() < args.lower:
        low = rng.randint(0, max(0, cap // 4))
    if cap is not None and rng.random() < args.neg:
        cost = -cost
    return (u, v, low, -1 if cap is None else cap, cost)


def netgen(rng, args):
    n, k = args.n, args.sources
    m = args.n * args.deg
    total = args.supply
    supplies = {}
    arcs = []
    sources = list(range(1, k + 1))
    sinks = list(range(n - k + 1, n + 1))
    middle = list(range(k + 1, n - k + 1))
    rng.shuffle(middle)
    share = total // k
    for i in range(k):
        supplies[sources[i]] = share
        supplies[sinks[i]] = -share
        chain = [sources[i]] + middle[i::k] + [sinks[i]]
        for u, v in zip(chain, chain[1:]):
            arcs.append(arc_line(rng, args, u, v, cap=total))
    while len(arcs) < m:
        u, v = rng.randint(1, n), rng.randint(1, n)
        if u != v:
            arcs.append(arc_line(rng, args, u, v))
    return n, supplies, arcs


def grid(rng, args):
    w = max(2, int(round(args.n ** 0.5)))
    n = w * w
    node = lambda x, y: y * w + x + 1
    arcs = []
    for y in range(w):
        for x in range(w):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < w and 0 <= y + dy < w:
                    arcs.append(arc_line(rng, args, node(x, y), node(x + dx, y + dy)))
    supplies = {}
    picks = rng.sample(range(1, n + 1), 2 * args.sources)
    share = args.supply // args.sources
    for s in picks[: args.sources]:
        supplies[s] = share
    for t in picks[args.sources :]:
        supplies[t] = -share
    return n, supplies, arcs


def transport(rng, args):
    k = args.sources
    sinks = max(1, args.n - k)
    n = k + sinks
    supplies = {}
    total = args.supply
    for s in range(1, k + 1):
        supplies[s] = total // k
    base, extra = divmod(total // k * k, sinks)
    for i, t in enumerate(range(k + 1, n + 1)):
        supplies[t] = -(base + (1 if i < extra else 0))
    arcs = [arc_line(rng, args, s, t) for s in range(1, k + 1) for t in range(k + 1, n + 1)]
    return n, supplies, arcs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=["netgen", "grid", "transport"])
    p.add_argument("--n", type=int, default=1000, help="node count (approximate for grid)")
    p.add_argument("--deg", type=int, default=8, help="average out-degree (netgen)")
    p.add_argument("--sources", type=int, default=10)
    p.add_argument("--supply", type=int, default=10000, help="total supply")
    p.add_argument("--maxcost", type=int, default=10000)
    p.add_argument("--maxcap", type=int, default=1000)
    p.add_argument("--inf", type=float, default=0.0, help="fraction of arcs with infinite capacity")
    p.add_argument("--lower", type=float, default=0.0, help="fraction of finite arcs with a lower bound")
    p.add_argument("--neg", type=float, default=0.0, help="fraction of finite arcs with negative cost")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n, supplies, arcs = {"netgen": netgen, "grid": grid, "transport": transport}[args.family](rng, args)

    out = sys.stdout
    out.write(f"c {' '.join(sys.argv[1:])}\n")
    out.write(f"p min {n} {len(arcs)}\n")
    for v in sorted(supplies):
        if supplies[v]:
            out.write(f"n {v} {supplies[v]}\n")
    for u, v, low, cap, cost in arcs:
        out.write(f"a {u} {v} {low} {cap} {cost}\n")


if __name__ == "__main__":
    main()
