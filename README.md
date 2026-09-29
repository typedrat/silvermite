# silvermite

Minimum cost flow solvers for Rust, ported from the network simplex and cost
scaling implementations in [LEMON](https://lemon.cs.elte.hu/) 1.3.1, which
Kovács's 2015 evaluation found to be among the fastest available.

The name comes from the citrus rust mite, *Phyllocoptruta oleivora*: a small
rust-colored thing that lives on lemons, whose damage turns their skin silver.

- `NetworkSimplex`: primal network simplex with all five of LEMON's pivot
  rules (block search by default). Handles lower bounds, negative costs,
  infinite capacities, and both GEQ and LEQ supply constraints. The default,
  and the faster solver on most inputs.
- `CostScaling`: push/augment-relabel cost scaling with price refinement and
  global update heuristics. Several times faster on some large synthetic
  families; see [Benchmarks](#benchmarks) before choosing it.

Every solve returns an optimal flow and optimal node potentials, which certify
optimality through complementary slackness.

```rust
use silvermite::{Algorithm, Capacity, Problem, solve};

// Ship 10 units from s to t over two routes, plus a pricier direct arc
// with no capacity limit.
let mut p = Problem::<i64, i64>::new(0);
let [s, a, b, t] = [10, 0, 0, -10].map(|supply| p.add_node(supply));
let sa = p.add_arc(s, a, 0, 6, 2); // source, target, lower, upper, cost
p.add_arc(a, t, 0, 6, 2);
p.add_arc(s, b, 0, 3, 3);
p.add_arc(b, t, 0, 3, 3);
let st = p.add_arc(s, t, 0, Capacity::Infinite, 7);

let solution = solve(&p, Algorithm::NetworkSimplex).unwrap();
assert_eq!(solution.flow(sa), 6);
assert_eq!(solution.flow(st), 1);
assert_eq!(solution.total_cost(), 6 * 4 + 3 * 6 + 7);
```

Flow and cost types can be any signed primitive integers. Nodes and arcs are
identified by `NodeId` and `ArcId`, numbered densely from zero in insertion
order, and upper bounds are a `Capacity`, which plain numbers convert into.
Solver instances keep their buffers between calls to `solve`, so reuse one
when solving many problems.

The crate is plain safe Rust, `no_std` (it needs only `alloc`), and works on
32-bit targets, including `wasm32-unknown-unknown`; the test suite passes on
`wasm32-wasip1`.

## petgraph

With the `petgraph` feature, `silvermite::petgraph::from_graph` builds a
`Problem` from any directed petgraph graph and returns an `IdMap` for reading
flows and potentials back by the graph's own edge and node ids. Graphs with index holes,
such as a `StableGraph` after removals, are compacted.

## Differences from LEMON

The solvers follow LEMON's code closely enough to take the same pivot and
relabel steps, with these exceptions:

- **Bug fixes in `CostScaling`.** Differential testing against the C++ build
  found three defects in LEMON 1.3.1, all fixed here:
  1. An arc with a lower bound and infinite capacity on a cycle (including a
     self-loop) could get a negative capacity, producing a flow that
     violates its lower bound and a wrong optimal cost.
  2. Arcs whose infinite capacity is replaced internally by a finite bound
     could be left with negative reduced cost, so the returned potentials
     were not a valid dual solution even when the flow was optimal.
  3. Price refinement could compute a node rank past the end of its bucket
     array, which is an out-of-bounds write in C++ (reproducible with
     `-D_GLIBCXX_ASSERTIONS` and `Method::Augment`).
- `CostScaling` supports `SupplyType::Leq` by solving the mirrored problem.
- An empty problem is solved (to an empty flow) instead of being reported
  infeasible.
- Inputs are validated: inconsistent bounds return `Error::InvalidBounds`,
  and cost scaling returns `Error::Overflow` when scaled costs would not fit
  its large cost type (`CostScaling::<V, C, i128>::default()` avoids that).
- Cost scaling's feasibility phase visits arcs newest-first, matching the
  order LEMON's graph types iterate in. Insertion order made that phase up to
  100 times slower on generator output.

## Testing

- `tests/lemon_suite.rs` ports LEMON's `min_cost_flow_test.cc` and runs it
  against every pivot rule and cost scaling method, for `i32` and `i64`.
- `tests/random.rs` cross-checks all eight solver configurations on 5,000
  random instances with lower bounds, infinite capacities, negative costs,
  self-loops, and all supply regimes, verifying every optimal solution's
  primal feasibility, complementary slackness, and dual objective.
- `tools/compare.py` runs the solvers against a C++ build of LEMON
  (`tools/lemon-ref`) on generated instances and on the benchmark suite.
- `tools/ab_builds.py` compares two builds of `solve_dimacs`, by interleaved
  timing or, with `--instructions`, by `perf stat` instruction counts, which
  are deterministic enough to catch regressions of a fraction of a percent.

## Benchmarks

`tools/benchmarks/fetch.sh` rebuilds the benchmark suite from Kovács,
*Minimum-Cost Flow Algorithms: an Experimental Evaluation* (Optimization
Methods and Software, 2015) using the original DIMACS generators and the
published generator parameters, and with `--real` downloads the ROAD and
VISION instances.

```sh
make -C tools/lemon-ref                       # LEMON reference build
cargo build --release --example solve_dimacs
tools/benchmarks/fetch.sh                     # generated families -> bench-data/
tools/compare.py suite bench-data/instances --csv results.csv
```

Measured with `tools/abtest.py`, which runs the Rust build and the LEMON
build (`g++ -O3 -march=native`) alternately so that load elsewhere on the
machine affects both equally; numbers are medians of five runs (three for the
real-world instances). Ratios below 1 mean the Rust port is faster.

| Instance | n | m | NS: LEMON ms | NS: Rust / LEMON | CS: LEMON ms | CS: Rust / LEMON |
|---|---:|---:|---:|---:|---:|---:|
| netgen_8_16a | 65,536 | 524,288 | 652 | 1.06 | 498 | 1.04 |
| netgen_sr_13a | 8,192 | 741,455 | 140 | 0.98 | 272 | 0.98 |
| netgen_lo_8_16a | 65,536 | 524,288 | 197 | 1.01 | 308 | 1.08 |
| netgen_lo_sr_13a | 8,192 | 741,455 | 49 | 1.04 | 204 | 0.98 |
| netgen_deg_07a | 4,096 | 524,288 | 67 | 0.86 | 152 | 0.98 |
| gridgen_8_16a | 65,537 | 524,296 | 888 | 0.95 | 701 | 1.27 (noisy) |
| gridgen_sr_13a | 8,191 | 745,381 | 155 | 0.94 | 400 | 0.93 |
| gridgen_deg_07a | 4,097 | 524,416 | 68 | 0.86 | 205 | 0.89 |
| goto_8_15a | 32,768 | 262,144 | 23,336 | 0.66 | 1,579 | 1.00 |
| goto_sr_12a | 4,096 | 262,144 | 493 | 0.87 | 452 | 1.01 |
| road_flow_03_NH_a | 116,920 | 265,402 | 492 | 0.90 | 1,284 | 0.91 |
| road_paths_03_NH_a | 116,920 | 265,402 | 454 | 0.82 | 853 | 1.03 |
| vision_rnd_01_a | 245,762 | 1,431,453 | 2,744 | 0.85 | 3,731 | 1.09 |

A single pass over the whole generated suite up to 2M arcs (79 instances,
ten families) plus 11 ROAD and VISION instances found every solver agreeing
on every optimal cost.

**Choosing a solver.** Network simplex was faster on every ROAD and VISION
instance (by 1.4 to 4 times, up to 1.4M arcs), on every family below about
16,000 nodes, and on all the dense families. Cost scaling won only on large
synthetic families: by up to 10 times on GOTO, and by 1.1 to 1.9 times on
NETGEN-8 and GRIDGEN-8 above 16,000 nodes. Neither node count, density, nor
the number of supply nodes separated the two groups, so the crate defaults to
network simplex instead of guessing; benchmark your own instances with
`examples/solve_dimacs` to decide.

## License

Boost Software License 1.0, the same as LEMON; see `LICENSE`, which carries
LEMON's copyright notice as the license requires for derivative works.
