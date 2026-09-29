//! Cross-checks every solver configuration against the others on random
//! instances and verifies each optimal solution's certificate.

mod common;

use common::{Rng, Solver, check_solution};
use silvermite::{Capacity, Error, Node, Number, Problem, SupplyType};

/// A small random instance mixing lower bounds (some negative), infinite
/// capacities, negative costs, self-loops, parallel arcs, and all three
/// supply regimes. Returns the problem and whether it has a negative-cost
/// arc with infinite capacity.
fn random_problem<T: Number>(
    rng: &mut Rng,
    allow_negative_infinite: bool,
) -> (Problem<T, T>, bool) {
    let t = |x: i64| T::from_i128(x as i128);
    let n = rng.range(1, 25) as usize;
    let m = rng.range(n as i64 - 1, 4 * n as i64) as usize;
    let mut p = Problem::new(n);
    let random_node = |rng: &mut Rng| Node::new(rng.range(0, n as i64 - 1) as usize);
    let mut negative_infinite = false;
    for _ in 0..m {
        let s = random_node(rng);
        let d = random_node(rng);
        let lower = if rng.chance(0.15) {
            rng.range(-5, 5)
        } else {
            0
        };
        let infinite = rng.chance(0.2);
        let upper = if infinite {
            Capacity::Infinite
        } else {
            Capacity::Finite(t(lower + rng.range(0, 20)))
        };
        let mut cost = rng.range(-10, 30);
        if infinite && cost < 0 {
            if allow_negative_infinite {
                negative_infinite = true;
            } else {
                cost = -cost;
            }
        }
        p.add_arc(s, d, t(lower), upper, t(cost));
    }

    // Random supplies, then nudge the total into the chosen regime.
    let mut total = 0;
    for v in p.nodes() {
        let s = if rng.chance(0.3) { rng.range(-8, 8) } else { 0 };
        p.set_supply(v, t(s));
        total += s;
    }
    let regime = rng.range(0, 2);
    let fix = match regime {
        0 => -total,
        1 => -total - rng.range(0, 5),
        _ => -total + rng.range(0, 5),
    };
    let v = random_node(rng);
    let supply = p.supply(v).to_i128() as i64;
    p.set_supply(v, t(supply + fix));
    p.set_supply_type(if regime == 2 {
        SupplyType::Leq
    } else {
        SupplyType::Geq
    });
    (p, negative_infinite)
}

fn outcome<T: Number>(r: &Result<silvermite::Solution<T, T>, Error>) -> Result<i128, Error> {
    r.as_ref().map(|s| s.total_cost()).map_err(|e| *e)
}

fn cross_check<T: Number>(seed: u64, count: usize, allow_negative_infinite: bool) {
    let mut rng = Rng::new(seed);
    let solvers = Solver::all();
    let mut feasible = 0;
    for case in 0..count {
        let (p, negative_infinite) = random_problem::<T>(&mut rng, allow_negative_infinite);
        let reference = outcome(&solvers[0].solve(&p));
        for &solver in &solvers {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solver.solve(&p)))
                    .unwrap_or_else(|_| {
                        panic!("seed {seed} case {case} {solver:?} panicked\n{p:?}")
                    });
            if let Ok(s) = &result {
                check_solution(&p, s)
                    .unwrap_or_else(|e| panic!("seed {seed} case {case} {solver:?}: {e}\n{p:?}"));
            }
            if negative_infinite && !solver.full_negative_cost_support() {
                assert_eq!(
                    result.err(),
                    Some(Error::Unbounded),
                    "seed {seed} case {case} {solver:?}"
                );
                continue;
            }
            assert_eq!(
                outcome(&result),
                reference,
                "seed {seed} case {case} {solver:?}\n{p:?}"
            );
        }
        feasible += (reference != Err(Error::Infeasible)) as usize;
    }
    // Guard against a generator that only produces infeasible instances.
    eprintln!("seed {seed}: {feasible} of {count} instances feasible");
    assert!(
        feasible > count / 3,
        "only {feasible} of {count} instances were feasible"
    );
}

#[test]
fn random_i64() {
    cross_check::<i64>(1, 3000, false);
}

#[test]
fn random_i32() {
    cross_check::<i32>(2, 1000, false);
}

#[test]
fn random_negative_infinite_arcs() {
    cross_check::<i64>(3, 1000, true);
}
