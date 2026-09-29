//! LEMON's `test/min_cost_flow_test.cc`, run against every solver
//! configuration with the same instances and expected results.

mod common;

use common::{Solver, check_solution};
use silvermite::{Capacity, Error, NodeId, Number, Problem, SupplyType};

// Columns: source, target, cost, cap, low1, low2, low3 (1-based node labels)
const ARCS: [[i32; 7]; 21] = [
    [1, 2, 70, 11, 0, 8, 8],
    [1, 3, 150, 3, 0, 1, 0],
    [1, 4, 80, 15, 0, 2, 2],
    [2, 8, 80, 12, 0, 0, 0],
    [3, 5, 140, 5, 0, 3, 1],
    [4, 6, 60, 10, 0, 1, 0],
    [4, 7, 80, 2, 0, 0, 0],
    [4, 8, 110, 3, 0, 0, 0],
    [5, 7, 60, 14, 0, 0, 0],
    [5, 11, 120, 12, 0, 0, 0],
    [6, 3, 0, 3, 0, 0, 0],
    [6, 9, 140, 4, 0, 0, 0],
    [6, 10, 90, 8, 0, 0, 0],
    [7, 1, 30, 5, 0, 0, -5],
    [8, 12, 60, 16, 0, 4, 3],
    [9, 12, 50, 6, 0, 0, 0],
    [10, 12, 70, 13, 0, 5, 2],
    [10, 2, 100, 7, 0, 0, 0],
    [10, 7, 60, 10, 0, 0, -3],
    [11, 10, 20, 14, 0, 6, -20],
    [12, 11, 30, 10, 0, 0, -10],
];

// Columns: sup1 .. sup6, one row per node
const SUPPLIES: [[i32; 6]; 12] = [
    [20, 27, 0, 30, 20, 30],
    [-4, 0, 0, 0, -8, -3],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 6, 11],
    [-6, 0, 0, 0, -5, -6],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3],
    [3, 0, 0, 0, 0, 0],
    [-2, 0, 0, 0, -7, -2],
    [0, 0, 0, 0, -10, 0],
    [-20, -27, 0, -30, -30, -20],
];

// Columns: source, target, cost, low1, low2
const NEG1_ARCS: [[i32; 5]; 9] = [
    [1, 2, 100, 0, 0],
    [1, 3, 30, 0, 0],
    [2, 4, 20, 0, 0],
    [3, 4, 80, 0, 0],
    [3, 2, 50, 0, 0],
    [5, 3, 10, 0, 0],
    [5, 6, 80, 0, 1000],
    [6, 7, 30, 0, -1000],
    [7, 5, -120, 0, 0],
];
const NEG1_SUPPLY: [i32; 7] = [100, 0, 0, -100, 0, 0, 0];

#[derive(Clone, Copy)]
enum Cost {
    Table,
    Unit,
}

#[derive(Clone, Copy)]
enum Upper {
    Cap,
    Infinite,
}

/// The node with a 1-based label from the tables above.
fn node(label: i32) -> NodeId {
    NodeId::new(label as usize - 1)
}

fn main_graph<T: Number>(lower: usize, upper: Upper, cost: Cost, supply: usize) -> Problem<T, T> {
    let mut p = Problem::new(0);
    for row in &SUPPLIES {
        p.add_node(T::from_i128(row[supply] as i128));
    }
    for a in &ARCS {
        let t = |x: i32| T::from_i128(x as i128);
        let low = if lower == 0 {
            T::zero()
        } else {
            t(a[3 + lower])
        };
        let up = match upper {
            Upper::Cap => Capacity::Finite(t(a[3])),
            Upper::Infinite => Capacity::Infinite,
        };
        let c = match cost {
            Cost::Table => t(a[2]),
            Cost::Unit => T::one(),
        };
        p.add_arc(node(a[0]), node(a[1]), low, up, c);
    }
    p
}

fn neg1_graph<T: Number>(lower: usize, upper: Option<i32>) -> Problem<T, T> {
    let t = |x: i32| T::from_i128(x as i128);
    let mut p = Problem::new(0);
    for &s in &NEG1_SUPPLY {
        p.add_node(t(s));
    }
    for a in &NEG1_ARCS {
        let up = upper.map_or(Capacity::Infinite, |u| Capacity::Finite(t(u)));
        p.add_arc(node(a[0]), node(a[1]), t(a[3 + lower]), up, t(a[2]));
    }
    p
}

fn neg2_graph<T: Number>(upper: Option<i32>) -> Problem<T, T> {
    let t = |x: i32| T::from_i128(x as i128);
    let mut p = Problem::new(0);
    let s = p.add_node(t(100));
    let d = p.add_node(t(-300));
    let up = upper.map_or(Capacity::Infinite, |u| Capacity::Finite(t(u)));
    p.add_arc(s, d, T::zero(), up, t(-1));
    p
}

fn with_type<T: Number>(mut p: Problem<T, T>, ty: SupplyType) -> Problem<T, T> {
    p.set_supply_type(ty);
    p
}

fn expect<T: Number>(solver: Solver, p: &Problem<T, T>, expected: Result<i128, Error>, id: &str) {
    let result = solver.solve(p);
    match (&result, expected) {
        (Ok(s), Ok(total)) => {
            assert_eq!(s.total_cost(), total, "{solver:?} {id}: wrong total cost");
            check_solution(p, s).unwrap_or_else(|e| panic!("{solver:?} {id}: {e}"));
        }
        (Err(e), Err(want)) => assert_eq!(*e, want, "{solver:?} {id}"),
        _ => panic!(
            "{solver:?} {id}: expected {expected:?}, got {:?}",
            result.map(|s| s.total_cost())
        ),
    }
}

fn run_geq_tests<T: Number>(solver: Solver) {
    use Cost::*;
    use Upper::*;
    let geq = SupplyType::Geq;

    // Basic tests
    expect(solver, &main_graph::<T>(0, Cap, Table, 0), Ok(5240), "1");
    expect(solver, &main_graph::<T>(0, Cap, Table, 1), Ok(7620), "2");
    expect(solver, &main_graph::<T>(2, Cap, Table, 0), Ok(5970), "3");
    expect(solver, &main_graph::<T>(2, Cap, Table, 1), Ok(8010), "4");
    expect(solver, &main_graph::<T>(0, Infinite, Unit, 0), Ok(74), "5");
    expect(solver, &main_graph::<T>(2, Infinite, Unit, 1), Ok(94), "6");
    expect(solver, &main_graph::<T>(0, Infinite, Unit, 2), Ok(0), "7");
    expect(
        solver,
        &main_graph::<T>(2, Cap, Unit, 2),
        Err(Error::Infeasible),
        "8",
    );
    expect(solver, &main_graph::<T>(3, Cap, Table, 3), Ok(6360), "9");

    // GEQ form
    expect(
        solver,
        &with_type(main_graph::<T>(0, Cap, Table, 4), geq),
        Ok(3530),
        "10",
    );
    expect(
        solver,
        &with_type(main_graph::<T>(2, Cap, Table, 4), geq),
        Ok(4540),
        "11",
    );
    expect(
        solver,
        &with_type(main_graph::<T>(2, Cap, Table, 5), geq),
        Err(Error::Infeasible),
        "12",
    );

    // Negative costs
    expect(
        solver,
        &neg1_graph::<T>(0, None),
        Err(Error::Unbounded),
        "13",
    );
    expect(solver, &neg1_graph::<T>(0, Some(5000)), Ok(-40000), "14");
    expect(
        solver,
        &neg1_graph::<T>(1, None),
        Err(Error::Unbounded),
        "15",
    );
    if solver.full_negative_cost_support() {
        expect(solver, &neg2_graph::<T>(None), Ok(-300), "16");
    } else {
        expect(solver, &neg2_graph::<T>(None), Err(Error::Unbounded), "17");
    }
    expect(solver, &neg2_graph::<T>(Some(1000)), Ok(-300), "18");

    // Empty graph
    let empty = solver
        .solve(&Problem::<T, T>::new(0))
        .expect("empty problem");
    assert_eq!(empty.total_cost(), 0);
}

fn run_leq_tests<T: Number>(solver: Solver) {
    use Cost::*;
    use Upper::*;
    let leq = SupplyType::Leq;
    expect(
        solver,
        &with_type(main_graph::<T>(0, Cap, Table, 5), leq),
        Ok(5080),
        "19",
    );
    expect(
        solver,
        &with_type(main_graph::<T>(2, Cap, Table, 5), leq),
        Ok(5930),
        "20",
    );
    expect(
        solver,
        &with_type(main_graph::<T>(2, Cap, Table, 4), leq),
        Err(Error::Infeasible),
        "21",
    );
}

#[test]
fn lemon_suite_i32() {
    for solver in Solver::all() {
        run_geq_tests::<i32>(solver);
        run_leq_tests::<i32>(solver);
    }
}

#[test]
fn lemon_suite_i64() {
    for solver in Solver::all() {
        run_geq_tests::<i64>(solver);
        run_leq_tests::<i64>(solver);
    }
}
