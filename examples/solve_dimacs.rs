//! Solves a DIMACS min-cost-flow file and prints the outcome in the same
//! format as the LEMON reference solver in `tools/lemon-ref`.
//!
//! Usage: solve_dimacs <ns|cs|cs-push|cs-augment> [--leq] [--repeat N] <file.min>
//! Output: "OPTIMAL <total cost> <solve ms>", "INFEASIBLE", or "UNBOUNDED".
//! With --repeat, the reported time is the fastest of N solves.

use std::time::Instant;

use silvermite::{
    Capacity, CostScaling, Error, Method, NetworkSimplex, NodeId, Problem, SupplyType,
};

fn read_dimacs(text: &str) -> Problem<i64, i64> {
    let mut problem = Problem::new(0);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let tag = fields.next();
        if !matches!(tag, Some("p" | "n" | "a")) {
            continue;
        }
        if tag == Some("p") {
            fields.next(); // problem type
        }
        let nums: Vec<i64> = fields.map(|f| f.parse().expect("bad number")).collect();
        // DIMACS node ids start at 1.
        let node = |id: i64| NodeId::new(id as usize - 1);
        match (tag, nums.as_slice()) {
            (Some("p"), &[n, m]) => problem = Problem::with_capacity(n as usize, m as usize),
            (Some("n"), &[id, supply]) => problem.set_supply(node(id), supply),
            (Some("a"), &[u, v, low, cap, cost]) => {
                // A capacity below the lower bound means "infinite".
                let upper = if cap >= low {
                    Capacity::Finite(cap)
                } else {
                    Capacity::Infinite
                };
                problem.add_arc(node(u), node(v), low, upper, cost);
            }
            _ => panic!("malformed line: {line}"),
        }
    }
    problem
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: solve_dimacs <ns|cs|cs-push|cs-augment> [--leq] [--repeat N] <file.min>");
        std::process::exit(2);
    }
    let algo = args[0].as_str();
    let leq = args.iter().any(|a| a == "--leq");
    let repeat: usize = args
        .iter()
        .position(|a| a == "--repeat")
        .map_or(1, |i| args[i + 1].parse().expect("bad repeat count"));
    let path = args.last().unwrap();

    let text = std::fs::read_to_string(path).expect("cannot read input");
    let mut problem = read_dimacs(&text);
    if leq {
        problem.set_supply_type(SupplyType::Leq);
    }

    let mut ns = NetworkSimplex::new();
    let mut cs = CostScaling::new();
    let mut best = f64::INFINITY;
    let mut result = Err(Error::Infeasible);
    for _ in 0..repeat {
        let start = Instant::now();
        result = match algo {
            "ns" => ns.solve(&problem),
            "cs" => cs.solve(&problem),
            "cs-push" => CostScaling::new().method(Method::Push).solve(&problem),
            "cs-augment" => CostScaling::new().method(Method::Augment).solve(&problem),
            other => panic!("unknown algorithm {other}"),
        };
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }

    match result {
        Ok(solution) => println!("OPTIMAL {} {best}", solution.total_cost()),
        Err(Error::Unbounded) => println!("UNBOUNDED"),
        Err(Error::Infeasible) => println!("INFEASIBLE"),
        Err(e) => println!("ERROR {e}"),
    }
}
