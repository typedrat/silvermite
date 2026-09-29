#![cfg(feature = "petgraph")]

mod common;

use common::check_solution;
use petgraph::stable_graph::StableDiGraph;
use silvermite::petgraph::{ArcData, from_graph};
use silvermite::{Algorithm, solve};

#[test]
fn stable_graph_with_holes() {
    // Node weights are supplies; edge weights are costs with capacity 10.
    let mut g = StableDiGraph::<i64, i64>::new();
    let s = g.add_node(8);
    let doomed = g.add_node(0);
    let a = g.add_node(0);
    let b = g.add_node(0);
    let t = g.add_node(-8);
    g.add_edge(s, doomed, 1);
    g.add_edge(doomed, t, 1);
    let sa = g.add_edge(s, a, 2);
    let at = g.add_edge(a, t, 2);
    let sb = g.add_edge(s, b, 3);
    let bt = g.add_edge(b, t, 3);
    // Removing the cheapest route leaves holes in both index spaces.
    g.remove_node(doomed);

    let (problem, ids) = from_graph(
        &g,
        |e| ArcData {
            lower: 0,
            upper: 5,
            cost: *e.weight(),
        },
        |n| g[n],
    );
    assert_eq!(problem.node_count(), 4);
    assert_eq!(problem.arc_count(), 4);

    for algorithm in [Algorithm::NetworkSimplex, Algorithm::CostScaling] {
        let solution = solve(&problem, algorithm).unwrap();
        check_solution(&problem, &solution).unwrap();
        assert_eq!(solution.total_cost(), 5 * 4 + 3 * 6);

        let flows: std::collections::HashMap<_, _> = ids.edge_flows(&solution).collect();
        assert_eq!(flows[&sa], 5);
        assert_eq!(flows[&at], 5);
        assert_eq!(flows[&sb], 3);
        assert_eq!(flows[&bt], 3);

        let potentials: std::collections::HashMap<_, _> = ids.node_potentials(&solution).collect();
        assert_eq!(potentials.len(), 4);
        assert!(potentials.contains_key(&t));
    }
}
