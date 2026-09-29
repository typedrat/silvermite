//! Building problems from [petgraph](https://docs.rs/petgraph) graphs.
//!
//! [`from_graph`] turns any directed petgraph graph into a [`Problem`] plus
//! an [`IdMap`] that translates solutions back to the graph's node and edge
//! ids. Graphs with index holes, such as `StableGraph` after removals, are
//! compacted.
//!
//! ```
//! use silvermite::petgraph::{ArcData, from_graph};
//! use silvermite::{Algorithm, solve};
//! use petgraph::graph::DiGraph;
//!
//! // Node weights are supplies; edge weights are (capacity, cost).
//! let mut g = DiGraph::<i64, (i64, i64)>::new();
//! let s = g.add_node(5);
//! let a = g.add_node(0);
//! let t = g.add_node(-5);
//! let sa = g.add_edge(s, a, (4, 1));
//! g.add_edge(a, t, (4, 1));
//! g.add_edge(s, t, (10, 3));
//!
//! let (problem, ids) = from_graph(
//!     &g,
//!     |e| ArcData { lower: 0, upper: e.weight().0, cost: e.weight().1 },
//!     |n| g[n],
//! );
//! let solution = solve(&problem, Algorithm::NetworkSimplex).unwrap();
//! assert_eq!(solution.total_cost(), 11);
//!
//! let flows: std::collections::HashMap<_, _> = ids.edge_flows(&solution).collect();
//! assert_eq!(flows[&sa], 4);
//! ```

use petgraph::Directed;
use petgraph::visit::{EdgeRef, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers, NodeIndexable};

use crate::{Number, Problem, Solution};

/// Bounds and cost of one arc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArcData<V, C> {
    pub lower: V,
    /// `V::max_value()` means infinite capacity.
    pub upper: V,
    pub cost: C,
}

/// Maps the dense node and arc indices of a [`Problem`] built by
/// [`from_graph`] back to the graph's ids.
#[derive(Clone, Debug)]
pub struct IdMap<N, E> {
    nodes: Vec<N>,
    edges: Vec<E>,
}

impl<N: Copy, E: Copy> IdMap<N, E> {
    /// The graph node behind problem node `node`.
    pub fn node_id(&self, node: usize) -> N {
        self.nodes[node]
    }

    /// The graph edge behind problem arc `arc`.
    pub fn edge_id(&self, arc: usize) -> E {
        self.edges[arc]
    }

    /// The flow on every graph edge.
    pub fn edge_flows<'a, V: Number, C: Number>(
        &'a self,
        solution: &'a Solution<V, C>,
    ) -> impl Iterator<Item = (E, V)> + 'a {
        self.edges
            .iter()
            .zip(solution.flows())
            .map(|(&e, &f)| (e, f))
    }

    /// The potential of every graph node.
    pub fn node_potentials<'a, V: Number, C: Number>(
        &'a self,
        solution: &'a Solution<V, C>,
    ) -> impl Iterator<Item = (N, C)> + 'a {
        self.nodes
            .iter()
            .zip(solution.potentials())
            .map(|(&n, &p)| (n, p))
    }
}

/// Builds a problem from a directed graph.
///
/// Problem nodes follow `node_identifiers()` order and problem arcs follow
/// `edge_references()` order. `arc` gives each edge's bounds and cost;
/// `supply` gives each node's supply (negative for demand).
pub fn from_graph<G, V, C>(
    graph: G,
    mut arc: impl FnMut(G::EdgeRef) -> ArcData<V, C>,
    mut supply: impl FnMut(G::NodeId) -> V,
) -> (Problem<V, C>, IdMap<G::NodeId, G::EdgeId>)
where
    G: IntoNodeIdentifiers + IntoEdgeReferences + NodeIndexable + GraphProp<EdgeType = Directed>,
    V: Number,
    C: Number,
{
    // Dense index of each graph index; u32::MAX marks holes.
    let mut dense = vec![u32::MAX; graph.node_bound()];
    let mut nodes = Vec::new();
    let mut problem = Problem::new(0);
    for n in graph.node_identifiers() {
        dense[graph.to_index(n)] = nodes.len() as u32;
        nodes.push(n);
        problem.add_node(supply(n));
    }

    let mut edges = Vec::new();
    for e in graph.edge_references() {
        let s = dense[graph.to_index(e.source())] as usize;
        let t = dense[graph.to_index(e.target())] as usize;
        let id = e.id();
        let ArcData { lower, upper, cost } = arc(e);
        problem.add_arc(s, t, lower, upper, cost);
        edges.push(id);
    }

    (problem, IdMap { nodes, edges })
}
