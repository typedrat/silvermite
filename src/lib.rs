//! Minimum cost flow solvers ported from the [LEMON] C++ graph library:
//! [`NetworkSimplex`] and [`CostScaling`].
//!
//! Build a [`Problem`] over dense node and arc indices, then call [`solve`]
//! or run a solver directly. Every solver returns an optimal flow together
//! with optimal node potentials, which certify optimality through
//! complementary slackness.
//!
//! Network simplex is the default and the better choice on most inputs,
//! including every real-world instance family measured so far. Cost scaling
//! can be several times faster on some large synthetic families (notably
//! grid-on-torus transportation problems), but neither network size nor
//! density predicts when, so benchmark on your own instances before
//! switching.
//!
//! ```
//! use silvermite::{Algorithm, Capacity, Problem, solve};
//!
//! // Ship 10 units from s to t over two routes, plus a pricier direct arc
//! // with no capacity limit.
//! let mut p = Problem::<i64, i64>::new(0);
//! let [s, a, b, t] = [10, 0, 0, -10].map(|supply| p.add_node(supply));
//! let sa = p.add_arc(s, a, 0, 6, 2); // source, target, lower, upper, cost
//! p.add_arc(a, t, 0, 6, 2);
//! p.add_arc(s, b, 0, 3, 3);
//! p.add_arc(b, t, 0, 3, 3);
//! let st = p.add_arc(s, t, 0, Capacity::Infinite, 7);
//!
//! let solution = solve(&p, Algorithm::NetworkSimplex).unwrap();
//! assert_eq!(solution.flow(sa), 6);
//! assert_eq!(solution.flow(st), 1);
//! assert_eq!(solution.total_cost(), 6 * 4 + 3 * 6 + 7);
//! ```
//!
//! Flow and cost types are any signed primitive integers (see [`Number`]).
//!
//! With the `petgraph` feature, the [`petgraph`](crate::petgraph) module
//! builds problems from petgraph graphs.
//!
//! The crate is `no_std` and needs only `alloc`.
//!
//! [LEMON]: https://lemon.cs.elte.hu/

#![no_std]

extern crate alloc;

mod circulation;
mod cost_scaling;
mod ivec;
mod network_simplex;
mod num;
mod problem;

#[cfg(feature = "petgraph")]
pub mod petgraph;

pub use cost_scaling::{CostScaling, Method};
pub use network_simplex::{NetworkSimplex, PivotRule};
pub use num::Number;
pub use problem::{Arc, Capacity, Error, Node, Problem, Solution, SupplyType};

/// Which solver [`solve`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// [`NetworkSimplex`] with its default settings.
    #[default]
    NetworkSimplex,
    /// [`CostScaling`] with its default settings and `i64` large costs.
    CostScaling,
}

/// Solves `problem` with the given algorithm and default solver settings.
///
/// To reuse buffers across many solves or change solver settings, use
/// [`NetworkSimplex`] or [`CostScaling`] directly.
pub fn solve<V: Number, C: Number>(
    problem: &Problem<V, C>,
    algorithm: Algorithm,
) -> Result<Solution<V, C>, Error> {
    match algorithm {
        Algorithm::NetworkSimplex => NetworkSimplex::new().solve(problem),
        Algorithm::CostScaling => CostScaling::<V, C>::new().solve(problem),
    }
}
