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
//! use silvermite::{Algorithm, Problem, solve};
//!
//! // Ship 10 units from node 0 to node 3 over two routes.
//! let mut p = Problem::<i64, i64>::new(4);
//! p.set_st_supply(0, 3, 10);
//! p.add_arc(0, 1, 0, 6, 2);
//! p.add_arc(1, 3, 0, 6, 2);
//! p.add_arc(0, 2, 0, 8, 3);
//! p.add_arc(2, 3, 0, 8, 3);
//!
//! let solution = solve(&p, Algorithm::NetworkSimplex).unwrap();
//! assert_eq!(solution.flows(), &[6, 6, 4, 4]);
//! assert_eq!(solution.total_cost(), 48);
//! ```
//!
//! Flow and cost types are any signed primitive integers (see [`Number`]);
//! `V::max_value()` as an upper bound means infinite capacity.
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
pub use problem::{Error, Problem, Solution, SupplyType};

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
