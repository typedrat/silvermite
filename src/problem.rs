use crate::Number;
use crate::ivec::{Idx, NodeId};
use alloc::vec;
use alloc::vec::Vec;

/// How node supplies constrain the flow balance `out(v) - in(v)` at each
/// node `v`.
///
/// When the supplies sum to zero, both variants mean the same thing: every
/// node's balance must equal its supply exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SupplyType {
    /// `out(v) - in(v) >= supply(v)`. All supply must be sent, but demand
    /// nodes may receive less than their demand. Requires the supplies to
    /// sum to zero or less.
    #[default]
    Geq,
    /// `out(v) - in(v) <= supply(v)`. All demand must be met, but supply
    /// nodes may send less than their supply. Requires the supplies to sum
    /// to zero or more.
    Leq,
}

/// A minimum cost flow problem over nodes and arcs numbered densely from
/// zero in insertion order.
///
/// An upper bound of `V::max_value()` means the arc has infinite capacity.
#[derive(Clone, Debug, Default)]
pub struct Problem<V, C> {
    pub(crate) supply: Vec<V>,
    pub(crate) source: Vec<NodeId>,
    pub(crate) target: Vec<NodeId>,
    pub(crate) lower: Vec<V>,
    pub(crate) upper: Vec<V>,
    pub(crate) cost: Vec<C>,
    pub(crate) supply_type: SupplyType,
}

impl<V: Number, C: Number> Problem<V, C> {
    /// Creates a problem with `node_count` nodes of zero supply and no arcs.
    pub fn new(node_count: usize) -> Self {
        Self::with_capacity(node_count, 0)
    }

    /// Creates a problem with `node_count` nodes of zero supply, reserving
    /// room for `arc_capacity` arcs.
    pub fn with_capacity(node_count: usize, arc_capacity: usize) -> Self {
        Problem {
            supply: vec![V::zero(); node_count],
            source: Vec::with_capacity(arc_capacity),
            target: Vec::with_capacity(arc_capacity),
            lower: Vec::with_capacity(arc_capacity),
            upper: Vec::with_capacity(arc_capacity),
            cost: Vec::with_capacity(arc_capacity),
            supply_type: SupplyType::Geq,
        }
    }

    pub fn node_count(&self) -> usize {
        self.supply.len()
    }

    pub fn arc_count(&self) -> usize {
        self.source.len()
    }

    /// Adds a node with the given supply (negative for demand) and returns
    /// its index.
    pub fn add_node(&mut self, supply: V) -> usize {
        self.supply.push(supply);
        self.supply.len() - 1
    }

    /// Adds an arc and returns its index.
    ///
    /// # Panics
    ///
    /// Panics if `source` or `target` is not a node of the problem.
    pub fn add_arc(&mut self, source: usize, target: usize, lower: V, upper: V, cost: C) -> usize {
        let n = self.node_count();
        assert!(
            source < n,
            "arc source {source} out of range (node count {n})"
        );
        assert!(
            target < n,
            "arc target {target} out of range (node count {n})"
        );
        self.source.push(NodeId::new(source));
        self.target.push(NodeId::new(target));
        self.lower.push(lower);
        self.upper.push(upper);
        self.cost.push(cost);
        self.source.len() - 1
    }

    pub fn supply(&self, node: usize) -> V {
        self.supply[node]
    }

    pub fn supplies(&self) -> &[V] {
        &self.supply
    }

    pub fn set_supply(&mut self, node: usize, supply: V) {
        self.supply[node] = supply;
    }

    /// Replaces all supplies with a single `amount` sent from `source` to
    /// `target`.
    pub fn set_st_supply(&mut self, source: usize, target: usize, amount: V) {
        self.supply.fill(V::zero());
        self.supply[source] = amount;
        self.supply[target] = -amount;
    }

    pub fn supply_type(&self) -> SupplyType {
        self.supply_type
    }

    pub fn set_supply_type(&mut self, supply_type: SupplyType) {
        self.supply_type = supply_type;
    }

    pub fn source(&self, arc: usize) -> usize {
        self.source[arc].index()
    }

    pub fn target(&self, arc: usize) -> usize {
        self.target[arc].index()
    }

    pub fn lower(&self, arc: usize) -> V {
        self.lower[arc]
    }

    pub fn upper(&self, arc: usize) -> V {
        self.upper[arc]
    }

    pub fn cost(&self, arc: usize) -> C {
        self.cost[arc]
    }

    pub fn set_bounds(&mut self, arc: usize, lower: V, upper: V) {
        self.lower[arc] = lower;
        self.upper[arc] = upper;
    }

    pub fn set_cost(&mut self, arc: usize, cost: C) {
        self.cost[arc] = cost;
    }

    /// Checks the invariants every solver relies on.
    pub(crate) fn validate(&self, extra_arcs_per_node: usize) -> Result<(), Error> {
        let n = self.node_count();
        let m = self.arc_count();
        // Internal indices are u32, with u32::MAX reserved as the niche for
        // optional indices.
        let internal_arcs = (m as u128) * 2 + (n as u128) * extra_arcs_per_node as u128;
        if n as u128 + 2 >= u32::MAX as u128 || internal_arcs >= u32::MAX as u128 {
            return Err(Error::TooLarge);
        }
        if let Some(arc) = self
            .lower
            .iter()
            .zip(&self.upper)
            .position(|(lower, upper)| upper < lower)
        {
            return Err(Error::InvalidBounds { arc });
        }
        Ok(())
    }

    /// Computes `sum(flow * cost)` without overflow.
    pub(crate) fn total_cost(&self, flow: &[V]) -> i128 {
        flow.iter()
            .zip(&self.cost)
            .map(|(&f, &c)| f.to_i128() * c.to_i128())
            .sum()
    }
}

/// An optimal flow together with optimal node potentials (dual solution).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution<V, C> {
    pub(crate) flow: Vec<V>,
    pub(crate) potential: Vec<C>,
    pub(crate) total_cost: i128,
}

impl<V: Number, C: Number> Solution<V, C> {
    pub fn flow(&self, arc: usize) -> V {
        self.flow[arc]
    }

    /// Flow on every arc, indexed by arc.
    pub fn flows(&self) -> &[V] {
        &self.flow
    }

    pub fn potential(&self, node: usize) -> C {
        self.potential[node]
    }

    /// Potential of every node, indexed by node.
    ///
    /// With `pi` as the potentials, every arc `(u, v)` satisfies complementary
    /// slackness for the reduced cost `cost + pi[u] - pi[v]`: arcs with
    /// positive reduced cost are at their lower bound and arcs with negative
    /// reduced cost are at their upper bound.
    pub fn potentials(&self) -> &[C] {
        &self.potential
    }

    /// Total cost of the flow, computed in 128 bits so it cannot overflow.
    pub fn total_cost(&self) -> i128 {
        self.total_cost
    }

    pub fn into_parts(self) -> (Vec<V>, Vec<C>) {
        (self.flow, self.potential)
    }
}

/// Why a problem has no optimal solution, or could not be solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No flow satisfies the bounds and supply constraints.
    #[error("the problem has no feasible flow")]
    Infeasible,
    /// The objective is unbounded below.
    ///
    /// Network simplex reports this only when a negative-cost cycle of
    /// infinite capacity is reachable. Cost scaling reports it for any arc
    /// with negative cost and infinite capacity, even if the objective is in
    /// fact bounded over the feasible flows.
    #[error("the objective is unbounded")]
    Unbounded,
    /// The arc's lower bound exceeds its upper bound.
    #[error("arc {arc} has a lower bound greater than its upper bound")]
    InvalidBounds { arc: usize },
    /// The problem has more nodes or arcs than the solvers can index.
    #[error("the problem has too many nodes or arcs")]
    TooLarge,
    /// Cost scaling's internal costs (arc costs multiplied by the node count
    /// and scaling factor) do not fit in its large cost type. Use a wider
    /// large cost type, such as `CostScaling<V, C, i128>`.
    #[error("scaled arc costs overflow the large cost type")]
    Overflow,
}
