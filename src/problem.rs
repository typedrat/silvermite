use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::Number;
use crate::ivec::{Idx, NodeIx};

macro_rules! handle {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            pub const fn new(index: usize) -> Self {
                $name(index)
            }

            pub const fn index(self) -> usize {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

handle!(
    /// Identifies a node of a [`Problem`]. Nodes are numbered densely from
    /// zero in the order they are added, so `NodeId::new(i)` is the `i`th.
    NodeId
);

handle!(
    /// Identifies an arc of a [`Problem`]. Arcs are numbered densely from
    /// zero in the order they are added, so `ArcId::new(i)` is the `i`th.
    ArcId
);

/// Upper bound on the flow along an arc.
///
/// Plain numbers convert into finite capacities, so `add_arc` accepts either
/// `5` or `Capacity::Infinite` as an upper bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capacity<V> {
    /// At most this much flow. `Finite(V::max_value())` is the same as
    /// `Infinite`, since no flow of type `V` can exceed it.
    Finite(V),
    Infinite,
}

impl<V> From<V> for Capacity<V> {
    fn from(v: V) -> Self {
        Capacity::Finite(v)
    }
}

impl<V: Number> Capacity<V> {
    /// The finite bound, or `None` for infinite capacity.
    pub fn finite(self) -> Option<V> {
        match self {
            Capacity::Finite(v) if v != V::max_value() => Some(v),
            _ => None,
        }
    }

    /// The solvers' representation, with `V::max_value()` for infinity.
    fn to_raw(self) -> V {
        self.finite().unwrap_or(V::max_value())
    }

    fn from_raw(v: V) -> Self {
        if v == V::max_value() {
            Capacity::Infinite
        } else {
            Capacity::Finite(v)
        }
    }
}

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

/// A minimum cost flow problem.
///
/// # Panics
///
/// Methods taking a [`NodeId`] or [`ArcId`] panic if it is not part of the
/// problem.
#[derive(Clone, Debug, Default)]
pub struct Problem<V, C> {
    pub(crate) supply: Vec<V>,
    pub(crate) source: Vec<NodeIx>,
    pub(crate) target: Vec<NodeIx>,
    pub(crate) lower: Vec<V>,
    // `V::max_value()` for infinite capacity
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

    /// Every node, in order.
    pub fn nodes(&self) -> impl DoubleEndedIterator<Item = NodeId> + ExactSizeIterator + use<V, C> {
        (0..self.node_count()).map(NodeId::new)
    }

    /// Every arc, in order.
    pub fn arcs(&self) -> impl DoubleEndedIterator<Item = ArcId> + ExactSizeIterator + use<V, C> {
        (0..self.arc_count()).map(ArcId::new)
    }

    /// Adds a node with the given supply (negative for demand).
    pub fn add_node(&mut self, supply: V) -> NodeId {
        self.supply.push(supply);
        NodeId::new(self.supply.len() - 1)
    }

    /// Adds an arc from `source` to `target` whose flow must lie between
    /// `lower` and `upper`.
    pub fn add_arc(
        &mut self,
        source: NodeId,
        target: NodeId,
        lower: V,
        upper: impl Into<Capacity<V>>,
        cost: C,
    ) -> ArcId {
        let n = self.node_count();
        assert!(
            source.index() < n,
            "arc source {source} out of range (node count {n})"
        );
        assert!(
            target.index() < n,
            "arc target {target} out of range (node count {n})"
        );
        self.source.push(NodeIx::new(source.index()));
        self.target.push(NodeIx::new(target.index()));
        self.lower.push(lower);
        self.upper.push(upper.into().to_raw());
        self.cost.push(cost);
        ArcId::new(self.source.len() - 1)
    }

    pub fn supply(&self, node: NodeId) -> V {
        self.supply[node.index()]
    }

    /// The supply of every node, indexed by [`NodeId::index`].
    pub fn supplies(&self) -> &[V] {
        &self.supply
    }

    pub fn set_supply(&mut self, node: NodeId, supply: V) {
        self.supply[node.index()] = supply;
    }

    /// Replaces all supplies with a single `amount` sent from `source` to
    /// `target`.
    pub fn set_st_supply(&mut self, source: NodeId, target: NodeId, amount: V) {
        self.supply.fill(V::zero());
        self.supply[source.index()] = amount;
        self.supply[target.index()] = -amount;
    }

    pub fn supply_type(&self) -> SupplyType {
        self.supply_type
    }

    pub fn set_supply_type(&mut self, supply_type: SupplyType) {
        self.supply_type = supply_type;
    }

    pub fn source(&self, arc: ArcId) -> NodeId {
        NodeId::new(self.source[arc.index()].index())
    }

    pub fn target(&self, arc: ArcId) -> NodeId {
        NodeId::new(self.target[arc.index()].index())
    }

    pub fn lower(&self, arc: ArcId) -> V {
        self.lower[arc.index()]
    }

    pub fn upper(&self, arc: ArcId) -> Capacity<V> {
        Capacity::from_raw(self.upper[arc.index()])
    }

    pub fn cost(&self, arc: ArcId) -> C {
        self.cost[arc.index()]
    }

    pub fn set_bounds(&mut self, arc: ArcId, lower: V, upper: impl Into<Capacity<V>>) {
        self.lower[arc.index()] = lower;
        self.upper[arc.index()] = upper.into().to_raw();
    }

    pub fn set_cost(&mut self, arc: ArcId, cost: C) {
        self.cost[arc.index()] = cost;
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
            return Err(Error::InvalidBounds {
                arc: ArcId::new(arc),
            });
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
    pub fn flow(&self, arc: ArcId) -> V {
        self.flow[arc.index()]
    }

    /// Flow on every arc, indexed by [`ArcId::index`].
    pub fn flows(&self) -> &[V] {
        &self.flow
    }

    pub fn potential(&self, node: NodeId) -> C {
        self.potential[node.index()]
    }

    /// Potential of every node, indexed by [`NodeId::index`].
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

    /// The flows and potentials, indexed like [`flows`](Self::flows) and
    /// [`potentials`](Self::potentials).
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
    InvalidBounds { arc: ArcId },
    /// The problem has more nodes or arcs than the solvers can index.
    #[error("the problem has too many nodes or arcs")]
    TooLarge,
    /// Cost scaling's internal costs (arc costs multiplied by the node count
    /// and scaling factor), or that multiplier itself, do not fit in its
    /// large cost type. Use a wider large cost type, such as
    /// `CostScaling<V, C, i128>`.
    #[error("scaled arc costs overflow the large cost type")]
    Overflow,
}
