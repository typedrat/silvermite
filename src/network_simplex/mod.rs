//! Primal network simplex, ported from LEMON's `NetworkSimplex`.
//!
//! The spanning tree is stored with the "augmented thread index" (ATI)
//! representation: parent/predecessor links, a preorder thread list with its
//! reverse, and the size and last node of every subtree. An artificial root
//! node connects to every node by an artificial arc, so the initial tree is
//! always a feasible basis for the extended problem.

use alloc::vec::Vec;

use itertools::izip;

use self::pivot_rules::{
    AlteringList, BestEligible, BlockSearch, CandidateList, FirstEligible, Pivot, PivotView,
};
use crate::ivec::{ArcId, IVec, Idx, NodeId};
use crate::{Error, Number, Problem, Solution, SupplyType};

mod pivot_rules;
mod setup;
mod tree;
mod warm_start;

/// Strategy for choosing the entering arc in each simplex iteration.
///
/// [`BlockSearch`](PivotRule::BlockSearch) is the most robust choice in
/// LEMON's experiments and is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PivotRule {
    /// Cyclically scans the arcs and takes the first eligible one.
    FirstEligible,
    /// Takes the eligible arc with the most negative reduced cost. Every
    /// iteration scans all arcs.
    BestEligible,
    /// Scans blocks of about `sqrt(m)` arcs cyclically and takes the best
    /// eligible arc from the first block that contains one.
    #[default]
    BlockSearch,
    /// Builds a list of eligible arcs, then takes the best arc from it for
    /// several minor iterations before rebuilding.
    CandidateList,
    /// Keeps a few of the best eligible arcs from the previous candidate list
    /// and extends it by block search each iteration.
    AlteringList,
}

/// Where an arc sits in the current basis.
///
/// A non-tree arc's discriminant is also the sign of the flow change that
/// would decrease its reduced cost, which lets the pivot rules compute
/// `state * reduced_cost` and look for negative values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
enum ArcState {
    /// Out of the tree, at its upper bound.
    Upper = -1,
    Tree = 0,
    /// Out of the tree, at its lower bound.
    Lower = 1,
}

impl ArcState {
    #[inline(always)]
    fn sign<T: Number>(self) -> T {
        T::from_i8(self as i8)
    }

    /// The state of a non-tree arc after it moves to its other bound.
    fn flipped(self) -> Self {
        match self {
            ArcState::Upper => ArcState::Lower,
            ArcState::Lower => ArcState::Upper,
            ArcState::Tree => unreachable!("tree arcs have no bound to flip to"),
        }
    }
}

/// Orientation of a tree arc relative to its child node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
enum Dir {
    /// From the parent to the child.
    Down = -1,
    /// From the child to the parent.
    Up = 1,
}

impl Dir {
    #[inline(always)]
    fn sign<T: Number>(self) -> T {
        T::from_i8(self as i8)
    }

    fn reversed(self) -> Self {
        match self {
            Dir::Down => Dir::Up,
            Dir::Up => Dir::Down,
        }
    }
}

/// Network simplex solver for minimum cost flow.
///
/// Supports negative costs, infinite capacities, lower bounds, and both
/// [`SupplyType`]s, and reports [`Error::Unbounded`] only when the objective
/// really is unbounded. The solver keeps its internal buffers between calls
/// to [`solve`](Self::solve), so reuse one instance for repeated solves.
///
/// ```
/// use silvermite::{Capacity, NetworkSimplex, Problem};
///
/// let mut p = Problem::<i64, i64>::new(0);
/// let [s, a, t] = [5, 0, -5].map(|supply| p.add_node(supply));
/// p.add_arc(s, a, 0, 4, 1);
/// p.add_arc(a, t, 0, 4, 1);
/// let direct = p.add_arc(s, t, 0, Capacity::Infinite, 3);
///
/// let solution = NetworkSimplex::new().solve(&p).unwrap();
/// assert_eq!(solution.flows(), &[4, 4, 1]);
/// assert_eq!(solution.flow(direct), 1);
/// assert_eq!(solution.total_cost(), 11);
/// ```
#[derive(Clone, Debug)]
pub struct NetworkSimplex<V, C> {
    pivot_rule: PivotRule,
    arc_mixing: bool,

    node_num: usize,
    arc_num: usize,
    all_arc_num: usize,
    search_arc_num: usize,

    has_lower: bool,
    stype: SupplyType,
    sum_supply: V,

    // Internal arc index of each problem arc; differs from the identity when
    // arc mixing is on.
    arc_id: Vec<ArcId>,
    source: IVec<ArcId, NodeId>,
    target: IVec<ArcId, NodeId>,

    lower: IVec<ArcId, V>,
    upper: IVec<ArcId, V>,
    cap: IVec<ArcId, V>,
    cost: IVec<ArcId, C>,
    supply: IVec<NodeId, V>,
    flow: IVec<ArcId, V>,
    pi: IVec<NodeId, C>,

    // Spanning tree. The root is its own parent.
    parent: IVec<NodeId, NodeId>,
    pred: IVec<NodeId, ArcId>,
    thread: IVec<NodeId, NodeId>,
    rev_thread: IVec<NodeId, NodeId>,
    succ_num: IVec<NodeId, u32>,
    last_succ: IVec<NodeId, NodeId>,
    pred_dir: IVec<NodeId, Dir>,
    state: IVec<ArcId, ArcState>,
    dirty_revs: Vec<NodeId>,
    root: NodeId,
}

impl<V: Number, C: Number> Default for NetworkSimplex<V, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Number, C: Number> NetworkSimplex<V, C> {
    pub fn new() -> Self {
        NetworkSimplex {
            pivot_rule: PivotRule::default(),
            arc_mixing: true,
            node_num: 0,
            arc_num: 0,
            all_arc_num: 0,
            search_arc_num: 0,
            has_lower: false,
            stype: SupplyType::Geq,
            sum_supply: V::zero(),
            arc_id: Vec::new(),
            source: IVec::slot(0),
            target: IVec::slot(1),
            lower: IVec::slot(2),
            upper: IVec::slot(3),
            cap: IVec::slot(4),
            cost: IVec::slot(5),
            supply: IVec::slot(6),
            flow: IVec::slot(7),
            pi: IVec::slot(8),
            parent: IVec::slot(9),
            pred: IVec::slot(10),
            thread: IVec::slot(11),
            rev_thread: IVec::slot(12),
            succ_num: IVec::slot(13),
            last_succ: IVec::slot(14),
            pred_dir: IVec::slot(15),
            state: IVec::slot(16),
            dirty_revs: Vec::new(),
            root: NodeId::default(),
        }
    }

    pub fn pivot_rule(mut self, rule: PivotRule) -> Self {
        self.pivot_rule = rule;
        self
    }

    /// Whether to store arcs internally in an interleaved order rather than
    /// problem order (on by default).
    ///
    /// Mixing usually performs about the same as problem order but avoids
    /// bad cases where arcs out of the same node are clustered, and is
    /// sometimes much faster.
    pub fn arc_mixing(mut self, enabled: bool) -> Self {
        self.arc_mixing = enabled;
        self
    }

    /// Solves `problem`, returning an optimal flow and optimal potentials.
    pub fn solve(&mut self, problem: &Problem<V, C>) -> Result<Solution<V, C>, Error> {
        problem.validate(2)?;
        let n = problem.node_count();
        if n == 0 {
            return Ok(Solution {
                flow: Vec::new(),
                potential: Vec::new(),
                total_cost: 0,
            });
        }

        self.load(problem);
        self.init()?;
        match self.pivot_rule {
            PivotRule::FirstEligible => self.start::<FirstEligible>()?,
            PivotRule::BestEligible => self.start::<BestEligible>()?,
            PivotRule::BlockSearch => self.start::<BlockSearch>()?,
            PivotRule::CandidateList => self.start::<CandidateList>()?,
            PivotRule::AlteringList => self.start::<AlteringList<C>>()?,
        }

        let flow: Vec<V> = self.arc_id.iter().map(|&i| self.flow[i]).collect();
        let potential = self.pi[..self.root].to_vec();
        let total_cost = problem.total_cost(&flow);
        Ok(Solution {
            flow,
            potential,
            total_cost,
        })
    }

    #[inline(always)]
    fn reduced_cost(&self, e: ArcId) -> C {
        self.state[e].sign::<C>()
            * (self.cost[e] + self.pi[self.source[e]] - self.pi[self.target[e]])
    }

    fn start<P: Pivot<C>>(&mut self) -> Result<(), Error> {
        let mut pivot = P::new(self.search_arc_num);

        self.initial_pivots()?;

        loop {
            let view = PivotView {
                source: &self.source,
                target: &self.target,
                cost: &self.cost,
                state: &self.state,
                pi: self.pi.as_ref(),
                search_arc_num: self.search_arc_num,
            };
            let Some(in_arc) = pivot.find_entering_arc(&view) else {
                break;
            };
            self.pivot(ArcId::new(in_arc))?;
        }

        // Flow left on an artificial arc outside the search range means some
        // supply could not be routed.
        let unsearched = ArcId::new(self.search_arc_num)..ArcId::new(self.all_arc_num);
        if self.flow[unsearched].iter().any(|&f| f != V::zero()) {
            return Err(Error::Infeasible);
        }

        // Transform the solution and the supply map to the original form
        if self.has_lower {
            let arcs = ..ArcId::new(self.arc_num);
            for (flow, &c, &source, &target) in izip!(
                &mut self.flow[arcs],
                &self.lower[arcs],
                &self.source[arcs],
                &self.target[arcs],
            ) {
                if c != V::zero() {
                    *flow += c;
                    self.supply[source] += c;
                    self.supply[target] -= c;
                }
            }
        }

        // Shift potentials to meet the sign requirements of the GEQ/LEQ
        // optimality conditions
        let pi = &mut self.pi[..self.root];
        if self.sum_supply == V::zero() {
            match self.stype {
                SupplyType::Geq => {
                    let max_pot = pi.iter().copied().max().unwrap_or(C::zero());
                    if max_pot > C::zero() {
                        pi.iter_mut().for_each(|p| *p -= max_pot);
                    }
                }
                SupplyType::Leq => {
                    let min_pot = pi.iter().copied().min().unwrap_or(C::zero());
                    if min_pot < C::zero() {
                        pi.iter_mut().for_each(|p| *p -= min_pot);
                    }
                }
            }
        }

        Ok(())
    }
}
