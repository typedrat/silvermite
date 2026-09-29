//! Primal network simplex, ported from LEMON's `NetworkSimplex`.
//!
//! The spanning tree is stored with the "augmented thread index" (ATI)
//! representation: parent/predecessor links, a preorder thread list with its
//! reverse, and the size and last node of every subtree. An artificial root
//! node connects to every node by an artificial arc, so the initial tree is
//! always a feasible basis for the extended problem.

use alloc::vec;
use alloc::vec::Vec;
use core::hint::cold_path;
use core::iter;
use core::ops::{ControlFlow, Range};

use itertools::izip;

use crate::ivec::{ArcId, IRef, IVec, IdVec, Idx, NodeId, first_ids};
use crate::{Error, Number, Problem, Solution, SupplyType};

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

/// The tree arc a pivot removes, and how the entering arc replaces it.
#[derive(Clone, Copy)]
struct Exchange {
    /// The entering arc's endpoint in the subtree cut off by removing the
    /// leaving arc.
    u_in: NodeId,
    /// The entering arc's other endpoint, which becomes `u_in`'s parent.
    v_in: NodeId,
    /// The child end of the leaving arc.
    u_out: NodeId,
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

    fn load(&mut self, p: &Problem<V, C>) {
        let n = p.node_count();
        let m = p.arc_count();
        self.node_num = n;
        self.arc_num = m;
        let all_node_num = n + 1;
        let max_arc_num = m + 2 * n;

        self.source.reset(max_arc_num, NodeId::default());
        self.target.reset(max_arc_num, NodeId::default());
        self.lower.reset(m, V::zero());
        self.upper.reset(m, V::zero());
        self.cap.reset(max_arc_num, V::zero());
        self.cost.reset(max_arc_num, C::zero());
        self.supply.reset(all_node_num, V::zero());
        self.flow.reset(max_arc_num, V::zero());
        self.pi.reset(all_node_num, C::zero());

        self.parent.reset(all_node_num, NodeId::default());
        self.pred.reset(all_node_num, ArcId::default());
        self.pred_dir.reset(all_node_num, Dir::Up);
        self.thread.reset(all_node_num, NodeId::default());
        self.rev_thread.reset(all_node_num, NodeId::default());
        self.succ_num.reset(all_node_num, 0);
        self.last_succ.reset(all_node_num, NodeId::default());
        self.state.reset(max_arc_num, ArcState::Tree);

        self.arc_id.clear();
        if self.arc_mixing && n > 1 {
            // Deal arcs round-robin into `skip` interleaved runs.
            let skip = (m / n).max(3);
            self.arc_id
                .extend((0..skip).flat_map(|j| (j..m).step_by(skip)).map(ArcId::new));
        } else {
            self.arc_id.extend(first_ids::<ArcId>(m));
        }

        for (&i, &source, &target, &lower, &upper, &cost) in izip!(
            &self.arc_id,
            &p.source,
            &p.target,
            &p.lower,
            &p.upper,
            &p.cost
        ) {
            self.source[i] = source;
            self.target[i] = target;
            self.lower[i] = lower;
            self.upper[i] = upper;
            self.cost[i] = cost;
        }
        self.supply[..NodeId::new(n)].copy_from_slice(&p.supply);
        self.has_lower = p.lower.iter().any(|&l| l != V::zero());
        self.stype = p.supply_type;
    }

    fn init(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let m = self.arc_num;
        let arcs = ..ArcId::new(m);
        let root = NodeId::new(n);
        self.root = root;
        let inf = V::max_value();
        let max = V::max_value();

        self.sum_supply = self.supply[..root]
            .iter()
            .fold(V::zero(), |sum, &s| sum + s);
        let feasible_type = match self.stype {
            SupplyType::Geq => self.sum_supply <= V::zero(),
            SupplyType::Leq => self.sum_supply >= V::zero(),
        };
        if !feasible_type {
            return Err(Error::Infeasible);
        }

        // Remove non-zero lower bounds
        if self.has_lower {
            for (cap, &c, &upper, &source, &target) in izip!(
                &mut self.cap[arcs],
                &self.lower[arcs],
                &self.upper[arcs],
                &self.source[arcs],
                &self.target[arcs],
            ) {
                *cap = if c >= V::zero() {
                    if upper < max { upper - c } else { inf }
                } else if upper < max + c {
                    upper - c
                } else {
                    inf
                };
                self.supply[source] -= c;
                self.supply[target] += c;
            }
        } else {
            self.cap[arcs].copy_from_slice(&self.upper[arcs]);
        }

        // Large enough that no optimal basis uses an artificial arc with
        // flow unless the problem is infeasible.
        let art_cost = C::max_value() / C::from_i8(2) + C::one();

        self.flow[arcs].fill(V::zero());
        self.state[arcs].fill(ArcState::Lower);

        // Start from the tree where every node hangs directly off the
        // artificial root, in thread order 0, 1, ..., n - 1. The root has no
        // pred arc, so its pred entry is never read.
        let first = NodeId::new(0);
        self.parent.fill(root);
        self.succ_num.fill(1);
        self.succ_num[root] = n as u32 + 1;
        self.thread[root] = first;
        self.rev_thread[first] = root;
        self.last_succ[root] = root.prev();
        self.supply[root] = -self.sum_supply;
        self.pi[root] = C::zero();

        // Join each node to the root by an artificial tree arc carrying its
        // supply: upwards for supply, downwards for demand. Tree arcs in the
        // direction the supply constraints do not allow slack in cost
        // `art_cost`, so the optimum drains them. With GEQ/LEQ constraints,
        // each costly tree arc gets a free non-tree twin in the other
        // direction, which the pivots search to absorb the slack.
        let has_slack = self.sum_supply != V::zero();
        let costly = if self.sum_supply < V::zero() {
            Dir::Up
        } else {
            Dir::Down
        };
        self.search_arc_num = if has_slack { m + n } else { m };
        let mut next_extra = ArcId::new(m + n);
        for u in first_ids::<NodeId>(n) {
            self.thread[u] = u.next();
            self.rev_thread[u.next()] = u;
            self.last_succ[u] = u;

            let supply = self.supply[u];
            let dir = if supply > V::zero() || (supply == V::zero() && costly == Dir::Down) {
                Dir::Up
            } else {
                Dir::Down
            };
            let cost = if dir == costly { art_cost } else { C::zero() };
            let mut e = ArcId::new(m + u.index());
            if has_slack && dir == costly {
                self.set_artificial(e, u, dir.reversed(), V::zero(), C::zero(), ArcState::Lower);
                e = next_extra;
                next_extra = next_extra.next();
            }
            let flow = dir.sign::<V>() * supply;
            self.set_artificial(e, u, dir, flow, cost, ArcState::Tree);
            self.pred[u] = e;
            self.pred_dir[u] = dir;
            self.pi[u] = -dir.sign::<C>() * cost;
        }
        self.all_arc_num = if has_slack { next_extra.index() } else { m + n };

        Ok(())
    }

    /// Sets up the infinite-capacity artificial arc `e` between node `u` and
    /// the root, oriented `dir` relative to `u`.
    fn set_artificial(&mut self, e: ArcId, u: NodeId, dir: Dir, flow: V, cost: C, state: ArcState) {
        let (source, target) = match dir {
            Dir::Up => (u, self.root),
            Dir::Down => (self.root, u),
        };
        self.source[e] = source;
        self.target[e] = target;
        self.cap[e] = V::max_value();
        self.flow[e] = flow;
        self.cost[e] = cost;
        self.state[e] = state;
    }

    #[inline(always)]
    fn reduced_cost(&self, e: ArcId) -> C {
        self.state[e].sign::<C>()
            * (self.cost[e] + self.pi[self.source[e]] - self.pi[self.target[e]])
    }

    /// The nearest common ancestor of `in_arc`'s endpoints, where the cycle
    /// it closes in the tree turns around.
    fn find_join_node(&self, in_arc: ArcId) -> NodeId {
        let succ_num = self.succ_num.as_ref();
        let parent = self.parent.as_ref();
        let mut u = self.source[in_arc];
        let mut v = self.target[in_arc];
        // The root's subtree is the largest, so only a non-root node ever
        // steps up.
        while u != v {
            if succ_num[u] < succ_num[v] {
                u = parent[u];
            } else {
                v = parent[v];
            }
        }
        u
    }

    /// Finds how much flow can be pushed around the cycle `in_arc` closes,
    /// and which arc blocks it. The exchange is `None` when `in_arc` blocks
    /// itself, so it only moves to its other bound.
    fn find_leaving_arc(&self, in_arc: ArcId, join: NodeId) -> (V, Option<Exchange>) {
        let state = self.state.as_ref();
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let cap = self.cap.as_ref();
        let pred = self.pred.as_ref();
        let flow = self.flow.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let parent = self.parent.as_ref();
        let inf = V::max_value();
        let max = V::max_value();

        // Orient the cycle along the direction flow will be pushed.
        let (first, second) = if state[in_arc] == ArcState::Lower {
            (source[in_arc], target[in_arc])
        } else {
            (target[in_arc], source[in_arc])
        };
        let mut delta = cap[in_arc];
        let mut exchange = None;

        // How far flow can be pushed along the pred arc of `u` when the
        // cycle runs through it in direction `along`.
        let residual = |u: NodeId, along: Dir| {
            let e = pred[u];
            let d = flow[e];
            if pred_dir[u] == along {
                d
            } else {
                let c = cap[e];
                if c >= max { inf } else { c - d }
            }
        };

        for u in path_up(parent, first, join) {
            let d = residual(u, Dir::Up);
            if d < delta {
                delta = d;
                exchange = Some(Exchange {
                    u_in: first,
                    v_in: second,
                    u_out: u,
                });
            }
        }

        // `<=` here and `<` above pick the last blocking arc along the cycle
        // direction, which keeps the tree strongly feasible.
        for u in path_up(parent, second, join) {
            let d = residual(u, Dir::Down);
            if d <= delta {
                delta = d;
                exchange = Some(Exchange {
                    u_in: second,
                    v_in: first,
                    u_out: u,
                });
            }
        }

        (delta, exchange)
    }

    /// Pushes `delta` around the cycle and updates the arc states.
    fn change_flow(&mut self, in_arc: ArcId, join: NodeId, delta: V, exchange: Option<Exchange>) {
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let pred = self.pred.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let parent = self.parent.as_ref();
        let mut state = self.state.as_mut();
        let mut flow = self.flow.as_mut();
        if delta > V::zero() {
            let val = state[in_arc].sign::<V>() * delta;
            flow[in_arc] += val;
            for u in path_up(parent, source[in_arc], join) {
                flow[pred[u]] -= pred_dir[u].sign::<V>() * val;
            }
            for u in path_up(parent, target[in_arc], join) {
                flow[pred[u]] += pred_dir[u].sign::<V>() * val;
            }
        }
        match exchange {
            Some(Exchange { u_out, .. }) => {
                state[in_arc] = ArcState::Tree;
                let out_arc = pred[u_out];
                state[out_arc] = if flow[out_arc] == V::zero() {
                    ArcState::Lower
                } else {
                    ArcState::Upper
                };
            }
            None => state[in_arc] = state[in_arc].flipped(),
        }
    }

    /// Replaces the leaving arc with `in_arc` in the spanning tree.
    fn update_tree_structure(&mut self, in_arc: ArcId, join: NodeId, exchange: Exchange) {
        let source = self.source.as_ref();
        let mut parent = self.parent.as_mut();
        let mut pred = self.pred.as_mut();
        let mut pred_dir = self.pred_dir.as_mut();
        let mut thread = self.thread.as_mut();
        let mut rev_thread = self.rev_thread.as_mut();
        let mut succ_num = self.succ_num.as_mut();
        let mut last_succ = self.last_succ.as_mut();
        let dirty_revs = &mut self.dirty_revs;
        let Exchange { u_in, v_in, u_out } = exchange;

        let old_rev_thread = rev_thread[u_out];
        let old_succ_num = succ_num[u_out];
        let old_last_succ = last_succ[u_out];
        let v_out = parent[u_out];
        let in_dir = if u_in == source[in_arc] {
            Dir::Up
        } else {
            Dir::Down
        };

        if u_in == u_out {
            // Update parent, pred, pred_dir
            parent[u_in] = v_in;
            pred[u_in] = in_arc;
            pred_dir[u_in] = in_dir;

            // Update thread and rev_thread
            if thread[v_in] != u_out {
                let mut after = thread[old_last_succ];
                thread[old_rev_thread] = after;
                rev_thread[after] = old_rev_thread;
                after = thread[v_in];
                thread[v_in] = u_out;
                rev_thread[u_out] = v_in;
                thread[old_last_succ] = after;
                rev_thread[after] = old_last_succ;
            }
        } else {
            // When old_rev_thread == v_in, join and v_out coincide too.
            let thread_continue = if old_rev_thread == v_in {
                thread[old_last_succ]
            } else {
                thread[v_in]
            };

            // Re-hang the stem (the path from u_in up to u_out) under v_in,
            // reversing parent links and splicing each stem node's subtree
            // into the thread after its new parent's.
            let mut stem = u_in;
            let mut par_stem = v_in;
            let mut last = last_succ[u_in];
            let mut after = thread[last];
            thread[v_in] = u_in;
            dirty_revs.clear();
            dirty_revs.push(v_in);
            while stem != u_out {
                // Insert the next stem node into the thread list
                let next_stem = parent[stem];
                thread[last] = next_stem;
                dirty_revs.push(last);

                // Remove the subtree of stem from the thread list
                let before = rev_thread[stem];
                thread[before] = after;
                rev_thread[after] = before;

                // Change the parent node and shift stem nodes
                parent[stem] = par_stem;
                par_stem = stem;
                stem = next_stem;

                // Update last and after
                last = if last_succ[stem] == last_succ[par_stem] {
                    rev_thread[par_stem]
                } else {
                    last_succ[stem]
                };
                after = thread[last];
            }
            parent[u_out] = par_stem;
            thread[last] = thread_continue;
            rev_thread[thread_continue] = last;
            last_succ[u_out] = last;

            // Remove the subtree of u_out from the thread list, unless
            // old_rev_thread == v_in, where it is already in place.
            if old_rev_thread != v_in {
                thread[old_rev_thread] = after;
                rev_thread[after] = old_rev_thread;
            }

            // Update rev_thread using the new thread values
            for &u in dirty_revs.iter() {
                rev_thread[thread[u]] = u;
            }

            // Update pred, pred_dir, last_succ and succ_num for the stem
            // nodes from u_out to u_in
            let mut tmp_sc = 0u32;
            let tmp_ls = last_succ[u_out];
            let mut u = u_out;
            while u != u_in {
                let p = parent[u];
                pred[u] = pred[p];
                pred_dir[u] = pred_dir[p].reversed();
                tmp_sc += succ_num[u] - succ_num[p];
                succ_num[u] = tmp_sc;
                last_succ[p] = tmp_ls;
                u = p;
            }
            pred[u_in] = in_arc;
            pred_dir[u_in] = in_dir;
            succ_num[u_in] = old_succ_num;
        }
        let parent = parent.as_ref();

        // Update last_succ from v_in towards the root
        let up_limit_out = (last_succ[join] == v_in).then_some(join);
        let last_succ_out = last_succ[u_out];
        for u in ancestors(parent, v_in) {
            if last_succ[u] != v_in {
                break;
            }
            last_succ[u] = last_succ_out;
        }

        // Update last_succ from v_out towards the root
        let new_last_succ = if join != old_rev_thread && v_in != old_rev_thread {
            Some(old_rev_thread)
        } else {
            (last_succ_out != old_last_succ).then_some(last_succ_out)
        };
        if let Some(new_last_succ) = new_last_succ {
            for u in ancestors(parent, v_out).take_while(|&u| Some(u) != up_limit_out) {
                if last_succ[u] != old_last_succ {
                    break;
                }
                last_succ[u] = new_last_succ;
            }
        }

        // Update succ_num from v_in and from v_out to join
        for u in path_up(parent, v_in, join) {
            succ_num[u] += old_succ_num;
        }
        for u in path_up(parent, v_out, join) {
            succ_num[u] -= old_succ_num;
        }
    }

    /// Shifts the potentials of the subtree that moved under v_in so the
    /// entering arc has zero reduced cost.
    fn update_potential(&mut self, in_arc: ArcId, exchange: Exchange) {
        let cost = self.cost.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let thread = self.thread.as_ref();
        let last_succ = self.last_succ.as_ref();
        let mut pi = self.pi.as_mut();
        let Exchange { u_in, v_in, .. } = exchange;
        let sigma = pi[v_in] - pi[u_in] - pred_dir[u_in].sign::<C>() * cost[in_arc];
        let end = thread[last_succ[u_in]];
        let mut u = u_in;
        while u != end {
            pi[u] += sigma;
            u = thread[u];
        }
    }

    /// Pivots `in_arc` into the basis. Fails if the cycle it closes has
    /// infinite capacity, i.e. the problem is unbounded.
    #[inline]
    fn pivot(&mut self, in_arc: ArcId) -> Result<(), Error> {
        let join = self.find_join_node(in_arc);
        let (delta, exchange) = self.find_leaving_arc(in_arc, join);
        if delta >= V::max_value() {
            return Err(Error::Unbounded);
        }
        self.change_flow(in_arc, join, delta, exchange);
        if let Some(exchange) = exchange {
            self.update_tree_structure(in_arc, join, exchange);
            self.update_potential(in_arc, exchange);
        }
        Ok(())
    }

    /// Heuristic warm start: pivots in arcs likely to carry flow in the
    /// optimum. Fails if the problem turns out to be unbounded.
    fn initial_pivots(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let m = self.arc_num;
        let arcs = ..ArcId::new(m);

        let nodes_where = |keep: fn(V) -> bool| -> Vec<NodeId> {
            first_ids(n).filter(|&u| keep(self.supply[u])).collect()
        };
        let supply_nodes = nodes_where(|s| s > V::zero());
        let demand_nodes = nodes_where(|s| s < V::zero());
        let mut total = supply_nodes
            .iter()
            .fold(V::zero(), |sum, &u| sum + self.supply[u]);
        if self.sum_supply > V::zero() {
            total -= self.sum_supply;
        }
        if total <= V::zero() {
            return Ok(());
        }

        let mut arc_vector: Vec<ArcId> = Vec::new();
        if self.sum_supply >= V::zero() {
            if let ([s], [t]) = (&supply_nodes[..], &demand_nodes[..]) {
                // Reverse DFS from the sink to the source over arcs that can
                // carry the whole amount. The incoming arcs of `v` are
                // `in_arcs[in_first[v]..in_first[v.next()]]`.
                let mut in_first = IdVec::<NodeId, usize>::filled(n + 1, 0);
                for &t in &self.target[arcs] {
                    in_first[t.next()] += 1;
                }
                let mut sum = 0;
                for count in in_first.iter_mut() {
                    sum += *count;
                    *count = sum;
                }
                let mut fill = in_first.clone();
                let mut in_arcs = vec![ArcId::default(); m];
                for (j, &t) in first_ids::<ArcId>(m).zip(&self.target[arcs]) {
                    in_arcs[fill[t]] = j;
                    fill[t] += 1;
                }

                let mut reached = IdVec::<NodeId, bool>::filled(n, false);
                let mut stack = vec![*t];
                reached[*t] = true;
                while let Some(v) = stack.pop() {
                    if v == *s {
                        break;
                    }
                    for &j in &in_arcs[in_first[v]..in_first[v.next()]] {
                        let u = self.source[j];
                        if !reached[u] && self.cap[j] >= total {
                            arc_vector.push(j);
                            reached[u] = true;
                            stack.push(u);
                        }
                    }
                }
            } else {
                // Find the min. cost incoming arc for each demand node
                let best = self.cheapest_arcs(&self.target[arcs]);
                arc_vector.extend(demand_nodes.iter().filter_map(|&v| best[v]));
            }
        } else {
            // Find the min. cost outgoing arc for each supply node
            let best = self.cheapest_arcs(&self.source[arcs]);
            arc_vector.extend(supply_nodes.iter().filter_map(|&u| best[u]));
        }

        for in_arc in arc_vector {
            if self.reduced_cost(in_arc) < C::zero() {
                self.pivot(in_arc)?;
            }
        }
        Ok(())
    }

    /// For each node, the cheapest original arc with that node as its
    /// endpoint in `endpoints`.
    fn cheapest_arcs(&self, endpoints: &[NodeId]) -> IdVec<NodeId, Option<ArcId>> {
        let mut best = IdVec::filled(self.node_num, None);
        let mut best_cost = IdVec::filled(self.node_num, C::max_value());
        for (j, &v, &c) in izip!(first_ids::<ArcId>(endpoints.len()), endpoints, &*self.cost) {
            if best[v].is_none() || c < best_cost[v] {
                best[v] = Some(j);
                best_cost[v] = c;
            }
        }
        best
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

/// The nodes from `from` up to, but not including, its ancestor `to`.
#[inline(always)]
fn path_up(
    parent: IRef<'_, NodeId, NodeId>,
    from: NodeId,
    to: NodeId,
) -> impl Iterator<Item = NodeId> + '_ {
    let mut u = from;
    iter::from_fn(move || {
        if u == to {
            return None;
        }
        let here = u;
        u = parent[u];
        Some(here)
    })
}

/// The nodes from `from` up to and including the root.
#[inline(always)]
fn ancestors(parent: IRef<'_, NodeId, NodeId>, from: NodeId) -> impl Iterator<Item = NodeId> + '_ {
    iter::successors(Some(from), move |&u| Some(parent[u]).filter(|&p| p != u))
}

/// The parts of the solver state a pivot rule reads.
struct PivotView<'a, C> {
    source: &'a [NodeId],
    target: &'a [NodeId],
    cost: &'a [C],
    state: &'a [ArcState],
    pi: IRef<'a, NodeId, C>,
    search_arc_num: usize,
}

impl<C: Number> PivotView<'_, C> {
    /// `state * reduced_cost`: negative exactly when the arc is eligible to
    /// enter the basis.
    #[inline(always)]
    fn eligibility(&self, e: usize) -> C {
        self.state[e].sign::<C>()
            * (self.cost[e] + self.pi[self.source[e]] - self.pi[self.target[e]])
    }

    /// Calls `f(e, eligibility(e))` for each arc in `range`, in order,
    /// stopping early with the value `f` breaks with.
    ///
    /// Zipping the arc arrays lets the compiler drop their bounds checks,
    /// which are most of the per-arc cost of the scan. Callers mark their
    /// "new minimum" branch with `cold_path`: it is rarely taken, and as a
    /// conditional move it would chain every iteration on the previous
    /// minimum, which costs up to 20% on scan-heavy inputs.
    #[inline(always)]
    fn scan<B>(
        &self,
        range: Range<usize>,
        mut f: impl FnMut(usize, C) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let arcs = self.state[range.clone()]
            .iter()
            .zip(&self.cost[range.clone()])
            .zip(&self.source[range.clone()])
            .zip(&self.target[range.clone()]);
        for (i, (((&state, &cost), &source), &target)) in arcs.enumerate() {
            let c = state.sign::<C>() * (cost + self.pi[source] - self.pi[target]);
            f(range.start + i, c)?;
        }
        ControlFlow::Continue(())
    }
}

trait Pivot<C> {
    fn new(search_arc_num: usize) -> Self;
    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize>;
}

// LEMON sizes its lists by floating-point factors of `sqrt(m)`; the integer
// forms below compute exactly the same values without `std`.

/// About `sqrt(m)`.
fn block_size(search_arc_num: usize) -> usize {
    const MIN_BLOCK_SIZE: usize = 10;
    search_arc_num.isqrt().max(MIN_BLOCK_SIZE)
}

struct FirstEligible {
    next_arc: usize,
}

impl<C: Number> Pivot<C> for FirstEligible {
    fn new(_: usize) -> Self {
        FirstEligible { next_arc: 0 }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let n = view.search_arc_num;
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    ControlFlow::Break(e)
                } else {
                    ControlFlow::Continue(())
                }
            });
            if let ControlFlow::Break(e) = found {
                self.next_arc = e + 1;
                return Some(e);
            }
        }
        None
    }
}

struct BestEligible;

impl<C: Number> Pivot<C> for BestEligible {
    fn new(_: usize) -> Self {
        BestEligible
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let mut min = C::zero();
        let mut in_arc = None;
        let _ = view.scan(0..view.search_arc_num, |e, c| {
            if c < min {
                cold_path();
                min = c;
                in_arc = Some(e);
            }
            ControlFlow::<()>::Continue(())
        });
        in_arc
    }
}

struct BlockSearch {
    block_size: usize,
    next_arc: usize,
}

impl<C: Number> Pivot<C> for BlockSearch {
    fn new(search_arc_num: usize) -> Self {
        BlockSearch {
            block_size: block_size(search_arc_num),
            next_arc: 0,
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let n = view.search_arc_num;
        let mut min = C::zero();
        let mut in_arc = 0;
        let mut cnt = self.block_size;
        let block_size = self.block_size;
        // Scan from next_arc, wrapping around, and stop at the end of the
        // first block that contains an eligible arc.
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < min {
                    cold_path();
                    min = c;
                    in_arc = e;
                }
                cnt -= 1;
                if cnt == 0 {
                    if min < C::zero() {
                        return ControlFlow::Break(e);
                    }
                    cnt = block_size;
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                self.next_arc = e;
                return Some(in_arc);
            }
        }
        (min < C::zero()).then_some(in_arc)
    }
}

struct CandidateList {
    candidates: Vec<usize>,
    list_length: usize,
    minor_limit: usize,
    minor_count: usize,
    next_arc: usize,
}

impl<C: Number> Pivot<C> for CandidateList {
    fn new(search_arc_num: usize) -> Self {
        const MIN_LIST_LENGTH: usize = 10;
        const MIN_MINOR_LIMIT: usize = 3;

        // A quarter of `sqrt(m)` long, rebuilt after a tenth as many minor
        // iterations.
        let list_length = (search_arc_num.isqrt() / 4).max(MIN_LIST_LENGTH);
        let minor_limit = (list_length / 10).max(MIN_MINOR_LIMIT);
        CandidateList {
            candidates: Vec::with_capacity(list_length),
            list_length,
            minor_limit,
            minor_count: 0,
            next_arc: 0,
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let mut in_arc = 0;
        if !self.candidates.is_empty() && self.minor_count < self.minor_limit {
            // Minor iteration: select the best eligible arc from the current
            // candidate list, dropping arcs that are no longer eligible.
            self.minor_count += 1;
            let mut min = C::zero();
            let mut i = 0;
            while i < self.candidates.len() {
                let e = self.candidates[i];
                let c = view.eligibility(e);
                if c < min {
                    cold_path();
                    min = c;
                    in_arc = e;
                } else if c >= C::zero() {
                    self.candidates.swap_remove(i);
                    continue;
                }
                i += 1;
            }
            if min < C::zero() {
                return Some(in_arc);
            }
        }

        // Major iteration: build a new candidate list
        let n = view.search_arc_num;
        let mut min = C::zero();
        self.candidates.clear();
        let (candidates, list_length) = (&mut self.candidates, self.list_length);
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    candidates.push(e);
                    if c < min {
                        cold_path();
                        min = c;
                        in_arc = e;
                    }
                    if candidates.len() == list_length {
                        return ControlFlow::Break(e);
                    }
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                self.minor_count = 1;
                self.next_arc = e;
                return Some(in_arc);
            }
        }
        if self.candidates.is_empty() {
            return None;
        }
        self.minor_count = 1;
        Some(in_arc)
    }
}

struct AlteringList<C> {
    block_size: usize,
    head_length: usize,
    next_arc: usize,
    candidates: Vec<usize>,
    cand_cost: Vec<C>,
}

impl<C: Number> Pivot<C> for AlteringList<C> {
    fn new(search_arc_num: usize) -> Self {
        const MIN_HEAD_LENGTH: usize = 3;

        // Keeps a hundredth of a block between iterations.
        let block_size = block_size(search_arc_num);
        let head_length = (block_size / 100).max(MIN_HEAD_LENGTH);
        AlteringList {
            block_size,
            head_length,
            next_arc: 0,
            candidates: Vec::with_capacity(head_length + block_size),
            cand_cost: vec![C::zero(); search_arc_num],
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        // Refresh the kept candidates, dropping ineligible ones
        let mut i = 0;
        while i < self.candidates.len() {
            let e = self.candidates[i];
            let c = view.eligibility(e);
            if c < C::zero() {
                self.cand_cost[e] = c;
                i += 1;
            } else {
                self.candidates.swap_remove(i);
            }
        }

        // Extend the list block by block. The first block must add more
        // than head_length candidates to stop; later blocks, any at all.
        let n = view.search_arc_num;
        let mut cnt = self.block_size;
        let mut limit = self.head_length;
        let mut stop_at = None;
        let (candidates, cand_cost, block_size) =
            (&mut self.candidates, &mut self.cand_cost, self.block_size);
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    cand_cost[e] = c;
                    candidates.push(e);
                }
                cnt -= 1;
                if cnt == 0 {
                    if candidates.len() > limit {
                        return ControlFlow::Break(e);
                    }
                    limit = 0;
                    cnt = block_size;
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                stop_at = Some(e);
                break;
            }
        }
        if stop_at.is_none() && self.candidates.is_empty() {
            return None;
        }
        if let Some(e) = stop_at {
            self.next_arc = e;
        }

        // Move the best head_length + 1 candidates to the front, best first
        let new_length = (self.head_length + 1).min(self.candidates.len());
        let cand_cost = &self.cand_cost;
        if new_length < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by_key(new_length - 1, |&e| cand_cost[e]);
        }
        self.candidates[..new_length].sort_unstable_by_key(|&e| cand_cost[e]);

        // Take the best as the entering arc and keep the rest of the head
        self.candidates.truncate(new_length);
        Some(self.candidates.swap_remove(0))
    }
}
