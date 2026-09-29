//! Primal network simplex, ported from LEMON's `NetworkSimplex`.
//!
//! The spanning tree is stored with the "augmented thread index" (ATI)
//! representation: parent/predecessor links, a preorder thread list with its
//! reverse, and the size and last node of every subtree. An artificial root
//! node connects to every node by an artificial arc, so the initial tree is
//! always a feasible basis for the extended problem.

use std::hint::cold_path;
use std::ops::{ControlFlow, Range};

use crate::ivec::{IVec, NONE};
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

// Arc states. A non-tree arc's state is also the sign of the flow change
// that would decrease its reduced cost, which lets the pivot rules compute
// `state * reduced_cost` and look for negative values.
const STATE_UPPER: i8 = -1;
const STATE_TREE: i8 = 0;
const STATE_LOWER: i8 = 1;

// Direction of a tree arc relative to its child node.
const DIR_DOWN: i8 = -1;
const DIR_UP: i8 = 1;

/// Network simplex solver for minimum cost flow.
///
/// Supports negative costs, infinite capacities, lower bounds, and both
/// [`SupplyType`]s, and reports [`Error::Unbounded`] only when the objective
/// really is unbounded. The solver keeps its internal buffers between calls
/// to [`solve`](Self::solve), so reuse one instance for repeated solves.
///
/// ```
/// use silvermite::{NetworkSimplex, Problem};
///
/// let mut p = Problem::<i64, i64>::new(3);
/// p.set_st_supply(0, 2, 5);
/// p.add_arc(0, 1, 0, 4, 1);
/// p.add_arc(1, 2, 0, 4, 1);
/// p.add_arc(0, 2, 0, i64::MAX, 3);
///
/// let solution = NetworkSimplex::new().solve(&p).unwrap();
/// assert_eq!(solution.flows(), &[4, 4, 1]);
/// assert_eq!(solution.total_cost(), 11);
/// ```
#[derive(Clone, Debug)]
pub struct NetworkSimplex<V, C> {
    pivot_rule: PivotRule,
    arc_mixing: bool,

    node_num: u32,
    arc_num: u32,
    all_arc_num: u32,
    search_arc_num: u32,

    has_lower: bool,
    stype: SupplyType,
    sum_supply: V,

    // Internal arc index of each problem arc; differs from the identity when
    // arc mixing is on.
    arc_id: Vec<u32>,
    source: IVec<u32>,
    target: IVec<u32>,

    lower: IVec<V>,
    upper: IVec<V>,
    cap: IVec<V>,
    cost: IVec<C>,
    supply: IVec<V>,
    flow: IVec<V>,
    pi: IVec<C>,

    // Spanning tree
    parent: IVec<u32>,
    pred: IVec<u32>,
    thread: IVec<u32>,
    rev_thread: IVec<u32>,
    succ_num: IVec<u32>,
    last_succ: IVec<u32>,
    pred_dir: IVec<i8>,
    state: IVec<i8>,
    dirty_revs: Vec<u32>,
    root: u32,

    // Current pivot
    in_arc: u32,
    join: u32,
    u_in: u32,
    v_in: u32,
    u_out: u32,
    v_out: u32,
    delta: V,
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
            root: 0,
            in_arc: 0,
            join: 0,
            u_in: 0,
            v_in: 0,
            u_out: 0,
            v_out: 0,
            delta: V::zero(),
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
        if !self.init() {
            return Err(Error::Infeasible);
        }
        match self.pivot_rule {
            PivotRule::FirstEligible => self.start::<FirstEligible>()?,
            PivotRule::BestEligible => self.start::<BestEligible>()?,
            PivotRule::BlockSearch => self.start::<BlockSearch>()?,
            PivotRule::CandidateList => self.start::<CandidateList>()?,
            PivotRule::AlteringList => self.start::<AlteringList<C>>()?,
        }

        let flow: Vec<V> = self.arc_id.iter().map(|&i| self.flow[i]).collect();
        let potential = (*self.pi)[..n].to_vec();
        let total_cost = problem.total_cost(&flow);
        Ok(Solution {
            flow,
            potential,
            total_cost,
        })
    }

    fn load(&mut self, p: &Problem<V, C>) {
        let n = p.node_count() as u32;
        let m = p.arc_count() as u32;
        self.node_num = n;
        self.arc_num = m;
        let all_node_num = (n + 1) as usize;
        let max_arc_num = (m + 2 * n) as usize;

        self.source.reset(max_arc_num, 0);
        self.target.reset(max_arc_num, 0);
        self.lower.reset(m as usize, V::zero());
        self.upper.reset(m as usize, V::zero());
        self.cap.reset(max_arc_num, V::zero());
        self.cost.reset(max_arc_num, C::zero());
        self.supply.reset(all_node_num, V::zero());
        self.flow.reset(max_arc_num, V::zero());
        self.pi.reset(all_node_num, C::zero());

        self.parent.reset(all_node_num, 0);
        self.pred.reset(all_node_num, 0);
        self.pred_dir.reset(all_node_num, 0);
        self.thread.reset(all_node_num, 0);
        self.rev_thread.reset(all_node_num, 0);
        self.succ_num.reset(all_node_num, 0);
        self.last_succ.reset(all_node_num, 0);
        self.state.reset(max_arc_num, 0);

        self.arc_id.clear();
        self.arc_id.resize(m as usize, 0);
        if self.arc_mixing && n > 1 {
            // Deal arcs round-robin into `skip` interleaved runs.
            let skip = (m / n).max(3);
            let (mut i, mut j) = (0u32, 0u32);
            for a in 0..m as usize {
                self.arc_id[a] = i;
                i += skip;
                if i >= m {
                    j += 1;
                    i = j;
                }
            }
        } else {
            for (a, id) in self.arc_id.iter_mut().enumerate() {
                *id = a as u32;
            }
        }

        for a in 0..m as usize {
            let i = self.arc_id[a];
            self.source[i] = p.source[a];
            self.target[i] = p.target[a];
            self.lower[i] = p.lower[a];
            self.upper[i] = p.upper[a];
            self.cost[i] = p.cost[a];
        }
        (*self.supply)[..n as usize].copy_from_slice(&p.supply);
        self.has_lower = p.lower.iter().any(|&l| l != V::zero());
        self.stype = p.supply_type;
    }

    fn init(&mut self) -> bool {
        let n = self.node_num;
        let m = self.arc_num;
        let inf = V::max_value();
        let max = V::max_value();

        self.sum_supply = V::zero();
        for i in 0..n {
            self.sum_supply += self.supply[i];
        }
        let feasible_type = match self.stype {
            SupplyType::Geq => self.sum_supply <= V::zero(),
            SupplyType::Leq => self.sum_supply >= V::zero(),
        };
        if !feasible_type {
            return false;
        }

        // Remove non-zero lower bounds
        if self.has_lower {
            for i in 0..m {
                let c = self.lower[i];
                let upper = self.upper[i];
                self.cap[i] = if c >= V::zero() {
                    if upper < max { upper - c } else { inf }
                } else if upper < max + c {
                    upper - c
                } else {
                    inf
                };
                self.supply[self.source[i]] -= c;
                self.supply[self.target[i]] += c;
            }
        } else {
            for i in 0..m {
                self.cap[i] = self.upper[i];
            }
        }

        // Large enough that no optimal basis uses an artificial arc with
        // flow unless the problem is infeasible.
        let art_cost = C::max_value() / C::from_i8(2) + C::one();

        for i in 0..m {
            self.flow[i] = V::zero();
            self.state[i] = STATE_LOWER;
        }

        // Set data for the artificial root node
        let root = n;
        self.root = root;
        self.parent[root] = NONE;
        self.pred[root] = NONE;
        self.thread[root] = 0;
        self.rev_thread[0] = root;
        self.succ_num[root] = n + 1;
        self.last_succ[root] = root - 1;
        self.supply[root] = -self.sum_supply;
        self.pi[root] = C::zero();

        // Add artificial arcs and initialize the spanning tree
        if self.sum_supply == V::zero() {
            // EQ supply constraints
            self.search_arc_num = m;
            self.all_arc_num = m + n;
            for u in 0..n {
                let e = m + u;
                self.parent[u] = root;
                self.pred[u] = e;
                self.thread[u] = u + 1;
                self.rev_thread[u + 1] = u;
                self.succ_num[u] = 1;
                self.last_succ[u] = u;
                self.cap[e] = inf;
                self.state[e] = STATE_TREE;
                if self.supply[u] >= V::zero() {
                    self.pred_dir[u] = DIR_UP;
                    self.pi[u] = C::zero();
                    self.source[e] = u;
                    self.target[e] = root;
                    self.flow[e] = self.supply[u];
                    self.cost[e] = C::zero();
                } else {
                    self.pred_dir[u] = DIR_DOWN;
                    self.pi[u] = art_cost;
                    self.source[e] = root;
                    self.target[e] = u;
                    self.flow[e] = -self.supply[u];
                    self.cost[e] = art_cost;
                }
            }
        } else if self.sum_supply > V::zero() {
            // LEQ supply constraints
            self.search_arc_num = m + n;
            let mut f = m + n;
            for u in 0..n {
                let e = m + u;
                self.parent[u] = root;
                self.thread[u] = u + 1;
                self.rev_thread[u + 1] = u;
                self.succ_num[u] = 1;
                self.last_succ[u] = u;
                if self.supply[u] >= V::zero() {
                    self.pred_dir[u] = DIR_UP;
                    self.pi[u] = C::zero();
                    self.pred[u] = e;
                    self.source[e] = u;
                    self.target[e] = root;
                    self.cap[e] = inf;
                    self.flow[e] = self.supply[u];
                    self.cost[e] = C::zero();
                    self.state[e] = STATE_TREE;
                } else {
                    self.pred_dir[u] = DIR_DOWN;
                    self.pi[u] = art_cost;
                    self.pred[u] = f;
                    self.source[f] = root;
                    self.target[f] = u;
                    self.cap[f] = inf;
                    self.flow[f] = -self.supply[u];
                    self.cost[f] = art_cost;
                    self.state[f] = STATE_TREE;
                    self.source[e] = u;
                    self.target[e] = root;
                    self.cap[e] = inf;
                    self.flow[e] = V::zero();
                    self.cost[e] = C::zero();
                    self.state[e] = STATE_LOWER;
                    f += 1;
                }
            }
            self.all_arc_num = f;
        } else {
            // GEQ supply constraints
            self.search_arc_num = m + n;
            let mut f = m + n;
            for u in 0..n {
                let e = m + u;
                self.parent[u] = root;
                self.thread[u] = u + 1;
                self.rev_thread[u + 1] = u;
                self.succ_num[u] = 1;
                self.last_succ[u] = u;
                if self.supply[u] <= V::zero() {
                    self.pred_dir[u] = DIR_DOWN;
                    self.pi[u] = C::zero();
                    self.pred[u] = e;
                    self.source[e] = root;
                    self.target[e] = u;
                    self.cap[e] = inf;
                    self.flow[e] = -self.supply[u];
                    self.cost[e] = C::zero();
                    self.state[e] = STATE_TREE;
                } else {
                    self.pred_dir[u] = DIR_UP;
                    self.pi[u] = -art_cost;
                    self.pred[u] = f;
                    self.source[f] = u;
                    self.target[f] = root;
                    self.cap[f] = inf;
                    self.flow[f] = self.supply[u];
                    self.state[f] = STATE_TREE;
                    self.cost[f] = art_cost;
                    self.source[e] = root;
                    self.target[e] = u;
                    self.cap[e] = inf;
                    self.flow[e] = V::zero();
                    self.cost[e] = C::zero();
                    self.state[e] = STATE_LOWER;
                    f += 1;
                }
            }
            self.all_arc_num = f;
        }

        true
    }

    #[inline(always)]
    fn reduced_cost(&self, e: u32) -> C {
        C::from_i8(self.state[e])
            * (self.cost[e] + self.pi[self.source[e]] - self.pi[self.target[e]])
    }

    fn find_join_node(&mut self) {
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let succ_num = self.succ_num.as_ref();
        let parent = self.parent.as_ref();
        let mut u = source[self.in_arc];
        let mut v = target[self.in_arc];
        while u != v {
            if succ_num[u] < succ_num[v] {
                u = parent[u];
            } else {
                v = parent[v];
            }
        }
        self.join = u;
    }

    /// Finds the leaving arc of the cycle closed by the entering arc.
    /// Returns false if the entering arc itself leaves (it just flips
    /// between its bounds).
    fn find_leaving_arc(&mut self) -> bool {
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

        let in_arc = self.in_arc;
        let join = self.join;

        // Orient the cycle along the direction flow will be pushed.
        let (first, second) = if state[in_arc] == STATE_LOWER {
            (source[in_arc], target[in_arc])
        } else {
            (target[in_arc], source[in_arc])
        };
        let mut delta = cap[in_arc];
        let mut u_out = self.u_out;
        let mut result = 0;

        let mut u = first;
        while u != join {
            let e = pred[u];
            let mut d = flow[e];
            if pred_dir[u] == DIR_DOWN {
                let c = cap[e];
                d = if c >= max { inf } else { c - d };
            }
            if d < delta {
                delta = d;
                u_out = u;
                result = 1;
            }
            u = parent[u];
        }

        // `<=` here and `<` above pick the last blocking arc along the cycle
        // direction, which keeps the tree strongly feasible.
        let mut u = second;
        while u != join {
            let e = pred[u];
            let mut d = flow[e];
            if pred_dir[u] == DIR_UP {
                let c = cap[e];
                d = if c >= max { inf } else { c - d };
            }
            if d <= delta {
                delta = d;
                u_out = u;
                result = 2;
            }
            u = parent[u];
        }
        self.delta = delta;
        self.u_out = u_out;

        if result == 1 {
            self.u_in = first;
            self.v_in = second;
        } else {
            self.u_in = second;
            self.v_in = first;
        }
        result != 0
    }

    fn change_flow(&mut self, change: bool) {
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let pred = self.pred.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let parent = self.parent.as_ref();
        let mut state = self.state.as_mut();
        let mut flow = self.flow.as_mut();
        let in_arc = self.in_arc;
        let join = self.join;
        if self.delta > V::zero() {
            let val = V::from_i8(state[in_arc]) * self.delta;
            flow[in_arc] += val;
            let mut u = source[in_arc];
            while u != join {
                flow[pred[u]] -= V::from_i8(pred_dir[u]) * val;
                u = parent[u];
            }
            let mut u = target[in_arc];
            while u != join {
                flow[pred[u]] += V::from_i8(pred_dir[u]) * val;
                u = parent[u];
            }
        }
        if change {
            state[in_arc] = STATE_TREE;
            let out_arc = pred[self.u_out];
            state[out_arc] = if flow[out_arc] == V::zero() {
                STATE_LOWER
            } else {
                STATE_UPPER
            };
        } else {
            state[in_arc] = -state[in_arc];
        }
    }

    fn update_tree_structure(&mut self) {
        let source = self.source.as_ref();
        let mut parent = self.parent.as_mut();
        let mut pred = self.pred.as_mut();
        let mut pred_dir = self.pred_dir.as_mut();
        let mut thread = self.thread.as_mut();
        let mut rev_thread = self.rev_thread.as_mut();
        let mut succ_num = self.succ_num.as_mut();
        let mut last_succ = self.last_succ.as_mut();
        let dirty_revs = &mut self.dirty_revs;
        let u_in = self.u_in;
        let v_in = self.v_in;
        let u_out = self.u_out;
        let in_arc = self.in_arc;
        let join = self.join;

        let old_rev_thread = rev_thread[u_out];
        let old_succ_num = succ_num[u_out];
        let old_last_succ = last_succ[u_out];
        let v_out = parent[u_out];
        self.v_out = v_out;

        if u_in == u_out {
            // Update parent, pred, pred_dir
            parent[u_in] = v_in;
            pred[u_in] = in_arc;
            pred_dir[u_in] = if u_in == source[in_arc] {
                DIR_UP
            } else {
                DIR_DOWN
            };

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
            let mut p = parent[u];
            while u != u_in {
                pred[u] = pred[p];
                pred_dir[u] = -pred_dir[p];
                tmp_sc += succ_num[u] - succ_num[p];
                succ_num[u] = tmp_sc;
                last_succ[p] = tmp_ls;
                u = p;
                p = parent[u];
            }
            pred[u_in] = in_arc;
            pred_dir[u_in] = if u_in == source[in_arc] {
                DIR_UP
            } else {
                DIR_DOWN
            };
            succ_num[u_in] = old_succ_num;
        }

        // Update last_succ from v_in towards the root
        let up_limit_out = if last_succ[join] == v_in { join } else { NONE };
        let last_succ_out = last_succ[u_out];
        let mut u = v_in;
        while u != NONE && last_succ[u] == v_in {
            last_succ[u] = last_succ_out;
            u = parent[u];
        }

        // Update last_succ from v_out towards the root
        if join != old_rev_thread && v_in != old_rev_thread {
            let mut u = v_out;
            while u != up_limit_out && last_succ[u] == old_last_succ {
                last_succ[u] = old_rev_thread;
                u = parent[u];
            }
        } else if last_succ_out != old_last_succ {
            let mut u = v_out;
            while u != up_limit_out && last_succ[u] == old_last_succ {
                last_succ[u] = last_succ_out;
                u = parent[u];
            }
        }

        // Update succ_num from v_in to join
        let mut u = v_in;
        while u != join {
            succ_num[u] += old_succ_num;
            u = parent[u];
        }
        // Update succ_num from v_out to join
        let mut u = v_out;
        while u != join {
            succ_num[u] -= old_succ_num;
            u = parent[u];
        }
    }

    /// Shifts the potentials of the subtree that moved under v_in so the
    /// entering arc has zero reduced cost.
    fn update_potential(&mut self) {
        let cost = self.cost.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let thread = self.thread.as_ref();
        let last_succ = self.last_succ.as_ref();
        let mut pi = self.pi.as_mut();
        let u_in = self.u_in;
        let sigma = pi[self.v_in] - pi[u_in] - C::from_i8(pred_dir[u_in]) * cost[self.in_arc];
        let end = thread[last_succ[u_in]];
        let mut u = u_in;
        while u != end {
            pi[u] += sigma;
            u = thread[u];
        }
    }

    /// Performs one pivot on `self.in_arc`. Returns false if the cycle has
    /// infinite capacity, i.e. the problem is unbounded.
    #[inline]
    fn pivot(&mut self) -> bool {
        self.find_join_node();
        let change = self.find_leaving_arc();
        if self.delta >= V::max_value() {
            return false;
        }
        self.change_flow(change);
        if change {
            self.update_tree_structure();
            self.update_potential();
        }
        true
    }

    /// Heuristic warm start: pivots in arcs likely to carry flow in the
    /// optimum. Returns false if the problem turns out to be unbounded.
    fn initial_pivots(&mut self) -> bool {
        let n = self.node_num;
        let m = self.arc_num;

        let mut total = V::zero();
        let mut supply_nodes = Vec::new();
        let mut demand_nodes = Vec::new();
        for u in 0..n {
            let curr = self.supply[u];
            if curr > V::zero() {
                total += curr;
                supply_nodes.push(u);
            } else if curr < V::zero() {
                demand_nodes.push(u);
            }
        }
        if self.sum_supply > V::zero() {
            total -= self.sum_supply;
        }
        if total <= V::zero() {
            return true;
        }

        let mut arc_vector: Vec<u32> = Vec::new();
        if self.sum_supply >= V::zero() {
            if supply_nodes.len() == 1 && demand_nodes.len() == 1 {
                // Reverse DFS from the sink to the source over arcs that can
                // carry the whole amount.
                let mut in_first = vec![0u32; n as usize + 1];
                for j in 0..m {
                    in_first[self.target[j] as usize + 1] += 1;
                }
                for v in 0..n as usize {
                    in_first[v + 1] += in_first[v];
                }
                let mut fill = in_first.clone();
                let mut in_arcs = vec![0u32; m as usize];
                for j in 0..m {
                    let t = self.target[j] as usize;
                    in_arcs[fill[t] as usize] = j;
                    fill[t] += 1;
                }

                let mut reached = vec![false; n as usize];
                let s = supply_nodes[0];
                let t = demand_nodes[0];
                let mut stack = vec![t];
                reached[t as usize] = true;
                while let Some(v) = stack.pop() {
                    if v == s {
                        break;
                    }
                    let range = in_first[v as usize] as usize..in_first[v as usize + 1] as usize;
                    for &j in &in_arcs[range] {
                        let u = self.source[j];
                        if reached[u as usize] {
                            continue;
                        }
                        if self.cap[j] >= total {
                            arc_vector.push(j);
                            reached[u as usize] = true;
                            stack.push(u);
                        }
                    }
                }
            } else {
                // Find the min. cost incoming arc for each demand node
                let best = self.cheapest_arcs(|ns, j| ns.target[j]);
                arc_vector.extend(demand_nodes.iter().filter_map(|&v| best[v as usize]));
            }
        } else {
            // Find the min. cost outgoing arc for each supply node
            let best = self.cheapest_arcs(|ns, j| ns.source[j]);
            arc_vector.extend(supply_nodes.iter().filter_map(|&u| best[u as usize]));
        }

        for in_arc in arc_vector {
            self.in_arc = in_arc;
            if self.reduced_cost(in_arc) >= C::zero() {
                continue;
            }
            if !self.pivot() {
                return false;
            }
        }
        true
    }

    /// For each node, the cheapest original arc with that node as its
    /// `endpoint`.
    fn cheapest_arcs(&self, endpoint: impl Fn(&Self, u32) -> u32) -> Vec<Option<u32>> {
        let mut best: Vec<Option<u32>> = vec![None; self.node_num as usize];
        let mut best_cost = vec![C::max_value(); self.node_num as usize];
        for j in 0..self.arc_num {
            let v = endpoint(self, j) as usize;
            let c = self.cost[j];
            if c < best_cost[v] {
                best_cost[v] = c;
                best[v] = Some(j);
            }
        }
        best
    }

    fn start<P: Pivot<C>>(&mut self) -> Result<(), Error> {
        let mut pivot = P::new(self.search_arc_num as usize);

        if !self.initial_pivots() {
            return Err(Error::Unbounded);
        }

        loop {
            let view = PivotView {
                source: &self.source,
                target: &self.target,
                cost: &self.cost,
                state: &self.state,
                pi: &self.pi,
                search_arc_num: self.search_arc_num as usize,
            };
            let Some(in_arc) = pivot.find_entering_arc(&view) else {
                break;
            };
            self.in_arc = in_arc as u32;
            if !self.pivot() {
                return Err(Error::Unbounded);
            }
        }

        // Flow left on an artificial arc outside the search range means some
        // supply could not be routed.
        for e in self.search_arc_num..self.all_arc_num {
            if self.flow[e] != V::zero() {
                return Err(Error::Infeasible);
            }
        }

        // Transform the solution and the supply map to the original form
        if self.has_lower {
            for i in 0..self.arc_num {
                let c = self.lower[i];
                if c != V::zero() {
                    self.flow[i] += c;
                    self.supply[self.source[i]] += c;
                    self.supply[self.target[i]] -= c;
                }
            }
        }

        // Shift potentials to meet the sign requirements of the GEQ/LEQ
        // optimality conditions
        let pi = &mut (*self.pi)[..self.node_num as usize];
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

/// The parts of the solver state a pivot rule reads.
struct PivotView<'a, C> {
    source: &'a [u32],
    target: &'a [u32],
    cost: &'a [C],
    state: &'a [i8],
    pi: &'a [C],
    search_arc_num: usize,
}

impl<C: Number> PivotView<'_, C> {
    /// `state * reduced_cost`: negative exactly when the arc is eligible to
    /// enter the basis.
    #[inline(always)]
    fn eligibility(&self, e: usize) -> C {
        C::from_i8(self.state[e])
            * (self.cost[e] + self.pi[self.source[e] as usize] - self.pi[self.target[e] as usize])
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
            let c =
                C::from_i8(state) * (cost + self.pi[source as usize] - self.pi[target as usize]);
            f(range.start + i, c)?;
        }
        ControlFlow::Continue(())
    }
}

trait Pivot<C> {
    fn new(search_arc_num: usize) -> Self;
    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize>;
}

fn block_size(search_arc_num: usize) -> usize {
    const BLOCK_SIZE_FACTOR: f64 = 1.0;
    const MIN_BLOCK_SIZE: usize = 10;
    ((BLOCK_SIZE_FACTOR * (search_arc_num as f64).sqrt()) as usize).max(MIN_BLOCK_SIZE)
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
        const LIST_LENGTH_FACTOR: f64 = 0.25;
        const MIN_LIST_LENGTH: usize = 10;
        const MINOR_LIMIT_FACTOR: f64 = 0.1;
        const MIN_MINOR_LIMIT: usize = 3;

        let list_length =
            ((LIST_LENGTH_FACTOR * (search_arc_num as f64).sqrt()) as usize).max(MIN_LIST_LENGTH);
        let minor_limit = ((MINOR_LIMIT_FACTOR * list_length as f64) as usize).max(MIN_MINOR_LIMIT);
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
        const HEAD_LENGTH_FACTOR: f64 = 0.01;
        const MIN_HEAD_LENGTH: usize = 3;

        let block_size = block_size(search_arc_num);
        let head_length = ((HEAD_LENGTH_FACTOR * block_size as f64) as usize).max(MIN_HEAD_LENGTH);
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
        (*self.candidates)[..new_length].sort_unstable_by_key(|&e| cand_cost[e]);

        // Take the best as the entering arc and keep the rest of the head
        let in_arc = self.candidates[0];
        self.candidates[0] = self.candidates[new_length - 1];
        self.candidates.truncate(new_length - 1);
        Some(in_arc)
    }
}
