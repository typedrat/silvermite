//! Cost scaling push/augment-relabel, ported from LEMON's `CostScaling`.
//!
//! Costs are multiplied by `(n + 1) * alpha` so that epsilon-optimality with
//! `epsilon = 1` implies exact optimality, and `epsilon` is divided by
//! `alpha` each phase. Every phase restores epsilon-optimality by pushing
//! excess along admissible (negative reduced cost) residual arcs and
//! relabeling nodes that have none. Two heuristics carry most of the
//! performance: price refinement, which can finish a phase without any
//! flow change, and global updates, which relabel nodes in bulk by a
//! bucket-based shortest path search towards deficit nodes.

use std::collections::VecDeque;

use itertools::izip;

use crate::circulation::{Layout, circulation};
use crate::ivec::{ArcId, IMut, IVec, IdVec, Idx, Link, NodeId, first_ids, ids};
use crate::{Error, Number, Problem, Solution, SupplyType};

/// The flow-moving operation used alongside relabeling.
///
/// [`PartialAugment`](Method::PartialAugment) was the fastest and most
/// robust in LEMON's experiments and is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Method {
    /// Local pushes: flow moves along one admissible arc at a time.
    Push,
    /// Flow moves along admissible paths from an excess node all the way to
    /// a deficit node.
    Augment,
    /// Flow moves along admissible paths of at most four arcs.
    #[default]
    PartialAugment,
}

/// Cost scaling solver for minimum cost flow.
///
/// `L` is the "large cost" type used internally for scaled costs and
/// potentials; its range must cover the largest absolute arc cost times
/// `(node_count + 1) * scaling_factor`. [`solve`](Self::solve) returns
/// [`Error::Overflow`] when it does not, in which case
/// `CostScaling::<V, C, i128>::default()` is the usual fix.
///
/// Arcs with negative cost and infinite capacity are not supported and make
/// [`solve`](Self::solve) return [`Error::Unbounded`], even when the objective
/// is bounded over the feasible flows. [`NetworkSimplex`](crate::NetworkSimplex)
/// handles those.
///
/// The solver keeps its internal buffers between calls to
/// [`solve`](Self::solve), so reuse one instance for repeated solves.
///
/// ```
/// use silvermite::{CostScaling, Problem};
///
/// let mut p = Problem::<i64, i64>::new(3);
/// p.set_st_supply(0, 2, 5);
/// p.add_arc(0, 1, 0, 4, 1);
/// p.add_arc(1, 2, 0, 4, 1);
/// p.add_arc(0, 2, 0, i64::MAX, 3);
///
/// let solution = CostScaling::new().solve(&p).unwrap();
/// assert_eq!(solution.flows(), &[4, 4, 1]);
/// assert_eq!(solution.total_cost(), 11);
/// ```
#[derive(Clone, Debug)]
pub struct CostScaling<V, C, L = i64> {
    method: Method,
    alpha: u32,

    node_num: usize,
    res_node_num: usize,
    res_arc_num: usize,
    root: NodeId,
    // Whether the problem was mirrored to turn LEQ constraints into GEQ.
    mirrored: bool,

    has_lower: bool,
    sum_supply: V,
    sup_node_num: usize,

    // Forward and backward residual arc of each problem arc
    arc_idf: Vec<ArcId>,
    arc_idb: Vec<ArcId>,
    // Node `u`'s arc block is `first_out[u]..first_out[u.next()]`.
    first_out: IVec<NodeId, ArcId>,
    forward: IVec<ArcId, bool>,
    source: IVec<ArcId, NodeId>,
    target: IVec<ArcId, NodeId>,
    reverse: IVec<ArcId, ArcId>,

    lower: IVec<ArcId, V>,
    upper: IVec<ArcId, V>,
    scost: IVec<ArcId, C>,
    supply: IVec<NodeId, V>,

    res_cap: IVec<ArcId, V>,
    // Forward arcs whose capacity was infinite before being bounded by the
    // total deficit.
    uncapped: IVec<ArcId, bool>,
    cost: IVec<ArcId, L>,
    pi: IVec<NodeId, L>,
    excess: IVec<NodeId, V>,
    next_out: IVec<NodeId, ArcId>,
    active_nodes: VecDeque<NodeId>,

    epsilon: L,

    // Bucket list heads, indexed by rank
    buckets: IVec<usize, Link<NodeId>>,
    bucket_next: IVec<NodeId, Link<NodeId>>,
    bucket_prev: IVec<NodeId, NodeId>,
    rank: IVec<NodeId, u32>,
    max_rank: u32,
}

impl<V: Number, C: Number> CostScaling<V, C> {
    /// Creates a solver with `i64` large costs. For another large cost type,
    /// use `CostScaling::<V, C, L>::default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V: Number, C: Number, L: Number> Default for CostScaling<V, C, L> {
    fn default() -> Self {
        CostScaling {
            method: Method::default(),
            alpha: 16,
            node_num: 0,
            res_node_num: 0,
            res_arc_num: 0,
            root: NodeId::default(),
            mirrored: false,
            has_lower: false,
            sum_supply: V::zero(),
            sup_node_num: 0,
            arc_idf: Vec::new(),
            arc_idb: Vec::new(),
            first_out: IVec::slot(0),
            forward: IVec::slot(1),
            source: IVec::slot(2),
            target: IVec::slot(3),
            reverse: IVec::slot(4),
            lower: IVec::slot(5),
            upper: IVec::slot(6),
            scost: IVec::slot(7),
            supply: IVec::slot(8),
            res_cap: IVec::slot(9),
            uncapped: IVec::slot(10),
            cost: IVec::slot(11),
            pi: IVec::slot(12),
            excess: IVec::slot(13),
            next_out: IVec::slot(14),
            active_nodes: VecDeque::new(),
            epsilon: L::zero(),
            buckets: IVec::slot(15),
            bucket_next: IVec::slot(16),
            bucket_prev: IVec::slot(17),
            rank: IVec::slot(18),
            max_rank: 0,
        }
    }
}

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Sets the factor epsilon is divided by in each phase (16 by default).
    ///
    /// # Panics
    ///
    /// Panics if `factor < 2`.
    pub fn scaling_factor(mut self, factor: u32) -> Self {
        assert!(factor >= 2, "the scaling factor must be at least 2");
        self.alpha = factor;
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
        if (n as u64 + 1) * self.alpha as u64 >= u32::MAX as u64 {
            return Err(Error::TooLarge);
        }

        self.load(problem);
        self.init()?;
        self.start();

        let flow: Vec<V> = self.arc_idb.iter().map(|&b| self.res_cap[b]).collect();
        let potential: Vec<C> = self.pi[..self.root]
            .iter()
            .map(|&p| {
                let p: C = p.cast();
                if self.mirrored { -p } else { p }
            })
            .collect();
        let total_cost = problem.total_cost(&flow);
        Ok(Solution {
            flow,
            potential,
            total_cost,
        })
    }

    /// Builds the residual graph and copies the problem data.
    ///
    /// Each real node's arc block holds its outgoing arcs (forward), its
    /// incoming arcs (backward), and a backward arc to the artificial root.
    /// The root's block holds the matching forward arcs to every node.
    fn load(&mut self, p: &Problem<V, C>) {
        let n = p.node_count();
        let m = p.arc_count();
        self.node_num = n;
        self.res_node_num = n + 1;
        self.res_arc_num = 2 * (m + n);
        let root = NodeId::new(n);
        self.root = root;
        let res_node_num = self.res_node_num;
        let res_arc_num = self.res_arc_num;

        // Mirroring every arc and negating supplies turns
        // `out - in <= supply` into `out - in >= -supply`.
        self.mirrored = p.supply_type == SupplyType::Leq;
        let (src, tgt) = if self.mirrored {
            (&p.target, &p.source)
        } else {
            (&p.source, &p.target)
        };

        self.first_out.reset(res_node_num + 1, ArcId::default());
        self.forward.reset(res_arc_num, false);
        self.source.reset(res_arc_num, NodeId::default());
        self.target.reset(res_arc_num, NodeId::default());
        self.reverse.reset(res_arc_num, ArcId::default());
        self.lower.reset(res_arc_num, V::zero());
        self.upper.reset(res_arc_num, V::max_value());
        self.scost.reset(res_arc_num, C::zero());
        self.supply.reset(res_node_num, V::zero());
        self.res_cap.reset(res_arc_num, V::zero());
        self.uncapped.reset(res_arc_num, false);
        self.cost.reset(res_arc_num, L::zero());
        self.pi.reset(res_node_num, L::zero());
        self.excess.reset(res_node_num, V::zero());
        self.next_out.reset(res_node_num, ArcId::default());

        // Block sizes, then block starts. `out_pos` and `in_pos` track the
        // next free forward and backward slot in each block.
        let mut outs = IdVec::<NodeId, usize>::filled(n, 0);
        let mut ins = IdVec::<NodeId, usize>::filled(n, 0);
        for (&s, &t) in src.iter().zip(tgt) {
            outs[s] += 1;
            ins[t] += 1;
        }
        let mut out_pos = IdVec::<NodeId, ArcId>::filled(n, ArcId::default());
        let mut in_pos = IdVec::<NodeId, ArcId>::filled(n, ArcId::default());
        let mut j = 0;
        for (first_out, out_pos, in_pos, &outs, &ins) in izip!(
            &mut *self.first_out,
            &mut *out_pos,
            &mut *in_pos,
            &*outs,
            &*ins,
        ) {
            *first_out = ArcId::new(j);
            *out_pos = ArcId::new(j);
            *in_pos = ArcId::new(j + outs);
            j += outs + ins + 1;
        }
        self.first_out[root] = ArcId::new(j);
        self.first_out[root.next()] = ArcId::new(res_arc_num);

        self.arc_idf.clear();
        self.arc_idb.clear();
        for (&s, &t, &lower, &upper, &cost) in izip!(src, tgt, &p.lower, &p.upper, &p.cost) {
            let f = out_pos[s];
            out_pos[s] = f.next();
            let b = in_pos[t];
            in_pos[t] = b.next();
            self.arc_idf.push(f);
            self.arc_idb.push(b);
            self.forward[f] = true;
            self.source[f] = s;
            self.target[f] = t;
            self.reverse[f] = b;
            self.source[b] = t;
            self.target[b] = s;
            self.reverse[b] = f;

            self.lower[f] = lower;
            self.upper[f] = upper;
            self.scost[f] = cost;
            self.scost[b] = -cost;
        }

        for (i, k) in first_ids::<NodeId>(n).zip(self.block(root)) {
            let j = self.first_out[i.next()].prev();
            self.source[j] = i;
            self.target[j] = root;
            self.reverse[j] = k;
            self.forward[k] = true;
            self.source[k] = root;
            self.target[k] = i;
            self.reverse[k] = j;
        }

        for (supply, &s) in self.supply.iter_mut().zip(&p.supply) {
            *supply = if self.mirrored { -s } else { s };
        }
        self.has_lower = p.lower.iter().any(|&l| l != V::zero());
    }

    fn init(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let root = self.root;
        let res_node_num = self.res_node_num;
        let max = V::max_value();

        self.sum_supply = self.supply[..root]
            .iter()
            .fold(V::zero(), |sum, &s| sum + s);
        if self.sum_supply > V::zero() {
            return Err(Error::Infeasible);
        }

        self.pi.fill(L::zero());
        self.excess.copy_from_slice(&self.supply);

        // Reject infinite capacities on negative arcs, and bound the total
        // flow any arc can carry to replace the remaining infinite ones.
        for &f in &self.arc_idf {
            let c = if self.scost[f] < C::zero() {
                self.upper[f]
            } else if self.has_lower {
                self.lower[f]
            } else {
                continue;
            };
            if c >= max {
                return Err(Error::Unbounded);
            }
            self.excess[self.source[f]] -= c;
            self.excess[self.target[f]] += c;
        }
        let max_cap = self
            .excess
            .iter()
            .filter(|&&ex| ex < V::zero())
            .fold(V::zero(), |sum, &ex| sum - ex);
        self.excess.fill(V::zero());
        // Relative to its lower bound (upper bound for negative arcs), no
        // arc needs to carry more than the total deficit in some optimal
        // flow, so that bounds every infinite arc above its lower bound.
        let root_arcs = self.first_out[root];
        for (j, upper, &lower, &forward, uncapped) in izip!(
            first_ids::<ArcId>(self.res_arc_num),
            &mut *self.upper,
            &*self.lower,
            &*self.forward,
            &mut *self.uncapped,
        ) {
            if *upper >= max {
                *uncapped = forward && j < root_arcs;
                *upper = lower.saturating_add(max_cap);
            }
        }

        // Scale the costs and set the initial epsilon
        let real_arcs = ..root_arcs;
        let alpha = self.alpha;
        let scale = res_node_num as i128 * alpha as i128;
        let max_abs_cost = self.scost[real_arcs]
            .iter()
            .map(|&c| c.to_i128().abs())
            .max()
            .unwrap_or(0);
        if max_abs_cost
            .checked_mul(scale)
            .is_none_or(|c| c > L::max_value().to_i128())
        {
            return Err(Error::Overflow);
        }
        let scale = L::from_i128(scale);
        for (cost, &scost) in self.cost[real_arcs].iter_mut().zip(&self.scost[real_arcs]) {
            *cost = scost.cast::<L>() * scale;
        }
        let max_cost = self.cost[real_arcs].iter().copied().max();
        self.epsilon = max_cost.unwrap_or(L::zero()).max(L::zero()) / L::from_i128(alpha as i128);

        // Find a feasible flow with lower bounds shifted to zero
        let mut cap = IdVec::<ArcId, V>::filled(self.res_arc_num, V::zero());
        let mut sup = IdVec::<NodeId, V>::filled(n, V::zero());
        sup.copy_from_slice(&self.supply[..root]);
        for &f in &self.arc_idf {
            let c = if self.has_lower {
                self.lower[f]
            } else {
                V::zero()
            };
            cap[f] = self.upper[f] - c;
            sup[self.source[f]] -= c;
            sup[self.target[f]] += c;
        }
        self.sup_node_num = sup.iter().filter(|&&s| s > V::zero()).count();

        let mut flow = IdVec::<ArcId, V>::filled(self.res_arc_num, V::zero());
        let layout = Layout {
            node_num: n,
            first_out: self.first_out.as_ref(),
            forward: self.forward.as_ref(),
            target: self.target.as_ref(),
            reverse: self.reverse.as_ref(),
        };
        circulation(
            &layout,
            &self.arc_idf,
            cap.as_ref(),
            sup.as_ref(),
            flow.as_mut(),
        )?;

        // Set residual capacities; with a supply surplus to absorb, route the
        // leftover deficits through the root.
        for (&f, &b) in self.arc_idf.iter().zip(&self.arc_idb) {
            let fa = flow[f];
            self.res_cap[f] = cap[f] - fa;
            self.res_cap[b] = fa;
            if self.sum_supply < V::zero() {
                sup[self.source[f]] -= fa;
                sup[self.target[f]] += fa;
            }
        }
        if self.sum_supply < V::zero() {
            self.excess[..root].copy_from_slice(&sup);
            for a in self.block(root) {
                let u = self.target[a];
                let ra = self.reverse[a];
                self.res_cap[a] = -self.sum_supply + V::one();
                self.res_cap[ra] = -self.excess[u];
                self.cost[a] = L::zero();
                self.cost[ra] = L::zero();
                self.excess[u] = V::zero();
            }
        } else {
            for a in self.block(root) {
                let ra = self.reverse[a];
                self.res_cap[a] = V::zero();
                self.res_cap[ra] = V::zero();
                self.cost[a] = L::zero();
                self.cost[ra] = L::zero();
            }
        }

        self.max_rank = alpha * res_node_num as u32;
        self.buckets.reset(self.max_rank as usize, Link::NONE);
        self.bucket_next.reset(res_node_num, Link::NONE);
        self.bucket_prev.reset(res_node_num, NodeId::default());
        self.rank.reset(res_node_num, 0);

        Ok(())
    }

    /// The arcs out of `u` in the residual graph.
    #[inline(always)]
    fn block(&self, u: NodeId) -> impl DoubleEndedIterator<Item = ArcId> + Clone + use<V, C, L> {
        ids(self.first_out[u]..self.first_out[u.next()])
    }

    /// Runs the scaling phases, then turns the potentials into an exact
    /// dual solution for the original costs.
    fn start(&mut self) {
        const MAX_PARTIAL_PATH_LENGTH: usize = 4;

        match self.method {
            Method::Push => self.start_push(),
            Method::Augment => self.start_augment(self.res_node_num - 1),
            Method::PartialAugment => self.start_augment(MAX_PARTIAL_PATH_LENGTH),
        }

        // Unscale the potentials, truncating like LEMON's conversion to the
        // cost type.
        let scale = L::from_i128(self.res_node_num as i128 * self.alpha as i128);
        for pi in self.pi.iter_mut() {
            *pi = (*pi / scale).cast::<C>().cast();
        }

        // Rounding can break exact optimality; repair it with shortest
        // paths in the residual graph if so. Originally infinite arcs count
        // as open even when saturated at their finite stand-in capacity, or
        // the potentials would not certify optimality for the real problem.
        let optimal = first_ids::<NodeId>(self.res_node_num).all(|i| {
            self.block(i).all(|j| {
                !self.is_open(j)
                    || self.scost[j].cast::<L>() + self.pi[i] - self.pi[self.target[j]] >= L::zero()
            })
        });
        if !optimal {
            let dist = self.bellman_ford();
            for (pi, &d) in self.pi.iter_mut().zip(&*dist) {
                *pi += d;
            }
        }

        // Shift potentials to meet the requirements of the GEQ type
        // optimality conditions
        let max_pot = self.pi.iter().copied().max().unwrap_or(L::zero());
        if max_pot != L::zero() {
            for p in self.pi.iter_mut() {
                *p -= max_pot;
            }
        }

        // Handle non-zero lower bounds
        if self.has_lower {
            for (&f, &b) in self.arc_idf.iter().zip(&self.arc_idb) {
                self.res_cap[b] += self.lower[f];
            }
        }
    }

    #[inline]
    fn is_open(&self, j: ArcId) -> bool {
        self.res_cap[j] > V::zero() || self.uncapped[j]
    }

    /// Shortest path distances in the residual graph under the reduced
    /// original costs, from a virtual source joined to every node by a
    /// zero-length arc.
    fn bellman_ford(&self) -> IdVec<NodeId, L> {
        let n = self.res_node_num;
        let mut dist = IdVec::filled(n, L::zero());
        let mut mask = IdVec::filled(n, true);
        let mut process: Vec<NodeId> = first_ids(n).collect();
        let mut next = Vec::new();
        for _ in 0..n.saturating_sub(1) {
            for &u in &process {
                mask[u] = false;
            }
            for &u in &process {
                let pi_u = self.pi[u];
                for j in self.block(u) {
                    if !self.is_open(j) {
                        continue;
                    }
                    let v = self.target[j];
                    let w = self.scost[j].cast::<L>() + pi_u - self.pi[v];
                    let relaxed = dist[u] + w;
                    if relaxed < dist[v] {
                        dist[v] = relaxed;
                        if !mask[v] {
                            mask[v] = true;
                            next.push(v);
                        }
                    }
                }
            }
            std::mem::swap(&mut process, &mut next);
            next.clear();
            if process.is_empty() {
                break;
            }
        }
        dist
    }

    /// Starts a phase: saturates every arc that violates epsilon-optimality
    /// and queues the nodes left with excess.
    fn init_phase(&mut self) {
        let pi = self.pi.as_ref();
        let target = self.target.as_ref();
        let cost = self.cost.as_ref();
        let reverse = self.reverse.as_ref();
        let first_out = self.first_out.as_ref();
        let mut excess = self.excess.as_mut();
        let mut res_cap = self.res_cap.as_mut();
        let mut next_out = self.next_out.as_mut();
        let active_nodes = &mut self.active_nodes;
        let res_node_num = self.res_node_num;
        for u in first_ids::<NodeId>(res_node_num) {
            let pi_u = pi[u];
            for a in ids(first_out[u]..first_out[u.next()]) {
                let delta = res_cap[a];
                if delta > V::zero() {
                    let v = target[a];
                    if cost[a] + pi_u - pi[v] < L::zero() {
                        excess[u] -= delta;
                        excess[v] += delta;
                        res_cap[a] = V::zero();
                        res_cap[reverse[a]] += delta;
                    }
                }
            }
        }

        active_nodes.extend(first_ids::<NodeId>(res_node_num).filter(|&u| excess[u] > V::zero()));
        next_out.copy_from_slice(&first_out[..NodeId::new(res_node_num)]);
    }

    /// Price refinement heuristic: tries to make the current flow
    /// epsilon-optimal by changing potentials only. Returns true if it
    /// succeeded, in which case the phase needs no flow changes.
    fn price_refinement(&mut self) -> bool {
        let res_node_num = self.res_node_num;
        let mut order = Vec::with_capacity(res_node_num);

        while self.topological_sort(&mut order) {
            let res_cap = self.res_cap.as_ref();
            let target = self.target.as_ref();
            let cost = self.cost.as_ref();
            let first_out = self.first_out.as_ref();
            let mut rank = self.rank.as_mut();
            let mut pi = self.pi.as_mut();
            let mut buckets = Buckets {
                first: self.buckets.as_mut(),
                next: self.bucket_next.as_mut(),
                prev: self.bucket_prev.as_mut(),
            };
            let (epsilon, max_rank) = (self.epsilon, self.max_rank);
            // Compute node ranks in the acyclic admissible network and store
            // the nodes in buckets
            rank.fill(0);
            buckets.first.fill(Link::NONE);
            let mut top_rank = 0u32;
            for &u in order.iter().rev() {
                let rank_u = rank[u];
                let pi_u = pi[u];
                for a in ids(first_out[u]..first_out[u.next()]) {
                    if res_cap[a] > V::zero() {
                        let v = target[a];
                        let rc = cost[a] + pi_u - pi[v];
                        if rc < L::zero() {
                            // floor((-rc - 0.5) / epsilon), computed exactly
                            let nrc = (-rc - L::one()) / epsilon;
                            if nrc < L::from_i128(max_rank as i128) {
                                let new_rank_v = rank_u as u64 + nrc.to_i128() as u64;
                                // Ranks past the bucket range only arise when
                                // the flow is far from epsilon-optimal; the
                                // heuristic cannot help then, so leave the
                                // phase to the regular algorithm.
                                if new_rank_v >= max_rank as u64 {
                                    return false;
                                }
                                let new_rank_v = new_rank_v as u32;
                                if new_rank_v > rank[v] {
                                    rank[v] = new_rank_v;
                                }
                            }
                        }
                    }
                }

                if rank_u > 0 {
                    top_rank = top_rank.max(rank_u);
                    buckets.link(u, rank_u);
                }
            }

            // The current flow is epsilon-optimal
            if top_rank == 0 {
                return true;
            }

            // Process buckets in top-down order
            for level in (1..=top_rank).rev() {
                while let Some(u) = buckets.pop(level) {
                    let pi_u = pi[u];
                    for a in ids(first_out[u]..first_out[u.next()]) {
                        if res_cap[a] <= V::zero() {
                            continue;
                        }
                        let v = target[a];
                        let old_rank_v = rank[v];
                        if old_rank_v >= level {
                            continue;
                        }

                        // Compute the new rank of node v
                        let rc = cost[a] + pi_u - pi[v];
                        let new_rank_v: i64 = if rc < L::zero() {
                            level as i64
                        } else {
                            let nrc = rc / epsilon;
                            if nrc < L::from_i128(max_rank as i128) {
                                level as i64 - 1 - nrc.to_i128() as i64
                            } else {
                                0
                            }
                        };

                        // Move v to its new bucket
                        if new_rank_v > old_rank_v as i64 {
                            let new_rank_v = new_rank_v as u32;
                            rank[v] = new_rank_v;
                            if old_rank_v > 0 {
                                buckets.unlink(v, old_rank_v);
                            }
                            buckets.link(v, new_rank_v);
                        }
                    }

                    // Refine potential of node u
                    pi[u] -= L::from_i128(level as i128) * epsilon;
                }
            }
        }

        false
    }

    /// Topologically sorts the admissible network by DFS into `order`,
    /// sources last. If the DFS finds an admissible cycle instead, cancels it
    /// and returns false.
    fn topological_sort(&mut self, order: &mut Vec<NodeId>) -> bool {
        let pi = self.pi.as_ref();
        let target = self.target.as_ref();
        let cost = self.cost.as_ref();
        let reverse = self.reverse.as_ref();
        let first_out = self.first_out.as_ref();
        let mut res_cap = self.res_cap.as_mut();
        let mut next_out = self.next_out.as_mut();
        const MAX_CYCLE_CANCEL: usize = 1;

        let n = self.res_node_num;
        let mut reached = IdVec::<NodeId, bool>::filled(n, false);
        let mut processed = IdVec::<NodeId, bool>::filled(n, false);
        let mut pred = IdVec::<NodeId, Link<NodeId>>::filled(n, Link::NONE);
        next_out.copy_from_slice(&first_out[..NodeId::new(n)]);
        order.clear();
        let pred_of = |pred: &IdVec<NodeId, Link<NodeId>>, u: NodeId| {
            pred[u].get().expect("DFS tree nodes have a pred")
        };

        let mut cycle_cnt = 0usize;
        for start in first_ids::<NodeId>(n) {
            if reached[start] {
                continue;
            }

            // Start DFS search from this start node
            pred[start] = Link::NONE;
            let mut tip = start;
            loop {
                // Check the outgoing arcs of the current tip node
                reached[tip] = true;
                let pi_tip = pi[tip];
                let mut a = next_out[tip];
                let mut last_out = first_out[tip.next()];
                while a != last_out {
                    if res_cap[a] > V::zero() {
                        let v = target[a];
                        if cost[a] + pi_tip - pi[v] < L::zero() {
                            if !reached[v] {
                                // A new node is reached
                                reached[v] = true;
                                pred[v] = Link::to(tip);
                                next_out[tip] = a;
                                tip = v;
                                a = next_out[tip];
                                last_out = first_out[tip.next()];
                                break;
                            } else if !processed[v] {
                                // A cycle is found
                                cycle_cnt += 1;
                                next_out[tip] = a;

                                // Find the minimum residual capacity along
                                // the cycle
                                let mut delta = res_cap[a];
                                let mut delta_node = tip;
                                let mut u = tip;
                                while u != v {
                                    u = pred_of(&pred, u);
                                    let d = res_cap[next_out[u]];
                                    if d <= delta {
                                        delta = d;
                                        delta_node = u;
                                    }
                                }

                                // Augment along the cycle
                                res_cap[a] -= delta;
                                res_cap[reverse[a]] += delta;
                                let mut u = tip;
                                while u != v {
                                    u = pred_of(&pred, u);
                                    let ca = next_out[u];
                                    res_cap[ca] -= delta;
                                    res_cap[reverse[ca]] += delta;
                                }

                                if cycle_cnt >= MAX_CYCLE_CANCEL {
                                    return false;
                                }

                                // Roll back search to delta_node
                                if delta_node != tip {
                                    let mut u = tip;
                                    while u != delta_node {
                                        reached[u] = false;
                                        u = pred_of(&pred, u);
                                    }
                                    tip = delta_node;
                                    a = next_out[tip].next();
                                    last_out = first_out[tip.next()];
                                    break;
                                }
                            }
                        }
                    }
                    a = a.next();
                }

                // Step back to the previous node
                if a == last_out {
                    processed[tip] = true;
                    order.push(tip);
                    let Some(p) = pred[tip].get() else {
                        break;
                    };
                    tip = p;
                    next_out[tip] = next_out[tip].next();
                }
            }
        }

        cycle_cnt == 0
    }

    /// Global update heuristic: relabels nodes by their reduced-cost
    /// distance (in units of epsilon) to the nearest deficit node, using a
    /// bucket-based Dijkstra over reversed residual arcs.
    fn global_update(&mut self) {
        let max_rank = self.max_rank;
        let epsilon = self.epsilon;
        let res_node_num = self.res_node_num;
        let first_out = self.first_out.as_ref();
        let reverse = self.reverse.as_ref();
        let source = self.source.as_ref();
        let res_cap = self.res_cap.as_ref();
        let cost = self.cost.as_ref();
        let excess = self.excess.as_ref();
        let mut pi = self.pi.as_mut();
        let mut next_out = self.next_out.as_mut();
        let mut rank = self.rank.as_mut();
        let mut buckets = Buckets {
            first: self.buckets.as_mut(),
            next: self.bucket_next.as_mut(),
            prev: self.bucket_prev.as_mut(),
        };

        buckets.first.fill(Link::NONE);
        let mut total_excess = V::zero();
        for i in first_ids::<NodeId>(res_node_num) {
            if excess[i] < V::zero() {
                rank[i] = 0;
                buckets.link(i, 0);
            } else {
                total_excess += excess[i];
                rank[i] = max_rank;
            }
        }
        if total_excess == V::zero() {
            return;
        }

        // Search the buckets
        let max_rank_l = L::from_i128(max_rank as i128);
        let mut r = 0u32;
        while r != max_rank {
            while let Some(u) = buckets.pop(r) {
                // Search the incoming arcs of u
                let pi_u = pi[u];
                for a in ids(first_out[u]..first_out[u.next()]) {
                    let ra = reverse[a];
                    if res_cap[ra] <= V::zero() {
                        continue;
                    }
                    let v = source[ra];
                    let old_rank_v = rank[v];
                    if r < old_rank_v {
                        let nrc = (cost[ra] + pi[v] - pi_u) / epsilon;
                        let mut new_rank_v = old_rank_v;
                        if nrc < max_rank_l {
                            new_rank_v = (r as i128 + 1 + nrc.to_i128()) as u32;
                        }

                        if new_rank_v < old_rank_v {
                            rank[v] = new_rank_v;
                            next_out[v] = first_out[v];
                            if old_rank_v < max_rank {
                                buckets.unlink(v, old_rank_v);
                            }
                            buckets.link(v, new_rank_v);
                        }
                    }
                }

                // Finish search if there are no more active nodes
                if excess[u] > V::zero() {
                    total_excess -= excess[u];
                    if total_excess <= V::zero() {
                        break;
                    }
                }
            }
            if total_excess <= V::zero() {
                break;
            }
            r += 1;
        }

        // Relabel nodes
        for (pi, next_out, &rank, &first_out) in
            izip!(&mut *pi, &mut *next_out, &*rank, &*first_out)
        {
            let k = rank.min(r);
            if k > 0 {
                *pi -= epsilon * L::from_i128(k as i128);
                *next_out = first_out;
            }
        }
    }

    /// Number of relabels between global updates. Counted in 64 bits: the
    /// quadratic term alone overflows a 32-bit `usize` on large inputs.
    fn global_update_interval(&self, factor: f64) -> u64 {
        let sup = self.sup_node_num as u64;
        (factor * (self.res_node_num as u64).saturating_add(sup.saturating_mul(sup)) as f64) as u64
    }

    #[inline]
    fn next_epsilon(&self) -> L {
        let alpha = L::from_i128(self.alpha as i128);
        if self.epsilon < alpha && self.epsilon > L::one() {
            L::one()
        } else {
            self.epsilon / alpha
        }
    }

    /// Runs the phases with (partial) augment and relabel operations.
    fn start_augment(&mut self, max_length: usize) {
        const PRICE_REFINEMENT_LIMIT: usize = 2;
        const GLOBAL_UPDATE_FACTOR: f64 = 1.0;
        let global_update_skip = self.global_update_interval(GLOBAL_UPDATE_FACTOR);
        let mut next_global_update_limit = global_update_skip;

        let mut path: Vec<ArcId> = Vec::new();
        let mut path_arc = IdVec::filled(self.res_arc_num, false);
        let mut relabel_cnt = 0u64;
        let mut eps_phase_cnt = 0usize;
        while self.epsilon >= L::one() {
            eps_phase_cnt += 1;

            if eps_phase_cnt >= PRICE_REFINEMENT_LIMIT && self.price_refinement() {
                self.epsilon = self.next_epsilon();
                continue;
            }

            self.init_phase();
            while !self.augment_until(
                max_length,
                &mut path,
                &mut path_arc,
                &mut relabel_cnt,
                next_global_update_limit,
            ) {
                self.global_update();
                next_global_update_limit =
                    next_global_update_limit.saturating_add(global_update_skip);
            }

            self.epsilon = self.next_epsilon();
        }
    }

    /// Performs (partial) augment and relabel steps until no node has
    /// excess, returning true, or until `relabel_cnt` reaches
    /// `relabel_limit`, returning false.
    fn augment_until(
        &mut self,
        max_length: usize,
        path: &mut Vec<ArcId>,
        path_arc: &mut IdVec<ArcId, bool>,
        relabel_cnt: &mut u64,
        relabel_limit: u64,
    ) -> bool {
        let epsilon = self.epsilon;
        let first_out = self.first_out.as_ref();
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let reverse = self.reverse.as_ref();
        let cost = self.cost.as_ref();
        let mut res_cap = self.res_cap.as_mut();
        let mut pi = self.pi.as_mut();
        let mut excess = self.excess.as_mut();
        let mut next_out = self.next_out.as_mut();
        let active_nodes = &mut self.active_nodes;

        loop {
            // Select an active node (FIFO selection)
            while let Some(&front) = active_nodes.front() {
                if excess[front] > V::zero() {
                    break;
                }
                active_nodes.pop_front();
            }
            let Some(&start) = active_nodes.front() else {
                return true;
            };

            // Find an augmenting path from the start node
            let mut tip = start;
            'path: while path.len() < max_length && excess[tip] >= V::zero() {
                let mut min_red_cost = L::max_value();
                let pi_tip = pi[tip];
                let last_out = first_out[tip.next()];
                for a in ids(next_out[tip]..last_out) {
                    if res_cap[a] > V::zero() {
                        let u = target[a];
                        let rc = cost[a] + pi_tip - pi[u];
                        if rc < L::zero() {
                            path.push(a);
                            next_out[tip] = a;
                            if path_arc[a] {
                                // A cycle is found, stop path search
                                break 'path;
                            }
                            tip = u;
                            path_arc[a] = true;
                            continue 'path;
                        } else if rc < min_red_cost {
                            min_red_cost = rc;
                        }
                    }
                }

                // Relabel tip node
                if tip != start {
                    let ra = reverse[*path.last().unwrap()];
                    min_red_cost = min_red_cost.min(cost[ra] + pi_tip - pi[target[ra]]);
                }
                for a in ids(first_out[tip]..next_out[tip]) {
                    if res_cap[a] > V::zero() {
                        let rc = cost[a] + pi_tip - pi[target[a]];
                        if rc < min_red_cost {
                            min_red_cost = rc;
                        }
                    }
                }
                pi[tip] -= min_red_cost + epsilon;
                next_out[tip] = first_out[tip];
                *relabel_cnt += 1;

                // Step back
                if tip != start {
                    let pa = path.pop().unwrap();
                    path_arc[pa] = false;
                    tip = source[pa];
                }
            }

            // Augment along the found path (as much flow as possible)
            let mut v = start;
            for &pa in path.iter() {
                let u = v;
                v = target[pa];
                path_arc[pa] = false;
                let delta = res_cap[pa].min(excess[u]);
                res_cap[pa] -= delta;
                res_cap[reverse[pa]] += delta;
                excess[u] -= delta;
                excess[v] += delta;
                if excess[v] > V::zero() && excess[v] <= delta {
                    active_nodes.push_back(v);
                }
            }
            path.clear();

            if *relabel_cnt >= relabel_limit {
                return false;
            }
        }
    }

    /// Runs the phases with push and relabel operations.
    fn start_push(&mut self) {
        const PRICE_REFINEMENT_LIMIT: usize = 2;
        const GLOBAL_UPDATE_FACTOR: f64 = 2.0;
        let global_update_skip = self.global_update_interval(GLOBAL_UPDATE_FACTOR);
        let mut next_global_update_limit = global_update_skip;

        // A "hyper" node received only part of a push because it could not
        // pass the whole amount on; it is processed next and relabeled even
        // without excess.
        let mut hyper = IdVec::<NodeId, bool>::filled(self.res_node_num, false);
        let mut hyper_cost = IdVec::<NodeId, L>::filled(self.res_node_num, L::zero());
        let mut relabel_cnt = 0u64;
        let mut eps_phase_cnt = 0usize;
        while self.epsilon >= L::one() {
            eps_phase_cnt += 1;

            if eps_phase_cnt >= PRICE_REFINEMENT_LIMIT && self.price_refinement() {
                self.epsilon = self.next_epsilon();
                continue;
            }

            self.init_phase();

            while !self.active_nodes.is_empty() {
                'next_node: loop {
                    // Select an active node (FIFO selection)
                    let n = self.active_nodes[0];
                    let last_out = self.first_out[n.next()];
                    let pi_n = self.pi[n];
                    let mut excess_used_up = false;

                    // Perform push operations if there are admissible arcs
                    if self.excess[n] > V::zero() {
                        let mut a = self.next_out[n];
                        while a != last_out {
                            let t = self.target[a];
                            if self.res_cap[a] > V::zero()
                                && self.cost[a] + pi_n - self.pi[t] < L::zero()
                            {
                                let delta = self.res_cap[a].min(self.excess[n]);

                                // Push-look-ahead heuristic
                                let mut ahead = -self.excess[t];
                                let pi_t = self.pi[t];
                                for ta in ids(self.next_out[t]..self.first_out[t.next()]) {
                                    if self.res_cap[ta] > V::zero()
                                        && self.cost[ta] + pi_t - self.pi[self.target[ta]]
                                            < L::zero()
                                    {
                                        ahead += self.res_cap[ta];
                                    }
                                    if ahead >= delta {
                                        break;
                                    }
                                }
                                if ahead < V::zero() {
                                    ahead = V::zero();
                                }

                                // Push flow along the arc
                                let r = self.reverse[a];
                                if ahead < delta && !hyper[t] {
                                    self.res_cap[a] -= ahead;
                                    self.res_cap[r] += ahead;
                                    self.excess[n] -= ahead;
                                    self.excess[t] += ahead;
                                    self.active_nodes.push_front(t);
                                    hyper[t] = true;
                                    hyper_cost[t] = self.cost[a] + pi_n - pi_t;
                                    self.next_out[n] = a;
                                    continue 'next_node;
                                }
                                self.res_cap[a] -= delta;
                                self.res_cap[r] += delta;
                                self.excess[n] -= delta;
                                self.excess[t] += delta;
                                if self.excess[t] > V::zero() && self.excess[t] <= delta {
                                    self.active_nodes.push_back(t);
                                }

                                if self.excess[n] == V::zero() {
                                    excess_used_up = true;
                                    break;
                                }
                            }
                            a = a.next();
                        }
                        self.next_out[n] = a;
                    }

                    // Relabel the node if it is still active (or hyper)
                    if !excess_used_up && (self.excess[n] > V::zero() || hyper[n]) {
                        let mut min_red_cost = if hyper[n] {
                            -hyper_cost[n]
                        } else {
                            L::max_value()
                        };
                        for a in ids(self.first_out[n]..last_out) {
                            if self.res_cap[a] > V::zero() {
                                let rc = self.cost[a] + pi_n - self.pi[self.target[a]];
                                if rc < min_red_cost {
                                    min_red_cost = rc;
                                }
                            }
                        }
                        self.pi[n] -= min_red_cost + self.epsilon;
                        self.next_out[n] = self.first_out[n];
                        hyper[n] = false;
                        relabel_cnt += 1;
                    }
                    break;
                }

                // Remove nodes that are neither active nor hyper
                while let Some(&front) = self.active_nodes.front() {
                    if self.excess[front] > V::zero() || hyper[front] {
                        break;
                    }
                    self.active_nodes.pop_front();
                }

                if relabel_cnt >= next_global_update_limit {
                    self.global_update();
                    hyper.fill(false);
                    next_global_update_limit =
                        next_global_update_limit.saturating_add(global_update_skip);
                }
            }

            self.epsilon = self.next_epsilon();
        }
    }
}

/// Doubly linked lists of nodes by rank; `first[r]` heads the list of rank
/// `r`. A list head's `prev` entry is stale and never read.
struct Buckets<'a> {
    first: IMut<'a, usize, Link<NodeId>>,
    next: IMut<'a, NodeId, Link<NodeId>>,
    prev: IMut<'a, NodeId, NodeId>,
}

impl Buckets<'_> {
    #[inline(always)]
    fn unlink(&mut self, v: NodeId, r: u32) {
        let next = self.next[v];
        let first = &mut self.first[r as usize];
        if *first == Link::to(v) {
            *first = next;
        } else {
            let prev = self.prev[v];
            self.next[prev] = next;
            if let Some(next) = next.get() {
                self.prev[next] = prev;
            }
        }
    }

    #[inline(always)]
    fn link(&mut self, v: NodeId, r: u32) {
        let first = &mut self.first[r as usize];
        let head = *first;
        *first = Link::to(v);
        self.next[v] = head;
        if let Some(head) = head.get() {
            self.prev[head] = v;
        }
    }

    /// Removes and returns the head of the list of rank `r`.
    #[inline(always)]
    fn pop(&mut self, r: u32) -> Option<NodeId> {
        let first = &mut self.first[r as usize];
        let head = first.get()?;
        *first = self.next[head];
        Some(head)
    }
}
