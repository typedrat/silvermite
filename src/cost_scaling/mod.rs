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

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::ivec::{ArcIx, IVec, IdVec, Idx, Link, NodeIx, first_ids, ids};
use crate::{Error, Number, Problem, Solution};

mod augment;
mod global_update;
mod push;
mod refine;
mod setup;

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
/// use silvermite::{Capacity, CostScaling, Problem};
///
/// let mut p = Problem::<i64, i64>::new(0);
/// let [s, a, t] = [5, 0, -5].map(|supply| p.add_node(supply));
/// p.add_arc(s, a, 0, 4, 1);
/// p.add_arc(a, t, 0, 4, 1);
/// let direct = p.add_arc(s, t, 0, Capacity::Infinite, 3);
///
/// let solution = CostScaling::new().solve(&p).unwrap();
/// assert_eq!(solution.flows(), &[4, 4, 1]);
/// assert_eq!(solution.flow(direct), 1);
/// assert_eq!(solution.total_cost(), 11);
/// ```
#[derive(Clone, Debug)]
pub struct CostScaling<V, C, L = i64> {
    method: Method,
    alpha: u32,

    node_num: usize,
    res_node_num: usize,
    res_arc_num: usize,
    root: NodeIx,
    // Whether the problem was mirrored to turn LEQ constraints into GEQ.
    mirrored: bool,

    has_lower: bool,
    sum_supply: V,
    sup_node_num: usize,

    // Forward and backward residual arc of each problem arc
    arc_idf: Vec<ArcIx>,
    arc_idb: Vec<ArcIx>,
    // Node `u`'s arc block is `first_out[u]..first_out[u.next()]`.
    first_out: IVec<NodeIx, ArcIx>,
    forward: IVec<ArcIx, bool>,
    source: IVec<ArcIx, NodeIx>,
    target: IVec<ArcIx, NodeIx>,
    reverse: IVec<ArcIx, ArcIx>,

    lower: IVec<ArcIx, V>,
    upper: IVec<ArcIx, V>,
    scost: IVec<ArcIx, C>,
    supply: IVec<NodeIx, V>,

    res_cap: IVec<ArcIx, V>,
    // Forward arcs whose capacity was infinite before being bounded by the
    // total deficit.
    uncapped: IVec<ArcIx, bool>,
    cost: IVec<ArcIx, L>,
    pi: IVec<NodeIx, L>,
    excess: IVec<NodeIx, V>,
    next_out: IVec<NodeIx, ArcIx>,
    active_nodes: VecDeque<NodeIx>,

    epsilon: L,

    // Bucket list heads, indexed by rank
    buckets: IVec<usize, Link<NodeIx>>,
    bucket_next: IVec<NodeIx, Link<NodeIx>>,
    bucket_prev: IVec<NodeIx, NodeIx>,
    rank: IVec<NodeIx, u32>,
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
            root: NodeIx::default(),
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
                basis: None,
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
            basis: None,
        })
    }

    /// The arcs out of `u` in the residual graph.
    #[inline(always)]
    fn block(&self, u: NodeIx) -> impl DoubleEndedIterator<Item = ArcIx> + Clone + use<V, C, L> {
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
        let optimal = first_ids::<NodeIx>(self.res_node_num).all(|i| {
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
    fn is_open(&self, j: ArcIx) -> bool {
        self.res_cap[j] > V::zero() || self.uncapped[j]
    }

    /// Shortest path distances in the residual graph under the reduced
    /// original costs, from a virtual source joined to every node by a
    /// zero-length arc.
    fn bellman_ford(&self) -> IdVec<NodeIx, L> {
        let n = self.res_node_num;
        let mut dist = IdVec::filled(n, L::zero());
        let mut mask = IdVec::filled(n, true);
        let mut process: Vec<NodeIx> = first_ids(n).collect();
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
            core::mem::swap(&mut process, &mut next);
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
        for u in first_ids::<NodeIx>(res_node_num) {
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

        active_nodes.extend(first_ids::<NodeIx>(res_node_num).filter(|&u| excess[u] > V::zero()));
        next_out.copy_from_slice(&first_out[..NodeIx::new(res_node_num)]);
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
}
