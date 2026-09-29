//! Loading a problem and building the initial spanning tree.

use itertools::izip;

use super::{ArcState, Dir, NetworkSimplex};
use crate::ivec::{ArcIx, Idx, NodeIx, first_ids};
use crate::{Error, Number, Problem, SupplyType};

impl<V: Number, C: Number> NetworkSimplex<V, C> {
    pub(super) fn load(&mut self, p: &Problem<V, C>) {
        let n = p.node_count();
        let m = p.arc_count();
        self.node_num = n;
        self.arc_num = m;
        let all_node_num = n + 1;
        let max_arc_num = m + 2 * n;

        self.source.reset(max_arc_num, NodeIx::default());
        self.target.reset(max_arc_num, NodeIx::default());
        self.lower.reset(m, V::zero());
        self.upper.reset(m, V::zero());
        self.cap.reset(max_arc_num, V::zero());
        self.cost.reset(max_arc_num, C::zero());
        self.supply.reset(all_node_num, V::zero());
        self.flow.reset(max_arc_num, V::zero());
        self.pi.reset(all_node_num, C::zero());

        self.parent.reset(all_node_num, NodeIx::default());
        self.pred.reset(all_node_num, ArcIx::default());
        self.pred_dir.reset(all_node_num, Dir::Up);
        self.thread.reset(all_node_num, NodeIx::default());
        self.rev_thread.reset(all_node_num, NodeIx::default());
        self.succ_num.reset(all_node_num, 0);
        self.last_succ.reset(all_node_num, NodeIx::default());
        self.state.reset(max_arc_num, ArcState::Tree);

        self.arc_id.clear();
        if self.arc_mixing && n > 1 {
            // Deal arcs round-robin into `skip` interleaved runs.
            let skip = (m / n).max(3);
            self.arc_id
                .extend((0..skip).flat_map(|j| (j..m).step_by(skip)).map(ArcIx::new));
        } else {
            self.arc_id.extend(first_ids::<ArcIx>(m));
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
        self.supply[..NodeIx::new(n)].copy_from_slice(&p.supply);
        self.has_lower = p.lower.iter().any(|&l| l != V::zero());
        self.stype = p.supply_type;
    }

    /// Builds the starting basis where every node hangs directly off the
    /// artificial root.
    pub(super) fn init(&mut self) -> Result<(), Error> {
        self.prepare()?;
        let n = self.node_num;
        let m = self.arc_num;
        let arcs = ..ArcIx::new(m);
        let root = self.root;

        self.flow[arcs].fill(V::zero());
        self.state[arcs].fill(ArcState::Lower);

        // Thread order 0, 1, ..., n - 1. The root has no pred arc, so its
        // pred entry is never read.
        let first = NodeIx::new(0);
        self.succ_num.fill(1);
        self.succ_num[root] = n as u32 + 1;
        self.thread[root] = first;
        self.rev_thread[first] = root;
        self.last_succ[root] = root.prev();

        let mut next_extra = ArcIx::new(m + n);
        for u in first_ids::<NodeIx>(n) {
            self.thread[u] = u.next();
            self.rev_thread[u.next()] = u;
            self.last_succ[u] = u;
            self.link_to_root(u, self.supply[u], &mut next_extra);
        }
        self.all_arc_num = next_extra.index();

        Ok(())
    }

    /// Checks the supply type, shifts out lower bounds, and sets up the
    /// root, leaving the spanning tree and the flow to the caller.
    pub(super) fn prepare(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let m = self.arc_num;
        let arcs = ..ArcIx::new(m);
        let root = NodeIx::new(n);
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

        self.parent[root] = root;
        self.supply[root] = -self.sum_supply;
        self.pi[root] = C::zero();
        self.search_arc_num = if self.has_slack() { m + n } else { m };
        Ok(())
    }

    /// Whether the supply constraints allow slack, i.e. are inequalities.
    fn has_slack(&self) -> bool {
        self.sum_supply != V::zero()
    }

    /// The direction of the artificial tree arcs that cost `art_cost`: the
    /// one the supply constraints do not allow slack in.
    fn costly_dir(&self) -> Dir {
        if self.sum_supply < V::zero() {
            Dir::Up
        } else {
            Dir::Down
        }
    }

    /// Hangs `u` directly off the root by an artificial tree arc carrying
    /// `excess`, the net flow `u`'s subtree sends to the root: upwards when
    /// positive, downwards when negative.
    ///
    /// Arcs in the costly direction cost enough that no optimal basis uses
    /// one with flow unless the problem is infeasible. With GEQ/LEQ
    /// constraints, each costly tree arc gets a free non-tree twin in the
    /// other direction, which the pivots search to absorb the slack; costly
    /// arcs themselves go past the searched range, at `next_extra`.
    pub(super) fn link_to_root(&mut self, u: NodeIx, excess: V, next_extra: &mut ArcIx) {
        let costly = self.costly_dir();
        let dir = if excess > V::zero() || (excess == V::zero() && costly == Dir::Down) {
            Dir::Up
        } else {
            Dir::Down
        };
        let cost = if dir == costly {
            C::max_value() / C::from_i8(2) + C::one()
        } else {
            C::zero()
        };
        let mut e = ArcIx::new(self.arc_num + u.index());
        if self.has_slack() && dir == costly {
            self.set_artificial(e, u, dir.reversed(), V::zero(), C::zero(), ArcState::Lower);
            e = *next_extra;
            *next_extra = next_extra.next();
        }
        self.set_artificial(e, u, dir, dir.sign::<V>() * excess, cost, ArcState::Tree);
        self.parent[u] = self.root;
        self.pred[u] = e;
        self.pred_dir[u] = dir;
        self.pi[u] = -dir.sign::<C>() * cost;
    }

    /// Leaves `u`'s artificial arc out of the tree, for a node that joins it
    /// through an original arc instead.
    pub(super) fn leave_unlinked(&mut self, u: NodeIx) {
        let e = ArcIx::new(self.arc_num + u.index());
        let dir = self.costly_dir().reversed();
        self.set_artificial(e, u, dir, V::zero(), C::zero(), ArcState::Lower);
    }

    /// Sets up the infinite-capacity artificial arc `e` between node `u` and
    /// the root, oriented `dir` relative to `u`.
    fn set_artificial(&mut self, e: ArcIx, u: NodeIx, dir: Dir, flow: V, cost: C, state: ArcState) {
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
}
