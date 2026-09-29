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

    pub(super) fn init(&mut self) -> Result<(), Error> {
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

        // Large enough that no optimal basis uses an artificial arc with
        // flow unless the problem is infeasible.
        let art_cost = C::max_value() / C::from_i8(2) + C::one();

        self.flow[arcs].fill(V::zero());
        self.state[arcs].fill(ArcState::Lower);

        // Start from the tree where every node hangs directly off the
        // artificial root, in thread order 0, 1, ..., n - 1. The root has no
        // pred arc, so its pred entry is never read.
        let first = NodeIx::new(0);
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
        let mut next_extra = ArcIx::new(m + n);
        for u in first_ids::<NodeIx>(n) {
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
            let mut e = ArcIx::new(m + u.index());
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
