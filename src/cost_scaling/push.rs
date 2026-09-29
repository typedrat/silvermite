//! Phases driven by push and relabel operations.

use super::CostScaling;
use crate::Number;
use crate::ivec::{IdVec, NodeIx, ids};

/// How a round of pushes out of a node ended.
enum Pushed {
    /// All of the node's excess moved on.
    UsedUp,
    /// Part of a push went to a node that could not pass it all on. That
    /// node is now hyper and queued in front, to be processed first.
    Deferred,
    /// The node had no excess, or its admissible arcs ran out first.
    Stuck,
}

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    /// Runs the phases with push and relabel operations.
    pub(super) fn start_push(&mut self) {
        const PRICE_REFINEMENT_LIMIT: usize = 2;
        const GLOBAL_UPDATE_FACTOR: f64 = 2.0;
        let global_update_skip = self.global_update_interval(GLOBAL_UPDATE_FACTOR);
        let mut next_global_update_limit = global_update_skip;

        // A "hyper" node received only part of a push because it could not
        // pass the whole amount on; it is processed next and relabeled even
        // without excess.
        let mut hyper = IdVec::<NodeIx, bool>::filled(self.res_node_num, false);
        let mut hyper_cost = IdVec::<NodeIx, L>::filled(self.res_node_num, L::zero());
        let mut relabel_cnt = 0u64;
        let mut eps_phase_cnt = 0usize;
        while self.epsilon >= L::one() {
            eps_phase_cnt += 1;

            if eps_phase_cnt >= PRICE_REFINEMENT_LIMIT && self.price_refinement() {
                self.epsilon = self.next_epsilon();
                continue;
            }

            self.init_phase();

            while let Some(&n) = self.active_nodes.front() {
                match self.push_excess(n, &mut hyper, &mut hyper_cost) {
                    // `n` stays at the front behind the new hyper node.
                    Pushed::Deferred => continue,
                    Pushed::UsedUp => {}
                    Pushed::Stuck => {
                        if self.excess[n] > V::zero() || hyper[n] {
                            self.relabel_push(n, hyper[n].then(|| hyper_cost[n]));
                            hyper[n] = false;
                            relabel_cnt += 1;
                        }
                    }
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

    /// Pushes excess out of `n` along admissible arcs, resuming from
    /// `next_out[n]`.
    #[inline]
    fn push_excess(
        &mut self,
        n: NodeIx,
        hyper: &mut IdVec<NodeIx, bool>,
        hyper_cost: &mut IdVec<NodeIx, L>,
    ) -> Pushed {
        if self.excess[n] <= V::zero() {
            return Pushed::Stuck;
        }
        let pi_n = self.pi[n];
        let mut a = self.next_out[n];
        let last_out = self.first_out[n.next()];
        while a != last_out {
            let t = self.target[a];
            if self.res_cap[a] > V::zero() && self.cost[a] + pi_n - self.pi[t] < L::zero() {
                let delta = self.res_cap[a].min(self.excess[n]);

                // Push-look-ahead heuristic: how much `t` can pass on
                let mut ahead = -self.excess[t];
                let pi_t = self.pi[t];
                for ta in ids(self.next_out[t]..self.first_out[t.next()]) {
                    if self.res_cap[ta] > V::zero()
                        && self.cost[ta] + pi_t - self.pi[self.target[ta]] < L::zero()
                    {
                        ahead += self.res_cap[ta];
                    }
                    if ahead >= delta {
                        break;
                    }
                }
                let ahead = ahead.max(V::zero());

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
                    return Pushed::Deferred;
                }
                self.res_cap[a] -= delta;
                self.res_cap[r] += delta;
                self.excess[n] -= delta;
                self.excess[t] += delta;
                if self.excess[t] > V::zero() && self.excess[t] <= delta {
                    self.active_nodes.push_back(t);
                }

                if self.excess[n] == V::zero() {
                    self.next_out[n] = a;
                    return Pushed::UsedUp;
                }
            }
            a = a.next();
        }
        self.next_out[n] = a;
        Pushed::Stuck
    }

    /// Lowers the potential of `n` just enough to make an arc out of it
    /// admissible. A hyper node also counts the arc its partial push came
    /// in by, whose reduced cost was `hyper_cost`.
    fn relabel_push(&mut self, n: NodeIx, hyper_cost: Option<L>) {
        let pi_n = self.pi[n];
        let mut min_red_cost = hyper_cost.map_or(L::max_value(), |c| -c);
        for a in self.block(n) {
            if self.res_cap[a] > V::zero() {
                let rc = self.cost[a] + pi_n - self.pi[self.target[a]];
                if rc < min_red_cost {
                    min_red_cost = rc;
                }
            }
        }
        self.pi[n] -= min_red_cost + self.epsilon;
        self.next_out[n] = self.first_out[n];
    }
}
