//! Phases driven by (partial) augment and relabel operations.

use alloc::vec::Vec;

use super::CostScaling;
use crate::Number;
use crate::ivec::{ArcIx, IdVec, ids};

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    /// Runs the phases with (partial) augment and relabel operations.
    pub(super) fn start_augment(&mut self, max_length: usize) {
        const PRICE_REFINEMENT_LIMIT: usize = 2;
        const GLOBAL_UPDATE_FACTOR: f64 = 1.0;
        let global_update_skip = self.global_update_interval(GLOBAL_UPDATE_FACTOR);
        let mut next_global_update_limit = global_update_skip;

        let mut path: Vec<ArcIx> = Vec::new();
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
        path: &mut Vec<ArcIx>,
        path_arc: &mut IdVec<ArcIx, bool>,
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
}
