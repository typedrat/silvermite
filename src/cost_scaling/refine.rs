//! Price refinement: finishing a phase by changing potentials alone.

use alloc::vec::Vec;
use core::iter;

use super::CostScaling;
use super::global_update::Buckets;
use crate::Number;
use crate::ivec::{IdVec, Idx, Link, NodeIx, first_ids, ids};

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    /// Price refinement heuristic: tries to make the current flow
    /// epsilon-optimal by changing potentials only. Returns true if it
    /// succeeded, in which case the phase needs no flow changes.
    pub(super) fn price_refinement(&mut self) -> bool {
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
    fn topological_sort(&mut self, order: &mut Vec<NodeIx>) -> bool {
        let pi = self.pi.as_ref();
        let target = self.target.as_ref();
        let cost = self.cost.as_ref();
        let reverse = self.reverse.as_ref();
        let first_out = self.first_out.as_ref();
        let mut res_cap = self.res_cap.as_mut();
        let mut next_out = self.next_out.as_mut();

        let n = self.res_node_num;
        let mut reached = IdVec::<NodeIx, bool>::filled(n, false);
        let mut processed = IdVec::<NodeIx, bool>::filled(n, false);
        let mut pred = IdVec::<NodeIx, Link<NodeIx>>::filled(n, Link::NONE);
        next_out.copy_from_slice(&first_out[..NodeIx::new(n)]);
        order.clear();

        for start in first_ids::<NodeIx>(n) {
            if reached[start] {
                continue;
            }

            // Depth-first search from `start`; `next_out[u]` is the arc the
            // search last left `u` by.
            pred[start] = Link::NONE;
            let mut tip = start;
            reached[tip] = true;
            loop {
                let pi_tip = pi[tip];
                let admissible = ids(next_out[tip]..first_out[tip.next()]).find(|&a| {
                    let v = target[a];
                    res_cap[a] > V::zero() && cost[a] + pi_tip - pi[v] < L::zero() && !processed[v]
                });

                let Some(a) = admissible else {
                    // Every admissible arc out of `tip` is explored: step back.
                    processed[tip] = true;
                    order.push(tip);
                    let Some(p) = pred[tip].get() else {
                        break;
                    };
                    tip = p;
                    next_out[tip] = next_out[tip].next();
                    continue;
                };

                next_out[tip] = a;
                let v = target[a];
                if !reached[v] {
                    reached[v] = true;
                    pred[v] = Link::to(tip);
                    tip = v;
                    continue;
                }

                // `v` is on the current path, so `a` closes an admissible
                // cycle through the path arcs from `v` to `tip`: saturate
                // its tightest arc.
                let cycle = || {
                    iter::once(a).chain(
                        iter::successors(Some(tip), |&u| pred[u].get())
                            .take_while(|&u| u != v)
                            .map(|u| next_out[pred[u].get().expect("path nodes have a pred")]),
                    )
                };
                let delta = cycle()
                    .map(|ca| res_cap[ca])
                    .min()
                    .expect("a cycle has arcs");
                for ca in cycle() {
                    res_cap[ca] -= delta;
                    res_cap[reverse[ca]] += delta;
                }
                return false;
            }
        }

        true
    }
}
