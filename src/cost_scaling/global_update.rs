//! Global updates: relabeling nodes in bulk by their distance to the
//! nearest deficit.

use itertools::izip;

use super::CostScaling;
use crate::Number;
use crate::ivec::{IMut, Link, NodeIx, first_ids, ids};

/// Doubly linked lists of nodes by rank; `first[r]` heads the list of rank
/// `r`. A list head's `prev` entry is stale and never read.
pub(super) struct Buckets<'a> {
    pub(super) first: IMut<'a, usize, Link<NodeIx>>,
    pub(super) next: IMut<'a, NodeIx, Link<NodeIx>>,
    pub(super) prev: IMut<'a, NodeIx, NodeIx>,
}

impl Buckets<'_> {
    #[inline(always)]
    pub(super) fn unlink(&mut self, v: NodeIx, r: u32) {
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
    pub(super) fn link(&mut self, v: NodeIx, r: u32) {
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
    pub(super) fn pop(&mut self, r: u32) -> Option<NodeIx> {
        let first = &mut self.first[r as usize];
        let head = first.get()?;
        *first = self.next[head];
        Some(head)
    }
}

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    /// Global update heuristic: relabels nodes by their reduced-cost
    /// distance (in units of epsilon) to the nearest deficit node, using a
    /// bucket-based Dijkstra over reversed residual arcs.
    pub(super) fn global_update(&mut self) {
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
        for i in first_ids::<NodeIx>(res_node_num) {
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
}
