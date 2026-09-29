//! Heuristic initial pivots.

use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;

use super::NetworkSimplex;
use crate::ivec::{ArcIx, IdVec, Idx, NodeIx, first_ids};
use crate::{Error, Number};

impl<V: Number, C: Number> NetworkSimplex<V, C> {
    /// Heuristic warm start: pivots in arcs likely to carry flow in the
    /// optimum. Fails if the problem turns out to be unbounded.
    pub(super) fn initial_pivots(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let m = self.arc_num;
        let arcs = ..ArcIx::new(m);

        let nodes_where = |keep: fn(V) -> bool| -> Vec<NodeIx> {
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

        let mut arc_vector: Vec<ArcIx> = Vec::new();
        if self.sum_supply >= V::zero() {
            if let ([s], [t]) = (&supply_nodes[..], &demand_nodes[..]) {
                // Reverse DFS from the sink to the source over arcs that can
                // carry the whole amount. The incoming arcs of `v` are
                // `in_arcs[in_first[v]..in_first[v.next()]]`.
                let mut in_first = IdVec::<NodeIx, usize>::filled(n + 1, 0);
                for &t in &self.target[arcs] {
                    in_first[t.next()] += 1;
                }
                let mut sum = 0;
                for count in in_first.iter_mut() {
                    sum += *count;
                    *count = sum;
                }
                let mut fill = in_first.clone();
                let mut in_arcs = vec![ArcIx::default(); m];
                for (j, &t) in first_ids::<ArcIx>(m).zip(&self.target[arcs]) {
                    in_arcs[fill[t]] = j;
                    fill[t] += 1;
                }

                let mut reached = IdVec::<NodeIx, bool>::filled(n, false);
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
    fn cheapest_arcs(&self, endpoints: &[NodeIx]) -> IdVec<NodeIx, Option<ArcIx>> {
        let mut best = IdVec::filled(self.node_num, None);
        let mut best_cost = IdVec::filled(self.node_num, C::max_value());
        for (j, &v, &c) in izip!(first_ids::<ArcIx>(endpoints.len()), endpoints, &*self.cost) {
            if best[v].is_none() || c < best_cost[v] {
                best[v] = Some(j);
                best_cost[v] = c;
            }
        }
        best
    }
}
