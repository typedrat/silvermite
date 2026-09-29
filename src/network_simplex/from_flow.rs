//! Building the starting spanning tree from a caller's guess at the flow.

use alloc::vec;
use alloc::vec::Vec;

use super::{ArcState, Dir, NetworkSimplex};
use crate::ivec::{ArcIx, IdVec, Idx, NodeIx, first_ids};
use crate::{Error, Number};

/// Disjoint sets of nodes, for growing a spanning forest.
struct Components(IdVec<NodeIx, NodeIx>);

impl Components {
    fn new(n: usize) -> Self {
        Components(first_ids(n).collect())
    }

    fn find(&mut self, mut u: NodeIx) -> NodeIx {
        let parent = &mut self.0;
        while parent[u] != u {
            let grandparent = parent[parent[u]];
            parent[u] = grandparent;
            u = grandparent;
        }
        u
    }

    /// Merges the sets holding `u` and `v`. Returns false if they were
    /// already the same set.
    fn union(&mut self, u: NodeIx, v: NodeIx) -> bool {
        let (a, b) = (self.find(u), self.find(v));
        self.0[a] = b;
        a != b
    }
}

/// Offsets into a flat list grouping items by node: the items of node `v`
/// are `items[first[v]..first[v.next()]]`.
fn group_by_node<T: Copy + Default>(
    n: usize,
    entries: impl Iterator<Item = (NodeIx, T)> + Clone,
) -> (IdVec<NodeIx, usize>, Vec<T>) {
    let mut first = IdVec::<NodeIx, usize>::filled(n + 1, 0);
    for (v, _) in entries.clone() {
        first[v.next()] += 1;
    }
    let mut sum = 0;
    for count in first.iter_mut() {
        sum += *count;
        *count = sum;
    }
    let mut fill = first.clone();
    let mut items = vec![T::default(); sum];
    for (v, item) in entries {
        items[fill[v]] = item;
        fill[v] += 1;
    }
    (first, items)
}

impl<V: Number, C: Number> NetworkSimplex<V, C> {
    /// Builds a starting basis whose flow stays as close to `guess` (in
    /// problem arc order) as a basis allows.
    ///
    /// Arcs strictly between their bounds must be tree arcs, so they form a
    /// spanning forest, with each component hung off the root by one
    /// artificial arc that absorbs the component's imbalance. Where that is
    /// impossible, arcs move to a bound:
    ///
    /// - an arc that would close a cycle in the forest, to its nearer bound;
    /// - a forest arc that would have to leave its bounds to route another
    ///   node's imbalance to the root, or would break strong feasibility, to
    ///   the bound it hits. Its subtree then hangs off the root directly.
    pub(super) fn init_from_flow(&mut self, guess: &[V]) -> Result<(), Error> {
        self.prepare()?;
        let n = self.node_num;
        let m = self.arc_num;
        let root = self.root;

        let mut components = Components::new(n);
        let mut forest = Vec::new();
        for (&e, &x) in self.arc_id.iter().zip(guess) {
            let cap = self.cap[e];
            let f = x.saturating_sub(self.lower[e]).max(V::zero()).min(cap);
            let (f, state) = if f == V::zero() {
                (f, ArcState::Lower)
            } else if f == cap {
                (f, ArcState::Upper)
            } else if components.union(self.source[e], self.target[e]) {
                forest.push(e);
                (f, ArcState::Tree)
            } else if f <= cap - f {
                (V::zero(), ArcState::Lower)
            } else {
                (cap, ArcState::Upper)
            };
            self.flow[e] = f;
            self.state[e] = state;
        }

        // `excess[v]` starts as what `v` must send through tree arcs given
        // the non-tree flows, and accumulates each subtree's total as the
        // tree is walked bottom-up.
        let mut excess: IdVec<NodeIx, V> = self.supply[..root].iter().copied().collect();
        let mut imbalance = excess.clone();
        for e in first_ids::<ArcIx>(m) {
            let (u, v, f) = (self.source[e], self.target[e], self.flow[e]);
            if self.state[e] != ArcState::Tree {
                excess[u] -= f;
                excess[v] += f;
            }
            imbalance[u] -= f;
            imbalance[v] += f;
        }

        // Root each component at its most imbalanced node, so that a guess
        // with only one imbalanced node per component keeps its tree flows.
        let mut top = IdVec::<NodeIx, Option<NodeIx>>::filled(n, None);
        for u in first_ids::<NodeIx>(n) {
            let c = components.find(u);
            if top[c].is_none_or(|t| imbalance[u].abs() > imbalance[t].abs()) {
                top[c] = Some(u);
            }
        }

        let (adj_first, adj) = group_by_node(
            n,
            forest.iter().flat_map(|&e| {
                [
                    (self.source[e], (self.target[e], e)),
                    (self.target[e], (self.source[e], e)),
                ]
            }),
        );
        let mut order = Vec::with_capacity(n);
        let mut stack = Vec::new();
        for &t in top.iter().flatten() {
            self.parent[t] = root;
            stack.push(t);
            while let Some(u) = stack.pop() {
                order.push(u);
                for &(v, e) in &adj[adj_first[u]..adj_first[u.next()]] {
                    if v != self.parent[u] {
                        self.parent[v] = u;
                        self.pred[v] = e;
                        self.pred_dir[v] = if self.source[e] == v {
                            Dir::Up
                        } else {
                            Dir::Down
                        };
                        stack.push(v);
                    }
                }
            }
        }

        let mut next_extra = ArcIx::new(m + n);
        for &v in order.iter().rev() {
            let p = self.parent[v];
            if p == root {
                self.link_to_root(v, excess[v], &mut next_extra);
                continue;
            }
            let e = self.pred[v];
            let dir = self.pred_dir[v];
            let cap = self.cap[e];
            let sub = excess[v];
            let f = dir.sign::<V>() * sub;
            // Positive flow must be able to go from `v` towards the root.
            let strongly_feasible = match dir {
                Dir::Up => f >= V::zero() && f < cap,
                Dir::Down => f > V::zero() && f <= cap,
            };
            if strongly_feasible {
                self.flow[e] = f;
                excess[p] += sub;
                self.leave_unlinked(v);
            } else {
                let (f, state) = if f <= V::zero() {
                    (V::zero(), ArcState::Lower)
                } else {
                    (cap, ArcState::Upper)
                };
                self.flow[e] = f;
                self.state[e] = state;
                let sent = dir.sign::<V>() * f;
                excess[p] += sent;
                self.link_to_root(v, sub - sent, &mut next_extra);
            }
        }
        self.all_arc_num = next_extra.index();

        self.build_thread();
        Ok(())
    }

    /// Rebuilds the thread, subtree sizes and last successors, and the
    /// potentials, from the parent and pred links.
    fn build_thread(&mut self) {
        let n = self.node_num;
        let root = self.root;
        let (child_first, children) =
            group_by_node(n + 1, first_ids::<NodeIx>(n).map(|u| (self.parent[u], u)));

        let mut order = Vec::with_capacity(n + 1);
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            order.push(u);
            stack.extend(children[child_first[u]..child_first[u.next()]].iter().rev());
        }

        for (&u, &next) in order.iter().zip(order.iter().cycle().skip(1)) {
            self.thread[u] = next;
            self.rev_thread[next] = u;
        }
        for &u in &order[1..] {
            let p = self.parent[u];
            let e = self.pred[u];
            self.pi[u] = self.pi[p] - self.pred_dir[u].sign::<C>() * self.cost[e];
        }
        self.succ_num.fill(1);
        for &u in order[1..].iter().rev() {
            let p = self.parent[u];
            self.succ_num[p] += self.succ_num[u];
        }
        for (i, &u) in order.iter().enumerate() {
            self.last_succ[u] = order[i + self.succ_num[u] as usize - 1];
        }
    }
}
