//! Building the starting spanning tree from a caller's guess at the flow,
//! or from the tree an earlier solve ended with.

use alloc::vec;
use alloc::vec::Vec;

use super::{ArcState, Dir, NetworkSimplex};
use crate::ivec::{ArcIx, IdVec, Idx, NodeIx, first_ids};
use crate::{Error, Number};

/// Where a warm start begins.
pub(super) enum Start<'a, V> {
    /// A guess at the flow, one entry per problem arc.
    Flow(&'a [V]),
    /// The basis an earlier solve ended with.
    Basis(&'a Basis),
}

/// The basis a network simplex solve ended with, by problem arc and node
/// index, which stays meaningful after the problem's values change or it
/// gains nodes and arcs.
#[derive(Clone, Debug)]
pub(crate) struct Basis {
    state: Vec<ArcState>,
    /// Whether each node hung directly off the root.
    root_child: Vec<bool>,
}

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
    /// The current basis, for a later warm start.
    pub(super) fn basis(&self) -> Basis {
        Basis {
            state: self.arc_id.iter().map(|&e| self.state[e]).collect(),
            root_child: self.parent[..self.root]
                .iter()
                .map(|&p| p == self.root)
                .collect(),
        }
    }

    /// Builds a starting basis from `start`.
    ///
    /// Tree arcs form a spanning forest, with each component hung off the
    /// root by one artificial arc that absorbs the component's imbalance.
    /// Tree flows follow from the supplies and the non-tree flows; a tree
    /// arc whose flow would leave its bounds or break strong feasibility
    /// moves to the bound it hits instead, and its subtree hangs off the
    /// root directly.
    pub(super) fn init_warm(&mut self, start: Start<'_, V>) -> Result<(), Error> {
        self.prepare()?;
        let n = self.node_num;
        let m = self.arc_num;
        let root = self.root;

        let mut components = Components::new(n);
        let mut forest = Vec::new();
        let priority = match start {
            Start::Flow(guess) => self.place_guess(guess, &mut components, &mut forest),
            Start::Basis(basis) => self.place_basis(basis, &mut components, &mut forest),
        };

        // `excess[v]` starts as what `v` must send through tree arcs given
        // the non-tree flows, and accumulates each subtree's total as the
        // tree is walked bottom-up.
        let mut excess: IdVec<NodeIx, V> = self.supply[..root].iter().copied().collect();
        for e in first_ids::<ArcIx>(m) {
            if self.state[e] != ArcState::Tree {
                let f = self.flow[e];
                excess[self.source[e]] -= f;
                excess[self.target[e]] += f;
            }
        }

        let mut top = IdVec::<NodeIx, Option<NodeIx>>::filled(n, None);
        for u in first_ids::<NodeIx>(n) {
            let c = components.find(u);
            if top[c].is_none_or(|t| priority[u] > priority[t]) {
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

    /// Sets the arc states and flows from a guess at the flow, clamped into
    /// each arc's bounds. Arcs strictly between their bounds join the
    /// forest, except that one closing a cycle moves to its nearer bound.
    ///
    /// Returns each node's priority for topping its component: its
    /// imbalance under the guess, so that a guess with only one imbalanced
    /// node per component keeps its tree flows.
    fn place_guess(
        &mut self,
        guess: &[V],
        components: &mut Components,
        forest: &mut Vec<ArcIx>,
    ) -> IdVec<NodeIx, V> {
        let mut imbalance: IdVec<NodeIx, V> = self.supply[..self.root].iter().copied().collect();
        for (&e, &x) in self.arc_id.iter().zip(guess) {
            let (u, v) = (self.source[e], self.target[e]);
            let cap = self.cap[e];
            let f = x.saturating_sub(self.lower[e]).max(V::zero()).min(cap);
            imbalance[u] -= f;
            imbalance[v] += f;
            let (f, state) = if f == V::zero() {
                (f, ArcState::Lower)
            } else if f == cap {
                (f, ArcState::Upper)
            } else if components.union(u, v) {
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
        imbalance.iter_mut().for_each(|b| *b = b.abs());
        imbalance
    }

    /// Sets the arc states from `basis`, with non-tree arcs at their current
    /// bounds. Arcs the basis does not cover start at their lower bound, as
    /// do tree arcs that would close a cycle, which only a basis from a
    /// different problem has.
    ///
    /// Returns each node's priority for topping its component: 1 if it hung
    /// off the root in `basis`, so the tree keeps its old orientation.
    fn place_basis(
        &mut self,
        basis: &Basis,
        components: &mut Components,
        forest: &mut Vec<ArcIx>,
    ) -> IdVec<NodeIx, V> {
        for (j, &e) in self.arc_id.iter().enumerate() {
            let cap = self.cap[e];
            let state = match basis.state.get(j) {
                Some(ArcState::Tree) if components.union(self.source[e], self.target[e]) => {
                    forest.push(e);
                    ArcState::Tree
                }
                Some(ArcState::Upper) if cap < V::max_value() => ArcState::Upper,
                _ => ArcState::Lower,
            };
            self.flow[e] = if state == ArcState::Upper {
                cap
            } else {
                V::zero()
            };
            self.state[e] = state;
        }
        first_ids::<NodeIx>(self.node_num)
            .map(|u| match basis.root_child.get(u.index()) {
                Some(true) => V::one(),
                _ => V::zero(),
            })
            .collect()
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
