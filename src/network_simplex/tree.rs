//! One pivot: finding the cycle the entering arc closes, pushing flow
//! around it, and exchanging the leaving arc for the entering arc in the
//! spanning tree.

use core::iter;

use super::{ArcState, Dir, NetworkSimplex};
use crate::ivec::{ArcIx, IRef, NodeIx};
use crate::{Error, Number};

/// The tree arc a pivot removes, and how the entering arc replaces it.
#[derive(Clone, Copy)]
struct Exchange {
    /// The entering arc's endpoint in the subtree cut off by removing the
    /// leaving arc.
    u_in: NodeIx,
    /// The entering arc's other endpoint, which becomes `u_in`'s parent.
    v_in: NodeIx,
    /// The child end of the leaving arc.
    u_out: NodeIx,
}

/// The nodes from `from` up to, but not including, its ancestor `to`.
#[inline(always)]
fn path_up(
    parent: IRef<'_, NodeIx, NodeIx>,
    from: NodeIx,
    to: NodeIx,
) -> impl Iterator<Item = NodeIx> + '_ {
    let mut u = from;
    iter::from_fn(move || {
        if u == to {
            return None;
        }
        let here = u;
        u = parent[u];
        Some(here)
    })
}

/// The nodes from `from` up to and including the root.
#[inline(always)]
fn ancestors(parent: IRef<'_, NodeIx, NodeIx>, from: NodeIx) -> impl Iterator<Item = NodeIx> + '_ {
    iter::successors(Some(from), move |&u| Some(parent[u]).filter(|&p| p != u))
}

impl<V: Number, C: Number> NetworkSimplex<V, C> {
    /// The nearest common ancestor of `in_arc`'s endpoints, where the cycle
    /// it closes in the tree turns around.
    fn find_join_node(&self, in_arc: ArcIx) -> NodeIx {
        let succ_num = self.succ_num.as_ref();
        let parent = self.parent.as_ref();
        let mut u = self.source[in_arc];
        let mut v = self.target[in_arc];
        // The root's subtree is the largest, so only a non-root node ever
        // steps up.
        while u != v {
            if succ_num[u] < succ_num[v] {
                u = parent[u];
            } else {
                v = parent[v];
            }
        }
        u
    }

    /// Finds how much flow can be pushed around the cycle `in_arc` closes,
    /// and which arc blocks it. The exchange is `None` when `in_arc` blocks
    /// itself, so it only moves to its other bound.
    fn find_leaving_arc(&self, in_arc: ArcIx, join: NodeIx) -> (V, Option<Exchange>) {
        let state = self.state.as_ref();
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let cap = self.cap.as_ref();
        let pred = self.pred.as_ref();
        let flow = self.flow.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let parent = self.parent.as_ref();
        let inf = V::max_value();
        let max = V::max_value();

        // Orient the cycle along the direction flow will be pushed.
        let (first, second) = if state[in_arc] == ArcState::Lower {
            (source[in_arc], target[in_arc])
        } else {
            (target[in_arc], source[in_arc])
        };
        let mut delta = cap[in_arc];
        let mut exchange = None;

        // How far flow can be pushed along the pred arc of `u` when the
        // cycle runs through it in direction `along`.
        let residual = |u: NodeIx, along: Dir| {
            let e = pred[u];
            let d = flow[e];
            if pred_dir[u] == along {
                d
            } else {
                let c = cap[e];
                if c >= max { inf } else { c - d }
            }
        };

        for u in path_up(parent, first, join) {
            let d = residual(u, Dir::Up);
            if d < delta {
                delta = d;
                exchange = Some(Exchange {
                    u_in: first,
                    v_in: second,
                    u_out: u,
                });
            }
        }

        // `<=` here and `<` above pick the last blocking arc along the cycle
        // direction, which keeps the tree strongly feasible.
        for u in path_up(parent, second, join) {
            let d = residual(u, Dir::Down);
            if d <= delta {
                delta = d;
                exchange = Some(Exchange {
                    u_in: second,
                    v_in: first,
                    u_out: u,
                });
            }
        }

        (delta, exchange)
    }

    /// Pushes `delta` around the cycle and updates the arc states.
    fn change_flow(&mut self, in_arc: ArcIx, join: NodeIx, delta: V, exchange: Option<Exchange>) {
        let source = self.source.as_ref();
        let target = self.target.as_ref();
        let pred = self.pred.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let parent = self.parent.as_ref();
        let mut state = self.state.as_mut();
        let mut flow = self.flow.as_mut();
        if delta > V::zero() {
            let val = state[in_arc].sign::<V>() * delta;
            flow[in_arc] += val;
            for u in path_up(parent, source[in_arc], join) {
                flow[pred[u]] -= pred_dir[u].sign::<V>() * val;
            }
            for u in path_up(parent, target[in_arc], join) {
                flow[pred[u]] += pred_dir[u].sign::<V>() * val;
            }
        }
        match exchange {
            Some(Exchange { u_out, .. }) => {
                state[in_arc] = ArcState::Tree;
                let out_arc = pred[u_out];
                state[out_arc] = if flow[out_arc] == V::zero() {
                    ArcState::Lower
                } else {
                    ArcState::Upper
                };
            }
            None => state[in_arc] = state[in_arc].flipped(),
        }
    }

    /// Replaces the leaving arc with `in_arc` in the spanning tree.
    fn update_tree_structure(&mut self, in_arc: ArcIx, join: NodeIx, exchange: Exchange) {
        let source = self.source.as_ref();
        let mut parent = self.parent.as_mut();
        let mut pred = self.pred.as_mut();
        let mut pred_dir = self.pred_dir.as_mut();
        let mut thread = self.thread.as_mut();
        let mut rev_thread = self.rev_thread.as_mut();
        let mut succ_num = self.succ_num.as_mut();
        let mut last_succ = self.last_succ.as_mut();
        let dirty_revs = &mut self.dirty_revs;
        let Exchange { u_in, v_in, u_out } = exchange;

        let old_rev_thread = rev_thread[u_out];
        let old_succ_num = succ_num[u_out];
        let old_last_succ = last_succ[u_out];
        let v_out = parent[u_out];
        let in_dir = if u_in == source[in_arc] {
            Dir::Up
        } else {
            Dir::Down
        };

        if u_in == u_out {
            // Update parent, pred, pred_dir
            parent[u_in] = v_in;
            pred[u_in] = in_arc;
            pred_dir[u_in] = in_dir;

            // Update thread and rev_thread
            if thread[v_in] != u_out {
                let mut after = thread[old_last_succ];
                thread[old_rev_thread] = after;
                rev_thread[after] = old_rev_thread;
                after = thread[v_in];
                thread[v_in] = u_out;
                rev_thread[u_out] = v_in;
                thread[old_last_succ] = after;
                rev_thread[after] = old_last_succ;
            }
        } else {
            // When old_rev_thread == v_in, join and v_out coincide too.
            let thread_continue = if old_rev_thread == v_in {
                thread[old_last_succ]
            } else {
                thread[v_in]
            };

            // Re-hang the stem (the path from u_in up to u_out) under v_in,
            // reversing parent links and splicing each stem node's subtree
            // into the thread after its new parent's.
            let mut stem = u_in;
            let mut par_stem = v_in;
            let mut last = last_succ[u_in];
            let mut after = thread[last];
            thread[v_in] = u_in;
            dirty_revs.clear();
            dirty_revs.push(v_in);
            while stem != u_out {
                // Insert the next stem node into the thread list
                let next_stem = parent[stem];
                thread[last] = next_stem;
                dirty_revs.push(last);

                // Remove the subtree of stem from the thread list
                let before = rev_thread[stem];
                thread[before] = after;
                rev_thread[after] = before;

                // Change the parent node and shift stem nodes
                parent[stem] = par_stem;
                par_stem = stem;
                stem = next_stem;

                // Update last and after
                last = if last_succ[stem] == last_succ[par_stem] {
                    rev_thread[par_stem]
                } else {
                    last_succ[stem]
                };
                after = thread[last];
            }
            parent[u_out] = par_stem;
            thread[last] = thread_continue;
            rev_thread[thread_continue] = last;
            last_succ[u_out] = last;

            // Remove the subtree of u_out from the thread list, unless
            // old_rev_thread == v_in, where it is already in place.
            if old_rev_thread != v_in {
                thread[old_rev_thread] = after;
                rev_thread[after] = old_rev_thread;
            }

            // Update rev_thread using the new thread values
            for &u in dirty_revs.iter() {
                rev_thread[thread[u]] = u;
            }

            // Update pred, pred_dir, last_succ and succ_num for the stem
            // nodes from u_out to u_in
            let mut tmp_sc = 0u32;
            let tmp_ls = last_succ[u_out];
            let mut u = u_out;
            while u != u_in {
                let p = parent[u];
                pred[u] = pred[p];
                pred_dir[u] = pred_dir[p].reversed();
                tmp_sc += succ_num[u] - succ_num[p];
                succ_num[u] = tmp_sc;
                last_succ[p] = tmp_ls;
                u = p;
            }
            pred[u_in] = in_arc;
            pred_dir[u_in] = in_dir;
            succ_num[u_in] = old_succ_num;
        }
        let parent = parent.as_ref();

        // Update last_succ from v_in towards the root
        let up_limit_out = (last_succ[join] == v_in).then_some(join);
        let last_succ_out = last_succ[u_out];
        for u in ancestors(parent, v_in) {
            if last_succ[u] != v_in {
                break;
            }
            last_succ[u] = last_succ_out;
        }

        // Update last_succ from v_out towards the root
        let new_last_succ = if join != old_rev_thread && v_in != old_rev_thread {
            Some(old_rev_thread)
        } else {
            (last_succ_out != old_last_succ).then_some(last_succ_out)
        };
        if let Some(new_last_succ) = new_last_succ {
            for u in ancestors(parent, v_out).take_while(|&u| Some(u) != up_limit_out) {
                if last_succ[u] != old_last_succ {
                    break;
                }
                last_succ[u] = new_last_succ;
            }
        }

        // Update succ_num from v_in and from v_out to join
        for u in path_up(parent, v_in, join) {
            succ_num[u] += old_succ_num;
        }
        for u in path_up(parent, v_out, join) {
            succ_num[u] -= old_succ_num;
        }
    }

    /// Shifts the potentials of the subtree that moved under v_in so the
    /// entering arc has zero reduced cost.
    fn update_potential(&mut self, in_arc: ArcIx, exchange: Exchange) {
        let cost = self.cost.as_ref();
        let pred_dir = self.pred_dir.as_ref();
        let thread = self.thread.as_ref();
        let last_succ = self.last_succ.as_ref();
        let mut pi = self.pi.as_mut();
        let Exchange { u_in, v_in, .. } = exchange;
        let sigma = pi[v_in] - pi[u_in] - pred_dir[u_in].sign::<C>() * cost[in_arc];
        let end = thread[last_succ[u_in]];
        let mut u = u_in;
        while u != end {
            pi[u] += sigma;
            u = thread[u];
        }
    }

    /// Pivots `in_arc` into the basis. Fails if the cycle it closes has
    /// infinite capacity, i.e. the problem is unbounded.
    #[inline]
    pub(super) fn pivot(&mut self, in_arc: ArcIx) -> Result<(), Error> {
        let join = self.find_join_node(in_arc);
        let (delta, exchange) = self.find_leaving_arc(in_arc, join);
        if delta >= V::max_value() {
            return Err(Error::Unbounded);
        }
        self.change_flow(in_arc, join, delta, exchange);
        if let Some(exchange) = exchange {
            self.update_tree_structure(in_arc, join, exchange);
            self.update_potential(in_arc, exchange);
        }
        Ok(())
    }
}
