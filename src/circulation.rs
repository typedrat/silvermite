//! Feasible circulation by highest-label push-relabel, ported from LEMON's
//! `Circulation` and `Elevator`, specialized to zero lower bounds.
//!
//! It runs directly on cost scaling's residual layout, where each real node's
//! arc block lists its outgoing arcs (forward), then its incoming arcs
//! (backward), then one arc to the artificial root.

use alloc::vec;
use alloc::vec::Vec;

use crate::ivec::{ArcId, IMut, IRef, IdVec, Idx, NodeId, first_ids, ids};
use crate::{Error, Number};

/// The residual graph layout the circulation reads, restricted to the real
/// nodes `0..node_num`.
pub(crate) struct Layout<'a> {
    pub node_num: usize,
    pub first_out: IRef<'a, NodeId, ArcId>,
    pub forward: IRef<'a, ArcId, bool>,
    pub target: IRef<'a, ArcId, NodeId>,
    pub reverse: IRef<'a, ArcId, ArcId>,
}

/// Searches for a flow with `0 <= flow <= cap` on every arc and
/// `out(v) - in(v) >= supply(v)` at every node.
///
/// `cap` and `flow` are indexed by forward residual arc; `arcs` lists every
/// forward arc once, in the order the greedy initialization visits them.
/// Fails with [`Error::Infeasible`] if no such flow exists.
pub(crate) fn circulation<V: Number>(
    g: &Layout<'_>,
    arcs: &[ArcId],
    cap: IRef<'_, ArcId, V>,
    supply: IRef<'_, NodeId, V>,
    mut flow: IMut<'_, ArcId, V>,
) -> Result<(), Error> {
    let n = g.node_num;
    let mut excess = IdVec::<NodeId, V>::filled(n, V::zero());
    excess.copy_from_slice(&supply[..NodeId::new(n)]);

    // Greedy initialization: send as much as the target still demands.
    // Arcs are visited newest-first; on inputs listed by source node, as
    // generator output usually is, insertion order leaves far more excess
    // for the push-relabel phase (100x slower on GOTO instances).
    for &e in arcs.iter().rev() {
        let t = g.target[e];
        let s = g.target[g.reverse[e]];
        let up = cap[e];
        if -excess[t] >= up {
            flow[e] = up;
            excess[t] += up;
            excess[s] -= up;
        } else if -excess[t] < V::zero() {
            flow[e] = V::zero();
        } else {
            let fc = -excess[t];
            flow[e] = fc;
            excess[t] = V::zero();
            excess[s] -= fc;
        }
    }

    let mut level = Elevator::new(n);
    for (v, &ex) in first_ids::<NodeId>(n).zip(&*excess) {
        if ex > V::zero() {
            level.activate(v);
        }
    }

    'active: while let Some(act) = level.highest_active() {
        let actlevel = level.level(act);
        let mut mlevel = n;
        let mut exc = excess[act];

        for j in ids(g.first_out[act]..g.first_out[act.next()]) {
            // Outgoing arcs carry flow forward; incoming arcs carry it by
            // cancelling their flow. The arc to the root is not part of the
            // circulation's graph.
            let v = g.target[j];
            let residual = if g.forward[j] {
                cap[j] - flow[j]
            } else if v.index() < n {
                flow[g.reverse[j]]
            } else {
                continue;
            };
            if residual <= V::zero() {
                continue;
            }
            if level.level(v) >= actlevel {
                mlevel = mlevel.min(level.level(v));
                continue;
            }

            let pushed = residual.min(exc);
            if g.forward[j] {
                flow[j] += pushed;
            } else {
                flow[g.reverse[j]] -= pushed;
            }
            excess[v] += pushed;
            if !level.active(v) && excess[v] > V::zero() {
                level.activate(v);
            }
            if residual >= exc {
                excess[act] = V::zero();
                level.deactivate(act);
                continue 'active;
            }
            exc -= residual;
        }

        excess[act] = exc;
        if exc <= V::zero() {
            level.deactivate(act);
        } else if mlevel == n {
            // No admissible arc can ever appear: `act` is behind a barrier.
            return Err(Error::Infeasible);
        } else {
            level.lift_highest_active(mlevel + 1);
            if level.on_level(actlevel) == 0 {
                // Emptying a level cuts every node above it off from the
                // deficits below, which also proves infeasibility.
                return Err(Error::Infeasible);
            }
        }
    }
    Ok(())
}

/// Bucketed node levels for push-relabel.
///
/// All items live in one array partitioned into contiguous per-level
/// segments `first[l]..first[l + 1]`; within a segment the active items come
/// first, at `first[l]..active_end[l]`.
struct Elevator {
    max_level: usize,
    items: Vec<NodeId>,
    where_: IdVec<NodeId, usize>,
    level: IdVec<NodeId, usize>,
    first: Vec<usize>,
    active_end: Vec<usize>,
    highest_active: Option<usize>,
}

impl Elevator {
    /// Creates an elevator over items `0..item_num`, all on level 0 and
    /// inactive, with levels `0..=item_num`.
    fn new(item_num: usize) -> Self {
        // Everything is on level 0, so every higher level starts at the end.
        let mut first = vec![item_num; item_num + 2];
        first[0] = 0;
        Elevator {
            max_level: item_num,
            items: first_ids(item_num).collect(),
            where_: (0..item_num).collect(),
            level: IdVec::filled(item_num, 0),
            active_end: first.clone(),
            first,
            highest_active: None,
        }
    }

    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.items.swap(i, j);
        self.where_[self.items[i]] = i;
        self.where_[self.items[j]] = j;
    }

    /// Moves the item at position `s` to position `p`.
    #[inline]
    fn copy_pos(&mut self, s: usize, p: usize) {
        if s != p {
            self.copy_item(self.items[s], p);
        }
    }

    #[inline]
    fn copy_item(&mut self, item: NodeId, p: usize) {
        self.items[p] = item;
        self.where_[item] = p;
    }

    fn level(&self, i: NodeId) -> usize {
        self.level[i]
    }

    fn active(&self, i: NodeId) -> bool {
        self.where_[i] < self.active_end[self.level(i)]
    }

    fn activate(&mut self, i: NodeId) {
        let l = self.level(i);
        self.swap(self.where_[i], self.active_end[l]);
        self.active_end[l] += 1;
        if self.highest_active.is_none_or(|h| l > h) {
            self.highest_active = Some(l);
        }
    }

    fn deactivate(&mut self, i: NodeId) {
        let l = self.level(i);
        self.active_end[l] -= 1;
        self.swap(self.where_[i], self.active_end[l]);
        self.drop_empty_highest();
    }

    fn drop_empty_highest(&mut self) {
        while let Some(h) = self.highest_active {
            if self.active_end[h] > self.first[h] {
                break;
            }
            self.highest_active = h.checked_sub(1);
        }
    }

    fn on_level(&self, l: usize) -> usize {
        self.first[l + 1] - self.first[l]
    }

    fn highest_active(&self) -> Option<NodeId> {
        self.highest_active
            .map(|h| self.items[self.active_end[h] - 1])
    }

    /// Lifts the highest active item to `new_level`, shifting the level
    /// boundaries in between down by one slot.
    fn lift_highest_active(&mut self, new_level: usize) {
        let ha = self.highest_active.expect("an item is active");
        self.active_end[ha] -= 1;
        let la = self.active_end[ha];
        let li = self.items[la];

        self.first[ha + 1] -= 1;
        self.copy_pos(self.first[ha + 1], la);
        // The levels in between have no active items, so their active
        // prefixes stay empty as their segments shift down.
        for l in ha + 1..new_level {
            self.first[l + 1] -= 1;
            self.copy_pos(self.first[l + 1], self.first[l]);
            self.active_end[l] -= 1;
        }
        self.copy_item(li, self.first[new_level]);
        self.level[li] = new_level;
        self.highest_active = Some(new_level);
        debug_assert!(new_level <= self.max_level);
    }
}
