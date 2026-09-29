//! Feasible circulation by highest-label push-relabel, ported from LEMON's
//! `Circulation` and `Elevator`, specialized to zero lower bounds.
//!
//! It runs directly on cost scaling's residual layout, where each real node's
//! arc block lists its outgoing arcs (forward), then its incoming arcs
//! (backward), then one arc to the artificial root.

use crate::Number;

/// The residual graph layout the circulation reads, restricted to the real
/// nodes `0..node_num`.
pub(crate) struct Layout<'a> {
    pub node_num: usize,
    pub first_out: &'a [u32],
    pub forward: &'a [bool],
    pub target: &'a [u32],
    pub reverse: &'a [u32],
}

/// Searches for a flow with `0 <= flow <= cap` on every arc and
/// `out(v) - in(v) >= supply(v)` at every node.
///
/// `cap` and `flow` are indexed by forward residual arc; `arcs` lists every
/// forward arc once, in the order the greedy initialization visits them.
/// Returns false if no such flow exists.
pub(crate) fn circulation<V: Number>(
    g: &Layout<'_>,
    arcs: &[u32],
    cap: &[V],
    supply: &[V],
    flow: &mut [V],
) -> bool {
    let n = g.node_num;
    let mut excess: Vec<V> = supply[..n].to_vec();

    // Greedy initialization: send as much as the target still demands.
    // Arcs are visited newest-first; on inputs listed by source node, as
    // generator output usually is, insertion order leaves far more excess
    // for the push-relabel phase (100x slower on GOTO instances).
    for &e in arcs.iter().rev() {
        let e = e as usize;
        let t = g.target[e] as usize;
        let s = g.target[g.reverse[e] as usize] as usize;
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
    for (v, &ex) in excess.iter().enumerate() {
        if ex > V::zero() {
            level.activate(v as u32);
        }
    }

    while let Some(act) = level.highest_active() {
        let act_u = act as usize;
        let actlevel = level.level(act);
        let mut mlevel = n as u32;
        let mut exc = excess[act_u];
        let mut discharged = false;

        let block = g.first_out[act_u] as usize..g.first_out[act_u + 1] as usize;
        for j in block {
            if g.forward[j] {
                // Outgoing arc: push along it.
                let v = g.target[j];
                let fc = cap[j] - flow[j];
                if fc <= V::zero() {
                    continue;
                }
                if level.level(v) < actlevel {
                    let v_u = v as usize;
                    if fc >= exc {
                        flow[j] += exc;
                        excess[v_u] += exc;
                        if !level.active(v) && excess[v_u] > V::zero() {
                            level.activate(v);
                        }
                        excess[act_u] = V::zero();
                        level.deactivate(act);
                        discharged = true;
                        break;
                    }
                    flow[j] = cap[j];
                    excess[v_u] += fc;
                    if !level.active(v) && excess[v_u] > V::zero() {
                        level.activate(v);
                    }
                    exc -= fc;
                } else if level.level(v) < mlevel {
                    mlevel = level.level(v);
                }
            } else if (g.target[j] as usize) < n {
                // Incoming arc: cancel flow on it. The arc to the root is
                // not part of the circulation's graph.
                let v = g.target[j];
                let e = g.reverse[j] as usize;
                let fc = flow[e];
                if fc <= V::zero() {
                    continue;
                }
                if level.level(v) < actlevel {
                    let v_u = v as usize;
                    if fc >= exc {
                        flow[e] -= exc;
                        excess[v_u] += exc;
                        if !level.active(v) && excess[v_u] > V::zero() {
                            level.activate(v);
                        }
                        excess[act_u] = V::zero();
                        level.deactivate(act);
                        discharged = true;
                        break;
                    }
                    flow[e] = V::zero();
                    excess[v_u] += fc;
                    if !level.active(v) && excess[v_u] > V::zero() {
                        level.activate(v);
                    }
                    exc -= fc;
                } else if level.level(v) < mlevel {
                    mlevel = level.level(v);
                }
            }
        }
        if discharged {
            continue;
        }

        excess[act_u] = exc;
        if exc <= V::zero() {
            level.deactivate(act);
        } else if mlevel == n as u32 {
            // No admissible arc can ever appear: `act` is behind a barrier.
            return false;
        } else {
            level.lift_highest_active(mlevel + 1);
            if level.on_level(actlevel) == 0 {
                // Emptying a level cuts every node above it off from the
                // deficits below, which also proves infeasibility.
                return false;
            }
        }
    }
    true
}

/// Bucketed node levels for push-relabel.
///
/// All items live in one array partitioned into contiguous per-level
/// segments; within a segment the active items come first. Positions are
/// signed because an empty active prefix is marked by
/// `last_active[l] == first[l] - 1`.
struct Elevator {
    max_level: u32,
    items: Vec<u32>,
    where_: Vec<isize>,
    level: Vec<u32>,
    first: Vec<isize>,
    last_active: Vec<isize>,
    highest_active: isize,
}

impl Elevator {
    /// Creates an elevator over items `0..item_num`, all on level 0 and
    /// inactive, with levels `0..=item_num`.
    fn new(item_num: usize) -> Self {
        let max_level = item_num as u32;
        let mut first = vec![0isize; item_num + 2];
        let mut last_active = vec![-1isize; item_num + 2];
        // Everything is on level 0, so every higher level starts at the end.
        for l in 1..=item_num + 1 {
            first[l] = item_num as isize;
            last_active[l] = item_num as isize - 1;
        }
        Elevator {
            max_level,
            items: (0..item_num as u32).collect(),
            where_: (0..item_num as isize).collect(),
            level: vec![0; item_num],
            first,
            last_active,
            highest_active: -1,
        }
    }

    #[inline]
    fn swap(&mut self, i: isize, j: isize) {
        let (i, j) = (i as usize, j as usize);
        let ti = self.items[i];
        let tj = self.items[j];
        self.items[i] = tj;
        self.items[j] = ti;
        self.where_[ti as usize] = j as isize;
        self.where_[tj as usize] = i as isize;
    }

    /// Moves the item at position `s` to position `p`.
    #[inline]
    fn copy_pos(&mut self, s: isize, p: isize) {
        if s != p {
            let item = self.items[s as usize];
            self.items[p as usize] = item;
            self.where_[item as usize] = p;
        }
    }

    #[inline]
    fn copy_item(&mut self, item: u32, p: isize) {
        self.items[p as usize] = item;
        self.where_[item as usize] = p;
    }

    fn level(&self, i: u32) -> u32 {
        self.level[i as usize]
    }

    fn active(&self, i: u32) -> bool {
        self.where_[i as usize] <= self.last_active[self.level[i as usize] as usize]
    }

    fn activate(&mut self, i: u32) {
        let l = self.level[i as usize] as usize;
        self.last_active[l] += 1;
        self.swap(self.where_[i as usize], self.last_active[l]);
        if l as isize > self.highest_active {
            self.highest_active = l as isize;
        }
    }

    fn deactivate(&mut self, i: u32) {
        let l = self.level[i as usize] as usize;
        self.swap(self.where_[i as usize], self.last_active[l]);
        self.last_active[l] -= 1;
        self.drop_empty_highest();
    }

    fn drop_empty_highest(&mut self) {
        while self.highest_active >= 0 {
            let h = self.highest_active as usize;
            if self.last_active[h] >= self.first[h] {
                break;
            }
            self.highest_active -= 1;
        }
    }

    fn on_level(&self, l: u32) -> isize {
        let l = l as usize;
        self.first[l + 1] - self.first[l]
    }

    fn highest_active(&self) -> Option<u32> {
        (self.highest_active >= 0)
            .then(|| self.items[self.last_active[self.highest_active as usize] as usize])
    }

    /// Lifts the highest active item to `new_level`, shifting the level
    /// boundaries in between down by one slot.
    fn lift_highest_active(&mut self, new_level: u32) {
        let ha = self.highest_active as usize;
        let li = self.items[self.last_active[ha] as usize];

        self.first[ha + 1] -= 1;
        let la = self.last_active[ha];
        self.last_active[ha] -= 1;
        self.copy_pos(self.first[ha + 1], la);
        for l in ha + 1..new_level as usize {
            self.first[l + 1] -= 1;
            self.copy_pos(self.first[l + 1], self.first[l]);
            self.last_active[l] -= 1;
        }
        self.copy_item(li, self.first[new_level as usize]);
        self.level[li as usize] = new_level;
        self.highest_active = new_level as isize;
        debug_assert!(new_level <= self.max_level);
    }
}
