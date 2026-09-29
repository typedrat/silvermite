//! Building the residual graph and finding an initial feasible flow.

use itertools::izip;

use super::CostScaling;
use crate::circulation::{Layout, circulation};
use crate::ivec::{ArcIx, IdVec, Idx, Link, NodeIx, first_ids};
use crate::{Error, Number, Problem, SupplyType};

impl<V: Number, C: Number, L: Number> CostScaling<V, C, L> {
    /// Builds the residual graph and copies the problem data.
    ///
    /// Each real node's arc block holds its outgoing arcs (forward), its
    /// incoming arcs (backward), and a backward arc to the artificial root.
    /// The root's block holds the matching forward arcs to every node.
    pub(super) fn load(&mut self, p: &Problem<V, C>) {
        let n = p.node_count();
        let m = p.arc_count();
        self.node_num = n;
        self.res_node_num = n + 1;
        self.res_arc_num = 2 * (m + n);
        let root = NodeIx::new(n);
        self.root = root;
        let res_node_num = self.res_node_num;
        let res_arc_num = self.res_arc_num;

        // Mirroring every arc and negating supplies turns
        // `out - in <= supply` into `out - in >= -supply`.
        self.mirrored = p.supply_type == SupplyType::Leq;
        let (src, tgt) = if self.mirrored {
            (&p.target, &p.source)
        } else {
            (&p.source, &p.target)
        };

        self.first_out.reset(res_node_num + 1, ArcIx::default());
        self.forward.reset(res_arc_num, false);
        self.source.reset(res_arc_num, NodeIx::default());
        self.target.reset(res_arc_num, NodeIx::default());
        self.reverse.reset(res_arc_num, ArcIx::default());
        self.lower.reset(res_arc_num, V::zero());
        self.upper.reset(res_arc_num, V::max_value());
        self.scost.reset(res_arc_num, C::zero());
        self.supply.reset(res_node_num, V::zero());
        self.res_cap.reset(res_arc_num, V::zero());
        self.uncapped.reset(res_arc_num, false);
        self.cost.reset(res_arc_num, L::zero());
        self.pi.reset(res_node_num, L::zero());
        self.excess.reset(res_node_num, V::zero());
        self.next_out.reset(res_node_num, ArcIx::default());

        // Block sizes, then block starts. `out_pos` and `in_pos` track the
        // next free forward and backward slot in each block.
        let mut outs = IdVec::<NodeIx, usize>::filled(n, 0);
        let mut ins = IdVec::<NodeIx, usize>::filled(n, 0);
        for (&s, &t) in src.iter().zip(tgt) {
            outs[s] += 1;
            ins[t] += 1;
        }
        let mut out_pos = IdVec::<NodeIx, ArcIx>::filled(n, ArcIx::default());
        let mut in_pos = IdVec::<NodeIx, ArcIx>::filled(n, ArcIx::default());
        let mut j = 0;
        for (first_out, out_pos, in_pos, &outs, &ins) in izip!(
            &mut *self.first_out,
            &mut *out_pos,
            &mut *in_pos,
            &*outs,
            &*ins,
        ) {
            *first_out = ArcIx::new(j);
            *out_pos = ArcIx::new(j);
            *in_pos = ArcIx::new(j + outs);
            j += outs + ins + 1;
        }
        self.first_out[root] = ArcIx::new(j);
        self.first_out[root.next()] = ArcIx::new(res_arc_num);

        self.arc_idf.clear();
        self.arc_idb.clear();
        for (&s, &t, &lower, &upper, &cost) in izip!(src, tgt, &p.lower, &p.upper, &p.cost) {
            let f = out_pos[s];
            out_pos[s] = f.next();
            let b = in_pos[t];
            in_pos[t] = b.next();
            self.arc_idf.push(f);
            self.arc_idb.push(b);
            self.forward[f] = true;
            self.source[f] = s;
            self.target[f] = t;
            self.reverse[f] = b;
            self.source[b] = t;
            self.target[b] = s;
            self.reverse[b] = f;

            self.lower[f] = lower;
            self.upper[f] = upper;
            self.scost[f] = cost;
            self.scost[b] = -cost;
        }

        for (i, k) in first_ids::<NodeIx>(n).zip(self.block(root)) {
            let j = self.first_out[i.next()].prev();
            self.source[j] = i;
            self.target[j] = root;
            self.reverse[j] = k;
            self.forward[k] = true;
            self.source[k] = root;
            self.target[k] = i;
            self.reverse[k] = j;
        }

        for (supply, &s) in self.supply.iter_mut().zip(&p.supply) {
            *supply = if self.mirrored { -s } else { s };
        }
        self.has_lower = p.lower.iter().any(|&l| l != V::zero());
    }

    pub(super) fn init(&mut self) -> Result<(), Error> {
        let n = self.node_num;
        let root = self.root;
        let res_node_num = self.res_node_num;
        let max = V::max_value();

        self.sum_supply = self.supply[..root]
            .iter()
            .fold(V::zero(), |sum, &s| sum + s);
        if self.sum_supply > V::zero() {
            return Err(Error::Infeasible);
        }

        self.pi.fill(L::zero());
        self.excess.copy_from_slice(&self.supply);

        // Reject infinite capacities on negative arcs, and bound the total
        // flow any arc can carry to replace the remaining infinite ones.
        for &f in &self.arc_idf {
            let c = if self.scost[f] < C::zero() {
                self.upper[f]
            } else if self.has_lower {
                self.lower[f]
            } else {
                continue;
            };
            if c >= max {
                return Err(Error::Unbounded);
            }
            self.excess[self.source[f]] -= c;
            self.excess[self.target[f]] += c;
        }
        let max_cap = self
            .excess
            .iter()
            .filter(|&&ex| ex < V::zero())
            .fold(V::zero(), |sum, &ex| sum - ex);
        self.excess.fill(V::zero());
        // Relative to its lower bound (upper bound for negative arcs), no
        // arc needs to carry more than the total deficit in some optimal
        // flow, so that bounds every infinite arc above its lower bound.
        let root_arcs = self.first_out[root];
        for (j, upper, &lower, &forward, uncapped) in izip!(
            first_ids::<ArcIx>(self.res_arc_num),
            &mut *self.upper,
            &*self.lower,
            &*self.forward,
            &mut *self.uncapped,
        ) {
            if *upper >= max {
                *uncapped = forward && j < root_arcs;
                *upper = lower.saturating_add(max_cap);
            }
        }

        // Scale the costs and set the initial epsilon
        let real_arcs = ..root_arcs;
        let alpha = self.alpha;
        let scale = res_node_num as i128 * alpha as i128;
        // At least 1, so the scale itself must fit too.
        let max_abs_cost = self.scost[real_arcs]
            .iter()
            .map(|&c| c.to_i128().abs())
            .max()
            .unwrap_or(0)
            .max(1);
        if max_abs_cost
            .checked_mul(scale)
            .is_none_or(|c| c > L::max_value().to_i128())
        {
            return Err(Error::Overflow);
        }
        let scale = L::from_i128(scale);
        for (cost, &scost) in self.cost[real_arcs].iter_mut().zip(&self.scost[real_arcs]) {
            *cost = scost.cast::<L>() * scale;
        }
        let max_cost = self.cost[real_arcs].iter().copied().max();
        self.epsilon = max_cost.unwrap_or(L::zero()).max(L::zero()) / L::from_i128(alpha as i128);

        // Find a feasible flow with lower bounds shifted to zero
        let mut cap = IdVec::<ArcIx, V>::filled(self.res_arc_num, V::zero());
        let mut sup = IdVec::<NodeIx, V>::filled(n, V::zero());
        sup.copy_from_slice(&self.supply[..root]);
        for &f in &self.arc_idf {
            let c = if self.has_lower {
                self.lower[f]
            } else {
                V::zero()
            };
            cap[f] = self.upper[f] - c;
            sup[self.source[f]] -= c;
            sup[self.target[f]] += c;
        }
        self.sup_node_num = sup.iter().filter(|&&s| s > V::zero()).count();

        let mut flow = IdVec::<ArcIx, V>::filled(self.res_arc_num, V::zero());
        let layout = Layout {
            node_num: n,
            first_out: self.first_out.as_ref(),
            forward: self.forward.as_ref(),
            target: self.target.as_ref(),
            reverse: self.reverse.as_ref(),
        };
        circulation(
            &layout,
            &self.arc_idf,
            cap.as_ref(),
            sup.as_ref(),
            flow.as_mut(),
        )?;

        // Set residual capacities; with a supply surplus to absorb, route the
        // leftover deficits through the root.
        for (&f, &b) in self.arc_idf.iter().zip(&self.arc_idb) {
            let fa = flow[f];
            self.res_cap[f] = cap[f] - fa;
            self.res_cap[b] = fa;
            if self.sum_supply < V::zero() {
                sup[self.source[f]] -= fa;
                sup[self.target[f]] += fa;
            }
        }
        if self.sum_supply < V::zero() {
            self.excess[..root].copy_from_slice(&sup);
            for a in self.block(root) {
                let u = self.target[a];
                let ra = self.reverse[a];
                self.res_cap[a] = -self.sum_supply + V::one();
                self.res_cap[ra] = -self.excess[u];
                self.cost[a] = L::zero();
                self.cost[ra] = L::zero();
                self.excess[u] = V::zero();
            }
        } else {
            for a in self.block(root) {
                let ra = self.reverse[a];
                self.res_cap[a] = V::zero();
                self.res_cap[ra] = V::zero();
                self.cost[a] = L::zero();
                self.cost[ra] = L::zero();
            }
        }

        self.max_rank = alpha * res_node_num as u32;
        self.buckets.reset(self.max_rank as usize, Link::NONE);
        self.bucket_next.reset(res_node_num, Link::NONE);
        self.bucket_prev.reset(res_node_num, NodeIx::default());
        self.rank.reset(res_node_num, 0);

        Ok(())
    }
}
