//! Pivot rules: strategies for choosing the entering arc.

use alloc::vec;
use alloc::vec::Vec;
use core::hint::cold_path;
use core::ops::{ControlFlow, Range};

use super::ArcState;
use crate::Number;
use crate::ivec::{IRef, NodeIx};

/// The parts of the solver state a pivot rule reads.
pub(super) struct PivotView<'a, C> {
    pub(super) source: &'a [NodeIx],
    pub(super) target: &'a [NodeIx],
    pub(super) cost: &'a [C],
    pub(super) state: &'a [ArcState],
    pub(super) pi: IRef<'a, NodeIx, C>,
    pub(super) search_arc_num: usize,
}

impl<C: Number> PivotView<'_, C> {
    /// `state * reduced_cost`: negative exactly when the arc is eligible to
    /// enter the basis.
    #[inline(always)]
    fn eligibility(&self, e: usize) -> C {
        self.state[e].sign::<C>()
            * (self.cost[e] + self.pi[self.source[e]] - self.pi[self.target[e]])
    }

    /// Calls `f(e, eligibility(e))` for each arc in `range`, in order,
    /// stopping early with the value `f` breaks with.
    ///
    /// Zipping the arc arrays lets the compiler drop their bounds checks,
    /// which are most of the per-arc cost of the scan. Callers mark their
    /// "new minimum" branch with `cold_path`: it is rarely taken, and as a
    /// conditional move it would chain every iteration on the previous
    /// minimum, which costs up to 20% on scan-heavy inputs.
    #[inline(always)]
    fn scan<B>(
        &self,
        range: Range<usize>,
        mut f: impl FnMut(usize, C) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let arcs = self.state[range.clone()]
            .iter()
            .zip(&self.cost[range.clone()])
            .zip(&self.source[range.clone()])
            .zip(&self.target[range.clone()]);
        for (i, (((&state, &cost), &source), &target)) in arcs.enumerate() {
            let c = state.sign::<C>() * (cost + self.pi[source] - self.pi[target]);
            f(range.start + i, c)?;
        }
        ControlFlow::Continue(())
    }
}

pub(super) trait Pivot<C> {
    fn new(search_arc_num: usize) -> Self;
    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize>;
}

// LEMON sizes its lists by floating-point factors of `sqrt(m)`; the integer
// forms below compute exactly the same values without `std`.

/// About `sqrt(m)`.
fn block_size(search_arc_num: usize) -> usize {
    const MIN_BLOCK_SIZE: usize = 10;
    search_arc_num.isqrt().max(MIN_BLOCK_SIZE)
}

pub(super) struct FirstEligible {
    next_arc: usize,
}

impl<C: Number> Pivot<C> for FirstEligible {
    fn new(_: usize) -> Self {
        FirstEligible { next_arc: 0 }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let n = view.search_arc_num;
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    ControlFlow::Break(e)
                } else {
                    ControlFlow::Continue(())
                }
            });
            if let ControlFlow::Break(e) = found {
                self.next_arc = e + 1;
                return Some(e);
            }
        }
        None
    }
}

pub(super) struct BestEligible;

impl<C: Number> Pivot<C> for BestEligible {
    fn new(_: usize) -> Self {
        BestEligible
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let mut min = C::zero();
        let mut in_arc = None;
        let _ = view.scan(0..view.search_arc_num, |e, c| {
            if c < min {
                cold_path();
                min = c;
                in_arc = Some(e);
            }
            ControlFlow::<()>::Continue(())
        });
        in_arc
    }
}

pub(super) struct BlockSearch {
    block_size: usize,
    next_arc: usize,
}

impl<C: Number> Pivot<C> for BlockSearch {
    fn new(search_arc_num: usize) -> Self {
        BlockSearch {
            block_size: block_size(search_arc_num),
            next_arc: 0,
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let n = view.search_arc_num;
        let mut min = C::zero();
        let mut in_arc = 0;
        let mut cnt = self.block_size;
        let block_size = self.block_size;
        // Scan from next_arc, wrapping around, and stop at the end of the
        // first block that contains an eligible arc.
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < min {
                    cold_path();
                    min = c;
                    in_arc = e;
                }
                cnt -= 1;
                if cnt == 0 {
                    if min < C::zero() {
                        return ControlFlow::Break(e);
                    }
                    cnt = block_size;
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                self.next_arc = e;
                return Some(in_arc);
            }
        }
        (min < C::zero()).then_some(in_arc)
    }
}

pub(super) struct CandidateList {
    candidates: Vec<usize>,
    list_length: usize,
    minor_limit: usize,
    minor_count: usize,
    next_arc: usize,
}

impl<C: Number> Pivot<C> for CandidateList {
    fn new(search_arc_num: usize) -> Self {
        const MIN_LIST_LENGTH: usize = 10;
        const MIN_MINOR_LIMIT: usize = 3;

        // A quarter of `sqrt(m)` long, rebuilt after a tenth as many minor
        // iterations.
        let list_length = (search_arc_num.isqrt() / 4).max(MIN_LIST_LENGTH);
        let minor_limit = (list_length / 10).max(MIN_MINOR_LIMIT);
        CandidateList {
            candidates: Vec::with_capacity(list_length),
            list_length,
            minor_limit,
            minor_count: 0,
            next_arc: 0,
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        let mut in_arc = 0;
        if !self.candidates.is_empty() && self.minor_count < self.minor_limit {
            // Minor iteration: select the best eligible arc from the current
            // candidate list, dropping arcs that are no longer eligible.
            self.minor_count += 1;
            let mut min = C::zero();
            let mut i = 0;
            while i < self.candidates.len() {
                let e = self.candidates[i];
                let c = view.eligibility(e);
                if c < min {
                    cold_path();
                    min = c;
                    in_arc = e;
                } else if c >= C::zero() {
                    self.candidates.swap_remove(i);
                    continue;
                }
                i += 1;
            }
            if min < C::zero() {
                return Some(in_arc);
            }
        }

        // Major iteration: build a new candidate list
        let n = view.search_arc_num;
        let mut min = C::zero();
        self.candidates.clear();
        let (candidates, list_length) = (&mut self.candidates, self.list_length);
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    candidates.push(e);
                    if c < min {
                        cold_path();
                        min = c;
                        in_arc = e;
                    }
                    if candidates.len() == list_length {
                        return ControlFlow::Break(e);
                    }
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                self.minor_count = 1;
                self.next_arc = e;
                return Some(in_arc);
            }
        }
        if self.candidates.is_empty() {
            return None;
        }
        self.minor_count = 1;
        Some(in_arc)
    }
}

pub(super) struct AlteringList<C> {
    block_size: usize,
    head_length: usize,
    next_arc: usize,
    candidates: Vec<usize>,
    cand_cost: Vec<C>,
}

impl<C: Number> Pivot<C> for AlteringList<C> {
    fn new(search_arc_num: usize) -> Self {
        const MIN_HEAD_LENGTH: usize = 3;

        // Keeps a hundredth of a block between iterations.
        let block_size = block_size(search_arc_num);
        let head_length = (block_size / 100).max(MIN_HEAD_LENGTH);
        AlteringList {
            block_size,
            head_length,
            next_arc: 0,
            candidates: Vec::with_capacity(head_length + block_size),
            cand_cost: vec![C::zero(); search_arc_num],
        }
    }

    fn find_entering_arc(&mut self, view: &PivotView<'_, C>) -> Option<usize> {
        // Refresh the kept candidates, dropping ineligible ones
        let mut i = 0;
        while i < self.candidates.len() {
            let e = self.candidates[i];
            let c = view.eligibility(e);
            if c < C::zero() {
                self.cand_cost[e] = c;
                i += 1;
            } else {
                self.candidates.swap_remove(i);
            }
        }

        // Extend the list block by block. The first block must add more
        // than head_length candidates to stop; later blocks, any at all.
        let n = view.search_arc_num;
        let mut cnt = self.block_size;
        let mut limit = self.head_length;
        let mut stop_at = None;
        let (candidates, cand_cost, block_size) =
            (&mut self.candidates, &mut self.cand_cost, self.block_size);
        for range in [self.next_arc..n, 0..self.next_arc] {
            let found = view.scan(range, |e, c| {
                if c < C::zero() {
                    cand_cost[e] = c;
                    candidates.push(e);
                }
                cnt -= 1;
                if cnt == 0 {
                    if candidates.len() > limit {
                        return ControlFlow::Break(e);
                    }
                    limit = 0;
                    cnt = block_size;
                }
                ControlFlow::Continue(())
            });
            if let ControlFlow::Break(e) = found {
                stop_at = Some(e);
                break;
            }
        }
        if stop_at.is_none() && self.candidates.is_empty() {
            return None;
        }
        if let Some(e) = stop_at {
            self.next_arc = e;
        }

        // Move the best head_length + 1 candidates to the front, best first
        let new_length = (self.head_length + 1).min(self.candidates.len());
        let cand_cost = &self.cand_cost;
        if new_length < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by_key(new_length - 1, |&e| cand_cost[e]);
        }
        self.candidates[..new_length].sort_unstable_by_key(|&e| cand_cost[e]);

        // Take the best as the entering arc and keep the rest of the head
        self.candidates.truncate(new_length);
        Some(self.candidates.swap_remove(0))
    }
}
