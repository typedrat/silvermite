use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Sentinel for "no node" / "no arc" in u32 index arrays.
pub(crate) const NONE: u32 = u32::MAX;

const CACHE_LINE: usize = 64;

/// A `Vec` indexed by `u32`, so the solvers can store node and arc indices
/// compactly and chain lookups like `thread[last_succ[u]]` without casts.
///
/// Each array in a solver gets a distinct `slot`, and its data starts that
/// many cache lines into its allocation. Large allocations are page-aligned,
/// so without the stagger, element `i` of every array would map to the same
/// cache set; walks that touch several arrays at one index then evict each
/// other, which measurably slows network simplex (by 40% on some inputs).
#[derive(Clone, Debug, Default)]
pub(crate) struct IVec<T> {
    buf: Vec<T>,
    off: usize,
    slot: usize,
}

impl<T> IVec<T> {
    pub(crate) fn slot(slot: usize) -> Self {
        IVec {
            buf: Vec::new(),
            off: 0,
            slot,
        }
    }

    pub(crate) fn as_mut(&mut self) -> IMut<'_, T> {
        IMut(&mut self.buf[self.off..])
    }

    pub(crate) fn as_ref(&self) -> IRef<'_, T> {
        IRef(&self.buf[self.off..])
    }
}

impl<T: Clone> IVec<T> {
    /// Resizes to `len` and overwrites every element with `value`.
    pub(crate) fn reset(&mut self, len: usize, value: T) {
        let per_line = (CACHE_LINE / size_of::<T>().max(1)).max(1);
        self.off = self.slot * per_line;
        self.buf.clear();
        self.buf.resize(self.off + len, value);
    }
}

impl<T> Index<u32> for IVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: u32) -> &T {
        &self.buf[self.off + i as usize]
    }
}

impl<T> IndexMut<u32> for IVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: u32) -> &mut T {
        &mut self.buf[self.off + i as usize]
    }
}

impl<T> Deref for IVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.buf[self.off..]
    }
}

impl<T> DerefMut for IVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf[self.off..]
    }
}

/// A mutable slice indexed by `u32`.
///
/// Hot loops borrow their arrays through these instead of indexing struct
/// fields, so the slice's pointer and length can stay in registers.
pub(crate) struct IMut<'a, T>(pub(crate) &'a mut [T]);

impl<T> Index<u32> for IMut<'_, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: u32) -> &T {
        &self.0[i as usize]
    }
}

impl<T> IndexMut<u32> for IMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: u32) -> &mut T {
        &mut self.0[i as usize]
    }
}

/// A shared slice indexed by `u32`; see [`IMut`].
#[derive(Clone, Copy)]
pub(crate) struct IRef<'a, T>(pub(crate) &'a [T]);

impl<T> Index<u32> for IRef<'_, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: u32) -> &T {
        &self.0[i as usize]
    }
}
