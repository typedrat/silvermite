use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeTo};

use nonmax::NonMaxU32;

/// A dense index into the arrays of one kind of solver entity.
pub(crate) trait Idx: Copy + Eq + Ord + Debug {
    fn new(i: usize) -> Self;

    fn index(self) -> usize;

    /// The indices in `range`, in order.
    ///
    /// Counting in the stored representation keeps the loop counter from
    /// being truncated and re-extended on every array access.
    fn ids(range: Range<Self>) -> impl DoubleEndedIterator<Item = Self> + Clone;
}

impl Idx for usize {
    #[inline(always)]
    fn new(i: usize) -> Self {
        i
    }

    #[inline(always)]
    fn index(self) -> usize {
        self
    }

    #[inline(always)]
    fn ids(range: Range<Self>) -> impl DoubleEndedIterator<Item = Self> + Clone {
        range
    }
}

macro_rules! id_type {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        ///
        /// Stored as a `u32` to halve the memory traffic of the index arrays
        /// the solvers are bound by. Problems too large to leave `u32::MAX`
        /// free are rejected before solving, so every real index fits and
        /// that value stays available as the niche of [`Link`].
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(u32);

        impl Idx for $name {
            #[inline(always)]
            fn new(i: usize) -> Self {
                debug_assert!(i < u32::MAX as usize);
                $name(i as u32)
            }

            #[inline(always)]
            fn index(self) -> usize {
                self.0 as usize
            }

            #[inline(always)]
            fn ids(range: Range<Self>) -> impl DoubleEndedIterator<Item = Self> + Clone {
                (range.start.0..range.end.0).map($name)
            }
        }

        impl $name {
            #[inline(always)]
            pub(crate) fn next(self) -> Self {
                $name(self.0 + 1)
            }

            #[inline(always)]
            pub(crate) fn prev(self) -> Self {
                $name(self.0 - 1)
            }
        }
    };
}

id_type!(
    /// A node of a solver's internal graph.
    NodeIx
);

id_type!(
    /// An arc of a solver's internal graph.
    ArcIx
);

/// The indices in `range`, in order.
#[inline(always)]
pub(crate) fn ids<I: Idx>(range: Range<I>) -> impl DoubleEndedIterator<Item = I> + Clone {
    I::ids(range)
}

/// The first `len` indices.
#[inline(always)]
pub(crate) fn first_ids<I: Idx>(len: usize) -> impl DoubleEndedIterator<Item = I> + Clone {
    I::ids(I::new(0)..I::new(len))
}

/// An optional index packed into four bytes, with `u32::MAX` as the niche
/// for `None`.
///
/// Decoding costs an xor, so arrays walked as long chains of dependent loads
/// are better off without it.
pub(crate) struct Link<I>(Option<NonMaxU32>, PhantomData<I>);

impl<I> Clone for Link<I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for Link<I> {}

impl<I> PartialEq for Link<I> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<I> Eq for Link<I> {}

impl<I> Debug for Link<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<I: Idx> Link<I> {
    pub(crate) const NONE: Self = Link(None, PhantomData);

    #[inline(always)]
    pub(crate) fn to(i: I) -> Self {
        Link(NonMaxU32::new(i.index() as u32), PhantomData)
    }

    #[inline(always)]
    pub(crate) fn get(self) -> Option<I> {
        self.0.map(|i| I::new(i.get() as usize))
    }
}

const CACHE_LINE: usize = 64;

/// A `Vec` indexed by `I`, so the solvers can store typed node and arc
/// indices compactly and chain lookups like `thread[last_succ[u]]`.
///
/// Each array in a solver gets a distinct `slot`, and its data starts that
/// many cache lines into its allocation. Large allocations are page-aligned,
/// so without the stagger, element `i` of every array would map to the same
/// cache set; walks that touch several arrays at one index then evict each
/// other, which measurably slows network simplex (by 40% on some inputs).
#[derive(Clone, Debug)]
pub(crate) struct IVec<I, T> {
    buf: Vec<T>,
    off: usize,
    slot: usize,
    _index: PhantomData<I>,
}

impl<I, T> Default for IVec<I, T> {
    fn default() -> Self {
        Self::slot(0)
    }
}

impl<I, T> IVec<I, T> {
    pub(crate) fn slot(slot: usize) -> Self {
        IVec {
            buf: Vec::new(),
            off: 0,
            slot,
            _index: PhantomData,
        }
    }

    pub(crate) fn as_mut(&mut self) -> IMut<'_, I, T> {
        IMut::new(&mut self.buf[self.off..])
    }

    pub(crate) fn as_ref(&self) -> IRef<'_, I, T> {
        IRef::new(&self.buf[self.off..])
    }
}

impl<I, T: Clone> IVec<I, T> {
    /// Resizes to `len` and overwrites every element with `value`.
    pub(crate) fn reset(&mut self, len: usize, value: T) {
        let per_line = (CACHE_LINE / size_of::<T>().max(1)).max(1);
        self.off = self.slot * per_line;
        self.buf.clear();
        self.buf.resize(self.off + len, value);
    }
}

impl<I: Idx, T> Index<I> for IVec<I, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: I) -> &T {
        &self.buf[self.off + i.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IVec<I, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: I) -> &mut T {
        &mut self.buf[self.off + i.index()]
    }
}

impl<I, T> Deref for IVec<I, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.buf[self.off..]
    }
}

impl<I, T> DerefMut for IVec<I, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf[self.off..]
    }
}

/// A plain `Vec` indexed by `I`, for scratch arrays.
///
/// Unlike [`IVec`], indexing it adds no stagger offset, so it suits arrays
/// indexed directly in hot loops rather than through a view.
#[derive(Clone, Debug)]
pub(crate) struct IdVec<I, T>(Vec<T>, PhantomData<I>);

impl<I, T: Clone> IdVec<I, T> {
    /// `len` copies of `value`.
    pub(crate) fn filled(len: usize, value: T) -> Self {
        IdVec(vec![value; len], PhantomData)
    }
}

impl<I, T> IdVec<I, T> {
    pub(crate) fn as_mut(&mut self) -> IMut<'_, I, T> {
        IMut::new(&mut self.0)
    }

    pub(crate) fn as_ref(&self) -> IRef<'_, I, T> {
        IRef::new(&self.0)
    }
}

impl<I, T> FromIterator<T> for IdVec<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        IdVec(iter.into_iter().collect(), PhantomData)
    }
}

impl<I: Idx, T> Index<I> for IdVec<I, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: I) -> &T {
        &self.0[i.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IdVec<I, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: I) -> &mut T {
        &mut self.0[i.index()]
    }
}

impl<I, T> Deref for IdVec<I, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<I, T> DerefMut for IdVec<I, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// A mutable slice indexed by `I`.
///
/// Hot loops borrow their arrays through these instead of indexing struct
/// fields, so the slice's pointer and length can stay in registers.
pub(crate) struct IMut<'a, I, T>(&'a mut [T], PhantomData<I>);

impl<'a, I, T> IMut<'a, I, T> {
    pub(crate) fn new(slice: &'a mut [T]) -> Self {
        IMut(slice, PhantomData)
    }

    pub(crate) fn as_ref(&self) -> IRef<'_, I, T> {
        IRef::new(self.0)
    }
}

impl<I: Idx, T> Index<I> for IMut<'_, I, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: I) -> &T {
        &self.0[i.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IMut<'_, I, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: I) -> &mut T {
        &mut self.0[i.index()]
    }
}

impl<I, T> Deref for IMut<'_, I, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.0
    }
}

impl<I, T> DerefMut for IMut<'_, I, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.0
    }
}

/// A shared slice indexed by `I`; see [`IMut`].
pub(crate) struct IRef<'a, I, T>(&'a [T], PhantomData<I>);

impl<'a, I, T> IRef<'a, I, T> {
    pub(crate) fn new(slice: &'a [T]) -> Self {
        IRef(slice, PhantomData)
    }
}

impl<I, T> Clone for IRef<'_, I, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I, T> Copy for IRef<'_, I, T> {}

impl<I: Idx, T> Index<I> for IRef<'_, I, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: I) -> &T {
        &self.0[i.index()]
    }
}

impl<I, T> Deref for IRef<'_, I, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.0
    }
}

// Indexing by a range of typed indices yields the plain sub-slice.
macro_rules! index_ranges {
    ($($range:ident),*) => {$(
        impl<I: Idx, T> Index<$range<I>> for IVec<I, T> {
            type Output = [T];

            #[inline(always)]
            fn index(&self, r: $range<I>) -> &[T] {
                &(**self)[r.to_positions()]
            }
        }

        impl<I: Idx, T> IndexMut<$range<I>> for IVec<I, T> {
            #[inline(always)]
            fn index_mut(&mut self, r: $range<I>) -> &mut [T] {
                &mut (**self)[r.to_positions()]
            }
        }

        impl<I: Idx, T> Index<$range<I>> for IdVec<I, T> {
            type Output = [T];

            #[inline(always)]
            fn index(&self, r: $range<I>) -> &[T] {
                &self.0[r.to_positions()]
            }
        }

        impl<I: Idx, T> IndexMut<$range<I>> for IdVec<I, T> {
            #[inline(always)]
            fn index_mut(&mut self, r: $range<I>) -> &mut [T] {
                &mut self.0[r.to_positions()]
            }
        }

        impl<I: Idx, T> Index<$range<I>> for IMut<'_, I, T> {
            type Output = [T];

            #[inline(always)]
            fn index(&self, r: $range<I>) -> &[T] {
                &self.0[r.to_positions()]
            }
        }

        impl<I: Idx, T> IndexMut<$range<I>> for IMut<'_, I, T> {
            #[inline(always)]
            fn index_mut(&mut self, r: $range<I>) -> &mut [T] {
                &mut self.0[r.to_positions()]
            }
        }

        impl<I: Idx, T> Index<$range<I>> for IRef<'_, I, T> {
            type Output = [T];

            #[inline(always)]
            fn index(&self, r: $range<I>) -> &[T] {
                &self.0[r.to_positions()]
            }
        }
    )*};
}

index_ranges!(Range, RangeFrom, RangeTo);

trait ToPositions {
    type Positions;

    fn to_positions(self) -> Self::Positions;
}

impl<I: Idx> ToPositions for Range<I> {
    type Positions = Range<usize>;

    #[inline(always)]
    fn to_positions(self) -> Range<usize> {
        self.start.index()..self.end.index()
    }
}

impl<I: Idx> ToPositions for RangeFrom<I> {
    type Positions = RangeFrom<usize>;

    #[inline(always)]
    fn to_positions(self) -> RangeFrom<usize> {
        self.start.index()..
    }
}

impl<I: Idx> ToPositions for RangeTo<I> {
    type Positions = RangeTo<usize>;

    #[inline(always)]
    fn to_positions(self) -> RangeTo<usize> {
        ..self.end.index()
    }
}
