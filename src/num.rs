use core::fmt::{Debug, Display};
use core::hash::Hash;

use num_traits::{AsPrimitive, NumAssign, PrimInt, Signed};

/// A signed primitive integer usable for flow amounts, capacities, supplies,
/// and costs.
///
/// Implemented for every type that satisfies the bounds, which in practice
/// means `i8` through `i128` and `isize`. Floating-point types are excluded
/// on purpose: the solvers rely on exact arithmetic, and `max_value()`
/// doubles as the "infinite capacity" marker.
pub trait Number:
    PrimInt + Signed + NumAssign + Hash + Default + Debug + Display + Send + Sync + 'static
{
    /// Converts with `as` semantics (truncating on overflow).
    fn from_i128(v: i128) -> Self;

    fn to_i128(self) -> i128;

    #[inline(always)]
    fn from_i8(v: i8) -> Self {
        Self::from_i128(v as i128)
    }

    #[inline(always)]
    fn from_usize(v: usize) -> Self {
        Self::from_i128(v as i128)
    }

    /// Converts between number types with `as` semantics.
    #[inline(always)]
    fn cast<T: Number>(self) -> T {
        T::from_i128(self.to_i128())
    }
}

impl<T> Number for T
where
    T: PrimInt
        + Signed
        + NumAssign
        + AsPrimitive<i128>
        + Hash
        + Default
        + Debug
        + Display
        + Send
        + Sync
        + 'static,
    i128: AsPrimitive<T>,
{
    #[inline(always)]
    fn from_i128(v: i128) -> Self {
        v.as_()
    }

    #[inline(always)]
    fn to_i128(self) -> i128 {
        self.as_()
    }
}
