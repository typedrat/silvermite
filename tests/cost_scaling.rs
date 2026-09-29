use silvermite::{CostScaling, Error, Problem};

#[test]
fn scale_overflow_with_zero_costs() {
    // With every cost zero, no scaled cost overflows, but the scale itself,
    // (15 + 1) * 16 = 256, does not fit in i8.
    let mut p = Problem::<i64, i64>::new(15);
    p.set_st_supply(0, 14, 3);
    for v in 0..14 {
        p.add_arc(v, v + 1, 0, 5, 0);
    }
    let result = CostScaling::<i64, i64, i8>::default().solve(&p);
    assert_eq!(result.err(), Some(Error::Overflow));

    let solution = CostScaling::<i64, i64, i16>::default().solve(&p).unwrap();
    assert_eq!(solution.total_cost(), 0);
}
