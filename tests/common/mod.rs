#![allow(dead_code)]

use silvermite::{
    CostScaling, Error, Method, NetworkSimplex, Number, PivotRule, Problem, Solution, SupplyType,
};

#[derive(Clone, Copy, Debug)]
pub enum Solver {
    NetworkSimplex(PivotRule),
    CostScaling(Method),
}

impl Solver {
    pub fn all() -> Vec<Solver> {
        use Method::*;
        use PivotRule::*;
        let ns = [
            FirstEligible,
            BestEligible,
            BlockSearch,
            CandidateList,
            AlteringList,
        ]
        .into_iter()
        .map(Solver::NetworkSimplex);
        let cs = [Push, Augment, PartialAugment]
            .into_iter()
            .map(Solver::CostScaling);
        ns.chain(cs).collect()
    }

    pub fn solve<V: Number, C: Number>(self, p: &Problem<V, C>) -> Result<Solution<V, C>, Error> {
        match self {
            Solver::NetworkSimplex(rule) => NetworkSimplex::new().pivot_rule(rule).solve(p),
            Solver::CostScaling(method) => CostScaling::<V, C>::new().method(method).solve(p),
        }
    }

    /// Whether the solver handles negative-cost arcs with infinite capacity
    /// when the objective is still bounded.
    pub fn full_negative_cost_support(self) -> bool {
        matches!(self, Solver::NetworkSimplex(_))
    }
}

/// Verifies a solution the way LEMON's test suite does: primal feasibility,
/// complementary slackness with the returned potentials, and equality of the
/// primal and dual objective values. Together these prove optimality.
pub fn check_solution<V: Number, C: Number>(
    p: &Problem<V, C>,
    s: &Solution<V, C>,
) -> Result<(), String> {
    let n = p.node_count();
    let m = p.arc_count();
    if s.flows().len() != m || s.potentials().len() != n {
        return Err("solution has the wrong shape".into());
    }
    let flow = |a: usize| s.flow(a).to_i128();
    let pi = |v: usize| s.potential(v).to_i128();
    let lower = |a: usize| p.lower(a).to_i128();
    let upper = |a: usize| p.upper(a).to_i128();
    let cost = |a: usize| p.cost(a).to_i128();
    let supply = |v: usize| p.supply(v).to_i128();
    let reduced = |a: usize| cost(a) + pi(p.source(a)) - pi(p.target(a));

    let mut balance = vec![0i128; n];
    for a in 0..m {
        if flow(a) < lower(a) || flow(a) > upper(a) {
            return Err(format!(
                "arc {a}: flow {} outside [{}, {}]",
                flow(a),
                lower(a),
                upper(a)
            ));
        }
        balance[p.source(a)] += flow(a);
        balance[p.target(a)] -= flow(a);
    }

    let total_supply: i128 = (0..n).map(supply).sum();
    let leq = p.supply_type() == SupplyType::Leq;
    for (v, &bal) in balance.iter().enumerate() {
        let ok = if total_supply == 0 {
            bal == supply(v)
        } else if leq {
            bal <= supply(v)
        } else {
            bal >= supply(v)
        };
        if !ok {
            return Err(format!(
                "node {v}: balance {bal} violates supply {}",
                supply(v)
            ));
        }
    }

    let total: i128 = (0..m).map(|a| flow(a) * cost(a)).sum();
    if total != s.total_cost() {
        return Err(format!(
            "reported cost {} but flow costs {total}",
            s.total_cost()
        ));
    }

    for a in 0..m {
        let rc = reduced(a);
        let ok = rc == 0 || (rc > 0 && flow(a) == lower(a)) || (rc < 0 && flow(a) == upper(a));
        if !ok {
            return Err(format!("arc {a}: reduced cost {rc} with flow {}", flow(a)));
        }
    }
    for (v, &bal) in balance.iter().enumerate() {
        let sign_ok = if leq { pi(v) >= 0 } else { pi(v) <= 0 };
        if !sign_ok || (bal != supply(v) && pi(v) != 0) {
            return Err(format!("node {v}: potential {} with balance {bal}", pi(v)));
        }
    }

    let mut dual: i128 = 0;
    let mut red_supply: Vec<i128> = (0..n).map(supply).collect();
    for a in 0..m {
        if lower(a) != 0 {
            dual += lower(a) * cost(a);
            red_supply[p.source(a)] -= lower(a);
            red_supply[p.target(a)] += lower(a);
        }
    }
    for (v, &rs) in red_supply.iter().enumerate() {
        dual -= rs * pi(v);
    }
    for a in 0..m {
        dual -= (upper(a) - lower(a)) * (-reduced(a)).max(0);
    }
    if dual != total {
        return Err(format!("dual objective {dual} differs from primal {total}"));
    }
    Ok(())
}

/// SplitMix64: small, deterministic, and good enough for test instances.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}
