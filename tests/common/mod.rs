#![allow(dead_code)]

use silvermite::{
    Arc, CostScaling, Error, Method, NetworkSimplex, Node, Number, PivotRule, Problem, Solution,
    SupplyType,
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
    if s.flows().len() != p.arc_count() || s.potentials().len() != p.node_count() {
        return Err("solution has the wrong shape".into());
    }
    let flow = |a: Arc| s.flow(a).to_i128();
    let pi = |v: Node| s.potential(v).to_i128();
    let lower = |a: Arc| p.lower(a).to_i128();
    // `None` for infinite capacity
    let upper = |a: Arc| p.upper(a).finite().map(|u| u.to_i128());
    let cost = |a: Arc| p.cost(a).to_i128();
    let supply = |v: Node| p.supply(v).to_i128();
    let reduced = |a: Arc| cost(a) + pi(p.source(a)) - pi(p.target(a));

    let mut balance = vec![0i128; p.node_count()];
    for a in p.arcs() {
        if flow(a) < lower(a) || upper(a).is_some_and(|u| flow(a) > u) {
            return Err(format!(
                "arc {a}: flow {} outside [{}, {:?}]",
                flow(a),
                lower(a),
                p.upper(a)
            ));
        }
        balance[p.source(a).index()] += flow(a);
        balance[p.target(a).index()] -= flow(a);
    }
    let balance = |v: Node| balance[v.index()];

    let total_supply: i128 = p.nodes().map(supply).sum();
    let leq = p.supply_type() == SupplyType::Leq;
    for v in p.nodes() {
        let ok = if total_supply == 0 {
            balance(v) == supply(v)
        } else if leq {
            balance(v) <= supply(v)
        } else {
            balance(v) >= supply(v)
        };
        if !ok {
            return Err(format!(
                "node {v}: balance {} violates supply {}",
                balance(v),
                supply(v)
            ));
        }
    }

    let total: i128 = p.arcs().map(|a| flow(a) * cost(a)).sum();
    if total != s.total_cost() {
        return Err(format!(
            "reported cost {} but flow costs {total}",
            s.total_cost()
        ));
    }

    for a in p.arcs() {
        let rc = reduced(a);
        let ok =
            rc == 0 || (rc > 0 && flow(a) == lower(a)) || (rc < 0 && upper(a) == Some(flow(a)));
        if !ok {
            return Err(format!("arc {a}: reduced cost {rc} with flow {}", flow(a)));
        }
    }
    for v in p.nodes() {
        let sign_ok = if leq { pi(v) >= 0 } else { pi(v) <= 0 };
        if !sign_ok || (balance(v) != supply(v) && pi(v) != 0) {
            return Err(format!(
                "node {v}: potential {} with balance {}",
                pi(v),
                balance(v)
            ));
        }
    }

    let mut dual: i128 = 0;
    let mut red_supply: Vec<i128> = p.nodes().map(supply).collect();
    for a in p.arcs() {
        if lower(a) != 0 {
            dual += lower(a) * cost(a);
            red_supply[p.source(a).index()] -= lower(a);
            red_supply[p.target(a).index()] += lower(a);
        }
    }
    for (v, &rs) in p.nodes().zip(&red_supply) {
        dual -= rs * pi(v);
    }
    // Complementary slackness, checked above, leaves every arc with negative
    // reduced cost at a finite upper bound.
    for a in p.arcs() {
        if let Some(u) = upper(a) {
            dual -= (u - lower(a)) * (-reduced(a)).max(0);
        }
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
