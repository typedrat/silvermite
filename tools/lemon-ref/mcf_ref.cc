// Reference solver for differential testing: runs LEMON's NetworkSimplex or
// CostScaling on a DIMACS min-cost-flow file and prints the outcome.
//
// Usage: mcf_ref <ns|cs|cs-push|cs-augment> [--leq] [--dump] [--repeat N] <file.min>
// Output: "OPTIMAL <total cost> <solve ms>", "INFEASIBLE", or "UNBOUNDED".
// With --repeat, the reported time is the fastest of N solves.
// With --dump, an optimal result is followed by "flow <f1> <f2> ..." and
// "potential <p1> <p2> ..." lines in DIMACS arc and node order.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include <lemon/cost_scaling.h>
#include <lemon/dimacs.h>
#include <lemon/network_simplex.h>
#include <lemon/smart_graph.h>

using namespace lemon;
typedef long long Num;
typedef SmartDigraph Digraph;

template <typename Mcf, typename Result>
int report(const Mcf& mcf, const Digraph& g, bool dump, Result r,
           Result optimal, Result unbounded, double ms) {
  if (r == optimal) {
    std::cout << "OPTIMAL " << mcf.totalCost() << " " << ms << "\n";
    if (dump) {
      std::cout << "flow";
      for (int i = 0; i < g.arcNum(); ++i)
        std::cout << " " << mcf.flow(g.arcFromId(i));
      std::cout << "\npotential";
      for (int i = 0; i < g.nodeNum(); ++i)
        std::cout << " " << mcf.potential(g.nodeFromId(i));
      std::cout << "\n";
    }
  } else if (r == unbounded) {
    std::cout << "UNBOUNDED\n";
  } else {
    std::cout << "INFEASIBLE\n";
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: mcf_ref <ns|cs|cs-push|cs-augment> [--leq] [--dump] [--repeat N] <file.min>\n";
    return 2;
  }
  std::string algo = argv[1];
  bool leq = false, dump = false;
  int repeat = 1;
  for (int i = 2; i < argc - 1; ++i) {
    if (std::strcmp(argv[i], "--leq") == 0) leq = true;
    if (std::strcmp(argv[i], "--dump") == 0) dump = true;
    if (std::strcmp(argv[i], "--repeat") == 0) repeat = std::atoi(argv[++i]);
  }
  std::ifstream in(argv[argc - 1]);

  Digraph g;
  Digraph::ArcMap<Num> lower(g), upper(g), cost(g);
  Digraph::NodeMap<Num> supply(g);
  readDimacsMin(in, g, lower, upper, cost, supply);

  double best = 1e300;
  auto elapsed = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };
  if (algo == "ns") {
    for (int k = 1;; ++k) {
      auto t0 = std::chrono::steady_clock::now();
      NetworkSimplex<Digraph, Num, Num> ns(g);
      ns.lowerMap(lower).upperMap(upper).costMap(cost).supplyMap(supply);
      if (leq) ns.supplyType(ns.LEQ);
      auto r = ns.run();
      best = std::min(best, elapsed(t0));
      if (k == repeat) return report(ns, g, dump, r, ns.OPTIMAL, ns.UNBOUNDED, best);
    }
  } else {
    for (int k = 1;; ++k) {
      auto t0 = std::chrono::steady_clock::now();
      CostScaling<Digraph, Num, Num> cs(g);
      cs.lowerMap(lower).upperMap(upper).costMap(cost).supplyMap(supply);
      auto method = algo == "cs-push" ? cs.PUSH
                  : algo == "cs-augment" ? cs.AUGMENT
                  : cs.PARTIAL_AUGMENT;
      auto r = cs.run(method);
      best = std::min(best, elapsed(t0));
      if (k == repeat) return report(cs, g, dump, r, cs.OPTIMAL, cs.UNBOUNDED, best);
    }
  }
}
