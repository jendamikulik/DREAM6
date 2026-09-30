#!/usr/bin/env python3
"""Independent exact finite audit of the core-entropy transfer theorem.

Python 3.10+, standard library only. No floating-point comparisons.
The accompanying note supplies all universal probability/operator arguments.
This program does not claim to formalize the entire proof.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
import hashlib
import json

counts = Counter()
case_count = forest_cases = marginal_atoms = 0


def check(category, condition):
    if not condition:
        raise AssertionError(category)
    counts[category] += 1


class Poly:
    """Exact Q[d,k], solely for a universal coefficient-identity check."""
    def __init__(self, x=0):
        self.terms = (dict(x.terms) if isinstance(x, Poly) else
                      {e: Q(c) for e, c in x.items() if c}
                      if isinstance(x, dict) else {(0, 0): Q(x)} if x else {})

    def __add__(self, other):
        result = self.terms.copy()
        for e, c in Poly(other).terms.items():
            result[e] = result.get(e, Q(0)) + c
        return Poly(result)
    __radd__ = __add__

    def __neg__(self):
        return Poly({e: -c for e, c in self.terms.items()})

    def __sub__(self, other):
        return self + -Poly(other)

    def __rsub__(self, other):
        return Poly(other) - self

    def __mul__(self, other):
        result = {}
        for (a, b), x in self.terms.items():
            for (c, d), y in Poly(other).terms.items():
                e = (a+c, b+d)
                result[e] = result.get(e, Q(0)) + x*y
        return Poly(result)
    __rmul__ = __mul__

    def __pow__(self, n):
        assert n >= 0
        result = Poly(1)
        for _ in range(n):
            result = result*self
        return result

    def at_k(self, value):
        result = Poly()
        for (a, b), c in self.terms.items():
            result += Poly({(a, 0): c}) * Poly(value)**b
        return result

    def __eq__(self, other):
        return self.terms == Poly(other).terms


d, k = Poly({(1, 0): 1}), Poly({(0, 1): 1})
dual = k*(k+1)*(d-k)*(d-k+1)
weight12 = 12*k*k-12*d*k+2*d*(d-1)
check('universal_quartic_interior',
      dual.at_k(k-1)-2*dual+dual.at_k(k+1) == weight12)
check('universal_quartic_left_endpoint', dual.at_k(1) == weight12.at_k(0))
check('universal_quartic_right_endpoint',
      dual.at_k(d-1) == weight12.at_k(d))
check('universal_quartic_zero_endpoints', dual.at_k(0) == 0 and dual.at_k(d) == 0)
check('entropy_barrier_constant', Q(3, 4)*Q(1, 36) == Q(1, 48))
check('two_half_widths', Q(1, 2)+Q(1, 2) == 1)


def independent_masks(n, edges, allowed=None):
    if allowed is None:
        allowed = (1 << n)-1
    return [m for m in range(1 << n)
            if not (m & ~allowed) and
            all(not (m >> u & 1 and m >> v & 1) for u, v in edges)]


def weights(masks, activities):
    result = {}
    for mask in masks:
        w = Q(1)
        for v, activity in enumerate(activities):
            if mask >> v & 1:
                w *= activity
        result[mask] = w
    return result


def moments(probabilities, observable):
    mu = sum((p*observable(mask) for mask, p in probabilities.items()), Q(0))
    var = sum((p*(observable(mask)-mu)**2
               for mask, p in probabilities.items()), Q(0))
    return mu, var


def is_forest(n, edges):
    parent = list(range(n))
    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v
    for u, v in edges:
        a, b = find(u), find(v)
        if a == b:
            return False
        parent[a] = b
    return True


def audit(n, edges, core, activities):
    global case_count, forest_cases, marginal_atoms
    case_count += 1
    whole = (1 << n)-1
    remainder = whole ^ core
    core_edges = [(u, v) for u, v in edges if core >> u & 1 and core >> v & 1]
    forest_edges = [(u, v) for u, v in edges
                    if remainder >> u & 1 and remainder >> v & 1]
    all_masks = independent_masks(n, edges)
    base_masks = independent_masks(n, forest_edges, remainder)
    core_masks = independent_masks(n, core_edges, core)
    wg = weights(all_masks, activities)
    wf = weights(base_masks, activities)
    wc = weights(core_masks, activities)
    zg, zf, C = sum(wg.values()), sum(wf.values()), sum(wc.values())
    pg = {m: w/zg for m, w in wg.items()}
    pf = {m: w/zf for m, w in wf.items()}
    marginal = dict.fromkeys(base_masks, Q(0))
    for mask, p in pg.items():
        marginal[mask & remainder] += p

    W = {}
    for j in base_masks:
        W[j] = sum((w for c, w in wc.items()
                    if all(not ((j | c) >> u & 1 and (j | c) >> v & 1)
                           for u, v in edges)), Q(0))
        check('completion_partition_bounds', 1 <= W[j] <= C)
    ew = sum((pf[j]*W[j] for j in base_masks), Q(0))
    check('partition_decomposition', zg == zf*ew)
    for j in base_masks:
        check('exact_marginal_reweighting', marginal[j] == pf[j]*W[j]/ew)
        check('pointwise_density_domination', marginal[j] <= C*pf[j])
        marginal_atoms += 1

    mu_f, var_f = moments(pf, int.bit_count)
    _, var_y = moments(marginal, int.bit_count)
    _, var_g = moments(pg, int.bit_count)
    _, var_t = moments(pg, lambda m: (m & core).bit_count())
    alpha = max(map(int.bit_count, core_masks))
    check('core_configuration_count', len(core_masks) >= 2**alpha)
    check('marginal_variance_domination', var_y <= C*var_f)
    check('core_range_variance', var_t <= Q(alpha*alpha, 4))
    centered = sum((p*(j.bit_count()-mu_f)**2 for j, p in marginal.items()), Q(0))
    check('centered_second_moment_domination', centered <= C*var_f)

    # v <= (sqrt(C var_f)+alpha/2)^2, without numerical square roots.
    excess = var_g-C*var_f-Q(alpha*alpha, 4)
    check('sharp_standard_deviation_transfer',
          excess <= 0 or excess*excess <= alpha*alpha*C*var_f)
    check('rational_relaxed_variance_transfer',
          var_g <= 2*C*var_f+Q(alpha*alpha, 2))

    # A dyadic activity ceiling permits an entirely rational entropy-cost check.
    ceiling = 1
    power = 0
    while ceiling < max(activities):
        ceiling *= 2
        power += 1
    check('core_activity_count_bound', C <= len(core_masks)*ceiling**alpha)
    check('core_entropy_count_bound', C <= len(core_masks)**(power+1))

    if is_forest(n, forest_edges):
        forest_cases += 1
        # Use ceiling >= 1 and integral B >= 2 sqrt(ceiling).
        B = 2
        while B*B < 4*ceiling:
            B += 1
        q = Q(ceiling, ceiling+1)
        for radius in range(4):
            bound = Q(n, 2)*B**radius+Q(n*n, 4)*q**(radius+1)
            check('forest_radius_variance_bound', var_f <= bound)


# Exhaust every labelled simple graph of orders 1 through 4, and every
# choice of core. The transfer lemma does not require the remainder to be
# a forest, so cyclic remainders are intentionally retained.
for n in range(1, 5):
    pairs = list(combinations(range(n), 2))
    activities_list = [[q]*n for q in (Q(1, 3), Q(1), Q(9))]
    activities_list.append([Q(v+1, (v % 2)+1) for v in range(n)])
    for edge_mask in range(1 << len(pairs)):
        edges = [e for j, e in enumerate(pairs) if edge_mask >> j & 1]
        for core in range(1 << n):
            for activities in activities_list:
                audit(n, edges, core, activities)


# Large-degree attachments and dense cores: choose the core first, put a
# path/star in the remainder, then add arbitrary cross-edges.
targets = [
    ('fan9', 9, [(j, j+1) for j in range(1, 8)]+[(0, j) for j in range(1, 9)], 1),
    ('clique3_binary7', 10,
     list(combinations(range(3), 2))+
     [(3+(j-1)//2, 3+j) for j in range(1, 7)]+
     [(u, v) for u in range(3) for v in range(3, 10) if (u+v) % 3 != 0], 7),
    ('clique4_star8', 12,
     list(combinations(range(4), 2))+[(4, j) for j in range(5, 12)]+
     [(u, v) for u in range(4) for v in range(4, 12) if (u*v+v) % 4 != 1], 15),
]
for _, n, edges, core in targets:
    for q in (Q(1, 7), Q(1), Q(16)):
        audit(n, edges, core, [q]*n)
    audit(n, edges, core, [Q(v % 7+1, v % 3+1) for v in range(n)])

report = {
    'status': 'PASS',
    'arithmetic': 'Exact rational arithmetic; universal quartic polynomial identity in Q[d,k]',
    'graph_core_activity_cases': case_count,
    'cases_with_forest_remainder': forest_cases,
    'marginal_atoms_checked': marginal_atoms,
    'checks_passed': sum(counts.values()),
    'checks_by_category': dict(sorted(counts.items())),
    'exhaustive_scope': 'All labelled simple graphs of orders 1..4; all vertex cores; four activity vectors',
    'additional_graphs': [name for name, *_ in targets],
    'universal_transfer': 'sd_G(|I|) <= sqrt(Z_G[S] * Var_G-S(|I|)) + alpha(G[S])/2',
    'proved_in_companion_note': 'kappa(G_n)=o(log n) implies all strict log-concavity failure runs have length o(n)',
    'limits': 'Finite audit, not a formal or exhaustive proof for arbitrary graphs. Publication priority is not established.',
    'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
target = Path(__file__).with_name('core_entropy_certificate.json')
target.write_text(json.dumps(report, indent=2)+'\n')
print(f'PASS: {case_count} graph/core/activity cases, {marginal_atoms} marginal atoms, '
      f'{sum(counts.values())} exact checks.')
