#!/usr/bin/env python3
"""Exact checks for fixed_branch_runs.tex. Python standard library only.
Run: python verify_fixed_branch_runs.py
Output: observed failure intervals in a prefix, and theorem-certified intervals.
This checks finite examples; the proof of unboundedness is in the note.
"""
from math import comb
import json


def coefficients(m, degree):
    """[x^j] ((1+x)^(2m) + x(1+3x+x^2)^m), through degree."""
    p = [1, 3*m]
    for k in range(1, degree):
        numerator = 3*(m-k)*p[k] + (2*m-k+1)*p[k-1]
        assert numerator % (k+1) == 0
        p.append(numerator//(k+1))
    return [comb(2*m, j) + (p[j-1] if j else 0)
            for j in range(degree+1)]


def multiply(a, b):
    c = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return c


def direct_coefficients(m):
    p = [1]
    for _ in range(m):
        p = multiply(p, [1, 3, 1])
    return [comb(2*m, j)+(p[j-1] if j else 0)
            for j in range(2*m+1)] + [p[-1]]


def intervals(indices):
    result = []
    for j in indices:
        if result and result[-1][1] == j-1:
            result[-1][1] = j
        else:
            result.append([j, j])
    return result


def criterion(m, j):
    """All comparisons are exact; no evaluation of logarithms."""
    return (j >= 48 and j*j*3**j >= 72*m*2**j
            and 8*3**j <= m*2**j and m >= 16*(j+1)**3)


def main():
    # Independent multiplication checks include every coefficient.
    for m in range(2, 21):
        direct = direct_coefficients(m)
        # Recurrence supports the full polynomial too; binomial beyond 2m is 0.
        assert coefficients(m, 2*m) == direct[:2*m+1]
        assert direct[-1] == 1
    # Explicit graph definition is the root plus m disjoint two-leaf branches.
    expected = {16: [[22,24]], 32: [[44,52]], 64: [[95,107]],
                128: [[200,217]], 256: [[416,435]]}
    rows = []
    for exponent, want in expected.items():
        m = 2**exponent
        degree = 2*exponent+20
        a = coefficients(m, degree)
        failures = [j for j in range(1, degree)
                    if a[j-1]*a[j+1] > a[j]*a[j]]
        observed = intervals(failures)
        assert observed == want, (exponent, observed)
        certified = [j for j in range(1, degree) if criterion(m,j)]
        assert set(certified).issubset(failures)
        rows.append({'m_power_of_two': exponent, 'vertices': str(3*m+1),
                     'prefix_degree': degree, 'observed_intervals': observed,
                     'finite_theorem_intervals': intervals(certified)})
        print(json.dumps(rows[-1]), flush=True)
    print('PASS: direct multiplication, integer gaps, finite theorem criterion.')
    return rows


if __name__ == '__main__':
    main()
