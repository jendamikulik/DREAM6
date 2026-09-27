#!/usr/bin/env python3
"""Exact certificate for the subcubic bulk log-concavity construction.

Python standard library only. Run: python3 verify_subcubic.py
All assertions use rational arithmetic. The JSON report certifies the uniform
limiting-moment inequalities, not an individual finite-height coefficient.
The accompanying paper supplies the convergence and all-large-height proof.
"""
from fractions import Fraction as F
from pathlib import Path
import json


class Interval:
    def __init__(self, lo, hi=None):
        self.lo = F(lo)
        self.hi = F(lo if hi is None else hi)
        assert self.lo <= self.hi

    def __add__(self, other):
        other = interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        other = interval(other)
        values = [a*b for a in (self.lo, self.hi)
                  for b in (other.lo, other.hi)]
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        assert not other.lo <= 0 <= other.hi
        return self * Interval(1/other.hi, 1/other.lo)

    def __rtruediv__(self, other):
        return interval(other) / self

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        if n == 0:
            return Interval(1)
        if n % 2:
            return Interval(self.lo**n, self.hi**n)
        lo = 0 if self.lo <= 0 <= self.hi else min(self.lo**n, self.hi**n)
        return Interval(lo, max(self.lo**n, self.hi**n))

    def json(self):
        return {"lower": str(self.lo), "upper": str(self.hi)}


def interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def limiting_moments(e):
    """Return central moments divided by e=1-q, factoring e before evaluation."""
    q = 1-e
    a, t, u, z = 1/(2*q**2), -1/(4*q**3), 1/(8*q**4), 3/(8*q**4)
    A0 = a*q/(1-a*e-a*a*q)
    A1 = a*A0
    B2 = e*A0 + q*A1 + q
    T0 = t*q*(3*e*(A1-A0)+1-2*q)/(1-t*e-t*t*q)
    T1 = t*T0
    C = (4*q*e*(T1-T0) + 6*q*e*(q*A0+e*A1)
         + q*(q**3+e**3))
    U0 = (u*(q*z*e*A0*A0+C)+z*e*B2*B2)/(1-u*e-u*u*q)
    U1 = u*U0+z*e*A0*A0
    return A0, A1, T0, T1, U0, U1


def convolution(a, b):
    out = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def add(a, b):
    return [(a[i] if i < len(a) else 0)+(b[i] if i < len(b) else 0)
            for i in range(max(len(a), len(b)))]


def statistics(poly, activity):
    weights = [F(c)*activity**k for k, c in enumerate(poly)]
    total = sum(weights)
    mean = sum(k*w for k, w in enumerate(weights))/total
    return (mean, *(sum((k-mean)**j*w for k, w in enumerate(weights))/total
                    for j in (2, 3, 4)))


def mix_moments(left, right, q):
    m0, v0, t0, u0 = left
    m1, v1, t1, u1 = right
    e, d = 1-q, m1-m0
    return (e*m0+q*m1,
            e*v0+q*v1+q*e*d*d,
            e*t0+q*t1+3*q*e*d*(v1-v0)+q*e*(1-2*q)*d**3,
            e*u0+q*u1+4*q*e*d*(t1-t0)
            +6*q*e*d*d*(q*v0+e*v1)+q*e*(q**3+e**3)*d**4)


def sum_two(moments, shift=0):
    m, v, t, u = moments
    return 2*m+shift, 2*v, 2*t, 2*u+6*v*v


def check_finite_recurrences():
    # A four-vertex path, rooted at an endpoint. Conditional polynomials:
    # root absent: I(P3); root present: x I(P2).
    p0, p1, activity = [1, 3, 1], [0, 1, 2], F(3, 2)
    for height in range(4):
        s0, s1 = statistics(p0, activity), statistics(p1, activity)
        z0 = sum(F(c)*activity**k for k, c in enumerate(p0))
        z1 = sum(F(c)*activity**k for k, c in enumerate(p1))
        q = z1/(z0+z1)
        mixed = mix_moments(s0, s1, q)
        assert mixed == statistics(add(p0, p1), activity)
        total = add(p0, p1)
        next0 = convolution(total, total)
        next1 = [0]+convolution(p0, p0)
        assert sum_two(mixed) == statistics(next0, activity)
        assert sum_two(s0, 1) == statistics(next1, activity)
        p0, p1 = next0, next1


def check_fixed_point_equations():
    assert limiting_moments(F(0)) == (F(2, 3), F(1, 3), F(4, 15),
                                      F(-1, 15), F(8, 63), F(1, 63))
    for q in (F(9, 10), F(99, 100), F(5000, 5001)):
        e = 1-q
        A0, A1, T0, T1, U0, U1 = (e*x for x in limiting_moments(e))
        # Means 0 and 1 realize D=1 in the normalized moment mixture.
        _, B2, B3, B4 = mix_moments((0, A0, T0, U0),
                                   (1, A1, T1, U1), q)
        rho = 2*q
        assert A0 == 2*B2/rho**2 and A1 == 2*A0/rho**2
        assert T0 == -2*B3/rho**3 and T1 == -2*T0/rho**3
        assert U0 == (2*B4+6*B2**2)/rho**4
        assert U1 == (2*U0+6*A0**2)/rho**4


def main():
    m = 10000
    e = Interval(0, F(2, m+2))
    q = 1-e
    names = ['A0_over_e', 'A1_over_e', 'T0_over_e', 'T1_over_e',
             'U0_over_e', 'U1_over_e']
    values = limiting_moments(e)
    U0, U1 = values[-2:]
    assert U0.lo >= 0 and U0.hi < F(16, 125)
    assert U1.lo >= 0 and U1.hi < F(2, 125)
    minor_over_e = 1-10000*e*U0
    major = q*(1-10000*e*U1)
    middle_over_e = F(625, 16)*(e*U0+q*U1)
    assert minor_over_e.lo > F(74, 100)
    assert major.lo > F(96, 100)
    assert middle_over_e.hi < F(63, 100)
    # Path pinning and nondegenerate growing mode, uniformly for every m>=10000.
    # At the chosen m these are exact rational inequalities.
    S = F(m*(m+1)*(2*m+1), 6)
    r_lower = F(m, 2)
    activity_lower = r_lower*(1+r_lower)**2
    assert m-S/activity_lower > r_lower
    assert S/(r_lower*activity_lower) < F(1, 4)
    assert 2*q.lo**2 > 1
    check_fixed_point_equations()
    check_finite_recurrences()
    report = {'m': m, 'relative_gap': str(F(1, 6*m+4)),
              'scope': 'Uniform limiting-moment certificate; see proof for finite-height convergence.',
              'q': q.json(), **{name: value.json() for name, value in zip(names, values)},
              'minor_window_mass_over_e_lower_bound': minor_over_e.json(),
              'major_window_mass_lower_bound': major.json(),
              'middle_window_mass_over_e_upper_bound': middle_over_e.json(),
              'finite_moment_recurrences_checked': True,
              'fixed_point_moment_equations_checked': True}
    target = Path(__file__).with_name('subcubic_certificate.json')
    target.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: exact interval certificate, fixed-point equations, finite polynomial checks.')
    print('Maximum degree: 3. Relative gap: 1/60004.')
    print('Minor outer mass/e > 0.74; major outer mass > 0.96; middle mass/e < 0.63 (limiting bounds).')
    print('Report:', target)


if __name__ == '__main__':
    main()
