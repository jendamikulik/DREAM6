"""Exact arithmetic audit for shifted_tail_orbit_2026-09-19.tex.

This checks identities and a first fan, NOT infinite head viability.
The all-depth identities and bounds are proved analytically in the TeX.
Only Python's standard library is used.
"""
from collections import defaultdict
from fractions import Fraction as F

P = tuple(F(w, 64) for w in (8, 16, 16, 16, 8))
Q = tuple(F(w, 64) for w in (7, 20, 10, 20, 7))
QINT = (7, 20, 10, 20, 7)


def convolutions(max_d):
    out = [(F(1),)]
    ints = [1]
    for d in range(1, max_d + 1):
        nxt = [0] * (len(ints) + 4)
        for y, a in enumerate(ints):
            for z, b in enumerate(QINT):
                nxt[y + z] += a * b
        ints = nxt
        out.append(tuple(F(a, 64 ** d) for a in ints))
    return out


def tail_mass(d, q):
    return sum(2 * v / (8 * d + 2 - y) for y, v in enumerate(q))


def boundary(d, s, q):
    r = 8 * d - s
    return {
        r + v: sum(
            2 * qy * Q[z]
            / ((8 * d + 2 - y + v - z - 1)
               * (8 * d + 2 - y + v - z))
            for y, qy in enumerate(q)
            for z in range(min(4, v - 1) + 1)
        )
        for v in range(1, 9)
    }


def harmonic(n):
    return sum((F(1, j) for j in range(1, n + 1)), F(0))


def root_fan(q1):
    supplies = boundary(0, 0, (F(1),))
    t = tail_mass(1, q1)
    remaining = [p * (1 - t) for p in P]
    heads = [defaultdict(F) for _ in P]
    x = 0
    for u, supply in supplies.items():
        while supply:
            while not remaining[x]:
                x += 1
            assert x <= u
            moved = min(supply, remaining[x])
            heads[x][u - x] += moved / P[x]
            remaining[x] -= moved
            supply -= moved
    assert all(v == 0 for v in remaining)
    assert all(sum(a.values()) == 1 - t for a in heads)
    # Entire finite head equality.
    for u in range(9):
        assert sum(P[x] * heads[x].get(u - x, F(0)) for x in range(5)) == supplies.get(u, F(0))
    # Entire tail equality, checked componentwise by poles, not by sampling j.
    source_poles = {10 - y: 2 * q1[y] for y in range(5)}
    reconstructed = defaultdict(F)
    for x in range(5):
        for y in range(5):
            phase = 10 - y
            weight = 2 * q1[y] / phase
            # After shifting child x back by x, every tail has shift 8.
            reconstructed[phase] += P[x] * weight * phase
    assert dict(reconstructed) == source_poles
    expected_mean = sum(P[x] * sum(j * v for j, v in heads[x].items()) for x in range(5))
    increment = sum((u - 2) * v for u, v in supplies.items())
    closed = 2 * sum(q1[y] * harmonic(10 - y) for y in range(5)) - 7 * t - 2
    assert expected_mean == increment == closed
    print('Exact first fan: ZR at every coefficient, masses, head mean verified.')
    print('T_1 =', t, '; mean of finite heads =', expected_mean)


def main():
    qs = convolutions(64)
    ts = [tail_mass(d, q) for d, q in enumerate(qs)]
    assert ts[0] == 1
    for d in range(1, 65):
        lo = F(2, 6 * d + 2)
        hi = lo + F(3 * d, (6 * d + 2) ** 2 * (4 * d + 2))
        assert lo <= ts[d] <= hi
        if 2 * d <= 64:
            assert ts[2 * d] <= ts[d] / 2 + F(19, 36) * ts[d] ** 2
    for d in (0, 1, 2, 4, 8, 16):
        for s in sorted({0, 2 * d, 4 * d}):
            births = boundary(d, s, qs[d])
            assert all(v > 0 for v in births.values())
            delta = ts[d] - ts[d + 1]
            assert sum(births.values()) == delta
            if d:
                assert F(8, (8 * d + 2) * (8 * d + 10)) <= delta
                assert delta <= F(16, (4 * d + 2) * (4 * d + 6))
                assert 0 < sum((u - 2) * v for u, v in births.items()) <= F(16, d)
            for x in range(5):
                got = defaultdict(F)
                for y, qy in enumerate(qs[d]):
                    for z, qz in enumerate(Q):
                        got[2 + s + x - y - z] += 2 * qy * qz
                expected = {2 + s + x - y: 2 * qy for y, qy in enumerate(qs[d + 1])}
                assert dict(got) == expected
    root_fan(qs[1])
    print('Rational checks passed: phase orbit, boundary source, scalar bounds.')
    print('No claim of an all-history positive head policy.')


if __name__ == '__main__':
    main()
