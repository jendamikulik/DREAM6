"""Exact all-threshold verification of a TWO-level Waring reference tree.

The finite checks certify all thresholds by the integral identity proved in
waring_phase_splitting_2026-09-19.tex. This is not an all-horizon certificate.
Uses only the Python standard library.
"""
from fractions import Fraction as F
from pathlib import Path
import json

P = tuple(F(w, 64) for w in (8, 16, 16, 16, 8))
Q = tuple(F(w, 64) for w in (7, 20, 10, 20, 7))


def decode(obj):
    return {F(a): F(w) for a, w in obj.items()}


def cdf(nu, k):
    if k <= 0:
        return F(0)
    return sum(w*F(k)/(a+k) for a, w in nu.items())


def capital(nu):
    return sum(a*w for a, w in nu.items())


def check(nu, kids):
    for law in [nu, *kids]:
        assert sum(law.values()) == 1
        assert all(a > 0 and w > 0 for a, w in law.items())
    assert capital(nu) == sum(P[x]*capital(k) for x, k in enumerate(kids))
    heads = [sum(P[x]*cdf(k, t-x) for x, k in enumerate(kids))
             - sum(Q[y]*cdf(nu, t-y) for y in range(5)) for t in range(1, 5)]
    assert min(heads) >= 0
    knots = sorted({a-y for a in nu for y in range(5)} |
                   {a-x for x, kid in enumerate(kids) for a in kid})
    gaps = []
    for t in knots:
        source = sum(w*a*Q[y]*max(F(0), t-a+y)
                     for a, w in nu.items() for y in range(5))
        target = sum(P[x]*w*a*max(F(0), t-a+x)
                     for x, kid in enumerate(kids) for a, w in kid.items())
        gaps.append(source-target)
    assert min(gaps) >= 0
    # Between knots the gap is affine. Before the first knot it is zero;
    # after the last it is constant because the capital masses agree.
    return heads, list(zip(knots, gaps))


def multiply(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i+j] += u*v
    return out


def main():
    path = Path(__file__).with_name('waring_phase_two_levels_2026-09-19.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    assert len(data) == 6
    laws = {'': {F(2): F(1)}}
    for record in data:
        h = record['history']
        nu = decode(record['parent'])
        kids = [decode(k) for k in record['children']]
        assert nu == laws[h]
        heads, gaps = check(nu, kids)
        for x, kid in enumerate(kids):
            laws[h+str(x)] = kid
        if h == '':
            assert heads == [F(7, 768), F(9, 1280), F(617, 23040), F(239, 12480)]
            assert gaps == [(F(-2), F(0)), (F(-1), F(0)), (F(-3, 4), F(7, 512)),
                            (F(0), F(11, 256)), (F(1), F(13, 256)), (F(2), F(129, 256))]
            print('Root CDF slacks:', ', '.join(map(str, heads)))
        else:
            margin = F(record['one_step_margin'])
            assert margin > 0
            for kid in kids:
                for k in range(2, 5):
                    assert sum(Q[y]*cdf(kid, k-y) for y in range(5))+margin <= sum(P[:k])
    assert set(h for h in laws if len(h) == 2) == {str(x)+str(y) for x in range(5) for y in range(5)}

    # Polynomial verification of the root's explicit tail formula.
    poles = [F(-1), F(-3, 4), F(0), F(1), F(2)]
    coeffs = [F(7), F(-2), F(-4), F(57), F(-58)]
    numerator = [F(0)]*5
    for i, coef in enumerate(coeffs):
        term = [coef]
        for j, pole in enumerate(poles):
            if i != j:
                term = multiply(term, [pole, F(1)])
        numerator = [a+b for a, b in zip(numerator, term)]
    assert numerator == [F(-6), F(81, 2), F(-177, 2), F(129, 2), F(0)]
    assert [F(-12), F(81), F(-177), F(129)] == [2*v for v in numerator[:4]]
    print('PASS: 6 exact capital-preserving phase splits, valid for EVERY threshold.')
    print('PASS: all 25 grandchildren and their strict one-step margins.')
    print('Scope: two phase levels; no claim for arbitrary depth.')


if __name__ == '__main__':
    main()
