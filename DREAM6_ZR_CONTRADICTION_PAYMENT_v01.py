#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6_ZR_CONTRADICTION_PAYMENT_v01.py
======================================

Contradiction formulation of the logarithmic "payment" for the coarse
quartic zero-repair pair

    P_b = (1,12,38,12,1)/64
    Q_b = (0,16,32,16,0)/64.

THIS IS NOT A SEARCH PROGRAM.
- no LP
- no tree enumeration
- no random objectives
- no finite support cutoff
- no attempt to construct a witness

It is a theorem/audit program for the following contradiction mechanism.

If a depth-n exact zero-repair tree with root mean m exists, then for every
block length k <= n/2 the Laplace/Jensen/Hellinger master inequality requires

    rho(a)^k
        >= exp(-a*m)
           - 2*Lambda(a)^(k/2) * sqrt(2*a*m*k/n).          (MASTER)

Choose

    k = ceil((a*m + log 2)/d),     d = -log rho(a) > 0,

so that

    rho(a)^k <= (1/2) exp(-a*m).

Then a feasible tree must pay enough mean m for the "quiet block" error term
to bridge at least the remaining half-gap.  If it does not, MASTER says that
the same block must simultaneously satisfy two incompatible inequalities.

That contradiction gives

    n <= 32*a*m*k*Lambda(a)^k*exp(2*a*m)

in the non-linear regime, hence

    log n <= Gamma(a)*m + O_a(log(m+1)),

where

    Gamma(a) = 2*a + a*log(Lambda(a))/d.

Therefore

    liminf B_n/log n >= 1/Gamma(a).

For the exact fixed tilt z = exp(-a) = 1/2 used here:

    G_P(1/2) = 289/1024
    G_Q(1/2) = 9/32 = 288/1024
    rho      = 288/289

    G_P(1/4) = 1681/16384
    Lambda   = 107584/83521.

So the quartic defect is visible as one exact missing unit at the tilt:
    G_P(1/2) - G_Q(1/2) = 1/1024.

The program can be used in two ways:

1) Audit the exact constants:
       python DREAM6_ZR_CONTRADICTION_PAYMENT_v01.py

2) Challenge a claimed cheap feasible tree:
       python DREAM6_ZR_CONTRADICTION_PAYMENT_v01.py --n 1000000 --m 1.0

   or a logarithmic schedule m = c log n:
       python DREAM6_ZR_CONTRADICTION_PAYMENT_v01.py --n 10**100 --c 0.01

Exit status:
    0 = arithmetic audit passed; queried candidate is not contradicted
        by this fixed-tilt certificate (or no candidate was supplied).
    2 = CONTRADICTION: no feasible depth-n tree can have the supplied mean m,
        assuming the proved MASTER inequality.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import mpmath as mp


# ---------------------------------------------------------------------------
# Exact coarse quartic pair
# ---------------------------------------------------------------------------

P = [F(1,64), F(12,64), F(38,64), F(12,64), F(1,64)]
Q = [F(0,64), F(16,64), F(32,64), F(16,64), F(0,64)]

DEFECT = [F(1,64), F(-4,64), F(6,64), F(-4,64), F(1,64)]


def pgf(law, z):
    return sum(p * (z ** j) for j, p in enumerate(law))


def mean(law):
    return sum(F(j) * p for j, p in enumerate(law))


def exact_audit():
    assert sum(P) == 1
    assert sum(Q) == 1
    assert mean(P) == mean(Q) == 2
    assert [P[j] - Q[j] for j in range(5)] == DEFECT

    z = F(1, 2)
    GP = pgf(P, z)
    GQ = pgf(Q, z)
    GP2 = pgf(P, z*z)

    assert GP == F(289, 1024)
    assert GQ == F(9, 32)
    assert GP - GQ == F(1, 1024)

    rho = GQ / GP
    Lam = GP2 / (GP * GP)

    assert rho == F(288, 289)
    assert GP2 == F(1681, 16384)
    assert Lam == F(107584, 83521)
    assert rho < 1
    assert Lam > 1

    return z, GP, GQ, GP2, rho, Lam


def mpf_fraction(x: F):
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)


def constants(dps: int = 100):
    mp.mp.dps = dps
    z, GP, GQ, GP2, rho_q, Lam_q = exact_audit()

    a = mp.log(2)
    rho = mpf_fraction(rho_q)
    Lam = mpf_fraction(Lam_q)
    d = -mp.log(rho)

    Gamma = 2*a + a*mp.log(Lam)/d
    cstar = 1/Gamma

    # An explicit finite constant for
    # n <= C_a (m+1)^2 exp(Gamma*m)
    alpha = a/d
    beta = 1 + mp.log(2)/d
    C_a = 32*a*(Lam**beta)*(alpha + beta)

    return {
        "z": z,
        "GP": GP,
        "GQ": GQ,
        "GP2": GP2,
        "rho_q": rho_q,
        "Lam_q": Lam_q,
        "a": a,
        "rho": rho,
        "Lambda": Lam,
        "d": d,
        "Gamma": Gamma,
        "cstar": cstar,
        "alpha": alpha,
        "beta": beta,
        "C_a": C_a,
    }


def parse_big_int(expr: str) -> int:
    """
    Accept either an integer literal or the restricted form BASE**EXP.
    No eval().
    """
    s = expr.strip().replace("_", "")
    if "**" in s:
        parts = s.split("**")
        if len(parts) != 2:
            raise ValueError("n expression must be INTEGER or BASE**EXP")
        base = int(parts[0])
        exp = int(parts[1])
        if base < 0 or exp < 0:
            raise ValueError("BASE and EXP must be nonnegative")
        return pow(base, exp)
    return int(s)


def contradiction_certificate(n: int, m, dps: int = 100):
    """
    Return a contradiction certificate for the hypothesis:

        "there exists a feasible depth-n tree with root mean <= m".

    The certificate uses only the proved MASTER inequality.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    C = constants(dps)
    a = C["a"]
    d = C["d"]
    rho = C["rho"]
    Lam = C["Lambda"]

    m = mp.mpf(m)
    if m < 0:
        return {
            "contradiction": True,
            "reason": "mean of a nonnegative buffer cannot be negative",
            "n": n,
            "m": m,
        }

    # The theorem chooses this k so rho^k <= 1/2 exp(-a m).
    x = (a*m + mp.log(2))/d
    k = int(mp.ceil(x))

    half_n = mp.mpf(n)/2

    out = {
        "n": n,
        "m": m,
        "k": k,
        "k_over_n": mp.mpf(k)/mp.mpf(n),
        "rho_k": rho**k,
        "half_root_laplace": mp.mpf("0.5")*mp.e**(-a*m),
        "regime": None,
        "contradiction": False,
        "reason": None,
    }

    # Case A from the proof: k > n/2 already forces a linear payment.
    if mp.mpf(k) > half_n:
        linear_lb = d*mp.mpf(n)/(2*a) - (d + mp.log(2))/a
        out["regime"] = "LINEAR"
        out["linear_lower_bound"] = linear_lb

        if m <= linear_lb:
            out["contradiction"] = True
            out["reason"] = (
                "chosen k exceeds n/2, but the supplied mean violates the "
                "resulting exact linear lower bound"
            )
        else:
            out["reason"] = (
                "candidate lies in the proof's linear-payment regime; "
                "this is already stronger than logarithmic growth"
            )
        return out

    # Case B: direct MASTER contradiction.
    err = 2*(Lam**(mp.mpf(k)/2))*mp.sqrt(
        2*a*m*mp.mpf(k)/mp.mpf(n)
    ) if m > 0 else mp.mpf("0")

    master_rhs = mp.e**(-a*m) - err
    master_lhs = rho**k
    gap = master_rhs - master_lhs

    # Equivalent rearranged capacity bound after using rho^k <= half exp(-am).
    n_cap = (
        32*a*m*mp.mpf(k)*(Lam**k)*mp.e**(2*a*m)
        if m > 0 else mp.mpf("0")
    )

    out.update({
        "regime": "MASTER",
        "hellinger_error_budget": err,
        "master_lhs": master_lhs,
        "master_rhs_required": master_rhs,
        "contradiction_gap": gap,
        "n_cap_from_half_gap": n_cap,
    })

    if gap > 0:
        out["contradiction"] = True
        out["reason"] = (
            "MASTER would require rho^k >= RHS, but RHS > rho^k. "
            "The same block would have to be both too quiet and sufficiently "
            "dissipative to absorb the fixed Laplace mismatch."
        )
    else:
        out["reason"] = (
            "this fixed-tilt contradiction certificate does not rule out "
            "the supplied finite (n,m) pair"
        )

    return out


def fmt(x, digits=22):
    if isinstance(x, int):
        return str(x)
    if isinstance(x, F):
        return f"{x.numerator}/{x.denominator}"
    try:
        return mp.nstr(x, digits)
    except Exception:
        return str(x)


def print_exact_ledger(C):
    print("=== DREAM6-ZR CONTRADICTION / PAYMENT LEDGER ===")
    print("mode             : theorem audit; NO SEARCH")
    print("pair             : coarse quartic P_b,Q_b")
    print()
    print("[EXACT]")
    print("P_b - Q_b        : (1-t)^4 / 64")
    print("mean(P_b)=mean(Q_b)=2")
    print(f"G_P(1/2)         : {fmt(C['GP'])}")
    print(f"G_Q(1/2)         : {fmt(C['GQ'])}")
    print("tilt gap          : 1/1024")
    print(f"rho               : {fmt(C['rho_q'])}")
    print(f"Lambda            : {fmt(C['Lam_q'])}")
    print()
    print("[CONTRADICTION THEOREM]")
    print("feasible depth-n tree + root mean m")
    print("    => MASTER inequality")
    print("choose k = ceil((a*m + log 2)/(-log rho))")
    print("    => rho^k <= (1/2) exp(-a*m)")
    print("    => insufficient Hellinger/Jensen budget is impossible")
    print("    => log n <= Gamma*m + O(log(m+1))")
    print()
    print("[PAYMENT]")
    print(f"a                 : {fmt(C['a'])}")
    print(f"-log(rho)         : {fmt(C['d'])}")
    print(f"Gamma             : {fmt(C['Gamma'])}")
    print(f"1/Gamma           : {fmt(C['cstar'])}")
    print()
    print("Therefore, for this fixed tilt,")
    print("    liminf B_n / log n >= 1/Gamma.")
    print("In particular any m(n)=o(log n) is eventually contradictory.")
    print()
    print("This is an Omega(log n) necessity certificate.")
    print("It is NOT an O(log n) construction.")


def print_certificate(cert):
    print("\n=== CANDIDATE CHEAP-TREE CHALLENGE ===")
    print(f"n                 : {cert['n']}")
    print(f"m                 : {fmt(cert['m'])}")
    if "k" in cert:
        print(f"chosen k          : {cert['k']}")
        print(f"k/n               : {fmt(cert['k_over_n'])}")
        print(f"regime            : {cert['regime']}")

    if cert.get("regime") == "LINEAR":
        print(f"linear lower bound: {fmt(cert['linear_lower_bound'])}")

    if cert.get("regime") == "MASTER":
        print(f"rho^k             : {fmt(cert['master_lhs'])}")
        print(f"required RHS      : {fmt(cert['master_rhs_required'])}")
        print(f"RHS - rho^k       : {fmt(cert['contradiction_gap'])}")
        print(f"Hellinger budget  : {fmt(cert['hellinger_error_budget'])}")
        print(f"n capacity        : {fmt(cert['n_cap_from_half_gap'])}")

    print()
    if cert["contradiction"]:
        print("VERDICT           : CONTRADICTION")
        print("No feasible depth-n exact zero-repair tree can have this root mean.")
        print("Reason            :", cert["reason"])
    else:
        print("VERDICT           : NOT CONTRADICTED BY THIS CERTIFICATE")
        print("Reason            :", cert["reason"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=str, default=None,
        help="depth, as INTEGER or restricted BASE**EXP, e.g. 10**100"
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--m", type=str, default=None,
        help="claimed root mean"
    )
    g.add_argument(
        "--c", type=str, default=None,
        help="claim m = c*log(n), natural logarithm"
    )
    ap.add_argument("--dps", type=int, default=100)
    args = ap.parse_args()

    C = constants(args.dps)
    print_exact_ledger(C)

    if args.n is None:
        return 0

    n = parse_big_int(args.n)

    if args.m is None and args.c is None:
        raise SystemExit("with --n, supply either --m or --c")

    if args.c is not None:
        m = mp.mpf(args.c) * mp.log(mp.mpf(n))
    else:
        m = mp.mpf(args.m)

    cert = contradiction_certificate(n, m, args.dps)
    print_certificate(cert)

    return 2 if cert["contradiction"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
