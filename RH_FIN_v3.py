#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH_FIN.py
---------
EVT-style MMC scan around the Riemann ξ-function:

- Presence vector v = (1, 0)  =>  S(σ,t) = ∂/∂σ log|ξ(σ+it)|
- MMC summary: jedna nula, uprostřed, s kladným sklonem, plus lineární pásmo.

Režimy:
  1) MMC:
     py.exe RH_FIN.py --t 10000 --sigma-min 0.2 --sigma-max 0.8 --sigma-steps 121 --out mmc_fin

  2) Demo:
     py.exe RH_FIN.py --demo
"""

import argparse
import json
from pathlib import Path
import mpmath as mp

# -----------------------------
# Nastavení přesnosti
# -----------------------------

mp.mp.dps = 80  # počet desetinných míst


# -----------------------------
# ξ(s) a S(σ,t)
# -----------------------------

def zeta(s):
    return mp.zeta(s)

def chi_star(s):
    # standard ξ(s) = 1/2 s(s-1) π^{-s/2} Γ(s/2) ζ(s)
    return 0.5 * s * (s - 1) * (mp.pi)**(-s/2) * mp.gamma(s/2)

def xi(s):
    return chi_star(s) * zeta(s)

def log_abs_xi(s):
    """log |ξ(s)| s bezpečností kolem nul."""
    val = xi(s)
    return mp.log(abs(val))


def S_sigma_t(sigma, t, h=1e-6):
    """
    S(σ,t) = ∂σ log|ξ(σ+it)| přes reálný symetrický rozdíl.
    t MUSÍ být číslo (float), žádné None.
    """
    s1 = (sigma + h) + 1j*t
    s2 = (sigma - h) + 1j*t
    f1 = log_abs_xi(s1)
    f2 = log_abs_xi(s2)
    return (f1 - f2) / (2*h)


# -----------------------------
# MMC-scan(t): nula + sklon
# -----------------------------

def mmc_scan(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    """
    1D scan S(σ,t) v intervalu [sigma_min, sigma_max],
    najde změnu znaménka, zpřesní nulu a spočítá S'(σ*,t).
    """
    sigmas = [sigma_min + i*(sigma_max - sigma_min)/max(1, steps-1)
              for i in range(steps)]
    values = [S_sigma_t(s, t) for s in sigmas]

    # najít intervaly, kde se mění znaménko
    zero_intervals = []
    for i in range(len(sigmas)-1):
        s1, s2 = sigmas[i], sigmas[i+1]
        f1, f2 = values[i], values[i+1]
        if f1 == 0:
            zero_intervals.append((s1, s1))
        elif f1 * f2 < 0:
            zero_intervals.append((s1, s2))

    if not zero_intervals:
        return {
            "zero_crossings": 0,
            "sigma_star": None,
            "S_prime": None,
            "details": "No sign change detected."
        }

    def dist_to_half(interval):
        a, b = interval
        mid = 0.5*(a+b)
        return abs(mid - 0.5)

    # vybereme interval nejblíž 1/2
    zero_intervals.sort(key=dist_to_half)
    a, b = zero_intervals[0]

    # bisection pro zpřesnění
    for _ in range(50):
        m = 0.5*(a + b)
        fa = S_sigma_t(a, t)
        fm = S_sigma_t(m, t)
        if fa * fm <= 0:
            b = m
        else:
            a = m
        if abs(b - a) < 1e-12:
            break

    sigma_star = 0.5*(a + b)

    # derivace S'(σ*,t) malým krokem
    dh = 1e-5
    Sp = (S_sigma_t(sigma_star + dh, t) - S_sigma_t(sigma_star - dh, t)) / (2*dh)

    result = {
        "zero_crossings": len(zero_intervals),
        "sigma_star": float(sigma_star),
        "S_prime": float(Sp),
    }
    return result


# -----------------------------
# Band & Exclusion (A_main, eps)
# -----------------------------

def A_main(t):
    """Stirling-like hlavní člen pro sign flux."""
    return 0.5 * mp.log(abs(t)/(2*mp.pi))

def band_and_exclusion(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    sigmas = [sigma_min + i*(sigma_max - sigma_min)/max(1, steps-1)
              for i in range(steps)]

    A = A_main(t)
    E_vals = []
    S_vals = []
    for s in sigmas:
        S_val = S_sigma_t(s, t)
        S_vals.append(S_val)
        E_vals.append(S_val - A*(s - 0.5))

    max_ratio = max(abs(E_vals[i]) / max(abs(A), 1e-12)
                    for i in range(len(sigmas)))
    eps = 2 * max_ratio

    return {
        "A": float(A),
        "eps": float(eps),
        "max_ratio": float(max_ratio),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
    }


# -----------------------------
# MMC summary (to, co už znáš)
# -----------------------------

def mmc_summary(t, sigma_min, sigma_max, sigma_steps, out_prefix=None):
    scan = mmc_scan(t, sigma_min, sigma_max, sigma_steps)
    band = band_and_exclusion(t, sigma_min, sigma_max, sigma_steps)

    MMC_pass = (
        scan["zero_crossings"] == 1
        and scan["sigma_star"] is not None
        and scan["S_prime"] is not None
        and 0.49 < scan["sigma_star"] < 0.51
        and scan["S_prime"] > 0.0  # tok roste přes kritickou přímku
    )

    summary = {
        "t": float(t),
        "zero_crossings": int(scan["zero_crossings"]),
        "sigma_star": scan["sigma_star"],
        "S_prime": scan["S_prime"],
        "A_main": band["A"],
        "eps": band["eps"],
        "MMC_pass_like": MMC_pass,
    }

    if out_prefix:
        Path(out_prefix).with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8"
        )

    print("=== EVT Edge-Vector MMC summary ===")
    print(f"t = {summary['t']}")
    print(f"zero crossings: {summary['zero_crossings']}")
    print(f"sigma* ≈ {summary['sigma_star']}")
    print(f"S'(sigma*, t) ≈ {summary['S_prime']}")
    print(f"A_main(t) ≈ {summary['A_main']}")
    print(f"eps(t) ≈ {summary['eps']}")
    print(f"MMC_pass_like = {'YES' if MMC_pass else 'NO'}")

    return summary


# -----------------------------
# DEMO: A–C Swiss-knife
# -----------------------------

def even_integral(f, a=0, b=mp.inf):
    """∫_{-b}^{b} f(x) dx pro sudé f."""
    return 2 * mp.quad(f, [a, b])

def sin_over_x_check():
    f1 = lambda x: mp.sin(x)/x
    f2 = lambda x: (mp.sin(x)**2)/(x**2)
    I1 = even_integral(f1, 0, mp.inf)
    I2 = even_integral(f2, 0, mp.inf)
    return I1, I2

def t4_plus_1_antiderivative(t):
    """∫ dt/(t⁴+1) v uzavřené formě (vektor/báze styl)."""
    sqrt2 = mp.sqrt(2)
    log_term = (sqrt2/8) * mp.log((t**2 + sqrt2*t + 1)/(t**2 - sqrt2*t + 1))
    atan_term = (sqrt2/4) * (mp.atan(sqrt2*t + 1) + mp.atan(sqrt2*t - 1))
    return log_term + atan_term

def t4_plus_1_integral(a, b):
    return t4_plus_1_antiderivative(b) - t4_plus_1_antiderivative(a)

def cosh_product(N=100):
    """P_N ≈ Π_{n≠0, |n|≤N} (1 + 1/cosh(2πn))."""
    prod = 1
    for n in range(-N, N+1):
        if n == 0:
            continue
        prod *= (1 + 1/mp.cosh(2*mp.pi*n))
    return prod

def cosh_product_target():
    return 2**(3/8) * mp.sqrt(1 + mp.sqrt(2))

def run_demo():
    print("Demo A: ∫ sin x / x  vs  ∫ sin² x / x² (even-integral check)")
    I1, I2 = sin_over_x_check()
    print("I1 =", I1)
    print("I2 =", I2)

    print("\nDemo B: ∫ dt/(t⁴+1) from -10 to 10")
    val = t4_plus_1_integral(-10, 10)
    print("∫_{-10}^{10} dt/(t⁴+1) ≈", val)

    print("\nDemo C: cosh-product")
    P = cosh_product(N=50)
    target = cosh_product_target()
    rel_err = abs(P - target) / target
    print("P_N ≈", P)
    print("target =", target)
    print("relative error ≈", rel_err)
    print()

    print("\nDemo D: Vindaloo VOICE qubit")
    vindaloo_demo(t_demo=10000.0)



def vindaloo_demo(t_demo=10000.0):
    sigma = 0.5
    S = S_sigma_t(sigma, t_demo)
    p_laugh = mp.cos(S/2)**2
    print(f"Vindaloo demo @ t = {t_demo}")
    print(f"S(1/2, t) ≈ {S}")
    print(f"p_laugh = cos^2(S/2) ≈ {p_laugh}")


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="EVT-style Edge Vector MMC scan around ξ(s)."
    )
    ap.add_argument("--t", type=float, help="Imaginary part t.")
    ap.add_argument("--sigma-min", type=float, default=0.2)
    ap.add_argument("--sigma-max", type=float, default=0.8)
    ap.add_argument("--sigma-steps", type=int, default=121)
    ap.add_argument("--out", type=str, default=None,
                    help="Prefix for JSON summary (optional).")
    ap.add_argument("--demo", action="store_true",
                    help="Run Swiss-knife demos instead of MMC.")
    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    mp.mp.dps = 80

    if args.demo:
        run_demo()
    else:
        if args.t is None:
            raise SystemExit("Error: --t is required unless you use --demo.")
        mmc_summary(
            t=args.t,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sigma_steps=args.sigma_steps,
            out_prefix=args.out,
        )
