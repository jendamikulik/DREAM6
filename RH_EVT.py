#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_vector_mmc.py
------------------
EVT-style MMC scan around the Riemann ξ-function:

- Presence vector v = (1, 0)  =>  D = ∂/∂σ
- X(σ,t) = ∇ log|ξ(σ+it)|
- S(σ,t) = ∂σ log|ξ(σ+it)| = <v, X>

Implements (roughly) the algorithms MMC-scan(t) and Band&Exclusion(t)
from the Edge Vector Theory Appendix C.
"""

import argparse
import json
from pathlib import Path
import mpmath as mp


# -----------------------------
# Riemann xi and sign-flux S
# -----------------------------

mp.mp.dps = 80  # set precision

def zeta(s):
    return mp.zeta(s)

def chi_star(s):
    # standard ξ(s) = 1/2 s(s-1) π^{-s/2} Γ(s/2) ζ(s)
    return 0.5 * s * (s - 1) * (mp.pi)**(-s/2) * mp.gamma(s/2)

def xi(s):
    return chi_star(s) * zeta(s)

def log_abs_xi(s):
    """log |ξ(s)| with safety around zeros."""
    val = xi(s)
    return mp.log(abs(val))

def S_sigma_t(sigma, t, h=1e-6j):
    """
    S(σ,t) = ∂σ log|ξ(σ+it)| via complex-step derivative.
    Uses presence vector v = (1,0) so we only perturb real part.
    """
    s = sigma + t*1j
    s_h = (sigma + h) + t*1j
    f = log_abs_xi(s)
    f_h = log_abs_xi(s_h)
    # complex-step: derivative ~ Im(f_h) / Im(h) if f is analytic,
    # but here we use a symmetric real-difference because log|xi| is real.
    # For log|xi| we can use standard finite diff with small real step:
    # derivative ~ (f(s+dh) - f(s-dh))/(2*dh)
    # But we keep interface to allow imaginary h if needed.
    # For now we just take real h:
    if h.imag != 0:
        # fallback: use small real step instead
        dh = 1e-6
        f1 = log_abs_xi((sigma + dh) + t*1j)
        f2 = log_abs_xi((sigma - dh) + t*1j)
        return (f1 - f2) / (2*dh)
    else:
        dh = h.real
        f1 = log_abs_xi((sigma + dh) + t*1j)
        f2 = log_abs_xi((sigma - dh) + t*1j)
        return (f1 - f2) / (2*dh)


# -----------------------------
# MMC-scan(t): find zero and slope
# -----------------------------

def mmc_scan(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    """
    1D scan of S(σ,t) in [sigma_min, sigma_max],
    locate sign-change near σ=1/2, refine root, compute S'(σ*,t).
    """
    sigmas = [sigma_min + i*(sigma_max-sigma_min)/max(1, steps-1)
              for i in range(steps)]
    values = [S_sigma_t(s, t) for s in sigmas]

    # find intervals where sign changes
    zero_intervals = []
    for i in range(len(sigmas)-1):
        s1, s2 = sigmas[i], sigmas[i+1]
        f1, f2 = values[i], values[i+1]
        if f1 == 0:
            zero_intervals.append((s1, s1))
        elif f1*f2 < 0:
            zero_intervals.append((s1, s2))

    # refine root near σ=1/2 (choose interval closest to 1/2)
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

    zero_intervals.sort(key=dist_to_half)
    a, b = zero_intervals[0]

    # bisection + small refinement
    for _ in range(50):
        m = 0.5*(a+b)
        fa = S_sigma_t(a, t)
        fm = S_sigma_t(m, t)
        if fa*fm <= 0:
            b = m
        else:
            a = m
        if abs(b-a) < 1e-12:
            break

    sigma_star = 0.5*(a+b)
    # derivative S'(σ*,t) via small step
    dh = 1e-5
    Sp = (S_sigma_t(sigma_star+dh, t) - S_sigma_t(sigma_star-dh, t)) / (2*dh)

    result = {
        "zero_crossings": len(zero_intervals),
        "sigma_star": float(sigma_star),
        "S_prime": float(Sp),
    }
    return result


# -----------------------------
# Band&Exclusion(t)
# -----------------------------

def A_main(t):
    # Stirling-like leading term for the sign flux
    # A(t) ≈ (1/2) log(|t| / (2π))
    return 0.5*mp.log(abs(t)/(2*mp.pi))

def band_and_exclusion(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    sigmas = [sigma_min + i*(sigma_max-sigma_min)/max(1, steps-1)
              for i in range(steps)]

    A = A_main(t)
    E_vals = []
    S_vals = []
    for s in sigmas:
        S_val = S_sigma_t(s, t)
        S_vals.append(S_val)
        E_vals.append(S_val - A*(s - 0.5))

    max_ratio = max(abs(E_vals[i]) / max(abs(A), 1e-12) for i in range(len(sigmas)))
    eps = 2 * max_ratio

    return {
        "A": float(A),
        "eps": float(eps),
        "max_ratio": float(max_ratio),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
    }


# -----------------------------
# MMC summary
# -----------------------------

def mmc_summary(t, sigma_min, sigma_max, sigma_steps, out_prefix=None):
    scan = mmc_scan(t, sigma_min, sigma_max, sigma_steps)
    band = band_and_exclusion(t, sigma_min, sigma_max, sigma_steps)

    MMC_pass = (
            scan["zero_crossings"] == 1
            and scan["sigma_star"] is not None
            and scan["S_prime"] is not None
            and scan["sigma_star"] > 0.49
            and scan["sigma_star"] < 0.51
            # tok roste přes osu, tj. S'(σ*,t) > 0
            and scan["S_prime"] > 0.0
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

    # Pretty print
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
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="EVT-style Edge Vector MMC scan around ξ(s)."
    )
    ap.add_argument("--t", type=float, required=True, help="Imaginary part t.")
    ap.add_argument("--sigma-min", type=float, default=0.2)
    ap.add_argument("--sigma-max", type=float, default=0.8)
    ap.add_argument("--sigma-steps", type=int, default=121)
    ap.add_argument("--out", type=str, default=None,
                    help="Prefix for JSON summary (optional).")

    args = ap.parse_args()

    # you can tweak precision here if needed
    mp.mp.dps = 80

    mmc_summary(
        t=args.t,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_steps=args.sigma_steps,
        out_prefix=args.out,
    )

if __name__ == "__main__":
    main()
