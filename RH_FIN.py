#!/usr/bin/env python3
# edge_vector_mmc.py

import argparse
import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 80  # working precision


# --------- Riemann xi and S(σ,t) --------- #

def zeta(s):
    return mp.zeta(s)

def xi(s):
    """Completed Riemann ξ-function."""
    return 0.5 * s * (s - 1) * (mp.pi ** (-s/2)) * mp.gamma(s/2) * zeta(s)

def log_abs_xi(s):
    """Logarithm of absolute value of ξ(s)."""
    return mp.log(abs(xi(s)))

def S_sigma_t(sigma, t, h=1e-6):
    """
    S(σ,t) = ∂/∂σ log|ξ(σ+it)|
    Approximated by a symmetric finite difference in the real direction.
    """
    s1 = (sigma + h) + 1j*t
    s2 = (sigma - h) + 1j*t
    f1 = log_abs_xi(s1)
    f2 = log_abs_xi(s2)
    return (f1 - f2) / (2*h)


# --------- MMC scan along one horizontal line --------- #

def mmc_scan(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    """
    Scan S(σ,t) for σ in [sigma_min, sigma_max], detect sign change,
    refine the zero near σ = 1/2 and compute slope S'(σ*,t).
    """
    # coarse sampling
    sigmas = [sigma_min + i*(sigma_max - sigma_min)/max(1, steps-1)
              for i in range(steps)]
    values = [S_sigma_t(s, t) for s in sigmas]

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
            "details": "No sign change detected in scan interval."
        }

    # choose interval closest to 1/2
    def dist_to_half(interval):
        a, b = interval
        mid = 0.5*(a+b)
        return abs(mid - 0.5)

    zero_intervals.sort(key=dist_to_half)
    a, b = zero_intervals[0]

    # refine root by bisection
    for _ in range(50):
        m = 0.5*(a+b)
        fa = S_sigma_t(a, t)
        fm = S_sigma_t(m, t)
        if fa * fm <= 0:
            b = m
        else:
            a = m
        if abs(b-a) < 1e-12:
            break

    sigma_star = 0.5*(a+b)

    # derivative S'(σ*,t)
    dh = 1e-5
    Sp = (S_sigma_t(sigma_star+dh, t) - S_sigma_t(sigma_star-dh, t)) / (2*dh)

    return {
        "zero_crossings": len(zero_intervals),
        "sigma_star": float(sigma_star),
        "S_prime": float(Sp),
    }


# --------- Exclusion band A(t)(σ-1/2) + E(σ,t) --------- #

def A_main(t):
    """Leading term from Stirling's formula."""
    return 0.5 * mp.log(abs(t) / (2*mp.pi))

def band_and_exclusion(t, sigma_min=0.2, sigma_max=0.8, steps=121):
    """
    Approximate E(σ,t) in S(σ,t) = A(t)(σ-1/2) + E(σ,t)
    and estimate a width ε(t) so that |E| ≤ ε A.
    """
    sigmas = [sigma_min + i*(sigma_max - sigma_min)/max(1, steps-1)
              for i in range(steps)]

    A = A_main(t)
    E_vals = []
    for s in sigmas:
        S_val = S_sigma_t(s, t)
        E_vals.append(S_val - A*(s - 0.5))

    # avoid division by zero for very small A
    denom = max(abs(A), 1e-12)
    max_ratio = max(abs(E) / denom for E in E_vals)
    eps = 2 * max_ratio  # quick-and-dirty safety factor

    return {
        "A": float(A),
        "eps": float(eps),
        "max_ratio": float(max_ratio),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
    }


# --------- Swiss-knife demos (optional) --------- #

def even_integral(f, a=0, b=mp.inf):
    """∫_{-b}^{b} f(x) dx for an even function f."""
    return 2 * mp.quad(f, [a, b])

def sin_over_x_demo():
    f1 = lambda x: mp.sin(x)/x
    f2 = lambda x: (mp.sin(x)**2)/(x**2)
    I1 = even_integral(f1, 0, mp.inf)
    I2 = even_integral(f2, 0, mp.inf)
    return I1, I2

def t4_plus_1_antiderivative(t):
    """Closed form for ∫ dt/(t⁴+1)."""
    sqrt2 = mp.sqrt(2)
    log_term = (sqrt2/8) * mp.log((t**2 + sqrt2*t + 1) / (t**2 - sqrt2*t + 1))
    atan_term = (sqrt2/4) * (mp.atan(sqrt2*t + 1) + mp.atan(sqrt2*t - 1))
    return log_term + atan_term

def t4_plus_1_integral(a, b):
    return t4_plus_1_antiderivative(b) - t4_plus_1_antiderivative(a)

def cosh_product(N=100):
    """Approximate Π_{n∈Z} (1 + 1/cosh(2πn)) by cutoff |n| ≤ N, n≠0."""
    prod = 1
    for n in range(-N, N+1):
        if n == 0:
            continue
        prod *= (1 + 1/mp.cosh(2*mp.pi*n))
    return prod

def cosh_product_target():
    return 2**(mp.mpf(3)/8) * mp.sqrt(1 + mp.sqrt(2))


# --------- High-level summary and CLI --------- #

def mmc_summary(t, sigma_min, sigma_max, sigma_steps, out_prefix=None):
    scan = mmc_scan(t, sigma_min, sigma_max, sigma_steps)
    band = band_and_exclusion(t, sigma_min, sigma_max, sigma_steps)

    MMC_pass = (
        scan["zero_crossings"] == 1
        and scan["sigma_star"] is not None
        and scan["S_prime"] is not None
        and 0.49 < scan["sigma_star"] < 0.51
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
        Path(f"{out_prefix}.summary.json").write_text(
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


def main():
    parser = argparse.ArgumentParser(
        description="EVT-style Edge Vector MMC scan around ξ(s)."
    )
    parser.add_argument("--t", type=float, help="Imaginary part t.")
    parser.add_argument("--sigma-min", type=float, default=0.2)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--sigma-steps", type=int, default=121)
    parser.add_argument("--out", type=str, default=None,
                        help="Prefix for JSON summary (optional).")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run Swiss-knife demos instead of MMC scan."
    )

    args = parser.parse_args()

    if args.demo:
        print("Demo A: ∫ sin x / x  vs  ∫ sin² x / x²")
        I1, I2 = sin_over_x_demo()
        print("I1 =", I1)
        print("I2 =", I2)
        print()

        print("Demo B: ∫ dt/(t⁴+1) from -10 to 10")
        val = t4_plus_1_integral(-10, 10)
        print("∫_{-10}^{10} dt/(t⁴+1) ≈", val)
        print()

        print("Demo C: cosh-product")
        P = cosh_product(N=50)
        target = cosh_product_target()
        print("P_N ≈", P)
        print("target =", target)
        print("relative error ≈", abs(P - target)/target)
    else:
        if args.t is None:
            raise SystemExit("Please provide --t (imaginary part) or use --demo.")
        mmc_summary(
            t=args.t,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sigma_steps=args.sigma_steps,
            out_prefix=args.out,
        )


if __name__ == "__main__":
    main()
