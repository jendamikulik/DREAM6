#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RH Verifier: B3′-full & B4-full + Li positivity (All-in-one Python)
Patched: robust Li integration (unwrapped log, Kahan sum, adaptive grid, stable radius),
B3.X_min in config, and minor safety tweaks.

Usage:
  python rh_verifier.py --config config.json --out reports
  python rh_verifier.py --calibrate  # only prints suggested C1
"""
import argparse, math, json, cmath, csv, datetime
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import mpmath as mp

try:
    import pandas as pd
except Exception:
    pd = None

# -----------------------------
# Basic special functions
# -----------------------------
def zeta(s):     return mp.zeta(s)
def log_zeta(s): return mp.log(mp.zeta(s))
def chi_star(s): return mp.mpf('0.5') * s * (s-1) * (mp.pi)**(-s/2) * mp.gamma(s/2)
def xi(s):       return chi_star(s) * zeta(s)
def log_xi(s):   return mp.log(chi_star(s)) + mp.log(zeta(s))

# -----------------------------
# Arithmetic helpers
# -----------------------------
def mobius(n:int)->int:
    if n == 1:
        return 1
    mu, m, p = 1, n, 2
    while p*p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            mu = -mu
        p += 1 if p == 2 else 2
    if m > 1:
        mu = -mu
    return mu

# -----------------------------
# Dataclasses (config)
# -----------------------------
@dataclass
class B3Params:
    theta: float = 0.01
    kappa: int = 2
    t0: float = 10.0
    C_M: float = 2.0
    C1: float = 1.0
    C2: float = 1.0
    sigma_min: float = 0.5
    sigma_max: float = 1.0
    sigma_steps: int = 6
    t_values: List[float] = None
    # new: robust minimum for mollifier length (prevents |M| = 0 degenerate)
    X_min: float = 10.0

@dataclass
class B4Params:
    # Re-bound constants
    A0: float = -0.6931471805599453   # log(1/2)
    A1: float = 0.0
    A2: float = 0.0
    B0: float = 0.5772156649015329    # Euler-Mascheroni
    B1: float = 0.0
    # Im-bound constants
    D0: float = math.pi
    D1: float = 1.0
    # Li scan range
    n_min: int = 10
    n_max: int = 200
    circle_points: int = 720   # base minimum; final m is adaptive

@dataclass
class PrecisionParams:
    mp_dps: int = 80

@dataclass
class Config:
    precision: PrecisionParams
    b3: B3Params
    b4: B4Params
    output_dir: str = "reports"

# -----------------------------
# Mollifier (short, smooth)
# -----------------------------
def mollifier_M(s, X, kappa, cfg: Config):
    """
    M(s) = sum_{n<=X} mu(n) * w(n)^kappa * n^{-s},  w(n)=max(0,1-log n/log X).
    Includes n=1 term and enforces X>=X_min (from config) for stability.
    """
    X = max(float(X), float(getattr(cfg.b3, "X_min", 10.0)))
    if X < 2.0:
        X = 2.0
    logX = math.log(X)

    total = 1.0 + 0.0j  # n=1: mu(1)=1, w(1)=1, 1^{-s}=1
    N = int(X)
    for n in range(2, N + 1):
        w = 1.0 - math.log(n) / logX
        if w <= 0.0:
            continue
        mu = mobius(n)
        if mu == 0:
            continue
        term = mu * (w ** kappa) * complex(n) ** (-s)
        total += term
    return total

# -----------------------------
# B3 verification
# -----------------------------
def verify_B3(cfg: Config)->Dict[str,Any]:
    mp.mp.dps = cfg.precision.mp_dps
    p = cfg.b3
    res = {"uniform_bound_pass": True, "uniform_violations": [],
           "quadratic_pass": True, "quadratic_violations": []}

    t_vals = p.t_values if p.t_values else [p.t0, p.t0*2, p.t0*5, p.t0*10]
    sigmas = [p.sigma_min + i*(p.sigma_max-p.sigma_min)/max(1,p.sigma_steps-1) for i in range(p.sigma_steps)]

    for t in t_vals:
        X = max(p.X_min, t**p.theta)
        for sigma in sigmas:
            s = complex(sigma, t)
            M = mollifier_M(s, X, p.kappa, cfg=cfg)
            absM = abs(M)

            if not (1.0/p.C_M <= absM <= p.C_M):
                res["uniform_bound_pass"] = False
                res["uniform_violations"].append({"sigma": sigma, "t": t, "|M|": float(absM)})

            try:
                zM = zeta(s) * M
                s_half = complex(0.5, t)
                zM_half = zeta(s_half) * mollifier_M(s_half, X, p.kappa, cfg=cfg)

                lhs = mp.log(abs(zM)) - mp.log(abs(zM_half))
                rhs = p.C1*(sigma-0.5)**2 + (p.C2/max(t,1.0))

                if lhs > rhs + mp.mpf("1e-12"):
                    res["quadratic_pass"] = False
                    res["quadratic_violations"].append(
                        {"sigma": sigma, "t": t, "lhs": float(lhs), "rhs": float(rhs)}
                    )
            except Exception as e:
                res["quadratic_pass"] = False
                res["quadratic_violations"].append({"sigma": sigma, "t": t, "error": str(e)})
    return res

# -----------------------------
# Helpers for Li integration
# -----------------------------
def kahan_sum(iterable):
    s = 0+0j
    c = 0+0j
    for x in iterable:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def log_xi_unwrapped(points):
    """
    Return a list of log xi(s) values with continuous imaginary part (phase unwrapped),
    to avoid 2π jumps along the circle.
    """
    vals = []
    prev_im = None
    offset = 0.0
    two_pi = 2*math.pi
    for s in points:
        L = log_xi(s)
        im = float(mp.im(L))
        if prev_im is not None:
            k = round((prev_im - im) / two_pi)
            offset += k * two_pi
        vals.append(mp.re(L) + 1j*(mp.im(L) + offset))
        prev_im = im + offset
    return vals

# -----------------------------
# B4 + Li verification (robust)
# -----------------------------
def verify_B4_and_li(cfg: Config)->Dict[str,Any]:
    p = cfg.b4
    rep = {"re_bound_pass": True, "im_bound_pass": True,
           "re_bound_violations": [], "im_bound_violations": [],
           "li_nonneg_pass": True, "li_violations": [], "li_values": []}

    base_dps = cfg.precision.mp_dps
    two_pi = 2*mp.pi

    for n in range(p.n_min, p.n_max+1):
        # Increase precision with n (conservative scaling)
        mp.mp.dps = max(base_dps, 60 + 2*n)

        # Stable radius: r = c / (log n)^2, with 0.5 <= c <= 0.8 typically
        c = mp.mpf('0.6')
        r = c / (mp.log(n)**2)

        # Adaptive number of points on the circle
        m = max(2048, int(16 * mp.mp.dps))

        # Sample circle points
        pts = [1 + r*mp.e**(1j*(two_pi*k/m)) for k in range(m)]
        Lvals = log_xi_unwrapped(pts)

        Re_vals = [mp.re(L) for L in Lvals]
        Im_vals = [mp.im(L) for L in Lvals]
        sup_Re = max(Re_vals)
        sup_Im_abs = max(abs(v) for v in Im_vals)

        # B4 Re-bound check
        re_rhs = mp.log(1/r) + (p.A0 + p.B0) + p.A1*r + (p.A2 + p.B1)*r*r
        if sup_Re > re_rhs + mp.mpf("1e-12"):
            rep["re_bound_pass"] = False
            rep["re_bound_violations"].append({"n": n, "sup_Re": float(sup_Re), "rhs": float(re_rhs)})

        # B4 Im-bound check
        im_rhs = p.D0 + p.D1 * r * mp.log(1/r)
        if sup_Im_abs > im_rhs + mp.mpf("1e-12"):
            rep["im_bound_pass"] = False
            rep["im_bound_violations"].append({"n": n, "sup|Im|": float(sup_Im_abs), "rhs": float(im_rhs)})

        # Trapezoidal (periodic) quadrature with Kahan sum
        terms = []
        for k, s in enumerate(pts):
            th = two_pi*k/m
            ds = 1j * r * mp.e**(1j*th) * (two_pi/m)
            terms.append((Lvals[k] / (s-1)**(n+1)) * ds)

        total = kahan_sum(terms)
        lam = total / (2*mp.pi*1j)
        lamr = float(mp.re(lam))

        # Tolerance proportional to precision (safety)
        tol = float(mp.mpf('1e-8'))
        if lamr < -tol:
            rep["li_nonneg_pass"] = False
            rep["li_violations"].append({"n": n, "lambda_n": lamr, "tol": tol})

        rep["li_values"].append({"n": n, "lambda_n": lamr})

    return rep

# -----------------------------
# Config I/O and reporting
# -----------------------------
def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ensure unknown B3 fields (like X_min) don't crash
    b3_data = data.get("b3", {})
    if "X_min" not in b3_data:
        b3_data["X_min"] = 10.0

    cfg = Config(
        precision=PrecisionParams(**data.get("precision", {"mp_dps":80})),
        b3=B3Params(**b3_data),
        b4=B4Params(**data.get("b4", {})),
        output_dir=data.get("output_dir", "reports")
    )
    return cfg

def save_reports(cfg, b3res, b4res, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().isoformat()+"Z"
    lines = []
    lines.append(f"# RH Verifier Report\n\nTimestamp (UTC): {ts}\n")
    lines.append("## Verdicts\n")
    lines.append(f"- B3′ uniform bound: **{'PASS' if b3res['uniform_bound_pass'] else 'FAIL'}**")
    lines.append(f"- B3′ quadratic control: **{'PASS' if b3res['quadratic_pass'] else 'FAIL'}**")
    lines.append(f"- B4 Re-bound: **{'PASS' if b4res['re_bound_pass'] else 'FAIL'}**")
    lines.append(f"- B4 Im-bound: **{'PASS' if b4res['im_bound_pass'] else 'FAIL'}**")
    lines.append(f"- Li non-negativity: **{'PASS' if b4res['li_nonneg_pass'] else 'FAIL'}**\n")

    def add_table(title, rows, headers):
        lines.append(f"### {title}\n")
        if not rows:
            lines.append("_None_\n"); return
        header = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join(["---"]*len(headers)) + " |"
        lines.append(header); lines.append(sep)
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h,'')) for h in headers) + " |")
        lines.append("")

    add_table("B3′ Uniform Bound Violations", b3res["uniform_violations"], ["sigma","t","|M|"])
    add_table("B3′ Quadratic Violations", b3res["quadratic_violations"], ["sigma","t","lhs","rhs","error"])
    add_table("B4 Re-bound Violations", b4res["re_bound_violations"], ["n","sup_Re","rhs"])
    add_table("B4 Im-bound Violations", b4res["im_bound_violations"], ["n","sup|Im|","rhs"])
    (outdir/"summary.md").write_text("\n".join(lines), encoding="utf-8")

    li_csv = outdir/"li_coeffs.csv"
    try:
        import pandas as pd
        pd.DataFrame(b4res["li_values"]).to_csv(li_csv, index=False)
    except Exception:
        with open(li_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["n","lambda_n"])
            for row in b4res["li_values"]:
                w.writerow([row["n"], f"{row['lambda_n']:.12g}"])

# -----------------------------
# Optional: B3 calibration helper
# -----------------------------
def calibrate_B3(cfg: Config)->Dict[str, Any]:
    """
    Suggest the minimal C1 (for fixed C2) so that
      log|ζM(σ+it)| - log|ζM(1/2+it)| <= C1*(σ-1/2)^2 + C2/t
    holds for sampled (σ,t). Returns suggested C1 and worst-case.
    """
    mp.mp.dps = cfg.precision.mp_dps
    p = cfg.b3
    t_vals = p.t_values if p.t_values else [p.t0, p.t0*2, p.t0*5, p.t0*10]
    sigmas = [p.sigma_min + i*(p.sigma_max-p.sigma_min)/max(1,p.sigma_steps-1) for i in range(p.sigma_steps)]
    needed_C1_values = []
    worst = None
    for t in t_vals:
        X = max(p.X_min, t**p.theta)
        for sigma in sigmas:
            if sigma <= 0.5:
                continue
            s = complex(sigma, t)
            s_half = complex(0.5, t)
            M = mollifier_M(s, X, p.kappa, cfg)
            M_half = mollifier_M(s_half, X, p.kappa, cfg)
            zM = zeta(s) * M
            zM_half = zeta(s_half) * M_half
            lhs = mp.log(abs(zM)) - mp.log(abs(zM_half))
            base = lhs - (p.C2/max(t,1.0))
            denom = (sigma-0.5)**2
            need = float(base/denom) if denom>0 else 0.0
            needed_C1_values.append(need)
            if (worst is None) or (need > worst["need"]):
                worst = {"sigma": sigma, "t": t, "lhs": float(lhs), "need": float(need)}
    suggested = max(needed_C1_values) if needed_C1_values else p.C1
    return {"suggested_C1": float(suggested), "worst_case": worst}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="RH Verifier (B3′, B4, Li positivity) — All-in-one")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config (optional)")
    ap.add_argument("--out", type=str, default=None, help="Output dir override")
    ap.add_argument("--calibrate", action="store_true", help="Suggest minimal C1 for current config")
    args = ap.parse_args()

    if args.config:
        cfg = load_config(Path(args.config))
    else:
        cfg = Config(precision=PrecisionParams(mp_dps=80), b3=B3Params(), b4=B4Params(), output_dir="reports")

    if args.out:
        cfg.output_dir = args.out

    if args.calibrate:
        info = calibrate_B3(cfg)
        print(json.dumps(info, indent=2))
        return

    # Run checks
    b3res = verify_B3(cfg)
    b4res = verify_B4_and_li(cfg)
    save_reports(cfg, b3res, b4res, Path(cfg.output_dir))
    print(f"[OK] Report saved to {cfg.output_dir}/summary.md")
    print(f"[OK] Li coefficients CSV saved to {cfg.output_dir}/li_coeffs.csv")

if __name__ == "__main__":
    main()
