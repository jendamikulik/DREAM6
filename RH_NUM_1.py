import csv
import math
import mpmath as mp

# =========================
# 0) Precision + globals
# =========================

# You can crank this up, but be mindful: zeta evaluations are expensive.
mp.mp.dps = 80

# Central difference step for zeta' in imaginary direction.
# We'll make it scale mildly with precision.
def default_h():
    # Safe-ish: smaller with more precision, but not absurdly small.
    return mp.mpf("1e-7") * mp.power(10, -(mp.mp.dps - 50) / 80)

# Skip thresholds (precision-scaled)
def zeta_skip_eps():
    # If |zeta| is below this, zeta'/zeta becomes numerically nasty.
    # Scale with precision so it adapts.
    return mp.power(10, -mp.mp.dps / 4)

ZETA_RATIO_CAP = mp.mpf("1e8")  # if |zeta'/zeta| explodes, skip that point

# =========================
# 1) Core math objects
# =========================

def zeta_prime_central(s, h=None):
    if h is None:
        h = default_h()
    # Differentiate w.r.t. imaginary part: s -> s + i h
    return (mp.zeta(s + mp.mpc(0, h)) - mp.zeta(s - mp.mpc(0, h))) / (2 * h)

def xi_log_derivative(s):
    """
    xi(s) = 1/2 * s(s-1) * pi^{-s/2} * Gamma(s/2) * zeta(s)
    so (xi'/xi)(s) = 1/s + 1/(s-1) - (1/2)log(pi) + (1/2)digamma(s/2) + zeta'(s)/zeta(s)
    """
    z = mp.zeta(s)
    if mp.fabs(z) < zeta_skip_eps():
        return None  # signal skip
    zp = zeta_prime_central(s)
    ratio = zp / z
    if mp.fabs(ratio) > ZETA_RATIO_CAP:
        return None
    return (1/s + 1/(s-1) - mp.log(mp.pi)/2 + mp.digamma(s/2)/2 + ratio)

def S(sigma, t):
    val = xi_log_derivative(mp.mpc(sigma, t))
    if val is None:
        return None
    return mp.re(val)

# =========================
# 2) Band + adaptive fit points
# =========================

def band_width(t, c):
    # band = c / log t  (use natural log)
    t = mp.mpf(t)
    if t <= 1:
        return mp.mpf("10")  # meaningless for tiny t; force "inside"
    return mp.mpf(c) / mp.log(t)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def adaptive_fit_sigmas(t, c, max_delta=mp.mpf("0.25")):
    """
    Choose symmetric fit points around 1/2, but outside the band.
    Start with d = max(2*band, 0.05) and then 2d, 3d.
    Cap deltas to max_delta to avoid hugging boundaries.
    """
    b = band_width(t, c)
    d0 = max(2*b, mp.mpf("0.05"))
    deltas = [d0, 2*d0, 3*d0]

    sigmas = []
    for d in deltas:
        d = min(d, max_delta)
        s1 = mp.mpf("0.5") - d
        s2 = mp.mpf("0.5") + d
        # Ensure within (0,1)
        if s1 > 0 and s2 < 1:
            sigmas += [s1, s2]

    # De-dup and sort
    sigmas = sorted(list({mp.nsum(lambda k: 0, [0, -1]) + s for s in sigmas}))  # harmless trick to force mp.mpf
    # (The above line just keeps mp types stable; you can ignore it.)
    # Actually: Let's just ensure mp.mpf conversion:
    sigmas = sorted([mp.mpf(s) for s in sigmas])

    return sigmas

def outside_band(sig, b):
    return (sig <= mp.mpf("0.5") - b) or (sig >= mp.mpf("0.5") + b)

# =========================
# 3) Robust linear drift fit: S(sigma,t) ≈ alpha(t)*(sigma-1/2) + beta(t)
# =========================

def fit_alpha_beta(t, sigmas):
    xs = []
    ys = []
    for s in sigmas:
        y = S(s, t)
        if y is None:
            continue
        xs.append(mp.mpf(s) - mp.mpf("0.5"))
        ys.append(y)

    if len(xs) < 2:
        return None

    n = mp.mpf(len(xs))
    xbar = mp.fsum(xs) / n
    ybar = mp.fsum(ys) / n
    num = mp.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = mp.fsum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return None
    alpha = num / den
    beta = ybar - alpha * xbar
    return alpha, beta

def Q_value(t, sigma, alpha, beta=None):
    """
    R = S - (alpha*(sigma-1/2)+beta)
    Q = |R| / (|alpha|*|sigma-1/2|)
    """
    s_val = S(sigma, t)
    if s_val is None:
        return None
    x = mp.fabs(mp.mpf(sigma) - mp.mpf("0.5"))
    if x == 0:
        return None
    if beta is None:
        pred = alpha * (mp.mpf(sigma) - mp.mpf("0.5"))
    else:
        pred = alpha * (mp.mpf(sigma) - mp.mpf("0.5")) + beta
    R = s_val - pred
    Q = mp.fabs(R) / (mp.fabs(alpha) * x) if alpha != 0 else None
    return Q, R, s_val

# =========================
# 4) Candidate sigma grid (outside band)
# =========================

def candidate_sigmas(b, n_each_side=30):
    """
    Sample outside band on both sides:
    [0, 0.5-b] and [0.5+b, 1]
    """
    left_hi  = mp.mpf("0.5") - b
    right_lo = mp.mpf("0.5") + b

    # If band is huge, might collapse; handle gracefully
    if left_hi <= 0 or right_lo >= 1:
        return []

    sigs = []
    # Avoid exactly 0 or 1
    eps = mp.mpf("1e-6")
    # Left
    for k in range(n_each_side):
        s = eps + (left_hi - eps) * mp.mpf(k) / mp.mpf(n_each_side - 1)
        sigs.append(s)
    # Right
    for k in range(n_each_side):
        s = right_lo + (mp.mpf("1") - eps - right_lo) * mp.mpf(k) / mp.mpf(n_each_side - 1)
        sigs.append(s)

    # Unique + sort
    sigs = sorted(list({mp.mpf(s) for s in sigs}))
    return sigs

# =========================
# 5) Main scan loop
# =========================

def bd_scan(
    T_max=mp.mpf("1e6"),
    T_steps=300,
    c_values=(mp.mpf("0.5"), mp.mpf("1.0"), mp.mpf("1.5")),
    n_sig_each_side=25,
    out_csv="bd_scan_v21.csv",
    diag_ts=(mp.mpf("1447.22187074113477"), mp.mpf("15630.754")),
    diag_window=mp.mpf("25.0"),
    diag_points=25
):
    # Log-spaced t grid, avoid t< ~50 where log effects are weird for this purpose
    t_min = mp.mpf("50")
    log_min = mp.log(t_min)
    log_max = mp.log(T_max)

    rows = []
    global_best = None  # (Qmax, t, sigma, c, band, alpha, beta)

    print("Starting global BD scan v2.1...")
    print(f"mp.dps={mp.mp.dps}, T_max={T_max}, T_steps={T_steps}, c_values={list(c_values)}")

    for idx in range(1, T_steps + 1):
        # log grid
        u = mp.mpf(idx - 1) / mp.mpf(T_steps - 1)
        t = mp.e ** (log_min + u * (log_max - log_min))

        # We'll track max over c for this t (for progress print)
        progress_Qmax = mp.mpf("0")
        progress_alpha = None

        for c in c_values:
            b = band_width(t, c)
            sig_fit = adaptive_fit_sigmas(t, c)
            # Keep only those truly outside band (they should be by construction, but safe)
            sig_fit = [s for s in sig_fit if outside_band(s, b)]
            fit = fit_alpha_beta(t, sig_fit)
            if fit is None:
                continue
            alpha, beta = fit

            # Evaluate Q over candidate sigmas outside band
            sig_cands = candidate_sigmas(b, n_each_side=n_sig_each_side)
            if not sig_cands:
                continue

            Qmax = mp.mpf("-1")
            argmax = None

            for sigma in sig_cands:
                q = Q_value(t, sigma, alpha, beta)
                if q is None:
                    continue
                Q, R, Sval = q
                if Qmax < 0 or Q > Qmax:
                    Qmax = Q
                    argmax = (sigma, R, Sval)

            if Qmax < 0:
                continue

            sigma_star, R_star, S_star = argmax
            row = {
                "t": str(t),
                "c": str(c),
                "band": str(b),
                "alpha": str(alpha),
                "beta": str(beta),
                "Qmax": str(Qmax),
                "sigma_at_Qmax": str(sigma_star),
                "R_at_Qmax": str(R_star),
                "S_at_Qmax": str(S_star),
            }
            rows.append(row)

            # Track per-t progress
            if Qmax > progress_Qmax:
                progress_Qmax = Qmax
                progress_alpha = alpha

            # Track global best = worst-case maximum (largest Qmax)
            if (global_best is None) or (Qmax > global_best[0]):
                global_best = (Qmax, t, sigma_star, c, b, alpha, beta)

        if idx % max(1, (T_steps // 10)) == 0:
            a_str = "None" if progress_alpha is None else str(progress_alpha)
            print(f"[{idx}/{T_steps}] t≈{t}  Qmax≈{progress_Qmax}  alpha≈{a_str}")

    # Write CSV
    fieldnames = [
        "t","c","band","alpha","beta","Qmax","sigma_at_Qmax","R_at_Qmax","S_at_Qmax"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== GLOBAL WORST (largest Qmax over all tested points) ===")
    if global_best is None:
        print("No valid points (too many skips). Try increasing mp.dps or loosening caps.")
    else:
        Qmax, t, sigma, c, b, alpha, beta = global_best
        print(f"Qmax = {Qmax}")
        print(f"t    = {t}")
        print(f"sigma= {sigma}")
        print(f"c    = {c}")
        print(f"band = {b}")
        print(f"alpha= {alpha}")
        print(f"beta = {beta}")

    print(f"\nCSV written to {out_csv}")

    # Diagnostics around specific t's
    print("\n=== DIAGNOSTIC DUMPS ===")
    for t0 in diag_ts:
        print(f"\n--- around t0 = {t0} ---")
        for c in c_values:
            b0 = band_width(t0, c)
            fit_sig = adaptive_fit_sigmas(t0, c)
            fit_sig = [s for s in fit_sig if outside_band(s, b0)]
            fit = fit_alpha_beta(t0, fit_sig)
            print(f"c={c} band={b0} fit_sigmas={list(map(str, fit_sig))}")
            if fit is None:
                print("  fit failed")
                continue
            alpha0, beta0 = fit
            print(f"  alpha={alpha0} beta={beta0}")

        # Sweep t near t0 to see alpha stability
        print("  local t-sweep:")
        for j in range(diag_points):
            tj = t0 - diag_window + (2*diag_window) * mp.mpf(j) / mp.mpf(diag_points - 1)
            c = c_values[0]
            bj = band_width(tj, c)
            sig_fit = adaptive_fit_sigmas(tj, c)
            sig_fit = [s for s in sig_fit if outside_band(s, bj)]
            fit = fit_alpha_beta(tj, sig_fit)
            if fit is None:
                print(f"    t={tj} fit failed")
                continue
            aj, bj0 = fit
            print(f"    t={tj} alpha={aj}")

    return global_best

# =========================
# 6) Run
# =========================

if __name__ == "__main__":
    # Suggested starting parameters: quick but meaningful
    bd_scan(
        T_max=mp.mpf("1e6"),
        T_steps=300,
        c_values=(mp.mpf("0.5"), mp.mpf("1.0"), mp.mpf("1.5")),
        n_sig_each_side=25,
        out_csv="bd_scan_v21.csv",
        diag_ts=(mp.mpf("1447.22187074113477"), mp.mpf("15630.754")),
        diag_window=mp.mpf("50.0"),
        diag_points=21
    )
