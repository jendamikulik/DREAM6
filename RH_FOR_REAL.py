import mpmath as mp
import math
from dataclasses import dataclass

mp.mp.dps = 50

# ---------- core analytic observable ----------
def xi_logderiv(s: mp.mpc) -> mp.mpc:
    # (xi'/xi)(s) using standard factorization:
    # xi(s)=1/2*s(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)
    # log-derivative:
    # (xi'/xi)(s)= 1/s + 1/(s-1) - (1/2)log(pi) + (1/2)psi(s/2) + zeta'(s)/zeta(s)
    return (1/s
            + 1/(s-1)
            - mp.log(mp.pi)/2
            + mp.digamma(s/2)/2
            + mp.diff(mp.zeta, s) / mp.zeta(s))

def S_sigma_t(sigma: mp.mpf, t: mp.mpf) -> mp.mpf:
    s = mp.mpc(sigma, t)
    return mp.re(xi_logderiv(s))

# ---------- grids ----------
def make_adaptive_grid(a=mp.mpf("0.135"), x0=mp.mpf("0.02"),
                       hin=mp.mpf("0.0005"), hout=mp.mpf("0.002")):
    xs = []
    # outer negative
    x = -a
    while x < -x0:
        xs.append(x)
        x += hout
    # inner
    x = -x0
    while x <= x0:
        xs.append(x)
        x += hin
    # outer positive
    x = x0 + hout
    while x <= a + mp.mpf("1e-18"):
        xs.append(x)
        x += hout
    # enforce symmetry by mirroring (avoid drift from step mismatch)
    xs = sorted(set(xs))
    # make perfectly symmetric by snapping pairs
    xs2 = []
    for x in xs:
        xs2.append(x)
        if -x not in xs2:
            xs2.append(-x)
    xs2 = sorted(set(xs2))
    return xs2

# ---------- least squares utilities ----------
def linfit_alpha_beta(xs, ys, mask):
    # Fit ys ~ alpha*x + beta over indices where mask[i]=True
    X = []
    Y = []
    for x, y, m in zip(xs, ys, mask):
        if m:
            X.append(x)
            Y.append(y)
    n = mp.mpf(len(X))
    if n < 2:
        return mp.mpf("0"), mp.mpf("0")
    xbar = mp.fsum(X)/n
    ybar = mp.fsum(Y)/n
    num = mp.fsum([(x-xbar)*(y-ybar) for x, y in zip(X, Y)])
    den = mp.fsum([(x-xbar)*(x-xbar) for x in X])
    if den == 0:
        return mp.mpf("0"), ybar
    alpha = num/den
    beta = ybar - alpha*xbar
    return alpha, beta

def quadfit_c2(xs, rs, mask):
    # Fit rs ~ c2*x^2 over masked points (even jet only)
    X = []
    Y = []
    for x, r, m in zip(xs, rs, mask):
        if m:
            X.append(x*x)
            Y.append(r)
    n = len(X)
    if n < 2:
        return mp.mpf("0")
    num = mp.fsum([xx*y for xx, y in zip(X, Y)])
    den = mp.fsum([xx*xx for xx in X])
    if den == 0:
        return mp.mpf("0")
    return num/den

# ---------- projection operators ----------
def smooth_odd_projection(xs, vals, sigma=mp.mpf("0.004")):
    # smooth with Gaussian then take odd part: (f - f(-x))/2
    # 1) gaussian smoothing on irregular grid by kernel weights
    # (O(n^2), OK for grid ~ few hundred; if larger, use convolution on uniform grid)
    n = len(xs)
    sm = [mp.mpf("0")]*n
    for i, xi in enumerate(xs):
        wsum = mp.mpf("0")
        acc = mp.mpf("0")
        for xj, fj in zip(xs, vals):
            w = mp.e**(-((xi-xj)**2)/(2*sigma*sigma))
            acc += w*fj
            wsum += w
        sm[i] = acc/wsum if wsum != 0 else vals[i]

    # 2) odd part
    # build map from x to index (grid is mp.mpf; use string key)
    idx = {str(x): i for i, x in enumerate(xs)}
    odd = [mp.mpf("0")]*n
    for i, x in enumerate(xs):
        j = idx.get(str(-x), None)
        if j is None:
            odd[i] = sm[i]
        else:
            odd[i] = (sm[i] - sm[j]) / 2
    return odd

# ---------- robust Lipschitz estimator ----------
def robust_max_slope(xs, vals, window=7):
    # local linear regression slope on sliding window
    n = len(xs)
    k = window//2
    best = mp.mpf("0")
    best_x = None
    for i in range(n):
        lo = max(0, i-k)
        hi = min(n, i+k+1)
        X = xs[lo:hi]
        Y = vals[lo:hi]
        m = len(X)
        if m < 2:
            continue
        xbar = mp.fsum(X)/m
        ybar = mp.fsum(Y)/m
        num = mp.fsum([(x-xbar)*(y-ybar) for x, y in zip(X, Y)])
        den = mp.fsum([(x-xbar)*(x-xbar) for x in X])
        if den == 0:
            continue
        slope = num/den
        if abs(slope) > best:
            best = abs(slope)
            best_x = xs[i]
    return best, best_x

# ---------- one window experiment ----------
def compute_Qlip_for_t(t,
                      a=mp.mpf("0.135"),
                      x0=mp.mpf("0.02"),
                      x1=mp.mpf("0.04")):
    xs = make_adaptive_grid(a=a, x0=x0)

    sigmas = [mp.mpf("0.5")+x for x in xs]
    Svals = [S_sigma_t(s, t) for s in sigmas]

    # drift fit outside center zone |x|<x1
    mask_drift = [abs(x) >= x1 for x in xs]
    alpha, beta = linfit_alpha_beta(xs, Svals, mask_drift)

    R1 = [Sv - (alpha*x + beta) for Sv, x in zip(Svals, xs)]

    # jet-even removal in |x|<=x0
    mask_loc = [abs(x) <= x0 for x in xs]
    c2 = quadfit_c2(xs, R1, mask_loc)
    R2 = []
    for x, r in zip(xs, R1):
        if abs(x) <= x0:
            R2.append(r - c2*(x*x))
        else:
            R2.append(r)

    # projected residual (smooth-odd)
    PR = smooth_odd_projection(xs, R2, sigma=mp.mpf("0.004"))

    # robust max slope of projected residual in x
    slope_max, x_star = robust_max_slope(xs, PR, window=7)

    # normalization ~ A(t) : use alpha as empirical A(t) proxy
    # your original Qlip ~ max |d/dx PR| / |A(t)|
    Ahat = abs(alpha) if alpha != 0 else mp.mpf("1e-30")
    Qlip = slope_max / Ahat
    return Qlip, x_star, alpha, beta, c2

# ---------- quick KPI sweep ----------
def kpi_run(T=mp.mpf("1e8"), centers=9, step=mp.mpf("2000")):
    out = []
    for j in range(centers):
        t = T + (j - centers//2)*step
        Q, x_star, alpha, beta, c2 = compute_Qlip_for_t(t)
        out.append((t, Q, x_star, alpha, c2))
        print(f"[{j+1}/{centers}] t={t}  Qlip={Q}  x*={x_star}")
    # summary
    Qs = sorted([q for _, q, _, _, _ in out])
    med = Qs[len(Qs)//2]
    gt1 = sum(1 for q in Qs if q > 1)/len(Qs)
    print("\n=== SUMMARY ===")
    print("median Qlip =", med)
    print("gt1_rate    =", gt1)
    # how many maxima near 0
    near0 = sum(1 for _, _, xstar, _, _ in out if xstar is not None and abs(xstar) < mp.mpf("0.005"))/len(out)
    print("x* near 0   =", near0)
    return out


mp.mp.dps = 40
kpi_run(T=mp.mpf("1e8"), centers=9, step=mp.mpf("2000"))
