#!/usr/bin/env python3
"""
QZR packet fan in coordinates (p, q, d), d = R_rho(M) - M(1).
Version 2: REALIZABLE packet selector.

Why v1 was unsound
------------------
v1 used d_G = (d - rho(1-4p-q)_+)_+, which is 0 in cell A although the packet
has mass 3p > 0 on j >= 2 (h = 0).  Any positive measure G on {2,3,...} has
d_G = sum_{j>=2} rho^{j-1} G(j) > 0, so (h, d_G) = (0, 0) is only a closure
point, not a packet.  It already occurs on the real child of W_{0,1} along
branch 2 (p = 1/16, q = 23/96, d ~ 0.13288 for rho = 1/2).

Selector used here (always realizable, for every law with the given state)
-----------------------------------------------------------------------
    h     = (4p+q-1)_+                        (minimal atom-1 part of the packet)
    theta = (3p-h)/(1-p-q)   in [0,1]
    G_{>=2} = theta * M_{>=2}                 (proportional tail packet)
  hence
    d_G = theta * d,     s = theta * r,       M - G has tail (1-theta) M_{>=2}.
Cells:
    A': 4p+q <= 1 :  h = 0,          theta = 3p/(1-p-q)
    C : 4p+q >= 1 :  h = 4p+q-1,     theta = 1  (d_G = d, s = r)

The only adversarial coordinate is r = M(2) in I(T,d) (closure of the
necessary interval; using the closure only enlarges the adversary: sound).

theta*d and theta*r are bilinear in cell A'.  Per box we introduce
w1 = theta*d, w2 = theta*r and the auxiliary theta with McCormick envelopes
(theta*T = 3p also relaxed by McCormick).  This is an over-approximation of
the exact image, so a CLOSED verdict is sound (up to floating-point LP).

Counterexample check
--------------------
A wall hit on the abstraction yields a chain of branches from the root.  The
exact (nonlinear) dynamics along this branch sequence is optimised over the
adversary choices r_t (multistart, bounded).  A strictly positive wall excess
found this way is reported as REAL (a realizable history up to closure
endpoints of the r-interval; with a strict margin it is realizable by nearby
interior values).  Otherwise the chain is treated as spurious and its boxes
are split.

Usage
-----
    python3 qzr_box_invariant_v2.py run [--rho 0.5 --hp 0.01 --hq 0.02 --hd 0.02
                                        --maxlev 6 --budget 200]
    python3 qzr_box_invariant_v2.py status
    python3 qzr_box_invariant_v2.py reset
State: qzr_box_state_v2.pkl (saved every 45 s and at the end).
"""
import sys, time, pickle, argparse, os
import numpy as np
from scipy.optimize import linprog, minimize

STATE = "qzr_box_state_v2.pkl"
EPS = 1e-9
NV = 6          # v = (p, q, d, r, w1, w2),  w1 = d_G = theta*d,  w2 = s = theta*r


# ----------------------------------------------------------------------------
# exact dynamics (used for the root, for concrete counterexample checks)
# ----------------------------------------------------------------------------
def interval_r(T, d, rho):
    if T <= 0:
        return 0.0, 0.0
    return max(0.0, (d - rho ** 2 * T) / (rho * (1 - rho))), min(T, d / rho)


def exact_child(x, branch, r, rho):
    p, q, d = x
    T = max(0.0, 1 - p - q)
    h = max(0.0, 4 * p + q - 1)
    TG = 3 * p - h
    theta = TG / T if T > 1e-15 else 1.0
    theta = min(max(theta, 0.0), 1.0)
    dG, s = theta * d, theta * r
    R, g = q + d, h + dG
    if branch == 0:
        pc, qc, Rc = 7 * p / 8, p / 2 + 7 * q / 8, p / 2 + ((7 + rho) * R - rho * g) / 8
    elif branch == 2:
        pc = (10 * p + 3 * q + h) / 16
        qc = (4 * p + 10 * q + 3 * r + s) / 16
        Rc = p * (4 + 2 * rho) / 16 + ((3 * d + dG) / rho + 10 * R + 3 * rho * R - rho * g) / 16
    else:
        pc = (5 * p + q + h) / 8
        qc = (7 * q + r + s) / 8
        Rc = ((d + dG) / rho + 7 * R) / 8
    return np.array([pc, qc, Rc - qc])


# ----------------------------------------------------------------------------
# affine expressions over v (NV-vector, const)
# ----------------------------------------------------------------------------
def E(i=None, c=0.0):
    v = np.zeros(NV)
    if i is not None:
        v[i] = 1.0
    return (v, float(c))


def comb(*terms):
    v, c = np.zeros(NV), 0.0
    for k, (a, b) in terms:
        v = v + k * a
        c += k * b
    return (v, c)


P_, Q_, D_, R_, W1, W2 = (E(i) for i in range(NV))
ONE, ZERO = E(None, 1.0), E(None, 0.0)


def cell_system(cell, rho, lo, hi):
    """Linear relaxation for a box [lo,hi] in (p,q,d) and a cell.
    Returns A, b (A v <= b) and the three children as affine maps of v."""
    T = comb((1, ONE), (-1, P_), (-1, Q_))
    m = comb((1, ONE), (-4, P_), (-1, Q_))          # 1-4p-q
    cons = []

    def le(a, b):
        cons.append(comb((1, a), (-1, b)))

    def ge(a, b):
        cons.append(comb((1, b), (-1, a)))

    # parent validity
    ge(P_, ZERO); ge(Q_, ZERO); ge(T, ZERO); ge(D_, ZERO); le(D_, comb((rho, T)))
    le(P_, E(None, 0.25))
    # adversary r in I(T,d) (closure)
    k = 1.0 / (rho * (1 - rho))
    ge(R_, ZERO); ge(R_, comb((k, D_), (-k * rho ** 2, T))); le(R_, T); le(R_, comb((1 / rho, D_)))

    if cell == "A":
        h = ZERO
        ge(m, ZERO)
        # theta bounds from the box (theta = 3p/(1-p-q) increasing in p and q)
        Tlo, Thi = 1 - hi[0] - hi[1], 1 - lo[0] - lo[1]
        if Tlo <= 1e-12:
            return None
        a, bth = 3 * lo[0] / Thi, min(1.0, 3 * hi[0] / Tlo)
        dl, dh = lo[2], hi[2]
        rl, rh = 0.0, min(Thi, dh / rho)
        # McCormick w1 = theta*d
        def mc(w, x, xl, xh):
            # w >= a x + xl th - a xl ; w >= b x + xh th - b xh ; w <= b x + xl th - b xl ; w <= a x + xh th - a xh
            # theta is not a variable; eliminate it using theta = 3p/T -> keep theta implicit via bounds:
            # use the weaker (theta-free) envelopes: a*x <= w <= b*x  (valid since x >= 0)
            ge(w, comb((a, x))); le(w, comb((bth, x)))
        mc(W1, D_, dl, dh)
        mc(W2, R_, rl, rh)
        # additional exact linear facts of the proportional packet
        le(W1, D_); le(W2, R_); ge(W1, ZERO); ge(W2, ZERO)
        le(W1, comb((3 * rho, P_)))                    # d_G <= rho * T_G = 3 rho p
        le(W2, comb((3, P_)))                          # s <= T_G = 3p
        # ratio consistency: w1 * T = 3p * d  and  w2 * T = 3p * r  (McCormick on both sides)
        for w, x, xl, xh in ((W1, D_, dl, dh), (W2, R_, rl, rh)):
            # w*T with w in [0, b*xh], T in [Tlo,Thi];  3p*x with p in [lo0,hi0], x in [xl,xh]
            wl, wh = a * xl, bth * xh
            pl, ph = lo[0], hi[0]
            # lower(w*T) <= upper(3 p x)  and  lower(3 p x) <= upper(w*T)
            # w*T >= Tlo*w + wl*T - Tlo*wl ; w*T >= Thi*w + wh*T - Thi*wh
            # w*T <= Thi*w + wl*T - Thi*wl ; w*T <= Tlo*w + wh*T - Tlo*wh
            # p*x similar
            lowWT = [comb((Tlo, w), (wl, T), (-Tlo * wl, ONE)), comb((Thi, w), (wh, T), (-Thi * wh, ONE))]
            upWT = [comb((Thi, w), (wl, T), (-Thi * wl, ONE)), comb((Tlo, w), (wh, T), (-Tlo * wh, ONE))]
            lowPX = [comb((3 * pl, x), (3 * xl, P_), (-3 * pl * xl, ONE)), comb((3 * ph, x), (3 * xh, P_), (-3 * ph * xh, ONE))]
            upPX = [comb((3 * ph, x), (3 * xl, P_), (-3 * ph * xl, ONE)), comb((3 * pl, x), (3 * xh, P_), (-3 * pl * xh, ONE))]
            for L in lowWT:
                for U in upPX:
                    le(L, U)
            for L in lowPX:
                for U in upWT:
                    le(L, U)
    else:  # cell C: theta = 1
        h = comb((-1, m))
        le(m, ZERO)
        le(W1, D_); ge(W1, D_); le(W2, R_); ge(W2, R_)
        le(h, Q_)
    dG, s = W1, W2
    Rr = comb((1, Q_), (1, D_))
    g = comb((1, h), (1, dG))
    p0 = comb((7 / 8, P_)); q0 = comb((1 / 2, P_), (7 / 8, Q_))
    R0 = comb((1 / 2, P_), ((7 + rho) / 8, Rr), (-rho / 8, g))
    p2 = comb((10 / 16, P_), (3 / 16, Q_), (1 / 16, h))
    q2 = comb((4 / 16, P_), (10 / 16, Q_), (3 / 16, R_), (1 / 16, s))
    R2 = comb(((4 + 2 * rho) / 16, P_), (3 / (16 * rho), D_), (1 / (16 * rho), dG),
              ((10 + 3 * rho) / 16, Rr), (-rho / 16, g))
    p4 = comb((5 / 8, P_), (1 / 8, Q_), (1 / 8, h))
    q4 = comb((7 / 8, Q_), (1 / 8, R_), (1 / 8, s))
    R4 = comb((1 / (8 * rho), D_), (1 / (8 * rho), dG), (7 / 8, Rr))
    kids = [(p0, q0, comb((1, R0), (-1, q0))),
            (p2, q2, comb((1, R2), (-1, q2))),
            (p4, q4, comb((1, R4), (-1, q4)))]
    A = np.array([c[0] for c in cons]); b = np.array([-c[1] for c in cons])
    return A, b, kids


# ----------------------------------------------------------------------------
class Explorer:
    def __init__(self, rho, hp, hq, hd, maxlev=6):
        self.rho, self.h, self.maxlev = rho, np.array([hp, hq, hd]), maxlev
        self.refined, self.done, self.wallsof, self.preds = set(), {}, {}, {}
        self.parent, self.queue, self.wallhit = {}, [], []
        self.active = set()
        self.real_cex, self.refinements, self.lps = None, 0, 0
        self.pseudo = {"q": (np.array([0, 1, 0.]), 0.40), "4p+q": (np.array([4, 1, 0.]), 0.85)}
        R0 = sum(rho ** (j - 1) * 2 / ((j + 1) * (j + 2)) for j in range(1, 20000))
        self.root = np.array([0.0, 1 / 3, R0 - 1 / 3])
        self.gc()

    # geometry --------------------------------------------------------------
    def size(self, lev):
        return self.h / (2 ** lev)

    def box_bounds(self, B):
        lev, i, j, k = B
        lo = np.array([i, j, k]) * self.size(lev)
        return lo, lo + self.size(lev)

    def leaf_of_point(self, x):
        B = (0,) + tuple(int(v) for v in np.floor(np.array(x) / self.h + 1e-12))
        while B in self.refined:
            lev = B[0] + 1
            B = (lev,) + tuple(int(v) for v in np.floor(np.array(x) / self.size(lev) + 1e-12))
        return B

    def leaves_in(self, lo, hi):
        out = []
        ilo = np.floor(lo / self.h + 1e-12).astype(int)
        ihi = np.floor(hi / self.h + 1e-12).astype(int)
        stack = [(0, i, j, k) for i in range(ilo[0], ihi[0] + 1)
                 for j in range(ilo[1], ihi[1] + 1) for k in range(ilo[2], ihi[2] + 1)]
        while stack:
            B = stack.pop()
            if B in self.refined:
                lev, i, j, k = B
                for a in (0, 1):
                    for b_ in (0, 1):
                        for c in (0, 1):
                            C = (lev + 1, 2 * i + a, 2 * j + b_, 2 * k + c)
                            cl, ch = self.box_bounds(C)
                            if np.all(cl <= hi + 1e-15) and np.all(ch >= lo - 1e-15):
                                stack.append(C)
            else:
                out.append(B)
        return out

    def lp(self, c, A, b, bounds):
        self.lps += 1
        return linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # abstract step -----------------------------------------------------------
    def process(self, B):
        lo, hi = self.box_bounds(B)
        bounds = [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])] + [(None, None)] * 3
        targets, walls = {}, []
        for cell in "AC":
            sysm = cell_system(cell, self.rho, lo, hi)
            if sysm is None:
                continue
            A, b, kids = sysm
            if self.lp(np.zeros(NV), A, b, bounds).status != 0:
                continue
            for bi, (kp, kq, kd) in enumerate(kids):
                M = np.vstack([kp[0], kq[0], kd[0]]); m0 = np.array([kp[1], kq[1], kd[1]])
                wl = [(np.array([1., 0, 0]), 0.25, None), (np.array([14., 4, 0]), 5.0, None)]
                wl += [(vec, lim, name) for name, (vec, lim) in self.pseudo.items()]
                for vec, lim, name in wl:
                    r = self.lp(-(vec @ M), A, b, bounds)
                    if r.status == 0 and -r.fun + vec @ m0 > lim + 1e-12:
                        walls.append((B, bi, -r.fun + vec @ m0 - lim, name))
                blo, bhi = [], []
                for row, c0 in zip(M, m0):
                    r1 = self.lp(row, A, b, bounds); r2 = self.lp(-row, A, b, bounds)
                    blo.append(r1.fun + c0); bhi.append(-r2.fun + c0)
                cand = self.leaves_in(np.array(blo) - EPS, np.array(bhi) + EPS)
                if len(cand) == 1:
                    targets.setdefault(cand[0], bi); continue
                for T_ in cand:
                    if T_ in targets:
                        continue
                    tlo, thi = self.box_bounds(T_)
                    A2 = np.vstack([A, M, -M])
                    b2 = np.concatenate([b, thi + EPS - m0, -(tlo - EPS - m0)])
                    if self.lp(np.zeros(NV), A2, b2, bounds).status == 0:
                        targets[T_] = bi
        return targets, walls

    # bookkeeping -----------------------------------------------------------
    def gc(self):
        rb = self.leaf_of_point(self.root)
        self.parent = {rb: None}
        active, frontier = {rb}, [rb]
        while frontier:
            nxt = []
            for B in frontier:
                for T_, bi in self.done.get(B, {}).items():
                    if T_ in self.refined or T_ in active:
                        continue
                    active.add(T_); self.parent[T_] = (B, bi); nxt.append(T_)
            frontier = nxt
        self.active = active
        self.queue = [B for B in active if B not in self.done]
        self.wallhit = []
        for B in active:
            if self.wallsof.get(B):
                self.wallhit = self.wallsof[B][:1]
                break

    def split(self, B):
        if B[0] >= self.maxlev or B in self.refined:
            return False
        self.refined.add(B); self.refinements += 1
        self.wallsof.pop(B, None)
        for T_ in self.done.pop(B, {}) or {}:
            self.preds.get(T_, set()).discard(B)
        for P0 in list(self.preds.pop(B, set())):
            self.wallsof.pop(P0, None)
            for T_ in self.done.pop(P0, {}) or {}:
                self.preds.get(T_, set()).discard(P0)
        return True

    def branch_chain(self, B, final_branch):
        seq, cur = [final_branch], B
        while self.parent.get(cur) is not None:
            P0, bi = self.parent[cur]
            seq.append(bi); cur = P0
        boxes, cur = [B], B
        while self.parent.get(cur) is not None:
            cur = self.parent[cur][0]; boxes.append(cur)
        return seq[::-1], boxes[::-1]

    def concrete(self, seq, wall):
        """Maximise the wall value along the exact branch sequence over adversary choices."""
        rho, n = self.rho, len(seq)

        def run(lam):
            x = self.root.copy(); worst = -np.inf
            for t, br in enumerate(seq):
                T = 1 - x[0] - x[1]
                lo, hi = interval_r(T, x[2], rho)
                r = lo + float(np.clip(lam[t], 0, 1)) * (hi - lo)
                x = exact_child(x, (0, 2, 4)[br], r, rho)
            return x

        def excess(x):
            vals = [x[0] - 0.25, 14 * x[0] + 4 * x[1] - 5]
            return max(vals)

        best = -np.inf; rng = np.random.default_rng(0)
        for trial in range(12):
            lam0 = rng.random(n) if trial else np.ones(n)
            res = minimize(lambda l: -excess(run(l)), lam0, method="L-BFGS-B",
                           bounds=[(0, 1)] * n, options={"maxiter": 200})
            best = max(best, -res.fun)
        return best

    def handle_wall(self, wall):
        B, bi, val, name = wall
        seq, boxes = self.branch_chain(B, bi)
        if name is None:
            v = self.concrete(seq, name)
            if v > 1e-9:
                self.real_cex = (seq, v)
                return "real"
        else:
            # pseudo wall: check if the exact dynamics really crosses it
            vec, lim = self.pseudo[name]
            rho, n = self.rho, len(seq)

            def run(lam):
                x = self.root.copy()
                for t, br in enumerate(seq):
                    lo, hi = interval_r(1 - x[0] - x[1], x[2], rho)
                    x = exact_child(x, (0, 2, 4)[br], lo + float(np.clip(lam[t], 0, 1)) * (hi - lo), rho)
                return x
            best = -np.inf; rng = np.random.default_rng(1)
            for trial in range(8):
                l0 = rng.random(n) if trial else np.ones(n)
                res = minimize(lambda l: -(vec @ run(l)), l0, method="L-BFGS-B", bounds=[(0, 1)] * n)
                best = max(best, -res.fun)
            if best > lim + 1e-9:
                self.pseudo[name] = (vec, lim + 0.05)
                for Bx in list(self.wallsof):
                    self.wallsof[Bx] = [w for w in self.wallsof[Bx] if w[3] != name]
                return "pseudo-relaxed"
        ok = False
        for Bx in boxes:
            ok = self.split(Bx) or ok
        return "refined" if ok else "maxlev"

    # driver --------------------------------------------------------------------
    def save(self):
        pickle.dump(self, open(STATE + ".tmp", "wb")); os.replace(STATE + ".tmp", STATE)

    def run(self, budget):
        t0 = last = time.time()
        while time.time() - t0 < budget:
            if time.time() - last > 45:
                self.save(); last = time.time()
            if self.wallhit:
                res = self.handle_wall(self.wallhit[0])
                if res in ("real", "maxlev"):
                    return self.status(res)
                self.gc()
                continue
            if not self.queue:
                break
            B = self.queue.pop(0)
            if B in self.done or B in self.refined:
                continue
            tg, walls = self.process(B)
            self.done[B] = tg; self.wallsof[B] = walls
            reconnect = False
            for T_, bi in tg.items():
                self.preds.setdefault(T_, set()).add(B)
                if T_ not in self.active:
                    self.active.add(T_); self.parent[T_] = (B, bi)
                    if T_ in self.done:
                        reconnect = True
                    else:
                        self.queue.append(T_)
            if reconnect:
                self.gc()
            if walls and not self.wallhit:
                self.wallhit = walls[:1]
        return self.status()

    def status(self, stop=None):
        boxes = [B for B in self.active if B in self.done]
        s = {"active explored leaves": len(boxes), "cached leaves": len(self.done),
             "queue": len(self.queue), "refinements": self.refinements, "lps": self.lps,
             "pending wall": bool(self.wallhit),
             "pseudo walls": {k: round(v[1], 3) for k, v in self.pseudo.items()}}
        if boxes:
            his = np.array([self.box_bounds(B)[1] for B in boxes])
            s["max p"] = round(float(his[:, 0].max()), 4)
            s["max q"] = round(float(his[:, 1].max()), 4)
            s["max 14p+4q (corners)"] = round(float((14 * his[:, 0] + 4 * his[:, 1]).max()), 4)
            s["levels used"] = sorted(set(B[0] for B in boxes))
        s["closed"] = (not self.queue) and not self.wallhit and self.real_cex is None
        if stop:
            s["stop"] = stop
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "status", "reset"])
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--hp", type=float, default=0.01)
    ap.add_argument("--hq", type=float, default=0.02)
    ap.add_argument("--hd", type=float, default=0.02)
    ap.add_argument("--maxlev", type=int, default=6)
    ap.add_argument("--budget", type=float, default=200)
    a = ap.parse_args()
    if a.mode == "reset":
        if os.path.exists(STATE):
            os.remove(STATE)
        print("state removed"); return
    ex = pickle.load(open(STATE, "rb")) if os.path.exists(STATE) else \
        Explorer(a.rho, a.hp, a.hq, a.hd, a.maxlev)
    st = ex.run(a.budget) if a.mode == "run" else ex.status()
    print(f"rho={ex.rho} base box={tuple(ex.h)} maxlev={ex.maxlev}")
    for k, v in st.items():
        print(f"  {k}: {v}")
    if ex.real_cex is not None:
        seq, v = ex.real_cex
        print(f"  REAL: exact history from W_(0,1) with branches {''.join(str((0,2,4)[b]) for b in seq)} "
              f"breaks a wall by {v:.4g} (adversary r at closure endpoints allowed)")
    if st.get("closed"):
        print("  CLOSED: union of active leaves is invariant for the realizable proportional packet "
              "(float LP, McCormick relaxation). Pseudo walls are only refinement triggers.")
    ex.save()


if __name__ == "__main__":
    main()
