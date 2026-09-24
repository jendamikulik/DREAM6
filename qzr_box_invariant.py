#!/usr/bin/env python3
"""
QZR packet fan in coordinates (p, q, d), d = R_rho(M) - M(1) = R - q.

Searches for a non-convex invariant set S = finite union of axis-parallel boxes
by forward abstract reachability from the box of the critical root W_{0,1}.

Controller (minimal packet, optimal against both walls):
    h   = (4p+q-1)_+
    d_G = (d - rho(1-4p-q)_+)_+
Cells:
    A: 4p+q<=1, d<=rho(1-4p-q):  h=0,        d_G=0
    B: 4p+q<=1, d>=rho(1-4p-q):  h=0,        d_G=d-rho(1-4p-q)
    C: 4p+q>=1:                  h=4p+q-1,   d_G=d

Adversary: every realizable second layer (r,s):
    r in I(T,d), s in I(T_G,d_G), r-s in I(T-T_G,d-d_G), s<=r,
    I(T,d) = [max(0,(d-rho^2 T)/(rho(1-rho))), min(T,d/rho)].
All these constraints are linear jointly in v=(p,q,d,r,s), so for a box X and a
cell, the image of branch x is the linear image of a polytope in v.

A target box B is marked as hit by branch x of box X iff the LP
    { v : x in X, cell, validity, polygon, child_x(v) in B (inflated by EPS) }
is feasible.  The walls p<=1/4 and 14p+4q<=5 are checked by LP maxima.

If the exploration closes (no new boxes, no wall hit), the union of explored
boxes is a robust invariant set containing the root (up to floating-point LP;
the file records every hit so an exact rational re-check can be added).
If a wall is hit, the offending box chain is reported; that is an abstraction
failure, not a proof that no S exists (boxes can be refined).

Usage:
    python3 qzr_box_invariant.py run  [--rho 0.5] [--hp 0.01 --hq 0.02 --hd 0.02]
                                      [--budget 250]   (seconds, resumable)
    python3 qzr_box_invariant.py status
State is kept in qzr_box_state.pkl, so repeated `run` calls continue the work.
"""
import sys, time, pickle, argparse, os
import numpy as np
from scipy.optimize import linprog

STATE = "qzr_box_state.pkl"
EPS = 1e-9          # inflation of target boxes (soundness against LP tolerance)


# --------------------------------------------------------------------------
# affine expressions over v = (p, q, d, r, s): (vector[5], const)
# --------------------------------------------------------------------------
def E(vec, c=0.0):
    return (np.array(vec, float), float(c))


def comb(*terms):
    v, c = np.zeros(5), 0.0
    for k, (a, b) in terms:
        v = v + k * a
        c += k * b
    return (v, c)


P, Q, D, Rv, Sv = (E(np.eye(5)[i]) for i in range(5))
ONE, ZERO = E(np.zeros(5), 1.0), E(np.zeros(5), 0.0)


def build_cell(cell, rho):
    """Return (constraints expr<=0, children [(p,q,d) exprs] for branches 0,2,4)."""
    T = comb((1, ONE), (-1, P), (-1, Q))
    m = comb((1, ONE), (-4, P), (-1, Q))           # 1-4p-q
    if cell == "A":
        h, dG = ZERO, ZERO
    elif cell == "B":
        h, dG = ZERO, comb((1, D), (-rho, m))
    else:
        h, dG = comb((-1, m)), D
    TG = comb((3, P), (-1, h))
    k = 1.0 / (rho * (1 - rho))
    cons = []

    def le(a, b):
        cons.append(comb((1, a), (-1, b)))

    def ge(a, b):
        cons.append(comb((1, b), (-1, a)))

    # validity of the parent state
    ge(P, ZERO); ge(Q, ZERO); ge(T, ZERO); ge(D, ZERO); le(D, comb((rho, T)))
    le(P, E(np.zeros(5), 0.25))
    # cell
    if cell in "AB":
        ge(m, ZERO)
        if cell == "A":
            le(D, comb((rho, m)))
        else:
            ge(D, comb((rho, m)))
    else:
        le(m, ZERO)
    # packet feasibility
    ge(h, ZERO); le(h, Q); le(h, comb((3, P)))
    ge(dG, ZERO); le(dG, D); le(dG, comb((rho, TG))); le(TG, T)
    # realizable polygon
    t = comb((1, Rv), (-1, Sv))
    TR, dR = comb((1, T), (-1, TG)), comb((1, D), (-1, dG))
    for var, TT, dd in ((Rv, T, D), (Sv, TG, dG), (t, TR, dR)):
        ge(var, ZERO)
        ge(var, comb((k, dd), (-k * rho ** 2, TT)))
        le(var, TT)
        le(var, comb((1 / rho, dd)))
    # children
    Rr = comb((1, Q), (1, D))
    g = comb((1, h), (1, dG))
    p0 = comb((7 / 8, P)); q0 = comb((1 / 2, P), (7 / 8, Q))
    R0 = comb((1 / 2, P), ((7 + rho) / 8, Rr), (-rho / 8, g))
    p2 = comb((10 / 16, P), (3 / 16, Q), (1 / 16, h))
    q2 = comb((4 / 16, P), (10 / 16, Q), (3 / 16, Rv), (1 / 16, Sv))
    R2 = comb(((4 + 2 * rho) / 16, P), (3 / (16 * rho), D), (1 / (16 * rho), dG),
              ((10 + 3 * rho) / 16, Rr), (-rho / 16, g))
    p4 = comb((5 / 8, P), (1 / 8, Q), (1 / 8, h))
    q4 = comb((7 / 8, Q), (1 / 8, Rv), (1 / 8, Sv))
    R4 = comb((1 / (8 * rho), D), (1 / (8 * rho), dG), (7 / 8, Rr))
    kids = [(p0, q0, comb((1, R0), (-1, q0))),
            (p2, q2, comb((1, R2), (-1, q2))),
            (p4, q4, comb((1, R4), (-1, q4)))]
    A = np.array([c[0] for c in cons])
    b = np.array([-c[1] for c in cons])
    return A, b, kids


# --------------------------------------------------------------------------
class Explorer:
    """Adaptive box partition (octree over a base grid) + CEGAR refinement."""
    def __init__(self, rho, hp, hq, hd, maxlev=6):
        self.rho, self.h = rho, np.array([hp, hq, hd])
        self.maxlev = maxlev
        self.cells = {c: build_cell(c, rho) for c in "ABC"}
        self.refined = set()     # boxes (lev,i,j,k) that have been split
        self.done = {}           # leaf -> {target leaf: (cell, branch)}
        self.preds = {}          # leaf -> set of leaves whose image hits it
        self.parent = {}         # leaf -> (pred leaf, (cell, branch)) first discovery
        self.queue = []
        self.wallhit = []
        self.real_cex = None
        self.refinements = 0
        self.lps = 0
        R0 = sum(rho ** (j - 1) * 2 / ((j + 1) * (j + 2)) for j in range(1, 20000))
        self.root = np.array([0.0, 1 / 3, R0 - 1 / 3])
        self.push_root()
        # pseudo walls: refinement triggers only (relaxed when a real path crosses them)
        self.pseudo = {"q": (np.array([0, 1, 0.]), 0.40), "4p+q": (np.array([4, 1, 0.]), 0.85)}

    # ---- geometry
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

    def push_root(self):
        rb = self.leaf_of_point(self.root)
        if rb not in self.done and rb not in self.queue:
            self.queue.insert(0, rb)
            self.parent[rb] = None

    def lp(self, c, A, b, bounds, **kw):
        self.lps += 1
        return linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", **kw)

    # ---- one abstract step (vertex enumeration, LP fallback)
    SUBST = {  # v = L z + l0, z = (p,q,d,r); s eliminated per cell
        "A": (np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]],float), np.zeros(5)),
        "B": (np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[4,1,0,1]],float), np.array([0,0,0,0,-1.])),
        "C": (np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]],float), np.zeros(5)),
    }

    def cell_vertices(self, A, b, bounds, cname=None):
        L, l0 = self.SUBST[cname]
        Az = A @ L; bz = b - A @ l0
        lo = np.array([bounds[i][0] for i in range(3)] + [-10.0])
        hi = np.array([bounds[i][1] for i in range(3)] + [10.0])
        Ab = np.vstack([Az, np.eye(4), -np.eye(4)]); bb = np.concatenate([bz, hi, -lo])
        nrm = np.linalg.norm(Ab, axis=1)
        keep = nrm > 1e-14
        if np.any(~keep & (bb < -1e-12)):
            return "empty"
        Ab, bb, nrm = Ab[keep], bb[keep], nrm[keep]
        r = linprog(np.r_[np.zeros(4), -1], A_ub=np.hstack([Ab, nrm[:, None]]), b_ub=bb,
                    bounds=[(None, None)] * 4 + [(0, 1)], method="highs")
        self.lps += 1
        if r.status != 0:
            return "empty"
        if r.x[4] < 1e-9:
            return None               # lower-dimensional: use LP path
        try:
            from scipy.spatial import HalfspaceIntersection
            hs = HalfspaceIntersection(np.hstack([Ab, -bb[:, None]]), r.x[:4])
            return hs.intersections @ L.T + l0
        except Exception:
            return None

    def process(self, B):
        lo, hi = self.box_bounds(B)
        bounds = [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]), (None, None), (None, None)]
        targets, walls = {}, []
        for cname, (A, b, kids) in self.cells.items():
            Vx = self.cell_vertices(A, b, bounds, cname)
            if isinstance(Vx, str):
                continue
            if Vx is None:
                t2, w2 = self.process_lp_cell(B, cname, A, b, kids, bounds)
                for k_, v_ in t2.items(): targets.setdefault(k_, v_)
                walls += w2
                continue
            for bi, (kp, kq, kd) in enumerate(kids):
                M = np.vstack([kp[0], kq[0], kd[0]]); m0 = np.array([kp[1], kq[1], kd[1]])
                img = Vx @ M.T + m0
                wl = [(np.array([1., 0, 0]), 0.25, None), (np.array([14., 4, 0]), 5.0, None)]
                for name, (vec, lim) in getattr(self, "pseudo", {}).items():
                    wl.append((vec, lim, name))
                for vec, lim, name in wl:
                    val = (img @ vec).max()
                    if val > lim + 1e-12:
                        walls.append((B, cname, bi, val - lim, vec @ M, lim - vec @ m0, name))
                cand = self.leaves_in(img.min(0) - EPS, img.max(0) + EPS)
                if len(cand) == 1:
                    targets.setdefault(cand[0], (cname, bi)); continue
                for T_ in cand:
                    if T_ in targets:
                        continue
                    tlo, thi = self.box_bounds(T_)
                    inside = np.all((img >= tlo - EPS) & (img <= thi + EPS), axis=1)
                    if inside.any():
                        targets[T_] = (cname, bi); continue
                    tlo, thi = self.box_bounds(T_)
                    A2 = np.vstack([A, M, -M])
                    b2 = np.concatenate([b, thi + EPS - m0, -(tlo - EPS - m0)])
                    if self.lp(np.zeros(5), A2, b2, bounds).status == 0:
                        targets[T_] = (cname, bi)
        return targets, walls

    def process_lp_cell(self, B, cname, A, b, kids, bounds):
        targets, walls = {}, []
        if True:
            if self.lp(np.zeros(5), A, b, bounds).status != 0:
                return targets, walls
            for bi, (kp, kq, kd) in enumerate(kids):
                wl = [(1.0 * kp[0], 0.25 - kp[1], None),
                      (14 * kp[0] + 4 * kq[0], 5 - 14 * kp[1] - 4 * kq[1], None)]
                for name, (vec, lim) in getattr(self, "pseudo", {}).items():
                    wv = vec[0] * kp[0] + vec[1] * kq[0] + vec[2] * kd[0]
                    wl.append((wv, lim - (vec[0] * kp[1] + vec[1] * kq[1] + vec[2] * kd[1]), name))
                for wvec, wmax, name in wl:
                    r = self.lp(-wvec, A, b, bounds)
                    if r.status == 0 and -r.fun > wmax + 1e-12:
                        walls.append((B, cname, bi, -r.fun - wmax, wvec, wmax, name))
                blo, bhi = [], []
                for ex in (kp, kq, kd):
                    r1 = self.lp(ex[0], A, b, bounds); r2 = self.lp(-ex[0], A, b, bounds)
                    blo.append(r1.fun + ex[1]); bhi.append(-r2.fun + ex[1])
                cand = self.leaves_in(np.array(blo) - EPS, np.array(bhi) + EPS)
                if len(cand) == 1:
                    targets.setdefault(cand[0], (cname, bi)); continue
                M = np.vstack([kp[0], kq[0], kd[0]]); m0 = np.array([kp[1], kq[1], kd[1]])
                for T_ in cand:
                    if T_ in targets:
                        continue
                    tlo, thi = self.box_bounds(T_)
                    A2 = np.vstack([A, M, -M])
                    b2 = np.concatenate([b, thi + EPS - m0, -(tlo - EPS - m0)])
                    if self.lp(np.zeros(5), A2, b2, bounds).status == 0:
                        targets[T_] = (cname, bi)
        return targets, walls

    # ---- exact path LP along a chain of boxes
    def path_lp(self, chain, upto, final=None):
        """chain: list of (box_t, (cell_t, branch_t)) for t=0..n-1, plus target box chain_boxes.
        Check steps 0..upto-1 with x_t in box_t (t>=1), x_0 = root.  If final=(cell,br,wvec,wmax)
        maximise the wall at the last state; else feasibility only."""
        boxes = [c[0] for c in chain]
        n = upto
        nv = 5 * (n + 1)
        Aub, bub, Aeq, beq = [], [], [], []
        for t in range(n):
            cell, br = chain[t][1]
            A, b, kids = self.cells[cell]
            blk = np.zeros((A.shape[0], nv)); blk[:, 5*t:5*t+5] = A
            Aub.append(blk); bub.append(b)
            for comp, ex in enumerate(kids[br]):
                row = np.zeros(nv); row[5*t:5*t+5] = ex[0]; row[5*(t+1)+comp] = -1
                Aeq.append(row); beq.append(-ex[1])
        for comp in range(3):
            row = np.zeros(nv); row[comp] = 1; Aeq.append(row); beq.append(self.root[comp])
        bounds = [(None, None)] * nv
        for t in range(1, n + 1):
            lo, hi = self.box_bounds(boxes[t])
            for comp in range(3):
                bounds[5*t+comp] = (lo[comp] - EPS, hi[comp] + EPS)
        c = np.zeros(nv)
        if final is not None:
            cell, br, wvec, wmax = final
            A, b, _ = self.cells[cell]
            blk = np.zeros((A.shape[0], nv)); blk[:, 5*n:5*n+5] = A
            Aub.append(blk); bub.append(b)
            c[5*n:5*n+5] = -wvec
        r = linprog(c, A_ub=np.vstack(Aub) if Aub else None, b_ub=np.concatenate(bub) if bub else None,
                    A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=bounds, method="highs")
        self.lps += 1
        if r.status != 0:
            return None
        return (-r.fun - final[3]) if final is not None else 0.0

    def chain_to(self, B):
        out = []
        cur = B
        while self.parent.get(cur) is not None:
            P_, how = self.parent[cur]
            out.append((P_, how))
            cur = P_
        out = out[::-1]
        return out, cur          # cur = first box (should contain the root)

    # ---- refinement
    def split(self, B):
        if B[0] >= self.maxlev or B in self.refined:
            return False
        self.refined.add(B)
        self.refinements += 1
        getattr(self, "wallsof", {}).pop(B, None)
        tg = self.done.pop(B, None)
        if tg:
            for T_ in tg:
                self.preds.get(T_, set()).discard(B)
        for P_ in list(self.preds.pop(B, set())):
            getattr(self, "wallsof", {}).pop(P_, None)
            old = self.done.pop(P_, None)
            if old:
                for T_ in old:
                    self.preds.get(T_, set()).discard(P_)
            if P_ not in self.queue:
                self.queue.insert(0, P_)
        if B in self.queue:
            self.queue.remove(B)
        self.parent.pop(B, None)
        return True

    def handle_wall(self, wall):
        B, cell, bi, val, wvec, wmax, pname = wall
        ch, first = self.chain_to(B)
        full = [(c[0], c[1]) for c in ch] + [(B, (cell, bi))]
        boxes = [first] + [c[0] for c in ch[1:]] + [B] if ch else [B]
        chain = []
        seq_boxes = [first] + [c[0] for c in ch][1:] + [B] if ch else [B]
        # rebuild aligned chain: box_t with how_t, t=0..n-1, and final box B
        chain = [(ch[t][0], ch[t][1]) for t in range(len(ch))] + [(B, (cell, bi))]
        n = len(chain) - 1        # transitions before the final state
        # first prefix whose exact path leaves the boxes
        for k in range(1, n + 1):
            if self.path_lp(chain, k) is None:
                ok = False
                for t in range(k - 1, len(chain)):
                    ok = self.split(chain[t][0]) or ok
                if not ok:
                    return "maxlev"
                self.push_root()
                return "spurious-prefix"
        v = self.path_lp(chain, n, final=(cell, bi, wvec, wmax))
        if v is not None and v > 1e-9:
            if pname is not None:        # pseudo wall is only a refinement trigger: relax it
                vec, lim = self.pseudo[pname]
                self.pseudo[pname] = (vec, lim + 0.05)
                self.relaxed = getattr(self, "relaxed", 0) + 1
                return "pseudo-relaxed"
            self.real_cex = (chain, v)
            return "real"
        if not self.split(B):
            return "maxlev"
        self.push_root()
        return "spurious-final"

    def gc(self):
        """Recompute the abstractly reachable (active) set from the root leaf by BFS over cached
        images; cached images of inactive leaves are kept (they may become active again)."""
        rb = self.leaf_of_point(self.root)
        self.parent = {rb: None}
        order, frontier, active = [rb], [rb], {rb}
        while frontier:
            nxt = []
            for B in frontier:
                for T_, how in self.done.get(B, {}).items():
                    if T_ in self.refined or T_ in active:
                        continue
                    active.add(T_); self.parent[T_] = (B, how); nxt.append(T_)
            frontier = nxt
        self.active = active
        self.queue = [B for B in active if B not in self.done]
        self.wallhit = []
        for B in active:
            w = getattr(self, "wallsof", {}).get(B)
            if w and B in self.done:
                self.wallhit = w[:1]
                break

    def save(self):
        cells = self.cells; self.cells = None
        pickle.dump(self, open(STATE + ".tmp", "wb")); os.replace(STATE + ".tmp", STATE)
        self.cells = cells

    def run(self, budget):
        t0 = time.time(); last = t0
        while time.time() - t0 < budget:
            if time.time() - last > 45:
                self.save(); last = time.time()
            if self.wallhit:
                res = self.handle_wall(self.wallhit.pop(0))
                self.wallhit = []
                if res in ("real", "maxlev"):
                    return self.status(extra=res)
                self.gc()
                continue
            if not self.queue:
                break
            B = self.queue.pop(0)
            if B in self.done or B in self.refined:
                continue
            tg, walls = self.process(B)
            self.done[B] = tg
            if not hasattr(self, "wallsof"): self.wallsof = {}
            self.wallsof[B] = walls
            if not hasattr(self, "active"): self.active = set(self.parent)
            self.active.add(B)
            for T_, how in tg.items():
                self.preds.setdefault(T_, set()).add(B)
                if T_ not in self.active:
                    self.active.add(T_); self.parent[T_] = (B, how)
                    if T_ in self.done:
                        self.gc(); break      # reconnects cached subgraph, re-collects walls
                    self.queue.append(T_)
            if walls and not self.wallhit:
                self.wallhit = walls[:1]
        return self.status()

    def status(self, extra=None):
        act = getattr(self, "active", set(self.done))
        boxes = [B for B in act if B in self.done]
        s = {"active explored leaves": len(boxes), "cached leaves": len(self.done), "queue": len(self.queue), "refinements": self.refinements,
             "lps": self.lps, "pending wall": bool(self.wallhit)}
        if boxes:
            his = np.array([self.box_bounds(B)[1] for B in boxes])
            s["max p"] = round(float(his[:, 0].max()), 4); s["max q"] = round(float(his[:, 1].max()), 4)
            s["max 14p+4q (corners)"] = round(float((14 * his[:, 0] + 4 * his[:, 1]).max()), 4)
            s["levels used"] = sorted(set(B[0] for B in boxes))
        s["pseudo walls"] = {k: round(v[1], 3) for k, v in getattr(self, "pseudo", {}).items()}
        s["closed"] = (not self.queue) and not self.wallhit and self.real_cex is None
        if extra:
            s["stop"] = extra
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "status", "reset"])
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--hp", type=float, default=0.01)
    ap.add_argument("--hq", type=float, default=0.02)
    ap.add_argument("--hd", type=float, default=0.02)
    ap.add_argument("--maxlev", type=int, default=6)
    ap.add_argument("--budget", type=float, default=250)
    a = ap.parse_args()
    if a.mode == "reset":
        if os.path.exists(STATE):
            os.remove(STATE)
        print("state removed"); return
    if os.path.exists(STATE):
        ex = pickle.load(open(STATE, "rb"))
        ex.cells = {c: build_cell(c, ex.rho) for c in "ABC"}
    else:
        ex = Explorer(a.rho, a.hp, a.hq, a.hd, a.maxlev)
    st = ex.run(a.budget) if a.mode == "run" else ex.status()
    print(f"rho={ex.rho} base box={tuple(ex.h)} maxlev={ex.maxlev}")
    for k, v in st.items():
        print(f"  {k}: {v}")
    if ex.real_cex is not None:
        chain, v = ex.real_cex
        print(f"  REAL realizable path from W_(0,1) breaks a wall by {v:.4g}; branches:",
              "".join(str((0, 2, 4)[h_[1]]) for _, h_ in chain), " cells:", "".join(h_[0] for _, h_ in chain))
    if st.get("closed"):
        print("  NOTE: pseudo walls are only triggers; a closed run is invariant regardless of them.")
        print("  CLOSED: union of explored leaves is a robust invariant set containing the root (float LP).")
    ex.save()


if __name__ == "__main__":
    main()
