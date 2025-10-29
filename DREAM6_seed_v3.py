
#!/usr/bin/env python3
# DREAM6_SEED v3  — A1/A2/A3 focused (offset geometry + Hadamard masks + bounded cross-terms)
# - De-aliased offsets near T/2 with stride s coprime with T (A1).
# - Walsh–Hadamard masks truncated to m slots, row/column indices coprime (A2).
# - Optional coupling over a d-regular circulant graph with kappa_S2-style attenuation (A3).
# - Power-iteration for principal eigenvector (implicit Gram).
# - Initial assignment + UNSAT report; optional tiny greedy micro-polish.
#
# God-mode defaults (from your message):
#   cR=10, L=3, sigma_up=0.045, rho=0.734296875, zeta0=0.4,
#   neighbor_atten=0.9495, seed=42, couple=1 (enable coupling), d=6.
#
# Usage (exact params you gave):
#   python3 DREAM6_seed_v3.py --cnf random_3sat_10000.cnf --godmode --polish 20000
#
# Or explicitly:
#   python3 DREAM6_seed_v3.py --cnf random_3sat_10000.cnf --mode unsat_hadamard \
#     --rho 0.734296875 --zeta0 0.4 --cR 10 --L 3 --sigma_up 0.045 \
#     --neighbor_atten 0.9495 --d 6 --seed 42 --couple 1 --power_iters 60 --polish 20000
#
# python3 DREAM6_seed_v3.py --cnf uf250-0100.cnf --godmode --sigma_up 0.024 --power_iters 120 --polish 500000 --cR 12
#   --L 4 --neighbor_atten 0.9495 --bias_weight 0.12 --score_norm_alpha 0.6 --rho 0.734296875 --zeta0 0.4

import argparse, math, os, random, sys, time
import numpy as np

# ---------- helpers ----------
def sigma_proxy(C, cR=10.0, L=3, eta_power=3, C_B=1.0):
    C = max(2, int(C))
    R = max(1, int(math.ceil(cR * math.log(C))))
    T = R * L
    eta = C ** (-eta_power)
    sigma_up = C_B * math.sqrt(max(1e-12, math.log(C / eta)) / max(1, T))
    return sigma_up, R, T

def wiring_neighbors_circulant(C, d=6):
    if d % 2 != 0 or d > C - 1:
        raise ValueError(f"d={d} must be even and < C-1 for circulant wiring.")
    nbrs = []
    for i in range(C):
        nbr_set = set()
        for step in range(1, d//2 + 1):
            nbr_set.add((i - step) % C)
            nbr_set.add((i + step) % C)
        nbrs.append(nbr_set)
    return nbrs

def gcd_coprime_stride_near_half(T):
    # choose s near T/2, coprime with T
    s = max(1, T//2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

# ---------- schedules ----------
def schedule_unsat_hadamard(C, R, rho=0.734296875, zeta0=0.40, L=3, sigma_up=0.045,
                             seed=42, couple=True, neighbor_atten=0.9495, d=6, verbose=False):
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(math.floor(rho * T))
    k = int(math.floor(zeta0 * m))

    # A1: de-aliased offsets with stride near T/2
    s = gcd_coprime_stride_near_half(T)
    offsets = [(j * s) % T for j in range(C)]
    lock_idx = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]
    Phi = np.full((T, C), np.pi, dtype=float)

    # A2: Walsh–Hadamard masks truncated to m, with coprime row/col indexing
    Hlen = 1
    while Hlen < m: Hlen <<= 1
    # Build H by recursion (small sizes only stored implicitly via indexing choice)
    # We'll assemble the signs by indexing into a synthetic Sylvester Hadamard using parity of bit dot.
    def hadamard_sign(row, col):
        # Sylvester H: sign = (-1)^{<rowbits, colbits>}
        return 1.0 if (bin(row & col).count("1") % 2 == 0) else -1.0

    # choose row step and column generator coprime
    row_step = (Hlen // 2) + 1
    if math.gcd(row_step, Hlen) != 1:
        row_step |= 1
        while math.gcd(row_step, Hlen) != 1:
            row_step += 2
    g = (Hlen // 3) | 1
    while math.gcd(g, Hlen) != 1:
        g += 2

    cols = np.mod(g * np.arange(m, dtype=int), Hlen)
    for j in range(C):
        row = (j * row_step) % Hlen
        # negatives to pi, positives to ~0 with tiny Gaussian sigma_up
        neg_idx = []
        for t in range(m):
            sgn = hadamard_sign(row, int(cols[t]))
            if sgn < 0: neg_idx.append(t)
        if len(neg_idx) >= k:
            mask_pi = rng.choice(np.array(neg_idx, dtype=int), size=k, replace=False)
        else:
            pool = np.setdiff1d(np.arange(m, dtype=int), np.array(neg_idx, dtype=int), assume_unique=False)
            extra = rng.choice(pool, size=k-len(neg_idx), replace=False) if k > len(neg_idx) else np.empty(0, dtype=int)
            mask_pi = np.concatenate([np.array(neg_idx, dtype=int), extra])
        mask_0 = np.setdiff1d(np.arange(m, dtype=int), mask_pi, assume_unique=False)
        slots = lock_idx[j]
        Phi[slots[mask_pi], j] = np.pi
        if len(mask_0) > 0:
            Phi[slots[mask_0], j] = rng.normal(loc=0.0, scale=sigma_up, size=len(mask_0))

    # A3: coupling (optional): attenuate overlaps along circulant neighbors
    if couple and (abs(neighbor_atten - 1.0) > 1e-12):
        neighbors = wiring_neighbors_circulant(C, d=d)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        # kappa_S2 proxy from your Lemma2 page
        # kappa = (1 - 2*ζ0)^2 + 2^(-⌊log2 m⌋/2) + 2/m + 1/T  (we'll use a mild safe version)
        kappa = (1.0 - 2.0*zeta0)**2 + (2.0**(-int(math.log2(max(2, m))) / 2.0)) + (2.0/max(1,m)) + (1.0/max(1,T))
        kappa = max(0.0, min(1.0, kappa))
        for j in range(C):
            o_j = offsets[j]
            Lj = lock_sets[j]
            for j_adj in neighbors[j]:
                if j_adj == j: continue
                La = lock_sets[j_adj]
                overlap = Lj.intersection(La)
                if not overlap: continue
                overlap_size = len(overlap)
                overlap_fraction = overlap_size / max(1, m)
                # attenuation formula biased by overlap and kappa proxy
                cross_term_weight = min(1.0, (len(neighbors[j]) * kappa) / max(1.0, C * (1 - 0.5*sigma_up)**2)
                                        * (1.0 + 3.0 * overlap_fraction))
                attenuation = max(0.70, neighbor_atten - 0.05 * overlap_size / (m * (1 + 0.25
                                        * math.sqrt(max(1e-9, math.log(C))) * overlap_fraction)) * (1 - cross_term_weight))
                idx = np.fromiter(overlap, dtype=int)
                Phi[idx, j_adj] *= attenuation
                if verbose and j % (max(1, C//20)) == 0:
                    print(f"[A3] j={j}→{j_adj} | overlap={overlap_size} attn={attenuation:.3f} κ≈{kappa:.4f}")
    return Phi

def schedule_sat_aligned(C, R, L=3):
    return np.zeros((R*L, C), dtype=float)

# ---------- spectral weight via power-iteration ----------
def principal_weight_power(Phi, iters=60):
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.normal(size=C) + 1j*rng.normal(size=C)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        y = Z @ x
        x = (Z.conj().T @ y) / T
        x /= (np.linalg.norm(x) + 1e-12)
    return np.abs(x)

# ---------- DIMACS ----------
def parse_dimacs(path):
    clauses, nvars = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "c%":
                continue
            if s[0] == "p":
                parts = s.split()
                if len(parts)>=4 and parts[1].lower()=="cnf":
                    nvars = int(parts[2])
                continue
            lits = [int(x) for x in s.split() if x!="0"]
            if lits:
                clauses.append(lits)
                for L in lits:
                    nvars = max(nvars, abs(L))
    return nvars, clauses

# ---------- scoring & seed ----------
def build_seed_assignment(clauses, nvars, mode='unsat_hadamard', cR=10.0, L=3,
                          rho=0.734296875, zeta0=0.40, sigma_up=0.045,
                          neighbor_atten=0.9495, seed=42, couple=True, d=6,
                          power_iters=60, score_norm_alpha=0.5, bias_weight=0.10):
    C = len(clauses)
    _, R, T = sigma_proxy(C, cR=cR, L=L)
    if mode == 'sat':
        Phi = schedule_sat_aligned(C, R, L)
    else:
        Phi = schedule_unsat_hadamard(C, R, rho, zeta0, L, sigma_up, seed,
                                      couple=couple, neighbor_atten=neighbor_atten, d=d, verbose=False)
    w_clause = principal_weight_power(Phi, iters=power_iters)
    w_clause = (w_clause / (w_clause.mean() + 1e-12)).clip(0.1, 10.0)

    pos = [[] for _ in range(nvars+1)]
    neg = [[] for _ in range(nvars+1)]
    pol = np.zeros(nvars+1, dtype=int)
    deg = np.zeros(nvars+1, dtype=int)
    for ci, cl in enumerate(clauses):
        for LIT in cl:
            v = abs(LIT)
            deg[v] += 1
            if LIT > 0:
                pos[v].append(ci); pol[v] += 1
            else:
                neg[v].append(ci); pol[v] -= 1

    score = np.zeros(nvars+1, dtype=float)
    for v in range(1, nvars+1):
        if pos[v]: score[v] += float(w_clause[pos[v]].sum())
        if neg[v]: score[v] -= float(w_clause[neg[v]].sum())
        if score_norm_alpha > 0.0:
            score[v] /= (deg[v]**score_norm_alpha + 1e-12)
        if bias_weight != 0.0:
            score[v] += bias_weight * float(pol[v]) / max(1, deg[v])

    rng = np.random.default_rng(seed+1337)
    dither = rng.uniform(-1e-7, 1e-7, size=nvars+1)
    assign = [(score[v] + dither[v]) >= 0.0 for v in range(1, nvars+1)]
    return assign, Phi

# ---------- UNSAT ----------
def count_unsat(clauses, assign_bool_list):
    unsat = 0
    for cl in clauses:
        ok = False
        for L in cl:
            v = abs(L); val = assign_bool_list[v-1]
            if L < 0: val = (not val)
            if val: ok = True; break
        if not ok: unsat += 1
    return unsat

def check_sat(clauses, model):
   assignment = {
       i + 1:
           bit == 1
       for i, bit in enumerate(model)
   }
   for clause in clauses:
       clause_satisfied = False
       try:
           clause_satisfied = any(
               (lit > 0 and assignment[abs(lit)]) or
               (lit < 0 and not assignment[abs(lit)])
               for lit in clause
           )
       except KeyError:
           return False
       if not clause_satisfied:
           return False
   return True

# ---------- micro-polish (optional) ----------

from typing import List, Optional

def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int = 20000,
    seed: int = 49,
    alpha: float = 2.4,     # probSAT make exponent
    beta: float = 0.9,      # probSAT break exponent
    epsilon: float = 1e-3,  # probSAT epsilon
    probsat_quota: int = 2000,  # max kroků pro probSAT fázi v rámci polish
) -> List[int]:
    """
    Micro-polish finisher:
      A) exhaust all zero-break flips (tie by max-make)
      B) min-break + max-make tie-break
      C) short probSAT burst if still UNSAT
    """
    rnd = random.Random(seed)
    nvars = len(assign01)
    C = len(clauses)

    # --- adjacency (1-indexed variables) ---
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for L in cl:
            (pos if L > 0 else neg)[abs(L)].append(ci)

    # --- state (1-indexed assign for speed) ---
    assign = [False] + [bool(b) for b in assign01]
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list: List[int] = []  # compact container of UNSAT clause indices

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

    # init sat_count / unsat
    for ci, cl in enumerate(clauses):
        cnt = 0
        for L in cl:
            v = abs(L)
            val = assign[v]
            if L < 0:
                val = (not val)
            if val:
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            add_unsat(ci)
    # compaction
    unsat_list = [ci for ci in unsat_list if in_unsat[ci]]

    # --- helpers ---
    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:  # True->False
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:          # False->True
            for ci in neg[v]:
                if sat_count[ci] == 1:
                    bc += 1
        return bc

    def makecount(v: int) -> int:
        mk = 0
        if assign[v]:
            for ci in neg[v]:
                if in_unsat[ci]:
                    mk += 1
        else:
            for ci in pos[v]:
                if in_unsat[ci]:
                    mk += 1
        return mk

    def flip_var(v: int):
        """Incremental flip with sat_count/unsat maintenance."""
        old = assign[v]
        assign[v] = not old
        if old:
            for ci in pos[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in neg[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)
        else:
            for ci in neg[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in pos[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)

    # utility to pick a random UNSAT clause quickly
    def pick_unsat_clause() -> Optional[int]:
        if not unsat_list:
            return None
        # fast cleanup-on-read
        i = rnd.randrange(len(unsat_list))
        for _ in range(3):  # up to 3 tries to hit a live one
            ci = unsat_list[i]
            if in_unsat[ci]:
                return ci
            i = rnd.randrange(len(unsat_list))
        # fallback: compact and retry once
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    def cur_unsat() -> int:
        # quick count without full compaction
        return sum(1 for ci in unsat_list if in_unsat[ci])

    # --- keep best-so-far ---
    best_assign = assign[:]
    best_uns = cur_unsat()

    steps = 0
    # -------- Phase A: exhaust freebies --------
    made_progress = True
    while made_progress and steps < flips:
        made_progress = False
        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]
        freebies = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            if bc == 0:
                freebies.append((makecount(v), v))
        if freebies:
            freebies.sort(reverse=True)  # prefer max make
            _, v = freebies[0]
            flip_var(v)
            steps += 1
            made_progress = True
            # update best
            u = cur_unsat()
            if u < best_uns:
                best_uns = u
                best_assign = assign[:]
            if u == 0:
                return [1 if b else 0 for b in assign[1:]]

    # -------- Phase B: min-break, max-make --------
    while steps < flips:
        if best_uns == 0:
            return [1 if b else 0 for b in best_assign[1:]]

        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]

        # preferentially do freebies if present
        v_choice = None
        freebies = []
        cand = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            mk = makecount(v)
            if bc == 0:
                freebies.append((mk, v))
            cand.append((bc, -mk, v))  # -mk to maximize make on tie
        if freebies:
            freebies.sort(reverse=True)
            v_choice = freebies[0][1]
        else:
            # min break, then max make
            v_choice = min(cand)[2]

        flip_var(v_choice)
        steps += 1

        u = cur_unsat()
        if u < best_uns:
            best_uns = u
            best_assign = assign[:]
        if u == 0:
            return [1 if b else 0 for b in assign[1:]]

        # malý „kick“: když se nic nezlepšilo 1k kroků, zkus probSAT burst
        if steps % 1000 == 0 and u >= best_uns:
            # -------- Phase C: short probSAT burst --------
            for _ in range(min(probsat_quota, flips - steps)):
                ci2 = pick_unsat_clause()
                if ci2 is None:
                    return [1 if b else 0 for b in assign[1:]]
                clause2 = clauses[ci2]
                # scores ~ (make+eps)^alpha / (break+eps)^beta
                scores = []
                tot = 0.0
                last_v = None
                for L in clause2:
                    v = abs(L)
                    mk = makecount(v)
                    bc = breakcount(v)
                    s = ((mk + epsilon) ** alpha) / ((bc + epsilon) ** beta)
                    scores.append((v, s))
                    tot += s
                    last_v = v
                r = rnd.random() * tot
                acc = 0.0
                pick = last_v
                for v, s in scores:
                    acc += s
                    if acc >= r:
                        pick = v
                        break
                flip_var(pick)
                steps += 1
                u2 = cur_unsat()
                if u2 < best_uns:
                    best_uns = u2
                    best_assign = assign[:]
                if u2 == 0 or steps >= flips:
                    return [1 if b else 0 for b in assign[1:]]
            # po burstu pokračujeme B-fází

    # vyčerpán budget – vrať nejlepší nalezené
    return [1 if b else 0 for b in (best_assign[1:] if best_uns < cur_unsat() else assign[1:])]


# --------------------- params -----------------------------------------

import math

def theory_params(C, want_sigma=None, cR=12, L=4, eta_power=3, zeta0=0.40,
                  rho_lock=0.734296875, neighbor_atten=0.9495,
                  tau_hint=0.40, mu_hint=0.002):
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (buď z want_sigma, nebo z C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T**0.5) / math.sqrt((1+eta_power)*math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1+eta_power)*math.log(C)) / (T**0.5)

    # 3) gamma0 kontrola
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    if gamma0 <= 0:
        # opravíme rho_lock minimálně tak, aby gamma0 > 0 s malou rezervou
        rho_lock = (0.5*sigma_up + 0.01)/zeta0
        gamma0 = rho_lock * zeta0 - 0.5*sigma_up

    # 4) bias a alpha z (tau, mu) – pokud máme hinty
    def clip(x,a,b): return max(a, min(b,x))
    if tau_hint is None: tau_hint = 0.40
    if mu_hint  is None: mu_hint  = 0.002

    bias_weight = clip(0.08 + 0.30*max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5*max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R, "T": T,
        "sigma_up": sigma_up, "C_B": CB,
        "rho": rho_lock, "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0
    }

# ------------------- finnisher -----------------------------

def spectral_pretest(nvars, clauses, samples=10000):
    C = len(clauses)
    if C == 0 or nvars == 0:
        return 0.0, 0.5, 0.5
    deg = [0]*(nvars+1)
    pol = [0]*(nvars+1)
    for cl in clauses:
        seen = set()
        for L in cl:
            v = abs(L)
            if v in seen:  # guard duplicate
                continue
            seen.add(v)
            deg[v] += 1
            pol[v] += 1 if L > 0 else -1
    tau0 = sum(abs(pol[v]) for v in range(1, nvars+1)) / max(1, sum(deg))
    x = np.random.rand(nvars) - 0.5
    x /= np.linalg.norm(x) + 1e-12
    sampler = []
    for cl in clauses:
        vs = [abs(L) for L in cl]
        k = len(vs)
        for i in range(k):
            for j in range(i+1, k):
                sampler.append((vs[i]-1, vs[j]-1))
    if len(sampler) > samples:
        random.shuffle(sampler)
        sampler = sampler[:samples]
    def Ax(xvec):
        y = np.zeros_like(xvec)
        for i, j in sampler:
            y[i] += xvec[j]
            y[j] += xvec[i]
        return y
    for _ in range(3):
        y = Ax(x)
        x = y / (np.linalg.norm(y) + 1e-12)
    num = float(np.dot(x, Ax(x)))
    den = float(np.dot(x, x)) + 1e-12
    lam = num/den
    mu = lam / max(1, C)
    base = 0.35
    p0 = base + 0.25*(mu - 0.002) - 0.20*(tau0 - 0.4)
    p0 = max(0.12, min(0.58, p0))
    return mu, tau0, p0
"""
class WalkSATFinisher:
    def __init__(self, nvars, clauses, start_assign, p=0.40, seed=42):
        self.n = nvars
        self.C = len(clauses)
        self.clauses = clauses
        self.p = p
        random.seed(seed)
        self.pos = [[] for _ in range(nvars+1)]
        self.neg = [[] for _ in range(nvars+1)]
        for ci, cl in enumerate(clauses):
            for L in cl:
                (self.pos if L>0 else self.neg)[abs(L)].append(ci)
        self.assign = list(start_assign)
        self.sat_count = [0]*self.C
        self.unsat = set()
        for ci, cl in enumerate(clauses):
            cnt = 0
            for L in cl:
                v = abs(L); val = self.assign[v-1]
                if L < 0: val = (not val)
                if val: cnt += 1
            self.sat_count[ci] = cnt
            if cnt == 0: self.unsat.add(ci)

    def _breakcount(self, v):
        bc = 0
        if self.assign[v-1]:
            for ci in self.pos[v]:
                if self.sat_count[ci] == 1: bc += 1
        else:
            for ci in self.neg[v]:
                if self.sat_count[ci] == 1: bc += 1
        return bc

    def _flip(self, v):
        old = self.assign[v-1]; new = not old
        self.assign[v-1] = new
        if old:
            for ci in self.pos[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.neg[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)
        else:
            for ci in self.neg[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.pos[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)

    def solve(self, max_flips=20_000_000, report_every=100000, novelty=0.30,
              restart_base=500_000, restart_mult=1.30, seed=42):
        best_unsat = len(self.unsat); best_state = self.assign[:]
        flips_since_best = 0
        next_restart = restart_base
        t0 = time.time()
        for flips in range(1, max_flips+1):
            if not self.unsat:
                return True, flips, time.time()-t0, self.assign[:]
            ci = random.choice(tuple(self.unsat))
            cl = self.clauses[ci]
            cand = [abs(L) for L in cl]
            if random.random() < self.p:
                v = random.choice(cand)
            else:
                best_bc, pool = 1e9, []
                for v0 in cand:
                    bc = self._breakcount(v0)
                    if bc < best_bc:
                        best_bc, pool = bc, [v0]
                    elif bc == best_bc:
                        pool.append(v0)
                v = random.choice(pool)
            self._flip(v)
            cur_unsat = len(self.unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_state = self.assign[:]
                flips_since_best = 0
            else:
                flips_since_best += 1
            if flips % report_every == 0:
                print(f"[finisher] flips={flips:,} unsat={cur_unsat:,} p≈{self.p:.6f}")
            if flips_since_best >= next_restart:
                # restart around the best state (keep 50%)
                rng = np.random.default_rng(seed)
                mask = rng.random(self.n) < 0.5
                self.assign = [ (best_state[i] if mask[i] else random.choice((False, True)))
                                for i in range(self.n) ]
                self.unsat.clear()
                self.sat_count = [0]*self.C
                for ci2, cl2 in enumerate(self.clauses):
                    cnt = 0
                    for L in cl2:
                        v2 = abs(L); val = self.assign[v2-1]
                        if L < 0: val = (not val)
                        if val: cnt += 1
                    self.sat_count[ci2] = cnt
                    if cnt == 0: self.unsat.add(ci2)
                flips_since_best = 0
                next_restart = int(next_restart * restart_mult)
        self.assign = best_state[:]
        return False, max_flips, time.time()-t0, self.assign[:]
"""
def check_sat(clauses, model):
   assignment = {i + 1: (bit == 1) for i, bit in enumerate(model)}
   for clause in clauses:
       clause_satisfied = False
       try:
           clause_satisfied = any(
               (lit > 0 and assignment[abs(lit)]) or
               (lit < 0 and not assignment[abs(lit)])
               for lit in clause
           )
       except KeyError:
           return False
       if not clause_satisfied:
           return False
   return True

# ----------------- verze 2 fix ----------------------------
# --- WalkSAT finisher (as in your code) ---
class WalkSATFinisher:
    def __init__(self, nvars, clauses, start_assign, p=0.40, seed=42):
        self.n = nvars
        self.C = len(clauses)
        self.clauses = clauses
        self.p = p
        random.seed(seed)
        self.pos = [[] for _ in range(nvars+1)]
        self.neg = [[] for _ in range(nvars+1)]
        for ci, cl in enumerate(clauses):
            for L in cl:
                (self.pos if L>0 else self.neg)[abs(L)].append(ci)
        self.assign = list(start_assign)
        self.sat_count = [0]*self.C
        self.unsat = set()
        for ci, cl in enumerate(clauses):
            cnt = 0
            for L in cl:
                v = abs(L); val = self.assign[v-1]
                if L < 0: val = (not val)
                if val: cnt += 1
            self.sat_count[ci] = cnt
            if cnt == 0: self.unsat.add(ci)

    def _breakcount(self, v):
        bc = 0
        if self.assign[v-1]:
            for ci in self.pos[v]:
                if self.sat_count[ci] == 1: bc += 1
        else:
            for ci in self.neg[v]:
                if self.sat_count[ci] == 1: bc += 1
        return bc

    def _flip(self, v):
        old = self.assign[v-1]; new = not old
        self.assign[v-1] = new
        if old:
            for ci in self.pos[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.neg[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)
        else:
            for ci in self.neg[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.pos[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)

    def solve(self, max_flips=600_000, report_every=50_000, novelty=0.30,
              restart_base=80_000, restart_mult=1.30, seed=42):
        best_unsat = len(self.unsat); best_state = self.assign[:]
        flips_since_best = 0
        next_restart = restart_base
        t0 = time.time()
        for flips in range(1, max_flips+1):
            if not self.unsat:
                return True, flips, time.time()-t0, self.assign[:]
            ci = random.choice(tuple(self.unsat))
            cl = self.clauses[ci]
            cand = [abs(L) for L in cl]
            if random.random() < self.p:
                v = random.choice(cand)
            else:
                best_bc, pool = 1e9, []
                for v0 in cand:
                    bc = self._breakcount(v0)
                    if bc < best_bc:
                        best_bc, pool = bc, [v0]
                    elif bc == best_bc:
                        pool.append(v0)
                v = random.choice(pool)
            self._flip(v)
            cur_unsat = len(self.unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_state = self.assign[:]
                flips_since_best = 0
            else:
                flips_since_best += 1
            if flips % report_every == 0:
                print(f"[finisher] flips={flips:,} unsat={cur_unsat:,} p≈{self.p:.3f}")
            if flips_since_best >= next_restart:
                rng = np.random.default_rng(seed)
                mask = rng.random(self.n) < 0.5
                self.assign = [ (best_state[i] if mask[i] else random.choice((False, True)))
                                for i in range(self.n) ]
                self.unsat.clear()
                self.sat_count = [0]*self.C
                for ci2, cl2 in enumerate(self.clauses):
                    cnt = 0
                    for L in cl2:
                        v2 = abs(L); val = self.assign[v2-1]
                        if L < 0: val = (not val)
                        if val: cnt += 1
                    self.sat_count[ci2] = cnt
                    if cnt == 0: self.unsat.add(ci2)
                flips_since_best = 0
                next_restart = int(next_restart * restart_mult)
        self.assign = best_state[:]
        return False, max_flips, time.time()-t0, self.assign[:]

def unsat_indices(clauses, assign01):
    ids = []
    for i, cl in enumerate(clauses):
        sat = False
        for L in cl:
            v = abs(L) - 1;
            val = bool(assign01[v])
            if L < 0: val = not val
            if val: sat = True; break
        if not sat: ids.append(i)
    return ids

# ---------- IO ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--mode", choices=["unsat_hadamard","sat"], default="unsat_hadamard")
    ap.add_argument("--rho", type=float, default=0.734296875)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--cR", type=float, default=10.0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--sigma_up", type=float, default=0.045)
    ap.add_argument("--neighbor_atten", type=float, default=0.9495)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--couple", type=int, default=1)  # 1=enable, 0=disable
    ap.add_argument("--power_iters", type=int, default=60)
    ap.add_argument("--score_norm_alpha", type=float, default=0.5)
    ap.add_argument("--bias_weight", type=float, default=0.10)
    ap.add_argument("--polish", type=int, default=20000)
    ap.add_argument("--dump_assign", type=str, default="")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--godmode", action="store_true",
        help="Apply the exact parameter pack you posted: cR=10, L=3, rho=0.734296875, zeta0=0.4, sigma_up=0.045, "
             "neighbor_atten=0.9495, seed=42, couple=1, d=6.")
    return ap.parse_args()


def main():

    print("\n*** SOLVER DEMO ***")




    print("\n[theory] generating theory params ...")

    args = parse_args()
    if args.godmode:
        args.cR = 10.0; args.L = 3; args.rho = 0.734296875; args.zeta0 = 0.40
        args.sigma_up = 0.045; args.neighbor_atten = 0.9495; args.seed = 42
        args.couple = 1; args.d = 6; args.mode = "unsat_hadamard"; args.power_iters = max(args.power_iters, 60)

    nvars, clauses = parse_dimacs(args.cnf)
    if args.subset and args.subset > 0:
        clauses = clauses[:args.subset]
    C = len(clauses)

    params = theory_params(C=C, want_sigma=0.02, cR=12, L=4, zeta0=0.4,
                      rho_lock=0.734296875, tau_hint=0.40, mu_hint=0.002)
    print("optimal params: ",params)

    print("\n[spectral] generating spectral seed ...")

    t0 = time.time()
    assign, Phi = build_seed_assignment(
        clauses, nvars, mode=args.mode, cR=args.cR, L=args.L,
        rho=args.rho, zeta0=args.zeta0, sigma_up=args.sigma_up,
        neighbor_atten=args.neighbor_atten, seed=args.seed, couple=bool(args.couple), d=args.d,
        power_iters=args.power_iters, score_norm_alpha=args.score_norm_alpha, bias_weight=args.bias_weight
    )


    t1 = time.time()
    if args.polish > 0:
        assign = greedy_polish(clauses, assign, flips=args.polish, seed=args.seed)
    unsat = count_unsat(clauses, assign)
    t2 = time.time()


    print("\n=== SPECTRAL REPORT ===")
    print(f"File              : {os.path.basename(args.cnf)}")
    print(f"Clauses (C)       : {C}")
    print(f"Vars (n)          : {nvars}")
    print(f"Mode              : {args.mode}")
    print(f"rho, zeta0        : {args.rho}, {args.zeta0}")
    print(f"cR, L             : {args.cR}, {args.L}")
    print(f"sigma_up          : {args.sigma_up}")
    print(f"neighbor_atten,d  : {args.neighbor_atten}, {args.d}")
    print(f"couple            : {args.couple}")
    print(f"score norm alpha  : {args.score_norm_alpha}")
    print(f"bias weight       : {args.bias_weight}")
    print(f"power iters       : {args.power_iters}")
    print(f"polish flips      : {args.polish}")
    print(f"Seed time         : {t1-t0:.3f}s")
    print(f"Check time        : {t2-t1:.3f}s")
    print(f"UNSAT clauses     : {unsat} / {C}  ({100.0 * unsat / max(1, C):.2f}%)")

    if unsat == 0:
        exit(0)

    # WalkSAT finisher
    print("\n[spectral] passing to finisher …\n")



    # spočti frekvence varů v aktuálních UNSAT
    u_ids = unsat_indices(clauses, assign)
    freq = [0] * nvars
    for ci in u_ids:
        for L in clauses[ci]:
            freq[abs(L) - 1] += 1

    # flipni 1–3 nejčastější
    K = min(8, sum(1 for f in freq if f > 0))
    tops = sorted(range(nvars), key=lambda i: freq[i], reverse=True)[:K]
    nudged = assign[:]
    for i in tops:
        nudged[i] = 1 - nudged[i]

    mu, tau0, p0 = spectral_pretest(nvars, clauses)

    fin = WalkSATFinisher(nvars, clauses, nudged, p=p0, seed=args.seed)
    ok, flips, tsec, model = fin.solve(
        max_flips=50_000_000,
        report_every=5_000_000,
        novelty=0.30,
        restart_base=4_000_000,   # dlouhý dwell
        restart_mult=1.15,        # pár restartů v průběhu
        seed=args.seed
    )

    sat = check_sat(clauses, model)
    unsat = count_unsat(clauses, model)

    print("\n=== FINISHER RESULT ===")
    print(f"UNSAT clauses  : {unsat} / {C}  ({100.0 * unsat / max(1, C):.2f}%)")
    print(f"Verified SAT   : {sat}")
    print(f"Solved flag    : {ok}")
    print(f"Flips          : {flips:,}")
    print(f"Time           : {tsec:.2f}s\n")


    #if args.dump_assign:
    #    with open(args.dump_assign, "w") as f:
    #        f.write("".join("1" if b else "0" for b in res["assignments"]))
    #    print(f"Wrote assignment to: {args.dump_assign}")

if __name__ == "__main__":
    main()
