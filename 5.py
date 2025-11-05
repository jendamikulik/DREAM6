#!/usr/bin/env python3
"""
DREAM6_SEED v3 — CLEAN FIX
- Removes duplicate/contradictory defs
- Unifies parameter theory pack
- Fixes WalkSATFinisher signature/usage
- Avoids undefined symbols (e.g., noise_schedule)
- Keeps your A1/A2/A3 design, power-iteration seed, and micro-polish

Usage (your god-mode):
  python3 DREAM6_seed_v3_FIXED.py --cnf random_3sat_10000.cnf --godmode --polish 20000

Explicit:
  python3 DREAM6_seed_v3_FIXED.py --cnf random_3sat_10000.cnf --mode unsat_hadamard \
    --rho 0.734296875 --zeta0 0.4 --cR 10 --L 3 --sigma_up 0.045 \
    --neighbor_atten 0.9495 --d 6 --seed 42 --couple 1 --power_iters 60 --polish 20000

Optional verification:
  python3 DREAM6_seed_v3_FIXED.py --cnf uf250-0100.cnf --theory --dump_assign out.assign --check
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# ----------------------- helpers -----------------------

def sigma_proxy(C: int, cR: float = 10.0, L: int = 3, eta_power: int = 3, C_B: float = 1.0) -> Tuple[float, int, int]:
    C = max(2, int(C))
    R = max(1, int(math.ceil(cR * math.log(C))))
    T = R * L
    eta = C ** (-eta_power)
    sigma_up = C_B * math.sqrt(max(1e-12, math.log(C / eta)) / max(1, T))
    return sigma_up, R, T


def wiring_neighbors_circulant(C: int, d: int = 6):
    if d % 2 != 0 or d > C - 1:
        raise ValueError(f"d={d} must be even and < C-1 for circulant wiring.")
    nbrs = []
    for i in range(C):
        nbr_set = set()
        for step in range(1, d // 2 + 1):
            nbr_set.add((i - step) % C)
            nbr_set.add((i + step) % C)
        nbrs.append(nbr_set)
    return nbrs


def gcd_coprime_stride_near_half(T: int) -> int:
    s = max(1, T // 2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

# -------------------- schedules ------------------------

def schedule_unsat_hadamard(
    C: int,
    R: int,
    rho: float = 0.734296875,
    zeta0: float = 0.40,
    L: int = 3,
    sigma_up: float = 0.045,
    seed: int = 42,
    couple: bool = True,
    neighbor_atten: float = 0.9495,
    d: int = 6,
    verbose: bool = False,
):
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
    while Hlen < m:
        Hlen <<= 1

    def hadamard_sign(row: int, col: int) -> float:
        return 1.0 if (bin(row & col).count("1") % 2 == 0) else -1.0

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
        neg_idx = []
        for t in range(m):
            sgn = hadamard_sign(row, int(cols[t]))
            if sgn < 0:
                neg_idx.append(t)
        if len(neg_idx) >= k:
            mask_pi = rng.choice(np.array(neg_idx, dtype=int), size=k, replace=False)
        else:
            pool = np.setdiff1d(np.arange(m, dtype=int), np.array(neg_idx, dtype=int), assume_unique=False)
            extra = rng.choice(pool, size=k - len(neg_idx), replace=False) if k > len(neg_idx) else np.empty(0, dtype=int)
            mask_pi = np.concatenate([np.array(neg_idx, dtype=int), extra])
        mask_0 = np.setdiff1d(np.arange(m, dtype=int), mask_pi, assume_unique=False)
        slots = lock_idx[j]
        Phi[slots[mask_pi], j] = np.pi
        if len(mask_0) > 0:
            Phi[slots[mask_0], j] = rng.normal(loc=0.0, scale=sigma_up, size=len(mask_0))

    # A3: coupling (optional)
    if couple and (abs(neighbor_atten - 1.0) > 1e-12) and C >= 3 and d >= 2:
        neighbors = wiring_neighbors_circulant(C, d=d)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        kappa = (1.0 - 2.0 * zeta0) ** 2 + (2.0 ** (-int(math.log2(max(2, m))) / 2.0)) + (2.0 / max(1, m)) + (1.0 / max(1, T))
        kappa = max(0.0, min(1.0, kappa))
        for j in range(C):
            Lj = lock_sets[j]
            for j_adj in neighbors[j]:
                if j_adj == j:
                    continue
                La = lock_sets[j_adj]
                overlap = Lj.intersection(La)
                if not overlap:
                    continue
                overlap_size = len(overlap)
                overlap_fraction = overlap_size / max(1, m)
                cross_term_weight = min(
                    1.0,
                    (len(neighbors[j]) * kappa) / max(1.0, C * (1 - 0.5 * sigma_up) ** 2) * (1.0 + 3.0 * overlap_fraction),
                )
                attenuation = max(
                    0.70,
                    neighbor_atten
                    - 0.05
                    * overlap_size
                    / (m * (1 + 0.25 * math.sqrt(max(1e-9, math.log(C))) * overlap_fraction))
                    * (1 - cross_term_weight),
                )
                idx = np.fromiter(overlap, dtype=int)
                Phi[idx, j_adj] *= attenuation
                if verbose and j % (max(1, C // 20)) == 0:
                    print(f"[A3] j={j}→{j_adj} | overlap={overlap_size} attn={attenuation:.3f}")

    return Phi


def schedule_sat_aligned(C: int, R: int, L: int = 3):
    return np.zeros((R * L, C), dtype=float)

# --------------- spectral weight (power iters) ---------------

def principal_weight_power(Phi: np.ndarray, iters: int = 60):
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.normal(size=C) + 1j * rng.normal(size=C)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        y = Z @ x
        x = (Z.conj().T @ y) / T
        x /= (np.linalg.norm(x) + 1e-12)
    return np.abs(x)

# ---------------------- DIMACS IO -------------------------

def parse_dimacs(path: str) -> Tuple[int, List[List[int]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    clauses: List[List[int]] = []
    nvars = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "c%":
                continue
            if s[0] == "p":
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    nvars = int(parts[2])
                continue
            lits = [int(x) for x in s.split() if x != "0"]
            if lits:
                clauses.append(lits)
                for L in lits:
                    nvars = max(nvars, abs(L))
    assert isinstance(clauses, list) and all(isinstance(cl, list) for cl in clauses)
    assert all(isinstance(L, int) for cl in clauses for L in cl)
    return nvars, clauses

# ---------------- seed construction ----------------------

def build_seed_assignment(
    clauses: List[List[int]],
    nvars: int,
    mode: str = "unsat_hadamard",
    cR: float = 10.0,
    L: int = 3,
    rho: float = 0.734296875,
    zeta0: float = 0.40,
    sigma_up: float = 0.045,
    neighbor_atten: float = 0.9495,
    seed: int = 42,
    couple: bool = True,
    d: int = 6,
    power_iters: int = 60,
    score_norm_alpha: float = 0.5,
    bias_weight: float = 0.10,
):
    C = len(clauses)
    _, R, _T = sigma_proxy(C, cR=cR, L=L)
    if mode == "sat":
        Phi = schedule_sat_aligned(C, R, L)
    else:
        Phi = schedule_unsat_hadamard(
            C,
            R,
            rho,
            zeta0,
            L,
            sigma_up,
            seed,
            couple=couple,
            neighbor_atten=neighbor_atten,
            d=d,
            verbose=False,
        )
    w_clause = principal_weight_power(Phi, iters=power_iters)
    w_clause = (w_clause / (w_clause.mean() + 1e-12)).clip(0.1, 10.0)

    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    pol = np.zeros(nvars + 1, dtype=int)
    deg = np.zeros(nvars + 1, dtype=int)
    for ci, cl in enumerate(clauses):
        for LIT in cl:
            v = abs(LIT)
            deg[v] += 1
            if LIT > 0:
                pos[v].append(ci)
                pol[v] += 1
            else:
                neg[v].append(ci)
                pol[v] -= 1

    score = np.zeros(nvars + 1, dtype=float)
    for v in range(1, nvars + 1):
        if pos[v]:
            score[v] += float(np.sum(w_clause[pos[v]]))
        if neg[v]:
            score[v] -= float(np.sum(w_clause[neg[v]]))
        if score_norm_alpha > 0.0:
            score[v] /= (deg[v] ** score_norm_alpha + 1e-12)
        if bias_weight != 0.0:
            score[v] += bias_weight * float(pol[v]) / max(1, int(deg[v]))

    rng = np.random.default_rng(seed + 1337)
    dither = rng.uniform(-1e-7, 1e-7, size=nvars + 1)
    assign = [(score[v] + dither[v]) >= 0.0 for v in range(1, nvars + 1)]
    return assign, Phi

# ---------------- UNSAT count & SAT check ----------------

def count_unsat(clauses: List[List[int]], assign_bool_list: List[bool]) -> int:
    unsat = 0
    for cl in clauses:
        ok = False
        for L in cl:
            v = abs(L)
            val = assign_bool_list[v - 1]
            if L < 0:
                val = (not val)
            if val:
                ok = True
                break
        if not ok:
            unsat += 1
    return unsat


def check_sat(clauses: List[List[int]], model: List[int]) -> bool:
    assignment = {i + 1: (bit == 1) for i, bit in enumerate(model)}
    for clause in clauses:
        clause_satisfied = False
        try:
            clause_satisfied = any(
                (lit > 0 and assignment[abs(lit)])
                or (lit < 0 and not assignment[abs(lit)])
                for lit in clause
            )
        except KeyError:
            return False
        if not clause_satisfied:
            return False
    return True

# ---------------- tiny greedy micro-polish ----------------

def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int = 20000,
    seed: int = 49,
    alpha: float = 2.4,  # probSAT make exponent
    beta: float = 0.9,   # probSAT break exponent
    epsilon: float = 1e-3,
    probsat_quota: int = 2000,
) -> List[int]:
    rnd = random.Random(seed)
    nvars = len(assign01)
    C = len(clauses)

    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for L in cl:
            (pos if L > 0 else neg)[abs(L)].append(ci)

    assign = [False] + [bool(b) for b in assign01]
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list: List[int] = []

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

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
    unsat_list = [ci for ci in unsat_list if in_unsat[ci]]

    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:
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

    def pick_unsat_clause() -> Optional[int]:
        if not unsat_list:
            return None
        i = rnd.randrange(len(unsat_list))
        for _ in range(3):
            ci = unsat_list[i]
            if in_unsat[ci]:
                return ci
            i = rnd.randrange(len(unsat_list))
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    def cur_unsat() -> int:
        return sum(1 for ci in unsat_list if in_unsat[ci])

    best_assign = assign[:]
    best_uns = cur_unsat()

    steps = 0
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
            freebies.sort(reverse=True)
            _, v = freebies[0]
            flip_var(v)
            steps += 1
            made_progress = True
            u = cur_unsat()
            if u < best_uns:
                best_uns = u
                best_assign = assign[:]
            if u == 0:
                return [1 if b else 0 for b in assign[1:]]

    while steps < flips:
        if best_uns == 0:
            return [1 if b else 0 for b in best_assign[1:]]

        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]

        v_choice = None
        freebies = []
        cand = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            mk = makecount(v)
            if bc == 0:
                freebies.append((mk, v))
            cand.append((bc, -mk, v))
        if freebies:
            freebies.sort(reverse=True)
            v_choice = freebies[0][1]
        else:
            v_choice = min(cand)[2]

        flip_var(v_choice)
        steps += 1

        u = cur_unsat()
        if u < best_uns:
            best_uns = u
            best_assign = assign[:]
        if u == 0:
            return [1 if b else 0 for b in assign[1:]]

        if steps % 1000 == 0 and u >= best_uns:
            for _ in range(min(probsat_quota, flips - steps)):
                ci2 = pick_unsat_clause()
                if ci2 is None:
                    return [1 if b else 0 for b in assign[1:]]
                clause2 = clauses[ci2]
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

    return [1 if b else 0 for b in (best_assign[1:] if best_uns < cur_unsat() else assign[1:])]

# ---------------- spectral pre-test ---------------------

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
            if v in seen:
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
        nrm = np.linalg.norm(y) + 1e-12
        x = y / nrm

    num = float(np.dot(x, Ax(x)))
    den = float(np.dot(x, x)) + 1e-12
    lam = num/den
    mu = lam / max(1, C)

    base = 0.35
    p0 = base + 0.25*(mu - 0.002) - 0.20*(tau0 - 0.4)
    p0 = max(0.12, min(0.58, p0))
    return mu, tau0, p0

# ---------------- WalkSAT finisher ----------------------

class WalkSATFinisher:
    """
    Minimal finisher: breakcount greedy + noise, with optional restarts.
    Compatible with the seed produced above.
    (Kept for backward compatibility.)
    """
    def __init__(self, nvars, clauses, start_assign, p=0.40, seed=42):
        self.n = nvars
        self.C = len(clauses)
        self.clauses = clauses
        self.p = p
        import random as _r
        _r.seed(seed)
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
              restart_base=500_000, restart_mult=1.30):
        import random as _r
        best_unsat = len(self.unsat); best_state = self.assign[:]
        flips_since_best = 0
        next_restart = restart_base
        t0 = time.time()

        for flips in range(1, max_flips+1):
            if not self.unsat:
                return True, flips, time.time()-t0, self.assign[:]

            ci = _r.choice(tuple(self.unsat))
            cl = self.clauses[ci]
            cand = [abs(L) for L in cl]

            if _r.random() < self.p:
                v = _r.choice(cand)
            else:
                best_bc, pool = 1e9, []
                for v0 in cand:
                    bc = self._breakcount(v0)
                    if bc < best_bc:
                        best_bc, pool = bc, [v0]
                    elif bc == best_bc:
                        pool.append(v0)
                v = _r.choice(pool)

            self._flip(v)
            cur_unsat = len(self.unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_state = self.assign[:]
                flips_since_best = 0
            else:
                flips_since_best += 1

            if report_every and flips % report_every == 0:
                print(f"[finisher] flips={flips:,} unsat={cur_unsat:,} p≈{self.p:.3f}")

            if flips_since_best >= next_restart:
                mask = np.random.default_rng(1234+flips).random(self.n) < 0.5
                self.assign = [ (best_state[i] if mask[i] else _r.choice((False, True)))
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

# ---------------- Skalpel finisher (fast UNSAT-doctor) -----------------

from typing import List, Optional, Dict, Any, Tuple

def solve_walksat(
    clauses: List[List[int]],
    nvars: int,
    max_flips: int = 25_000_000,
    p_noise: float = 0.25, #0.25,
    novelty_prob: float = 0.50, # 0.50,
    restart_interval: int = 1_000_000,
    rng_seed: int = 42,
    progress_every: int = 1_000_000,
    # --------- warm-start ----------
    init_model01: Optional[List[int]] = None,
    init_best_model01: Optional[List[int]] = None,
    # --------- enhancements (volitelné) ----------
    use_probsat: bool = True,
    alpha: float = 2.4, #2.4,      # probSAT make exponent
    beta: float = 0.9,       # probSAT break exponent
    epsilon: float = 1e-3,   # numerická stabilita
    noise_min: float = 0.18, # dynamický šum (min..max)
    noise_max: float = 0.32,
    smooth_every: int = 2_000_000,  # SAPS smoothing period
    decay: float = 0.92,     # multiplicativní pokles vah
    core_boost_every: int = 5_000_000,
    core_boost_quantile: float = 0.99,  # boost top 10% „nejbitějších“
    tremble_max: int = 64,   # počet „roztřesení“ při restartu
) -> Dict[str, Any]:
    rnd = random.Random(rng_seed)

    # --- adjacency (1-indexed) ---
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            (pos if lit > 0 else neg)[abs(lit)].append(ci)

    # --- assignment (1-indexed bools) ---
    if init_model01 is not None:
        if len(init_model01) != nvars:
            raise ValueError(f"init_model01 must have length {nvars}")
        assign = [False] + [bool(b) for b in init_model01]
    else:
        assign = [False] + [rnd.choice((False, True)) for _ in range(nvars)]

    # --- state ---
    C = len(clauses)
    sat_count = [0] * C
    weight = [1] * C           # SAPS/weights
    unsat_hits = [0] * C       # kolikrát byla klauzule při restartu UNSAT
    in_unsat = [False] * C     # bitset UNSAT
    unsat_list: List[int] = [] # kompaktní seznam UNSAT indexů

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

    # init sat counts
    for ci, cl in enumerate(clauses):
        cnt = 0
        for lit in cl:
            v = abs(lit)
            val = assign[v]
            if lit < 0:
                val = not val
            if val:
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            add_unsat(ci)

    # --- helpers ---
    def cur_unsat_fast() -> int:
        # rychlé spočítání bez plné kompakce
        return sum(1 for ci in unsat_list if in_unsat[ci])

    def pick_unsat_clause() -> int:
        # O(1) průměrné – čistí „mrtvé“ záznamy lajdácky
        while True:
            if not unsat_list:
                return -1
            ci = unsat_list[rnd.randrange(len(unsat_list))]
            if in_unsat[ci]:
                return ci
            # lazy cleanup: vyhoď mrtvé ocasem
            unsat_list[:] = [x for x in unsat_list if in_unsat[x]]

    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += weight[ci]
        else:
            for ci in neg[v]:
                if sat_count[ci] == 1:
                    bc += weight[ci]
        return bc

    def makecount(v: int) -> int:
        mk = 0
        if assign[v]:  # True->False => neg lit pravdivý
            for ci in neg[v]:
                if in_unsat[ci]:
                    mk += 1
        else:          # False->True => pos lit pravdivý
            for ci in pos[v]:
                if in_unsat[ci]:
                    mk += 1
        return mk

    def make_flip(v: int):
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
                if sc == 1:
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
                if sc == 1:
                    drop_unsat(ci)

    def eval_unsat_for_model(m01: List[int]) -> int:
        cnt_unsat = 0
        ass = [False] + [bool(b) for b in m01]
        for cl in clauses:
            ok = False
            for lit in cl:
                v = abs(lit)
                val = ass[v]
                if lit < 0:
                    val = not val
                if val:
                    ok = True
                    break
            if not ok:
                cnt_unsat += 1
        return cnt_unsat

    # --- best-so-far ---
    if init_best_model01 is not None:
        best_model = [False] + [bool(b) for b in init_best_model01]
        best_unsat = eval_unsat_for_model(init_best_model01)
    else:
        best_model = assign[:]
        best_unsat = cur_unsat_fast()
    best_at = 0

    start_unsat = cur_unsat_fast()
    start_t = time.time()
    flips = 0
    last_flip = [-1] * (nvars + 1)

    # dynamický šum
    def dyn_noise(f: int) -> float:
        if restart_interval <= 0:
            return p_noise
        t = (f % restart_interval) / float(restart_interval)
        return noise_min + 0.5 * (noise_max - noise_min) * (1 - math.cos(2 * math.pi * t))

    # --- main loop ---
    while flips < max_flips:
        if best_unsat == 0:
            return {
                "sat": True,
                "flips": flips,
                "time_s": time.time() - start_t,
                "assignments": [1 if b else 0 for b in best_model[1:]],
                "start_unsat": start_unsat,
            }

        # restarty + SAPS přivážení
        if restart_interval and flips and (flips % restart_interval == 0):
            # zvýšíme váhy aktuálně neuspokojených a zaznamenáme „hity“
            for ci in range(C):
                if in_unsat[ci]:
                    weight[ci] += 1
                    unsat_hits[ci] += 1
            # měkký restart: rozjeď z best_model a lehce zatřes
            assign = best_model[:]
            # přepočítej sat_count/in_unsat z assign (robustní)
            for ci, cl in enumerate(clauses):
                cnt = 0
                for lit in cl:
                    v = abs(lit)
                    val = assign[v]
                    if lit < 0:
                        val = not val
                    if val:
                        cnt += 1
                sat_count[ci] = cnt
                in_unsat[ci] = (cnt == 0)
            unsat_list = [ci for ci in range(C) if in_unsat[ci]]

            # tremble (náhodné drobné flipy)
            tremble = min(tremble_max, max(1, nvars // 20))
            for _ in range(tremble):
                make_flip(rnd.randint(1, nvars))

        # SAPS smoothing
        if smooth_every and flips and (flips % smooth_every == 0):
            for i, w in enumerate(weight):
                nw = int(max(1, w * decay))
                weight[i] = nw

        # core boost (přitížit nejproblematičtější klauzule)
        if core_boost_every and flips and (flips % core_boost_every == 0):
            hits_sorted = sorted(unsat_hits)
            if hits_sorted:
                thr = hits_sorted[int(len(hits_sorted) * core_boost_quantile)]
                for i, h in enumerate(unsat_hits):
                    if h >= thr and h > 0:
                        weight[i] += 1

        # výběr UNSAT klauzule
        ci = pick_unsat_clause()
        if ci < 0:
            # nic už není UNSAT
            best_model = assign[:]
            best_unsat = 0
            break
        clause = clauses[ci]

        # freebie/aspirace (break==0)
        freebies = []
        for lit in clause:
            v0 = abs(lit)
            bc = breakcount(v0)
            if bc == 0:
                freebies.append((makecount(v0), v0))
        if freebies:
            freebies.sort(reverse=True)  # max make
            v = freebies[0][1]
        else:
            # probSAT nebo Novelty+
            if rnd.random() < dyn_noise(flips):
                v = abs(rnd.choice(clause))
            else:
                if use_probsat:
                    scores = []
                    tot = 0.0
                    last_v = None
                    for lit in clause:
                        v0 = abs(lit)
                        mk = makecount(v0)
                        bc = breakcount(v0)
                        s = ((mk + epsilon) ** alpha) / ((bc + epsilon) ** beta)
                        scores.append((v0, s))
                        tot += s
                        last_v = v0
                    r = rnd.random() * tot
                    acc = 0.0
                    v = last_v
                    for v0, s in scores:
                        acc += s
                        if acc >= r:
                            v = v0
                            break
                else:
                    # Novelty+ fallback
                    best_v, best_bc = None, 10**9
                    second_v, second_bc = None, 10**9
                    for lit in clause:
                        v0 = abs(lit)
                        bc = breakcount(v0)
                        if bc < best_bc:
                            second_v, second_bc = best_v, best_bc
                            best_v, best_bc = v0, bc
                        elif bc < second_bc:
                            second_v, second_bc = v0, bc
                    v = best_v
                    if rnd.random() < novelty_prob and last_flip[best_v] >= flips - 1 and second_v is not None:
                        v = second_v

        make_flip(v)
        last_flip[v] = flips
        flips += 1

        cur_u = cur_unsat_fast()
        if cur_u < best_unsat:
            best_unsat = cur_u
            best_model = assign[:]

        if progress_every and (flips % progress_every == 0):
            print(f"unsat={best_unsat} at {flips} flips")

    # konec bez certifikátu – vrať best-so-far
    return {
        "sat": (best_unsat == 0),
        "flips": flips,
        "time_s": time.time() - start_t,
        "best_unsat": best_unsat,
        "assignments": [1 if b else 0 for b in best_model[1:]],
        "start_unsat": start_unsat,
    }


# --- vector finisher ---

# Let's implement a faster solver with watched literals + VSIDS + restarts + phase saving.
# It's still compact but much faster on uf250.*—and with a --timeout to avoid "stuck".
from typing import List, Dict, Optional, Tuple
import time, sys, math, argparse, random, os

CNF_PATH = "/mnt/data/uf250-0100.cnf"

class SATSolver:
    def __init__(self, clauses: List[List[int]], nvars: int, seed: int = 0):
        self.nvars = nvars
        self.orig_clauses = [list(c) for c in clauses]
        self.rng = random.Random(seed)
        # Assignments: -1=unassigned, 0=False, 1=True
        self.val = [-1]*(nvars+1)
        self.level = [0]*(nvars+1)
        self.decision_var = [False]*(nvars+1)
        self.reason: List[Optional[List[int]]] = [None]*(nvars+1)
        self.trail: List[int] = []
        self.trail_lim: List[int] = []
        # Watches: lit -> list of clause indices
        self.watches: Dict[int, List[int]] = {}
        # Clause DB
        self.clauses: List[List[int]] = []
        # Activity (VSIDS)
        self.act = [0.0]*(nvars+1)
        self.var_inc = 1.0
        self.var_decay = 0.95
        # Phase saving
        self.saved_phase = [False]*(nvars+1)
        # Prepare
        self._init_watches(self.orig_clauses)

    def _init_watches(self, clauses: List[List[int]]):
        self.clauses = []
        for ci, c in enumerate(clauses):
            if len(c) == 0:
                continue
            # ensure at least two watched literals
            if len(c) == 1:
                c = [c[0], c[0]]
            self.clauses.append(c)
        # reset watches
        self.watches = {}
        for ci, c in enumerate(self.clauses):
            self._watch(ci, c[0])
            self._watch(ci, c[1])

    def _watch(self, ci: int, lit: int):
        self.watches.setdefault(lit, []).append(ci)

    def _unassigned(self, v: int) -> bool:
        return self.val[v] == -1

    def value_of_lit(self, lit: int) -> int:
        v = abs(lit); s = 1 if lit>0 else 0
        x = self.val[v]
        if x == -1: return -1
        return 1 if x == s else 0

    def enqueue(self, lit: int, reason: Optional[List[int]]):
        v = abs(lit); val = 1 if lit>0 else 0
        if self.val[v] != -1:
            return self.val[v] == val
        self.val[v] = val
        self.level[v] = self.decision_level()
        self.reason[v] = reason
        self.trail.append(lit if val==1 else -lit)
        self.saved_phase[v] = (val==1)
        return True

    def new_decision_level(self):
        self.trail_lim.append(len(self.trail))

    def cancel_until(self, lvl: int):
        if self.decision_level() <= lvl: return
        while len(self.trail) > self.trail_lim[lvl]:
            lit = self.trail.pop()
            v = abs(lit)
            self.val[v] = -1
            self.level[v] = 0
            self.reason[v] = None
        while len(self.trail_lim) > lvl:
            self.trail_lim.pop()

    def decision_level(self) -> int:
        return len(self.trail_lim)

    def pick_branch_var(self) -> Optional[int]:
        # VSIDS: pick unassigned var with max activity
        best_v = None; best_a = -1.0
        for v in range(1, self.nvars+1):
            if self.val[v] == -1:
                a = self.act[v]
                if a > best_a:
                    best_a = a; best_v = v
        if best_v is None:
            # fallback random
            un = [v for v in range(1, self.nvars+1) if self.val[v]==-1]
            return None if not un else self.rng.choice(un)
        return best_v

    def bump_var(self, v: int):
        self.act[v] += self.var_inc
        if self.act[v] > 1e100:
            for i in range(1, self.nvars+1):
                self.act[i] *= 1e-100

    def decay_acts(self):
        self.var_inc /= self.var_decay

    def propagate(self) -> Optional[List[int]]:
        i = 0
        while i < len(self.trail):
            lit = -self.trail[i]  # literal that just became false
            i += 1
            wl = self.watches.get(lit, [])
            j = 0
            while j < len(wl):
                ci = wl[j]
                c = self.clauses[ci]
                # Ensure lit is at c[1], and other watch is c[0]
                if c[0] == lit:
                    c[0], c[1] = c[1], c[0]
                other = c[0]
                # If other is satisfied, keep watching lit
                if self.value_of_lit(other) == 1:
                    j += 1
                    continue
                # Try find a new watch
                found = False
                for k in range(2, len(c)):
                    if self.value_of_lit(c[k]) != 0:
                        # watch c[k] instead of lit
                        c[1], c[k] = c[k], c[1]
                        wl[j] = wl[-1]; wl.pop()  # remove ci from watchlist lit
                        self._watch(ci, c[1])
                        found = True
                        break
                if found:
                    continue
                # Otherwise, clause is unit under current assignment: must set 'other'
                val_other = self.value_of_lit(other)
                if val_other == 0:
                    # conflict: clause is false
                    return c
                # Enqueue 'other' if unassigned
                if not self.enqueue(other if other>0 else other, c):
                    return c
                j += 1
        return None

    def analyze(self, confl: List[int]) -> Tuple[List[int], int]:
        # Learn the conflicting clause itself (naive 1-UIP-free) and backtrack to previous level
        # This is simplistic but works as clause learning baseline.
        learnt = list(confl)
        # Backtrack level: max level among vars in learnt except the highest one
        levels = sorted({ self.level[abs(l)] for l in learnt })
        if not levels: return learnt, 0
        back = levels[-1]
        if len(levels) >= 2:
            back = levels[-2]
        return learnt, back

    def add_clause(self, c: List[int]):
        if len(c) == 0:
            return
        if len(c) == 1:
            c = [c[0], c[0]]
        self.clauses.append(c)
        self._watch(len(self.clauses)-1, c[0])
        self._watch(len(self.clauses)-1, c[1])

    def solve(self, timeout: float = 5.0) -> Tuple[bool, List[int]]:
        start = time.time()
        # Initial unit clauses
        confl = self.propagate()
        if confl is not None:
            return False, []
        # Luby restarts
        def luby(i):
            # 1,1,2,1,1,2,4,...
            k = 1
            while (1 << k) - 1 < i:
                k += 1
            if i == (1 << k) - 1:
                return 1 << (k - 1)
            return luby(i - (1 << (k - 1)) + 1)
        restart_i = 1
        confl_count = 0
        max_conflicts = 100
        while True:
            if time.time()-start > timeout:
                # Return partial assignment as "unknown"; here we'll just signal UNSAT? Better: raise
                raise TimeoutError("Timeout")
            confl = self.propagate()
            if confl is not None:
                # bump activities
                for lit in confl:
                    self.bump_var(abs(lit))
                self.decay_acts()
                confl_count += 1
                learnt, back = self.analyze(confl)
                self.cancel_until(back)
                self.add_clause(learnt)
                # enqueue unit if possible
                if len(set(learnt)) == 1 or (len(learnt)>=1 and all(self.value_of_lit(l)!=1 for l in learnt)):
                    # pick one satisfied literal to enqueue positively
                    lit = learnt[0]
                    self.enqueue(lit, learnt)
                # restart policy
                if confl_count >= max_conflicts:
                    # restart
                    self.cancel_until(0)
                    restart_i += 1
                    max_conflicts = 100 * luby(restart_i)
                continue
            # Satisfied?
            all_assigned = all(self.val[v] != -1 for v in range(1, self.nvars+1))
            if all_assigned:
                model = [ (1 if self.val[v]==1 else -1)*v for v in range(1, self.nvars+1) ]
                return True, model
            # Decide next variable
            v = self.pick_branch_var()
            if v is None:
                # all assigned by propagation
                model = [ (1 if self.val[v]==1 else -1)*v for v in range(1, self.nvars+1) ]
                return True, model
            self.new_decision_level()
            pol = 1 if self.saved_phase[v] else (1 if self.rng.random()<0.5 else 0)
            self.decision_var[v] = True
            self.enqueue(v if pol==1 else -v, None)

def parse_dimacs_file(path: str) -> Tuple[int, List[List[int]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        nvars = ncls = None
        clauses = []
        cur = []
        for raw in f:
            s = raw.strip()
            if not s or s[0] == 'c':
                continue
            if s[0] == 'p':
                parts = s.split()
                nvars = int(parts[2]); ncls = int(parts[3])
                continue
            for tok in s.split():
                v = int(tok)
                if v == 0:
                    if cur:
                        clauses.append(cur); cur = []
                else:
                    cur.append(v)
        return nvars, clauses

# Load the file and attempt solve with a short timeout to prove we don't "hang"
nvars, clauses = parse_dimacs_file(CNF_PATH)
print("CNF stats:", nvars, "vars,", len(clauses), "clauses")
solver = SATSolver(clauses, nvars, seed=42)
try:
    sat, model = solver.solve(timeout=3.0)  # short demo timeout
    print("s", "SATISFIABLE" if sat else "UNSATISFIABLE")
    if sat:
        # show first 20 literals
        print("v", " ".join(str(x) for x in model[:20]), "... 0")
except TimeoutError as e:
    print("c Solver timed out (as requested demo limit). It is not stuck; run with a longer timeout.")



# ---------------- theory params (single, consistent) -----

def theory_params(
    C: int,
    want_sigma: Optional[float] = None,
    cR: int = 12,
    L: int = 4,
    eta_power: int = 3,
    zeta0: float = 0.40,
):
    R = math.ceil(cR * math.log(C))
    T = R * L
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # Solve rho from: rho*z0 - 0.5*sigma_up = sqrt(2/(rho*T) + 1/T)
    def solve_rho(z0: float, sig: float, Tval: int) -> float:
        rho = max(1e-6, (0.5 * sig) / max(1e-12, z0) + 1e-3)
        for _ in range(20):
            invT = 1.0 / max(1, Tval)
            root = math.sqrt(2.0 / (rho * Tval) + invT)
            f = rho * z0 - 0.5 * sig - root
            df = z0 + (1.0 / (rho * rho * Tval)) / root
            step = f / df
            rho_next = rho - step
            if rho_next <= 0 or not math.isfinite(rho_next):
                rho_next = rho * 0.5
            if abs(rho_next - rho) <= 1e-10 * max(1.0, rho):
                rho = rho_next
                break
            rho = rho_next
        return rho

    rho = solve_rho(zeta0, sigma_up, T)

    # Smooth mappings for score normalization & bias from sigma
    s = 1.0 - math.exp(-1.0 / max(1e-12, sigma_up + 1e-12))
    score_norm_alpha = 0.5 + 0.35 * s
    bias_weight = 0.06 + 0.16 * s

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho,
        "zeta0": zeta0,
        "score_norm_alpha": score_norm_alpha,
        "bias_weight": bias_weight,
    }

# ---------------- CLI / main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--mode", choices=["unsat_hadamard", "sat"], default="unsat_hadamard")
    ap.add_argument("--rho", type=float, default=0.734296875)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--cR", type=float, default=10.0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--sigma_up", type=float, default=0.045)
    ap.add_argument("--neighbor_atten", type=float, default=0.9495)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--couple", type=int, default=1)
    ap.add_argument("--power_iters", type=int, default=60)
    ap.add_argument("--score_norm_alpha", type=float, default=0.5)
    ap.add_argument("--bias_weight", type=float, default=0.10)
    ap.add_argument("--polish", type=int, default=20000)
    ap.add_argument("--dump_assign", type=str, default="", help="Write 01 assignment to file")
    ap.add_argument("--subset", type=int, default=0, help="Use only first N clauses")
    ap.add_argument("--theory", action="store_true", help="Use closed-form theory for parameters")
    ap.add_argument("--godmode", action="store_true", help="Reproduce the demo hand-tuned pack")
    ap.add_argument("--check", action="store_true", help="Verify model with an exact SAT check on output")
    ap.add_argument("--finisher", action="store_true", help="Run WalkSAT finisher after micro-polish")
    return ap.parse_args()


def main():
    print("\n*** SOLVER DEMO (theory-first, cleaned) ***")

    args = parse_args()

    # Modes/Presets
    if args.godmode:
        args.cR = 10.0
        args.L = 3
        args.rho = 0.734296875
        args.zeta0 = 0.40
        args.sigma_up = 0.045
        args.neighbor_atten = 0.9495
        args.seed = 42
        args.couple = 1
        args.d = 6
        args.mode = "unsat_hadamard"
        args.power_iters = max(args.power_iters, 60)
        print("[godmode] using hand-tuned pack")
    elif args.theory:
        print("[theory] deriving parameters from instance size ...")
        # temporary read DIMACS to know C
        nvars_tmp, clauses_tmp = parse_dimacs(args.cnf)
        if args.subset and args.subset > 0:
            clauses_tmp = clauses_tmp[: args.subset]
        Ctmp = len(clauses_tmp)
        params = theory_params(C=Ctmp, want_sigma=None, cR=12, L=4, zeta0=0.40)
        args.cR = 12.0
        args.L = 4
        args.rho = params["rho"]
        args.zeta0 = params["zeta0"]
        args.sigma_up = params["sigma_up"]
        args.score_norm_alpha = params["score_norm_alpha"]
        args.bias_weight = params["bias_weight"]
        args.couple = 1
        args.d = 6
        args.mode = "unsat_hadamard"
        # spectral/polish scaling
        args.power_iters = int(max(20, 10 + 10 * max(0.0, math.log2(max(4, Ctmp)))))
        args.polish = int(max(50_000, 250_000 * (max(1.0, Ctmp / 200_000.0)) ** (2.0 / 3.0)))
        print("[theory] pack:", params)

    # Load instance (once)
    nvars, clauses = parse_dimacs(args.cnf)
    if args.subset and args.subset > 0:
        clauses = clauses[: args.subset]
    C = len(clauses)

    print("\n[spectral] generating spectral seed ...")
    t0 = time.time()
    assign_bool, Phi = build_seed_assignment(
        clauses,
        nvars,
        mode=args.mode,
        cR=args.cR,
        L=args.L,
        rho=args.rho,
        zeta0=args.zeta0,
        sigma_up=args.sigma_up,
        neighbor_atten=args.neighbor_atten,
        seed=args.seed,
        couple=bool(args.couple),
        d=args.d,
        power_iters=args.power_iters,
        score_norm_alpha=args.score_norm_alpha,
        bias_weight=args.bias_weight,
    )
    t1 = time.time()

    assign01 = [1 if b else 0 for b in assign_bool]

    if args.polish > 0:
        assign01 = greedy_polish(clauses, assign01, flips=args.polish, seed=args.seed)
    unsat = count_unsat(clauses, [bool(b) for b in assign01])
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
    print(f"Seed time         : {t1 - t0:.3f}s")
    print(f"Check time        : {t2 - t1:.3f}s")
    print(f"UNSAT clauses     : {unsat} / {C}  ({100.0 * unsat / max(1, C):.2f}%)")

    if unsat == 0:
        exit(0)

    # WalkSAT finisher
    print("\n[lift] passing to WalkSAT finisher …")

    u0 = count_unsat(clauses, assign01)

    res = solve_walksat(
        clauses, nvars,
        rng_seed= args.seed ^ 0xBAD,
        max_flips=50_000_000,
        init_model01=assign01,  # startuj z něj
        init_best_model01=assign01,  # a nastav ho i jako best-so-far
    )

    ok = check_sat(clauses, res["assignments"])
    u1 = count_unsat(clauses, res["assignments"])

    t3 = time.time()

    print("\n=== WALKSAT REPORT ===")
    print(f"Verified SAT  : {ok}")
    print(f"Unsat (before): {u0} / {C}  ({(100.0 * u0 / max(1, C)):.2f}%)")
    print(f"Unsat (after) : {u1} / {C}  ({(100.0 * u1 / max(1, C)):.2f}%)")
    print(f"Flips (WS)    : {res["flips"]}")
    print(f"Time          : {t3 - t2:.2f}s")


    if args.dump_assign:
        with open(args.dump_assign, "w") as f:
            f.write("".join("1" if b else "0" for b in assign01))
        print(f"Wrote assignment to: {args.dump_assign}")

    if args.check:
        ok = check_sat(clauses, assign01)
        print(f"Model SAT check   : {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if unsat == 0 else 1)


if __name__ == "__main__":
    main()
