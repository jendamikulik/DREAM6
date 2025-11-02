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
#   python3 DREAM6_seed_v3_fixed.py --cnf random_3sat_10000.cnf --godmode --polish 20000
#
# Or explicitly:
#   python3 DREAM6_seed_v3_fixed.py --cnf random_3sat_10000.cnf --mode unsat_hadamard \
#     --rho 0.734296875 --zeta0 0.4 --cR 10 --L 3 --sigma_up 0.045 \
#     --neighbor_atten 0.9495 --d 6 --seed 42 --couple 1 --power_iters 60 --polish 20000
#
# Example:
#   python3 DREAM6_seed_v3_fixed.py --cnf uf250-0100.cnf --theory --dump_assign out.assign --check

from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# ---------- helpers ----------
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
    # choose s near T/2, coprime with T
    s = max(1, T // 2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

# ---------- schedules ----------
def schedule_unsat_hadamard(C: int, R: int, rho: float = 0.734296875, zeta0: float = 0.40, L: int = 3, sigma_up: float = 0.045,
                             seed: int = 42, couple: bool = True, neighbor_atten: float = 0.9495, d: int = 6, verbose: bool = False):
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

    # A3: coupling (optional): attenuate overlaps along circulant neighbors
    if couple and (abs(neighbor_atten - 1.0) > 1e-12) and C >= 3 and d >= 2:
        neighbors = wiring_neighbors_circulant(C, d=d)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        # kappa_S2 proxy
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
                    (len(neighbors[j]) * kappa)
                    / max(1.0, C * (1 - 0.5 * sigma_up) ** 2)
                    * (1.0 + 3.0 * overlap_fraction),
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
                    print(f"[A3] j={j}→{j_adj} | overlap={overlap_size} attn={attenuation:.3f} κ≈{kappa:.4f}")
    return Phi

def schedule_sat_aligned(C: int, R: int, L: int = 3):
    return np.zeros((R * L, C), dtype=float)

# ---------- spectral weight via power-iteration ----------
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

# ---------- DIMACS ----------
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
    # sanity (your asserts)
    assert isinstance(clauses, list) and all(isinstance(cl, list) for cl in clauses)
    assert all(isinstance(L, int) for cl in clauses for L in cl)
    return nvars, clauses

# ---------- scoring & seed ----------
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

# ---------- UNSAT ----------
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

# ---------- micro-polish (optional) ----------

def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int = 20000,
    seed: int = 49,
    alpha: float = 2.4,  # probSAT make exponent
    beta: float = 0.9,   # probSAT break exponent
    epsilon: float = 1e-3,  # probSAT epsilon
    probsat_quota: int = 2000,  # max steps for probSAT burst
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
        else:  # False->True
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

        # small "kick": if stagnating, do a short probSAT burst
        if steps % 1000 == 0 and u >= best_uns:
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

    # budget exhausted – return best found
    return [1 if b else 0 for b in (best_assign[1:] if best_uns < cur_unsat() else assign[1:])]


# --------------------- params -----------------------------------------

def theory_params(
    C: int,
    want_sigma: Optional[float] = None,
    cR: int = 12,
    L: int = 4,
    eta_power: int = 3,
    zeta0: float = 0.40,
    rho_lock: float = 0.734296875,
    neighbor_atten: float = 0.9495,
    tau_hint: float = 0.40,
    mu_hint: float = 0.002,
):
    """
    FULL-PURE theory pack (no clamps, no heuristics):
      - σ_up = √((1+η) log C)/√T with η=C^{-eta_power}
      - ρ solves exactly:  ρ·ζ₀ − 0.5 σ_up = √( 2/(ρT) + 1/T )  on ρ>0
        (Newton w/ analytic derivative; deterministic; no clipping)
    Everything reduces to log C / √T.
    """
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) Solve for rho > 0 from the implicit equation without any clamps
    def solve_rho(z0: float, sig: float, Tval: int) -> float:
        # f(ρ) = ρ z0 − 0.5 sig − sqrt( 2/(ρT) + 1/T ) = 0
        # f'(ρ) = z0 + (1/(ρ^2 T)) / sqrt( 2/(ρT) + 1/T )
        rho = max(1e-6, (0.5 * sig) / max(1e-12, z0) + 1e-3)
        for _ in range(20):
            invT = 1.0 / max(1, Tval)
            root = math.sqrt(2.0 / (rho * Tval) + invT)
            f = rho * z0 - 0.5 * sig - root
            df = z0 + (1.0 / (rho * rho * Tval)) / root
            step = f / df
            rho_next = rho - step
            if rho_next <= 0 or not math.isfinite(rho_next):
                rho_next = rho * 0.5  # backoff, keeps positivity
            if abs(rho_next - rho) <= 1e-10 * max(1.0, rho):
                rho = rho_next
                break
            rho = rho_next
        return rho

    rho = solve_rho(zeta0, sigma_up, T)
    chi = math.sqrt(2.0 / (rho * T) + 1.0 / max(1, T))
    gamma0 = rho * zeta0 - 0.5 * sigma_up  # equals χ at convergence

    # 4) bias & alpha: keep deterministic closed form tied to σ_up (no knobs)
    # Map σ_up∈(0,∞) into stable ranges without explicit clamps by smooth saturations.
    s = 1.0 - math.exp(-1.0 / max(1e-12, sigma_up + 1e-12))  # in (0,1)
    score_norm_alpha = 0.5 + 0.35 * s   # ∈ (0.5,0.85)
    bias_weight = 0.06 + 0.16 * s       # ∈ (0.06,0.22)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
        "chi": chi,
    }
    """
    Closed-form, parameter-free theory pack.

    Key change: ρ (lock ratio) is no longer a constant. We derive it
    from instance complexity so it self-tightens as the spectral gap closes.

    Definitions (no tunables):
      - T = R·L,  R = ceil(cR·log C)
      - σ_up = √((1+η) log C) / √T  with η = C^{-eta_power}
      - Complexity index χ = √( 2/(ρT) + 1/T )  (ρ appears ⇒ do a short fixed-point
        update; start from ρ₀=0.70, three deterministic iterations)
      - ρ = clip( (0.5·σ_up + χ) / ζ₀ , lower=0.55, upper=0.96 )

    This enforces γ₀ = ρ·ζ₀ − 0.5·σ_up ≥ χ ≥ 0, with χ shrinking as T grows or the
    instance is easier; hence ρ automatically relaxes on easy cases and tightens on
    hard ones. No magic epsilons.
    """
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (either want_sigma, or from C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) Self-consistent rho from complexity (fixed-point, 3 iterations, no knobs)
    def clip(x, a, b):
        return max(a, min(b, x))

    rho = 0.70  # seed, deterministic
    for _ in range(3):
        # guard against tiny T at toy scale
        invT = 1.0 / max(1, T)
        chi = math.sqrt(max(0.0, 2.0 / max(1.0, rho * T) + invT))
        rho = clip((0.5 * sigma_up + chi) / max(1e-12, zeta0), 0.55, 0.96)

    gamma0 = rho * zeta0 - 0.5 * sigma_up  # should be >= chi

    # 4) bias and alpha from (tau, mu)
    if tau_hint is None:
        tau_hint = 0.40
    if mu_hint is None:
        mu_hint = 0.002

    bias_weight = clip(0.08 + 0.30 * max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5 * max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
    }
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (either want_sigma, or from C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) gamma0 guard
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    if gamma0 <= 0:
        rho_lock = (0.5 * sigma_up + 0.01) / zeta0
        gamma0 = rho_lock * zeta0 - 0.5 * sigma_up

    # 4) bias and alpha from (tau, mu)
    def clip(x, a, b):
        return max(a, min(b, x))

    if tau_hint is None:
        tau_hint = 0.40
    if mu_hint is None:
        mu_hint = 0.002

    bias_weight = clip(0.08 + 0.30 * max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5 * max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho_lock,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
    }

# -------------------- utils ---------------------------------------------

def unsat_indices(clauses: List[List[int]], assign01: List[int]):
    ids = []
    for i, cl in enumerate(clauses):
        sat = False
        for L in cl:
            v = abs(L) - 1
            val = bool(assign01[v])
            if L < 0:
                val = not val
            if val:
                sat = True
                break
        if not sat:
            ids.append(i)
    return ids


# ----------------------- fix ---------------------------------------------

def theory_params(
    C: int,
    want_sigma: Optional[float] = None,
    cR: int = 12,
    L: int = 4,
    eta_power: int = 3,
    zeta0: float = 0.40,
    rho_lock: float = 0.734296875,
    neighbor_atten: float = 0.9495,
    tau_hint: float = 0.40,
    mu_hint: float = 0.002,
):
    """
    FULL-PURE theory pack (no clamps, no heuristics):
      - σ_up = √((1+η) log C)/√T with η=C^{-eta_power}
      - ρ solves exactly:  ρ·ζ₀ − 0.5 σ_up = √( 2/(ρT) + 1/T )  on ρ>0
        (Newton w/ analytic derivative; deterministic; no clipping)
    Everything reduces to log C / √T.
    """
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) Solve for rho > 0 from the implicit equation without any clamps
    def solve_rho(z0: float, sig: float, Tval: int) -> float:
        # f(ρ) = ρ z0 − 0.5 sig − sqrt( 2/(ρT) + 1/T ) = 0
        # f'(ρ) = z0 + (1/(ρ^2 T)) / sqrt( 2/(ρT) + 1/T )
        rho = max(1e-6, (0.5 * sig) / max(1e-12, z0) + 1e-3)
        for _ in range(20):
            invT = 1.0 / max(1, Tval)
            root = math.sqrt(2.0 / (rho * Tval) + invT)
            f = rho * z0 - 0.5 * sig - root
            df = z0 + (1.0 / (rho * rho * Tval)) / root
            step = f / df
            rho_next = rho - step
            if rho_next <= 0 or not math.isfinite(rho_next):
                rho_next = rho * 0.5  # backoff, keeps positivity
            if abs(rho_next - rho) <= 1e-10 * max(1.0, rho):
                rho = rho_next
                break
            rho = rho_next
        return rho

    rho = solve_rho(zeta0, sigma_up, T)
    chi = math.sqrt(2.0 / (rho * T) + 1.0 / max(1, T))
    gamma0 = rho * zeta0 - 0.5 * sigma_up  # equals χ at convergence

    # 4) bias & alpha: keep deterministic closed form tied to σ_up (no knobs)
    # Map σ_up∈(0,∞) into stable ranges without explicit clamps by smooth saturations.
    s = 1.0 - math.exp(-1.0 / max(1e-12, sigma_up + 1e-12))  # in (0,1)
    score_norm_alpha = 0.5 + 0.35 * s   # ∈ (0.5,0.85)
    bias_weight = 0.06 + 0.16 * s       # ∈ (0.06,0.22)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
        "chi": chi,
    }
    """
    Closed-form, parameter-free theory pack.

    Key change: ρ (lock ratio) is no longer a constant. We derive it
    from instance complexity so it self-tightens as the spectral gap closes.

    Definitions (no tunables):
      - T = R·L,  R = ceil(cR·log C)
      - σ_up = √((1+η) log C) / √T  with η = C^{-eta_power}
      - Complexity index χ = √( 2/(ρT) + 1/T )  (ρ appears ⇒ do a short fixed-point
        update; start from ρ₀=0.70, three deterministic iterations)
      - ρ = clip( (0.5·σ_up + χ) / ζ₀ , lower=0.55, upper=0.96 )

    This enforces γ₀ = ρ·ζ₀ − 0.5·σ_up ≥ χ ≥ 0, with χ shrinking as T grows or the
    instance is easier; hence ρ automatically relaxes on easy cases and tightens on
    hard ones. No magic epsilons.
    """
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (either want_sigma, or from C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) Self-consistent rho from complexity (fixed-point, 3 iterations, no knobs)
    def clip(x, a, b):
        return max(a, min(b, x))

    rho = 0.70  # seed, deterministic
    for _ in range(3):
        # guard against tiny T at toy scale
        invT = 1.0 / max(1, T)
        chi = math.sqrt(max(0.0, 2.0 / max(1.0, rho * T) + invT))
        rho = clip((0.5 * sigma_up + chi) / max(1e-12, zeta0), 0.55, 0.96)

    gamma0 = rho * zeta0 - 0.5 * sigma_up  # should be >= chi

    # 4) bias and alpha from (tau, mu)
    if tau_hint is None:
        tau_hint = 0.40
    if mu_hint is None:
        mu_hint = 0.002

    bias_weight = clip(0.08 + 0.30 * max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5 * max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
    }
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (either want_sigma, or from C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T ** 0.5) / math.sqrt((1 + eta_power) * math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1 + eta_power) * math.log(C)) / (T ** 0.5)

    # 3) gamma0 guard
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    if gamma0 <= 0:
        rho_lock = (0.5 * sigma_up + 0.01) / zeta0
        gamma0 = rho_lock * zeta0 - 0.5 * sigma_up

    # 4) bias and alpha from (tau, mu)
    def clip(x, a, b):
        return max(a, min(b, x))

    if tau_hint is None:
        tau_hint = 0.40
    if mu_hint is None:
        mu_hint = 0.002

    bias_weight = clip(0.08 + 0.30 * max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5 * max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R,
        "T": T,
        "sigma_up": sigma_up,
        "C_B": CB,
        "rho": rho_lock,
        "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0,
    }



# ---------- IO ----------

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
    ap.add_argument("--couple", type=int, default=1)  # 1=enable, 0=disable
    ap.add_argument("--power_iters", type=int, default=60)
    ap.add_argument("--score_norm_alpha", type=float, default=0.5)
    ap.add_argument("--bias_weight", type=float, default=0.10)
    ap.add_argument("--polish", type=int, default=20000)
    ap.add_argument("--dump_assign", type=str, default="", help="Write 01 assignment to file")
    ap.add_argument("--subset", type=int, default=0, help="Use only first N clauses")
    ap.add_argument("--theory", action="store_true", help="Use closed-form theory for parameters (default unless --godmode)")
    ap.add_argument("--godmode", action="store_true", help="Reproduce the demo hand-tuned pack")
    ap.add_argument("--check", action="store_true", help="Verify model with an exact SAT check on output")
    return ap.parse_args()


def main():
    print("\n*** SOLVER DEMO (theory-first) ***")

    args = parse_args()
    if not args.godmode:
        args.theory = True
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

    nvars, clauses = parse_dimacs(args.cnf)
    if args.subset and args.subset > 0:
        clauses = clauses[: args.subset]
    C = len(clauses)

    # unified theory pack
    use_theory = args.theory and not args.godmode
    if use_theory:
        params = theory_params(
            C=C,
            want_sigma=None,  # σ_up = √((1+η) log C)/√T
            cR=12,
            L=4,
            zeta0=0.40,
            rho_lock=0.734296875,  # legacy default; dynamic rho overrides
            neighbor_atten=0.9495,
            tau_hint=0.40,
            mu_hint=0.002,
        )
        """
        log C)/√T
            cR=12,
            L=4,
            zeta0=0.40,
            rho_lock=0.734296875,
            neighbor_atten=0.9495,
            tau_hint=0.40,
            mu_hint=0.002,
        )
        """
        # push the theory into the runtime args
        args.cR = 12.0
        args.L = 4
        args.rho = params["rho"]  # dynamic, complexity-aware
        args.zeta0 = params["zeta0"]
        args.sigma_up = params["sigma_up"]
        args.neighbor_atten = params["neighbor_atten"]
        args.score_norm_alpha = params["score_norm_alpha"]
        args.bias_weight = params["bias_weight"]
        args.couple = 1
        args.d = 6
        args.mode = "unsat_hadamard"
        # spectral iterations & polish from log / scaling laws
        args.power_iters = int(max(20, 10 + 10 * max(0.0, math.log2(max(4, C)))))
        args.polish = int(max(50_000, 250_000 * (max(1.0, C / 200_000.0)) ** (2.0 / 3.0)))
        print("\n[theory] optimal pack:", params)

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

    if args.dump_assign:
        with open(args.dump_assign, "w") as f:
            f.write("".join("1" if b else "0" for b in assign01))
        print(f"Wrote assignment to: {args.dump_assign}")

    if args.check:
        ok = check_sat(clauses, assign01)
        print(f"Model SAT check   : {'PASS' if ok else 'FAIL'}")

    # exit code: 0 if SAT, 1 otherwise
    sys.exit(0 if unsat == 0 else 1)


if __name__ == "__main__":
    main()
