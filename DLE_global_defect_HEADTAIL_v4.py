#!/usr/bin/env python3
"""
DLE global-defect Bellman solver -- HEAD/TAIL v4
================================================

Exact finite-support lazy Bellman/Benders solver for the B8 dominating-leaf
extension problem.  Compared with DLE_global_defect_bellman.py, this version
adds a DREAM6-inspired HEAD/TAIL active-set architecture and v4 anti-thrashing/memoization:

  * HEAD: a small set of high-influence affine cuts inserted into each LP;
  * TAIL: every other valid cut, retained permanently and checked in one
          vectorized separation scan;
  * violated tail cuts are promoted monotonically inside an oracle call;
  * promoted cuts remain sticky across calls until a large hard-cap compaction,
    avoiding the v3 promote/demote thrashing seen with a fixed 48-cut head;
  * exact oracle values/supporting cuts are memoized by exact float64 state;
  * therefore HEAD/TAIL changes computation, not the feasible set or soundness.

It also uses single-child lazy recursive refinement: if certification of one
child discovers a new lower-level cut, the parent is immediately re-solved
instead of needlessly certifying the other four stale children.

For fixed support cutoff M, the global defect is

    d_0(b) = max(0, b_0-c_0, b_0+b_1-c_1),

    d_{r+1}(a)
      = min_{b^(0),...,b^(4) in Delta_M}
          sum_x P(x) d_r(b^(x))
          + lambda || Q*a - sum_x P(x) delta_x*b^(x) ||_1.

For every lambda > 0,

    d_r(a) = 0

iff a has an exact depth-r ZR continuation whose leaves dominate B8.

The code is exact with respect to the currently accumulated affine cut pools
up to LP/certification tolerances.  The HEAD/TAIL reduction is safe because a
restricted-head LP is accepted only after the candidate solution satisfies
EVERY cut in the full pool by vectorized separation.

This is still a finite-support computation.  M=8 must not be identified with
the unrestricted-support value without a separate support-cutoff audit.

Examples
--------
  python DLE_global_defect_HEADTAIL_v4.py --max-r 5
  python DLE_global_defect_HEADTAIL_v4.py --max-r 8 --max-seconds 280
  python DLE_global_defect_HEADTAIL_v4.py --max-r 8 --resume

Dependencies: numpy, scipy
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------

P = np.array([1/8, 1/4, 1/4, 1/4, 1/8], dtype=float)
Q = np.array([7/64, 5/16, 5/32, 5/16, 7/64], dtype=float)

B8_WEIGHTS = np.array(
    [
        0.53562333995590705,
        0.42450840479209673,
        0.039868255251995686,
    ],
    dtype=float,
)
B8_MEAN = 0.50424491529608806

KNOWN = {
    1: 0.5626050126445069,
    2: 0.58954728,
    3: 0.6017042293187882,
    4: 0.60760931,
    5: 0.61096453,
}


class DeadlineExceeded(RuntimeError):
    pass


@dataclass
class Cut:
    g: np.ndarray
    h: float


class GlobalDefectHeadTail:
    def __init__(
        self,
        M: int = 8,
        lam: float = 1.0,
        lp_tol: float = 1e-9,
        dedupe_tol: float = 1e-10,
        sep_tol: float = 1e-9,
        active_tol: float = 1e-7,
        head_limit: int = 96,
        promote_batch: int = 16,
        headtail_threshold: int = 2000,
        head_soft_cap: int = 384,
        head_hard_cap: int = 1024,
        cache_limit: int = 20000,
        verbose: bool = True,
    ):
        if M < 2:
            raise ValueError("M must be at least 2")
        if lam <= 0:
            raise ValueError("lambda must be strictly positive")
        if head_limit < 3:
            raise ValueError("head_limit must be at least 3")
        if promote_batch < 1:
            raise ValueError("promote_batch must be positive")
        if head_soft_cap < head_limit:
            raise ValueError("head_soft_cap must be >= head_limit")
        if head_hard_cap < head_soft_cap:
            raise ValueError("head_hard_cap must be >= head_soft_cap")
        if cache_limit < 0:
            raise ValueError("cache_limit must be nonnegative")

        self.M = int(M)
        self.N = self.M + 1
        self.L = self.M + 5
        self.lam = float(lam)
        self.lp_tol = float(lp_tol)
        self.dedupe_tol = float(dedupe_tol)
        self.sep_tol = float(sep_tol)
        self.active_tol = float(active_tol)
        self.head_limit = int(head_limit)
        self.promote_batch = int(promote_batch)
        self.headtail_threshold = int(headtail_threshold)
        self.head_soft_cap = int(head_soft_cap)
        self.head_hard_cap = int(head_hard_cap)
        self.cache_limit = int(cache_limit)
        self.verbose = bool(verbose)

        self.P = P.copy()
        self.Q = Q.copy()

        # C a = Q*a on support 0,...,M+4.
        self.C = np.zeros((self.L, self.N), dtype=float)
        for u in range(self.L):
            for j in range(self.N):
                k = u - j
                if 0 <= k <= 4:
                    self.C[u, j] = self.Q[k]

        # S[x] b = delta_x*b on support 0,...,M+4.
        self.S = np.zeros((5, self.L, self.N), dtype=float)
        for x in range(5):
            for j in range(self.N):
                self.S[x, j + x, j] = 1.0

        self.c0 = float(B8_WEIGHTS[0])
        self.c1 = float(B8_WEIGHTS[0] + B8_WEIGHTS[1])

        z = np.zeros(self.N)
        e0 = np.zeros(self.N); e0[0] = 1.0
        e01 = np.zeros(self.N); e01[:2] = 1.0

        self.pools: Dict[int, List[Cut]] = {
            0: [Cut(z.copy(), 0.0), Cut(e0, self.c0), Cut(e01, self.c1)]
        }

        # Stable cut indices are important for HEAD/TAIL bookkeeping.
        # We therefore never physically delete an existing cut.
        self.heads: Dict[int, set[int]] = {0: {0, 1, 2}}
        self.cut_hits: Dict[int, List[float]] = {0: [1.0, 1.0, 1.0]}
        self.cut_birth: Dict[int, List[int]] = {0: [0, 0, 0]}
        self.clock = 0
        self.pool_version: Dict[int, int] = {0: 0}

        # Dense matrices are cached only for vectorized tail scans.
        self._matrix_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.solved: Dict[int, dict] = {}
        self.stats = {
            "eval_calls": {},
            "primal_solves": {},
            "dual_solves": {},
            "root_solves": {},
            "cuts_added": {},
            "tail_scans": {},
            "promotions": {},
            "stale_child_aborts": {},
            "head_compactions": {},
            "cache_hits": {},
            "cache_misses": {},
            "cache_reinstalls": {},
        }

        # Exact-state memoization.  A cache entry stores not only the certified
        # value but also a supporting cut active at that exact state.  On a
        # cache hit the cut can be re-installed if later pool compaction removed
        # it.  Keys are exact float64 bytes: no approximate-state substitution.
        self.exact_cache: Dict[int, OrderedDict[bytes, tuple]] = {}
        self.deadline: Optional[float] = None

    # ------------------------------------------------------------------
    # Utilities / cut pool
    # ------------------------------------------------------------------

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise DeadlineExceeded("Time budget exhausted")

    def _inc(self, key: str, r: int, amount: int = 1) -> None:
        d = self.stats[key]
        d[r] = d.get(r, 0) + amount

    def ensure_pool(self, r: int) -> List[Cut]:
        if r not in self.pools or not self.pools[r]:
            self.pools[r] = [Cut(np.zeros(self.N), 0.0)]
            self.heads[r] = {0}
            self.cut_hits[r] = [1.0]
            self.cut_birth[r] = [self.clock]
            self.pool_version[r] = self.pool_version.get(r, 0) + 1
            self._matrix_cache.pop(r, None)
        else:
            self.heads.setdefault(r, {0})
            self.cut_hits.setdefault(r, [0.0] * len(self.pools[r]))
            self.cut_birth.setdefault(r, [0] * len(self.pools[r]))
            self.pool_version.setdefault(r, 0)
            while len(self.cut_hits[r]) < len(self.pools[r]):
                self.cut_hits[r].append(0.0)
                self.cut_birth[r].append(self.clock)
        return self.pools[r]

    def _pool_arrays(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        self.ensure_pool(r)
        cached = self._matrix_cache.get(r)
        if cached is not None and cached[0].shape[0] == len(self.pools[r]):
            return cached
        G = np.vstack([c.g for c in self.pools[r]])
        h = np.asarray([c.h for c in self.pools[r]], dtype=float)
        self._matrix_cache[r] = (G, h)
        return G, h

    def _canonicalize(self, g: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        # Gauge exact on simplex: (g+c1)^T a-(h+c)=g^T a-h.
        g = np.asarray(g, dtype=float).copy()
        c = float(np.mean(g))
        return g - c, float(h - c)

    def _cache_key(self, a: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        return arr.tobytes()

    def _cache_store(self, r: int, a: np.ndarray, value: float, g: np.ndarray, h: float) -> None:
        if self.cache_limit <= 0:
            return
        cache = self.exact_cache.setdefault(r, OrderedDict())
        key = self._cache_key(a)
        gc, hc = self._canonicalize(g, h)
        cache[key] = (float(value), gc.copy(), float(hc))
        cache.move_to_end(key)
        while len(cache) > self.cache_limit:
            cache.popitem(last=False)

    def _cache_lookup(self, r: int, a: np.ndarray, cert_tol: float) -> Optional[float]:
        cache = self.exact_cache.get(r)
        if not cache:
            self._inc("cache_misses", r)
            return None
        key = self._cache_key(a)
        entry = cache.get(key)
        if entry is None:
            self._inc("cache_misses", r)
            return None

        value, g, h = entry
        cache.move_to_end(key)
        self._inc("cache_hits", r)

        # If the current pool no longer reaches the cached exact value at this
        # point, restore the cached supporting cut.  This is safe and ensures
        # an upper-level stale-child detector sees a pool_version change.
        G, hh = self._pool_arrays(r)
        env = float(np.max(G @ np.asarray(a, dtype=float) - hh))
        if value - env > max(cert_tol, 10.0 * self.lp_tol):
            if self.add_cut(r, g, h):
                self._inc("cache_reinstalls", r)
        return float(value)

    def add_cut(self, r: int, g: np.ndarray, h: float) -> bool:
        """
        Add a valid cut and safely remove globally dominated old cuts.

        Pairwise domination is exact on the simplex because an affine
        difference attains its maximum at a simplex vertex.  Whenever the
        pool is compacted, head indices are remapped and pool_version changes;
        upper Bellman levels therefore know that their current child LP became
        stale even if the pool length happens to remain unchanged.
        """
        pool = self.ensure_pool(r)
        g, h = self._canonicalize(g, h)

        # Duplicate / new cut dominated by an existing cut.
        for cut in pool:
            if max(np.max(np.abs(cut.g - g)), abs(cut.h - h)) <= self.dedupe_tol:
                return False
            gap = np.max(g - cut.g) - (h - cut.h)
            if gap <= self.dedupe_tol:
                return False

        old_head = set(self.heads.get(r, {0}))
        old_hits = self.cut_hits[r]
        old_birth = self.cut_birth[r]

        keep_old = [0]
        for i, cut in enumerate(pool[1:], start=1):
            # Existing cut dominated everywhere by the new cut?
            gap = np.max(cut.g - g) - (cut.h - h)
            if gap > self.dedupe_tol:
                keep_old.append(i)

        new_pool = [pool[i] for i in keep_old]
        new_hits = [old_hits[i] for i in keep_old]
        new_birth = [old_birth[i] for i in keep_old]
        remap = {old_i: new_i for new_i, old_i in enumerate(keep_old)}
        new_head = {remap[i] for i in old_head if i in remap}
        new_head.add(0)

        self.clock += 1
        new_idx = len(new_pool)
        new_pool.append(Cut(g.copy(), float(h)))
        new_hits.append(1.0)
        new_birth.append(self.clock)
        new_head.add(new_idx)

        self.pools[r] = new_pool
        self.cut_hits[r] = new_hits
        self.cut_birth[r] = new_birth
        self.heads[r] = new_head
        self.pool_version[r] = self.pool_version.get(r, 0) + 1
        self._matrix_cache.pop(r, None)
        self._inc("cuts_added", r)
        return True

    def d0(self, a: np.ndarray) -> float:
        return max(0.0, float(a[0] - self.c0), float(a[0] + a[1] - self.c1))

    def _trim_head(self, r: int, force: bool = False) -> None:
        """
        Compact only a VERY large persistent head.

        v3 trimmed back to a tiny fixed head after every successful oracle
        call, which caused the same cuts to be promoted thousands of times.
        v4 keeps promoted cuts sticky across calls.  During an individual
        primal/root separation loop the working set is strictly monotone.

        Normal operation compacts only when head_hard_cap is exceeded; a forced
        compaction (used only on explicit request/load hygiene) targets
        head_soft_cap.  Full-pool separation still makes this performance-only.
        """
        self.ensure_pool(r)
        if r == 0:
            self.heads[r] = set(range(len(self.pools[r])))
            return

        H = set(self.heads.get(r, {0}))
        H.add(0)
        trigger = self.head_soft_cap if force else self.head_hard_cap
        if len(H) <= trigger:
            self.heads[r] = H
            return

        target = min(self.head_soft_cap, len(self.pools[r]))
        candidates = [i for i in H if i != 0]
        hits = self.cut_hits[r]
        birth = self.cut_birth[r]

        n_keep = max(0, target - 1)
        # Influence first, recency as tie-breaker.
        ranked = sorted(candidates, key=lambda i: (hits[i], birth[i]), reverse=True)
        keep = set(ranked[:n_keep])
        self.heads[r] = {0} | keep
        self._inc("head_compactions", r)

    def _mark_influence(self, r: int, values: np.ndarray) -> None:
        """
        values shape [K,npoints].  Reward cuts lying within active_tol of the
        upper envelope at any inspected point.
        """
        if values.size == 0:
            return
        env = np.max(values, axis=0, keepdims=True)
        near = np.any(values >= env - self.active_tol, axis=1)
        for i in np.flatnonzero(near):
            self.cut_hits[r][int(i)] += 1.0

    def _promote(self, r: int, indices: Sequence[int]) -> int:
        H = self.heads.setdefault(r, {0})
        before = len(H)
        for i in indices:
            H.add(int(i))
            self.cut_hits[r][int(i)] += 1.0
        added = len(H) - before
        if added:
            self._inc("promotions", r, added)
        return added

    # ------------------------------------------------------------------
    # Restricted-head primal + exact full-tail separation
    # ------------------------------------------------------------------

    def _solve_primal_with_indices(
        self, r: int, a: np.ndarray, indices: Sequence[int]
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        pool = self.ensure_pool(r - 1)
        cuts = [pool[i] for i in indices]
        self._inc("primal_solves", r)

        N, L = self.N, self.L
        off_t = 5 * N
        off_ep = off_t + 5
        off_em = off_ep + L
        nv = off_em + L

        c = np.zeros(nv)
        c[off_t:off_t + 5] = self.P
        c[off_ep:off_ep + L] = self.lam
        c[off_em:off_em + L] = self.lam

        Aeq = []
        beq = []

        for x in range(5):
            row = np.zeros(nv)
            row[x * N:(x + 1) * N] = 1.0
            Aeq.append(row); beq.append(1.0)

        Ca = self.C @ a
        for u in range(L):
            row = np.zeros(nv)
            for x in range(5):
                row[x * N:(x + 1) * N] = self.P[x] * self.S[x, u]
            row[off_ep + u] = 1.0
            row[off_em + u] = -1.0
            Aeq.append(row); beq.append(Ca[u])

        Aub = []
        bub = []
        for x in range(5):
            for cut in cuts:
                row = np.zeros(nv)
                row[x * N:(x + 1) * N] = cut.g
                row[off_t + x] = -1.0
                Aub.append(row); bub.append(cut.h)

        bounds = (
            [(0.0, None)] * (5 * N)
            + [(None, None)] * 5
            + [(0.0, None)] * (2 * L)
        )
        options = {
            "dual_feasibility_tolerance": self.lp_tol,
            "primal_feasibility_tolerance": self.lp_tol,
        }

        res = linprog(
            c,
            A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
            A_eq=np.asarray(Aeq), b_eq=np.asarray(beq),
            bounds=bounds, method="highs", options=options,
        )
        if not res.success:
            raise RuntimeError(f"Bellman primal failed at r={r}: {res.message}")

        children = np.vstack([res.x[x*N:(x+1)*N] for x in range(5)])
        t = res.x[off_t:off_t + 5]
        ep = res.x[off_ep:off_ep + L]
        em = res.x[off_em:off_em + L]
        residual_cost = float(self.lam * np.sum(ep + em))
        return float(res.fun), children, t, residual_cost

    def primal_headtail(
        self, r: int, a: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, float, Tuple[int, ...]]:
        """
        Solve on a small head; scan ALL cuts in one matrix multiplication.
        Promote violated tail cuts until the restricted solution is feasible for
        the full current cut pool.  At return, lb is exactly the full-pool LP
        optimum up to sep_tol/LP tolerances.
        """
        level = r - 1
        self.ensure_pool(level)

        # For small pools, HEAD/TAIL bookkeeping costs more than it saves.
        # Use the full pool directly and activate separation only after the
        # pool crosses the configured threshold.
        if len(self.pools[level]) <= self.headtail_threshold:
            indices = tuple(range(len(self.pools[level])))
            lb, children, t, residual_cost = self._solve_primal_with_indices(r, a, indices)
            G, h = self._pool_arrays(level)
            self._mark_influence(level, G @ children.T - h[:, None])
            return lb, children, t, residual_cost, indices

        self._trim_head(level)
        local_head = set(self.heads[level])
        local_head.add(0)

        while True:
            self.check_deadline()
            indices = tuple(sorted(local_head))
            lb, children, t, residual_cost = self._solve_primal_with_indices(r, a, indices)

            G, h = self._pool_arrays(level)
            vals = G @ children.T - h[:, None]  # [K,5]
            self._inc("tail_scans", level)

            violation = vals - t[None, :]
            worst_by_cut = np.max(violation, axis=1)
            if indices:
                worst_by_cut[np.asarray(indices, dtype=int)] = -np.inf

            bad = np.flatnonzero(worst_by_cut > self.sep_tol)
            if bad.size == 0:
                self._mark_influence(level, vals)
                self.heads[level] = set(local_head)
                self._trim_head(level)
                return lb, children, t, residual_cost, indices

            order = bad[np.argsort(worst_by_cut[bad])[::-1]]
            promote = order[:self.promote_batch]
            for i in promote:
                local_head.add(int(i))
            self._promote(level, promote)

    # ------------------------------------------------------------------
    # Fenchel dual using exactly the certified restricted head
    # ------------------------------------------------------------------

    def dual_cut(
        self, r: int, a: np.ndarray, indices: Sequence[int]
    ) -> Tuple[float, np.ndarray, float]:
        self.check_deadline()
        pool_all = self.ensure_pool(r - 1)
        pool = [pool_all[i] for i in indices]
        K = len(pool)
        N, L = self.N, self.L
        self._inc("dual_solves", r)

        off_alpha = L
        off_mu = L + 5 * K
        nv = off_mu + 5

        c = np.zeros(nv)
        c[:L] = -(self.C @ a)
        for x in range(5):
            for i, cut in enumerate(pool):
                c[off_alpha + x*K + i] = self.P[x] * cut.h
            c[off_mu + x] = self.P[x]

        Aeq = []
        beq = []
        for x in range(5):
            row = np.zeros(nv)
            row[off_alpha + x*K:off_alpha + (x+1)*K] = 1.0
            Aeq.append(row); beq.append(1.0)

        Aub = []
        bub = []
        for x in range(5):
            for j in range(N):
                row = np.zeros(nv)
                row[:L] = self.S[x, :, j]
                for i, cut in enumerate(pool):
                    row[off_alpha + x*K + i] = -cut.g[j]
                row[off_mu + x] = -1.0
                Aub.append(row); bub.append(0.0)

        bounds = (
            [(-self.lam, self.lam)] * L
            + [(0.0, None)] * (5*K)
            + [(None, None)] * 5
        )
        options = {
            "dual_feasibility_tolerance": self.lp_tol,
            "primal_feasibility_tolerance": self.lp_tol,
        }
        res = linprog(
            c,
            A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
            A_eq=np.asarray(Aeq), b_eq=np.asarray(beq),
            bounds=bounds, method="highs", options=options,
        )
        if not res.success:
            raise RuntimeError(f"Bellman dual failed at r={r}: {res.message}")

        y = res.x[:L]
        g = self.C.T @ y
        hnew = 0.0
        for x in range(5):
            alpha = res.x[off_alpha + x*K:off_alpha + (x+1)*K]
            mu = res.x[off_mu + x]
            hnew += self.P[x] * (
                mu + sum(alpha[i] * pool[i].h for i in range(K))
            )

        return float(g @ a - hnew), g, float(hnew)

    # ------------------------------------------------------------------
    # Certified nested evaluation, with stale-child early abort
    # ------------------------------------------------------------------

    def exact_eval(
        self,
        r: int,
        a: np.ndarray,
        cert_tol: float = 1e-7,
        max_refine: int = 2000,
    ) -> float:
        self.check_deadline()
        self._inc("eval_calls", r)
        a = np.asarray(a, dtype=float)

        if r == 0:
            return self.d0(a)

        self.ensure_pool(r)
        cached = self._cache_lookup(r, a, cert_tol)
        if cached is not None:
            return cached

        for _ in range(max_refine):
            self.check_deadline()
            lb, children, t, _residual, head_indices = self.primal_headtail(r, a)

            # Most consequential children first.  If one adds a new lower-level
            # cut, all remaining child certifications are stale and skipped.
            priority = self.P * (np.maximum(t, 0.0) + 1e-12)
            order = list(np.argsort(-priority))

            stale = False
            child_values = np.full(5, np.nan)
            for x in order:
                self.check_deadline()
                self.ensure_pool(r - 1)
                v_before = self.pool_version.get(r - 1, 0)
                child_values[x] = self.exact_eval(r - 1, children[x], cert_tol, max_refine)
                v_after = self.pool_version.get(r - 1, 0)

                if v_after != v_before:
                    self._inc("stale_child_aborts", r)
                    stale = True
                    break

                if child_values[x] - t[x] > cert_tol:
                    # With no new cut this should be rare/numerical; force a
                    # parent re-solve rather than silently accepting it.
                    stale = True
                    break

            if stale:
                continue

            dual_value, g, h = self.dual_cut(r, a, head_indices)
            if abs(dual_value - lb) > max(cert_tol, 10.0*self.lp_tol):
                continue

            value = max(0.0, float(lb))
            self.add_cut(r, g, h)
            self._cache_store(r, a, value, g, h)
            return value

        raise RuntimeError(f"Could not certify d_{r}(a) after {max_refine} rounds")

    # ------------------------------------------------------------------
    # Root master with HEAD/TAIL separation
    # ------------------------------------------------------------------

    def _solve_root_with_indices(
        self, r: int, indices: Sequence[int]
    ) -> Tuple[float, np.ndarray]:
        pool = self.ensure_pool(r)
        cuts = [pool[i] for i in indices]
        self._inc("root_solves", r)

        c = np.arange(self.N, dtype=float)
        Aub = np.asarray([cut.g for cut in cuts])
        bub = np.asarray([cut.h for cut in cuts])
        options = {
            "dual_feasibility_tolerance": self.lp_tol,
            "primal_feasibility_tolerance": self.lp_tol,
        }
        res = linprog(
            c,
            A_ub=Aub, b_ub=bub,
            A_eq=np.ones((1, self.N)), b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * self.N,
            method="highs", options=options,
        )
        if not res.success:
            raise RuntimeError(f"Root master failed at r={r}: {res.message}")
        return float(res.fun), res.x

    def root_master_headtail(self, r: int) -> Tuple[float, np.ndarray]:
        self.ensure_pool(r)
        if len(self.pools[r]) <= self.headtail_threshold:
            indices = tuple(range(len(self.pools[r])))
            obj, a = self._solve_root_with_indices(r, indices)
            G, h = self._pool_arrays(r)
            self._mark_influence(r, (G @ a - h)[:, None])
            return obj, a

        self._trim_head(r)
        local_head = set(self.heads[r]); local_head.add(0)

        while True:
            self.check_deadline()
            indices = tuple(sorted(local_head))
            obj, a = self._solve_root_with_indices(r, indices)

            G, h = self._pool_arrays(r)
            vals = G @ a - h
            self._inc("tail_scans", r)

            tail_violation = vals.copy()
            tail_violation[np.asarray(indices, dtype=int)] = -np.inf
            bad = np.flatnonzero(tail_violation > self.sep_tol)
            if bad.size == 0:
                self._mark_influence(r, vals[:, None])
                self.heads[r] = set(local_head)
                self._trim_head(r)
                return obj, a

            order = bad[np.argsort(tail_violation[bad])[::-1]]
            promote = order[:self.promote_batch]
            for i in promote:
                local_head.add(int(i))
            self._promote(r, promote)

    def solve_D(
        self,
        r: int,
        cert_tol: float = 1e-7,
        max_outer: int = 500,
        checkpoint: Optional[Path] = None,
    ) -> Tuple[float, np.ndarray, int]:
        self.ensure_pool(r)

        for outer in range(max_outer):
            self.check_deadline()
            objective, a = self.root_master_headtail(r)
            defect = self.exact_eval(r, a, cert_tol=cert_tol)

            if self.verbose:
                pool_sizes = {k: len(v) for k, v in sorted(self.pools.items())}
                head_sizes = {k: len(self.heads.get(k, ())) for k in sorted(self.pools)}
                cache_sizes = {k: len(v) for k, v in sorted(self.exact_cache.items()) if v}
                print(
                    f"[r={r:2d} outer={outer:3d}] mean={objective:.12f} "
                    f"defect={defect:.3e} pools={pool_sizes} heads={head_sizes} "
                    f"cache={cache_sizes}",
                    flush=True,
                )

            if checkpoint is not None:
                self.save(checkpoint)

            if defect <= cert_tol:
                self.solved[r] = {
                    "value": float(objective),
                    "root": a.copy(),
                    "outer_rounds": outer + 1,
                    "cert_tol": float(cert_tol),
                }
                if checkpoint is not None:
                    self.save(checkpoint)
                return float(objective), a, outer + 1

        raise RuntimeError(f"Outer master did not converge for r={r}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "format": "DLE_HEADTAIL_v4",
            "M": self.M,
            "lam": self.lam,
            "lp_tol": self.lp_tol,
            "dedupe_tol": self.dedupe_tol,
            "sep_tol": self.sep_tol,
            "active_tol": self.active_tol,
            "head_limit": self.head_limit,
            "promote_batch": self.promote_batch,
            "headtail_threshold": self.headtail_threshold,
            "head_soft_cap": self.head_soft_cap,
            "head_hard_cap": self.head_hard_cap,
            "cache_limit": self.cache_limit,
            "pool_version": self.pool_version,
            "pools": {
                int(r): [(cut.g.copy(), float(cut.h)) for cut in pool]
                for r, pool in self.pools.items()
            },
            "heads": {int(r): sorted(map(int, H)) for r, H in self.heads.items()},
            "cut_hits": self.cut_hits,
            "cut_birth": self.cut_birth,
            "clock": self.clock,
            "solved": self.solved,
            "stats": self.stats,
            "exact_cache": {
                int(r): [(k, v, g.copy(), h) for k, (v, g, h) in cache.items()]
                for r, cache in self.exact_cache.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["M"]) != self.M:
            raise ValueError(f"checkpoint M={state['M']} != requested M={self.M}")
        if abs(float(state["lam"]) - self.lam) > 1e-15:
            raise ValueError(
                f"checkpoint lambda={state['lam']} != requested lambda={self.lam}"
            )

        self.pools = {}
        for r, cuts in state["pools"].items():
            self.pools[int(r)] = [
                Cut(np.asarray(g, dtype=float), float(h)) for g, h in cuts
            ]

        # Backward-compatible load from v1 checkpoints: create conservative
        # heads from zero + most recent cuts.
        heads_state = state.get("heads")
        self.heads = {}
        for r, pool in self.pools.items():
            if heads_state is not None and (r in heads_state or str(r) in heads_state):
                raw = heads_state.get(r, heads_state.get(str(r)))
                H = {int(i) for i in raw if 0 <= int(i) < len(pool)}
                self.heads[r] = H | {0}
            elif r == 0:
                self.heads[r] = set(range(len(pool)))
            else:
                lo = max(1, len(pool) - (self.head_limit - 1))
                self.heads[r] = {0} | set(range(lo, len(pool)))

        self.cut_hits = {}
        old_hits = state.get("cut_hits", {})
        self.cut_birth = {}
        old_birth = state.get("cut_birth", {})
        for r, pool in self.pools.items():
            rh = old_hits.get(r, old_hits.get(str(r), [0.0] * len(pool)))
            rb = old_birth.get(r, old_birth.get(str(r), list(range(len(pool)))))
            self.cut_hits[r] = list(map(float, rh[:len(pool)])) + [0.0] * max(0, len(pool)-len(rh))
            self.cut_birth[r] = list(map(int, rb[:len(pool)])) + [0] * max(0, len(pool)-len(rb))

        self.clock = int(state.get("clock", max((max(v) if v else 0 for v in self.cut_birth.values()), default=0)))
        pv = state.get("pool_version", {})
        self.pool_version = {
            int(r): int(pv.get(r, pv.get(str(r), 0))) for r in self.pools
        }
        self.solved = state.get("solved", {})

        old_stats = state.get("stats", {})
        for key in self.stats:
            self.stats[key] = old_stats.get(key, {})

        self.exact_cache = {}
        for r0, entries in state.get("exact_cache", {}).items():
            r = int(r0)
            cache = OrderedDict()
            for key, value, g, h in entries[-self.cache_limit:] if self.cache_limit > 0 else []:
                cache[bytes(key)] = (float(value), np.asarray(g, dtype=float), float(h))
            self.exact_cache[r] = cache

        self._matrix_cache = {}

        if 0 not in self.pools:
            z = np.zeros(self.N)
            e0 = np.zeros(self.N); e0[0] = 1.0
            e01 = np.zeros(self.N); e01[:2] = 1.0
            self.pools[0] = [Cut(z, 0.0), Cut(e0, self.c0), Cut(e01, self.c1)]
            self.heads[0] = {0,1,2}
            self.cut_hits[0] = [1.0,1.0,1.0]
            self.cut_birth[0] = [0,0,0]

        for r in list(self.pools):
            self._trim_head(r)

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            self.load_state_dict(pickle.load(f))


def format_root(a: np.ndarray, threshold: float = 1e-11) -> str:
    return "{" + ", ".join(
        f"{j}:{p:.16g}" for j, p in enumerate(a) if abs(p) > threshold
    ) + "}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--lp-tol", type=float, default=1e-9)
    ap.add_argument("--dedupe-tol", type=float, default=1e-10)
    ap.add_argument("--sep-tol", type=float, default=1e-9)
    ap.add_argument("--active-tol", type=float, default=1e-7)
    ap.add_argument("--head-limit", type=int, default=96)
    ap.add_argument("--promote-batch", type=int, default=16)
    ap.add_argument("--headtail-threshold", type=int, default=2000)
    ap.add_argument("--head-soft-cap", type=int, default=384)
    ap.add_argument("--head-hard-cap", type=int, default=1024)
    ap.add_argument("--cache-limit", type=int, default=20000)
    ap.add_argument("--max-r", type=int, default=8)
    ap.add_argument("--start-r", type=int, default=1)
    ap.add_argument("--max-outer", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument(
        "--checkpoint", type=Path,
        default=Path("dle_global_defect_HEADTAIL_v4_state.pkl")
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    solver = GlobalDefectHeadTail(
        M=args.M,
        lam=args.lam,
        lp_tol=args.lp_tol,
        dedupe_tol=args.dedupe_tol,
        sep_tol=args.sep_tol,
        active_tol=args.active_tol,
        head_limit=args.head_limit,
        promote_batch=args.promote_batch,
        headtail_threshold=args.headtail_threshold,
        head_soft_cap=args.head_soft_cap,
        head_hard_cap=args.head_hard_cap,
        cache_limit=args.cache_limit,
        verbose=not args.quiet,
    )

    if args.resume and args.checkpoint.exists():
        solver.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    if args.max_seconds is not None:
        solver.deadline = time.monotonic() + args.max_seconds

    print(
        f"M={solver.M}, lambda={solver.lam:g}, tol={args.tol:g}, "
        f"head_limit={solver.head_limit}, promote_batch={solver.promote_batch}, "
        f"headtail_threshold={solver.headtail_threshold}, "
        f"head_soft/hard={solver.head_soft_cap}/{solver.head_hard_cap}, "
        f"cache_limit={solver.cache_limit}, "
        f"B8 mean={B8_MEAN:.17g}",
        flush=True,
    )

    overall_start = time.time()
    try:
        for r in range(args.start_r, args.max_r + 1):
            if r in solver.solved:
                print(f"r={r}: already solved: {solver.solved[r]['value']:.12f}")
                continue

            t0 = time.time()
            value, root, rounds = solver.solve_D(
                r,
                cert_tol=args.tol,
                max_outer=args.max_outer,
                checkpoint=args.checkpoint,
            )
            elapsed = time.time() - t0
            premium = value - B8_MEAN
            known = KNOWN.get(r)

            print("\n" + "="*76)
            print(f"r={r}")
            print(f"D_r^(M)(B8) = {value:.15f}")
            print(f"premium       = {premium:.15f}")
            print(f"root          = {format_root(root)}")
            print(f"outer rounds  = {rounds}")
            print(f"elapsed       = {elapsed:.3f} s")
            print("pool sizes     = " + str({k:len(v) for k,v in sorted(solver.pools.items())}))
            print("head sizes     = " + str({k:len(solver.heads.get(k,())) for k in sorted(solver.pools)}))
            if known is not None:
                print(f"known         = {known:.15f}")
                print(f"difference    = {value-known:+.6e}")
            print("="*76 + "\n", flush=True)
            solver.save(args.checkpoint)

    except DeadlineExceeded:
        solver.save(args.checkpoint)
        print(f"\nTime budget exhausted. State saved to {args.checkpoint}.", flush=True)
        print("Resume with the same M/lambda using --resume.", flush=True)
    finally:
        print(f"Total wall time: {time.time()-overall_start:.3f} s", flush=True)
        print("Stats:", solver.stats, flush=True)


if __name__ == "__main__":
    main()
