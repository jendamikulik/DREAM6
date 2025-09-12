#!/usr/bin/env python3
# python DREAM6.py --mode unsat --report_s2 true --C 1000 --cR 10 --sigma_up 0.045 --rho_lock 0.734296875 --zeta0 0.4 --neighbor_atten 0.9495 --seed 42 --couple 1
# python DREAM6.py --mode unsat --report_s2 ture --C 200 --cR 10 --sigma_up 0.045 --rho_lock 0.734296875 --zeta0 0.4 --neighbor_atten 0.9495 --seed 42 --couple 1
# python DREAM6.py --mode unsat --report_s2 true --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --neighbor_atten 0.75 --seed 42 --couple 1
# python DREAM6.py --mode unsat --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --neighbor_atten 0.75 --seed 42 --couple 1
# python DREAM6.py --mode unsat_hadamard --report_s2 true --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --neighbor_atten 0.75 --seed 42 --couple 1
# python DREAM6.py --mode sat --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --neighbor_atten 0.75 --seed 42 --couple 1
# python DREAM6.py --mode theory --C 1000
# python DREAM6.py --mode report_s2 --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --neighbor_atten 0.9495 --seed 42 --couple 1
# python DREAM6.py --mode sweep
# Deterministic A2/A3, theory bands, SAT/UNSAT samples, S2 report + UNSAT noise & neighbor attenuation wiring.
# Nekonzistence $ d = 6 $ v definici wiring_neighbors_circulant vs. $ d = 4 $ v run_sample není v diagramu zřejmá, což může být přehlížení.

import argparse, math, numpy as np
from numpy.linalg import eigh

# ---------- Theory bands ----------
def predictors(eps_lock=0.01, rho_lock=0.60, zeta0=0.30, sigma_up=0.10):
    alpha = (1.0 - eps_lock)**2
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    beta = (1.0 - gamma0)**2
    gamma_spec = 0.5 * (alpha + beta)
    delta_spec = 0.5 * (alpha - beta)
    return dict(alpha=alpha, beta=beta, gamma_spec=gamma_spec,
                delta_spec=delta_spec, gamma0=gamma0)

# ---------- Sigma proxy (Matrix Bernstein shape) ----------
def sigma_proxy(C, cR=15.0, L=3, eta_power=3, C_B=1.0):
    C = max(2, int(C))
    R = max(1, int(math.ceil(cR * math.log(C))))  # Zvýšeno na 15 pro lepší stabilitu
    T = R * L
    eta = C ** (-eta_power)
    sigma_up = C_B * math.sqrt(math.log(C / eta) / T)
    return sigma_up, R, T

# ---------- Deterministic A3 wiring (d-regular circulant, adjustable) ----------
def wiring_neighbors_circulant(C, d=4):  # Zvýšeno na d=6 pro lepší kontrolu cross-termů
    if d % 2 != 0 or d > C - 1:
        raise ValueError(f"d={d} must be even and < C-1 for circulant wiring.")
    s = 2
    while math.gcd(s, C) != 1:
        s += 1
    nbrs = []
    for i in range(C):
        nbr_set = set()
        for step in range(1, (d // 2) + 1):
            nbr_set.add((i - step) % C)
            nbr_set.add((i + step) % C)
        nbrs.append(nbr_set)
    return nbrs  # list[set[int]]

# ---------- Deterministic A2 schedule (UNSAT model) + noise + enhanced neighbor attenuation ----------
def schedule_unsat_det(
    C, R, rho_lock=0.50, zeta0=0.40, L=3,
    sigma_up=0.10, neighbor_atten=0.80, seed=42, use_coupling=True
):
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(round(rho_lock * T))
    k = int(round(zeta0 * m))

    offsets = [(j * T) // C for j in range(C)]
    lock_idx = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]
    Phi = np.full((T, C), np.pi, dtype=float)

    for j in range(C):
        li = lock_idx[j]
        pi_slots = li[:k]
        zero_slots = li[k:m]
        Phi[pi_slots, j] = np.pi
        if zero_slots.size > 0:
            Phi[zero_slots, j] = rng.normal(loc=0.0, scale=sigma_up, size=zero_slots.size)

    if use_coupling and (abs(neighbor_atten - 1.0) > 1e-12):
        neighbors = wiring_neighbors_circulant(C, d=4)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        bands = predictors(eps_lock=0.01, rho_lock=rho_lock, zeta0=zeta0, sigma_up=sigma_up)
        kappa = max(0.0, (1.0 - rho_lock * zeta0)**2 - 0.5 * sigma_up)

        for j in range(C):
            o_j = offsets[j]
            for j_adj in neighbors[j]:
                if j_adj == j:
                    continue
                o_adj = offsets[j_adj]
                overlap = set()
                for t in range(m):
                    t_j = (o_j + t) % T
                    for t_adj in range(m):
                        t_adj_full = (o_adj + t_adj) % T
                        if t_j == t_adj_full and t_j in lock_sets[j] and t_adj_full in lock_sets[j_adj]:
                            overlap.add(t_j)
                if overlap:
                    idx = np.fromiter(overlap, dtype=int)
                    overlap_size = len(overlap)
                    overlap_fraction = overlap_size / m
                    cross_term_weight = min(1.0, (len(neighbors[j]) * kappa) / (C * (1 - bands['gamma0'])**2) * (1 + 3.0 * overlap_fraction))
                    attenuation = max(0.70, neighbor_atten - 0.0500 * overlap_size / (m * (1 + 0.250 * np.sqrt(np.log(C)) * overlap_fraction)) * (1 - cross_term_weight))
                    Phi[idx, j_adj] *= attenuation

                    print(f"Clause {j} vs {j_adj}: overlap_size={overlap_size}, attenuation={attenuation:.4f}, cross_term_weight={cross_term_weight:.4f}")

    return Phi

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H],[H, -H]])
    return H
"""
def schedule_unsat_hadamard(C,R,rho_lock=0.5,zeta0=0.4,L=3,seed=42):
    rng = np.random.default_rng(seed)
    T = R*L; m = int(round(rho_lock*T)); k = int(round(zeta0*m))
    # offsets s krokem koprimárním s T (de-aliased)
    s = max(1, T//C)
    while math.gcd(s, T) != 1: s += 1
    offsets = [(j*s) % T for j in range(C)]
    lock = [np.array([(offsets[j]+t)%T for t in range(m)], int) for j in range(C)]
    Phi = np.full((T,C), np.pi, float)
    Hlen = 1
    while Hlen < m: Hlen <<= 1
    H = hadamard(Hlen); step_rows = 2*int(Hlen//max(1,C)) + 1  # liché, coprime s Hlen
    for j in range(C):
        row = H[(j*step_rows) % Hlen, :m]
        neg = np.flatnonzero(row < 0)
        if len(neg) < k:
            extra = rng.choice(np.setdiff1d(np.arange(m), neg), size=k-len(neg), replace=False)
            mask_pi = np.concatenate([neg, extra])
        else:
            mask_pi = neg[:k]
        mask_0 = np.setdiff1d(np.arange(m), mask_pi)
        slots = lock[j]
        Phi[slots[mask_pi], j] = np.pi
        Phi[slots[mask_0], j]  = 0.0
    #return Phi
    return (Phi, lock)
"""

def schedule_unsat_hadamard(
    C, R, rho_lock=0.5, zeta0=0.4, L=3, seed=42,
    return_locks=True, s_stride=None, row_step=None, col_stride=None, col_offset=0
):
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(round(rho_lock * T))
    k = int(round(zeta0 * m))

    # --- OFFSETS: stride blízko T/2 a koprimární s T (menší overlap sousedů) ---
    if s_stride is None:
        s = max(1, T // 2 - 1)
        while math.gcd(s, T) != 1:
            s -= 1
            if s <= 0:
                s = 1
                break
    else:
        s = int(s_stride)
        if math.gcd(s, T) != 1:
            raise ValueError("s_stride must be coprime with T")
    offsets = [(j * s) % T for j in range(C)]
    locks = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]

    # --- Hadamard masky s nízkou korelací i po truncaci na m ---
    Phi = np.full((T, C), np.pi, dtype=float)
    Hlen = 1
    while Hlen < m:
        Hlen <<= 1
    H = hadamard(Hlen)  # ±1

    # řádky: velký lichý krok, koprimární s Hlen
    #if row_step is None:
    #    row_step = (Hlen // 2) + 1
    #if math.gcd(row_step, Hlen) != 1:
    #    # posun na nejbližší liché koprimární
    #    row_step = (row_step | 1)
    #    while math.gcd(row_step, Hlen) != 1:
    #        row_step += 2
    row_step = (Hlen // 2) + 1
    if math.gcd(row_step, Hlen) != 1:
        row_step |= 1
        while math.gcd(row_step, Hlen) != 1:
            row_step += 2

    # sloupce: rozprostři m indexů s lichým stride g koprimárním s Hlen
    #if col_stride is None:
    #    g = (Hlen // 3) | 1
    #else:
    #    g = int(col_stride) | 1
    #while math.gcd(g, Hlen) != 1:
    #    g += 2
    #cols = (int(col_offset) + g * np.arange(m)) % Hlen
    g = (Hlen // 3) | 1
    while math.gcd(g, Hlen) != 1:
        g += 2
    cols = (0 + g * np.arange(m)) % Hlen

    #for j in range(C):
    #    row = H[(j * row_step) % Hlen, cols]
    #    neg = np.flatnonzero(row < 0.0)
    #    if len(neg) >= k:
    #        mask_pi = rng.choice(neg, size=k, replace=False)
    #    else:
    #        extra = rng.choice(np.setdiff1d(np.arange(m), neg), size=k - len(neg), replace=False)
    #        mask_pi = np.concatenate([neg, extra])
    for j in range(C):
        row = H[(j * row_step) % Hlen, cols]
        neg = np.flatnonzero(row < 0.0)
        if len(neg) >= k:
            mask_pi = rng.choice(neg, size=k, replace=False)  # náhodně z negativních
        else:
            extra = rng.choice(np.setdiff1d(np.arange(m), neg), size=k - len(neg), replace=False)
            mask_pi = np.concatenate([neg, extra])

        mask_0 = np.setdiff1d(np.arange(m), mask_pi)
        slots = locks[j]
        Phi[slots[mask_pi], j] = np.pi
        Phi[slots[mask_0], j] = 0.0

    return (Phi, locks) if return_locks else Phi



# ---------- SAT "envelope" schedule ----------
def schedule_sat_aligned(C, R, L=3):
    T = R * L
    Phi = np.zeros((T, C), dtype=float)  # Synchronizované fáze pro SAT
    return Phi

# ---------- Gram & mu ----------
#def gram_from_phases(Phi):
#    Z = np.exp(1j * Phi)
#    G = np.abs(Z.conj().T @ Z) / Phi.shape[0]
#    G = 0.5 * (G + G.T)  # symmetrize
#    np.fill_diagonal(G, 1.0)  # set diag exactly 1
#    return G

def gram_from_phases(Phi):
    Z = np.exp(1j * Phi)
    G = (Z.conj().T @ Z) / Phi.shape[0]   # komplexní Hermitian, žádné |.| !
    G = 0.5 * (G + G.conj().T)           # numerická symetrizace
    np.fill_diagonal(G, 1.0 + 0j)
    return G
"""
def gram_lock_only(Phi, locks):
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    G = np.zeros((C, C), dtype=np.complex128)
    for i in range(C):
        Li = set(locks[i].tolist())
        for j in range(i, C):
            Lj = set(locks[j].tolist())
            inter = np.array(sorted(Li & Lj), dtype=int)
            val = 0.0 if inter.size == 0 else (Z[inter, i].conj() * Z[inter, j]).mean()
            G[i, j] = val; G[j, i] = np.conjugate(val)
    np.fill_diagonal(G, 1.0+0j)
    return G
"""

def gram_lock_only(Phi, locks):
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    m = len(locks[0])  # délka lock okna
    G = np.zeros((C, C), dtype=np.complex128)
    for i in range(C):
        Li = set(locks[i].tolist())
        for j in range(i, C):
            Lj = set(locks[j].tolist())
            inter = np.array(sorted(Li & Lj), dtype=int)
            if inter.size == 0:
                val = 0.0
            else:
                # POZOR: normalizuj přes m (ne přes |inter|)
                val = (Z[inter, i].conj() * Z[inter, j]).sum() / m
            G[i, j] = val; G[j, i] = np.conjugate(val)
    np.fill_diagonal(G, 1.0 + 0j)
    return G

#def mu_clause(G):
#    evals = eigh(G, UPLO='U')[0]
#    lam = float(evals[-1])
#    return lam / G.shape[0], lam

def mu_clause(G):
    evals = eigh(G)[0]  # funguje i pro komplexní Hermitian
    lam = float(evals[-1])
    return lam / G.shape[0], lam


# ---------- S2 helpers ----------
def kappa_bound(rho_lock, zeta0, sigma_up):
    base = (1.0 - rho_lock * zeta0)**2 - 0.5 * sigma_up
    return max(0.0, min(1.0, base))

def s2_metrics_from_G(G, neighbors):
    C = G.shape[0]
    edges_vals = []
    row_sums = np.zeros(C, dtype=float)
    for i in range(C):
        for j in neighbors[i]:
            if j <= i:  # count each edge once
                continue
            edges_vals.append(abs(G[i, j]))
        row_sums[i] = sum(abs(G[i, j]) for j in neighbors[i])
    edges_vals = np.array(edges_vals) if edges_vals else np.array([0.0])
    return dict(
        max_edge=float(np.max(edges_vals)),
        avg_edge=float(np.mean(edges_vals)),
        max_row_sum_neighbors=float(np.max(row_sums)),
        avg_row_sum_neighbors=float(np.mean(row_sums)),
        row_sums_neighbors=row_sums,
    )

# ---------- Run blocks ----------
def run_theory(C, cR, eps, rho, zeta0, sigma_up_opt, d):
    if sigma_up_opt is None:
        sigma_up, R, T = sigma_proxy(C, cR, L=3)
    else:
        sigma_up = float(sigma_up_opt)
        R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
        T = R * 3
    bands = predictors(eps, rho, zeta0, sigma_up)
    kap = kappa_bound(rho, zeta0, sigma_up)
    print(f"Constants: eps_lock={eps}, rho_lock={rho}, zeta0={zeta0}, "
          f"sigma_up={sigma_up:.4f}, R≈{cR} log C, L=3")
    print(f"C={C}, R={R}, T={T}")
    print(f"Predicted: mu_SAT={bands['alpha']:.4f}, mu_UNSAT={bands['beta']:.4f}, "
          f"Delta_spec={bands['delta_spec']:.4f}, gamma0={bands['gamma0']:.4f}")
    print(f"S2 helper (bound): kappa≤{kap:.4f}; with d={len(wiring_neighbors_circulant(C, d=d)[0])}, d*kappa≤{len(wiring_neighbors_circulant(C, d=d)[0])*kap:.4f}")

    return R, T, sigma_up, bands, kap



def run_sample(mode, C, cR, eps, rho, zeta0, sigma_up_opt, neighbor_atten, seed, couple, report_s2, d):

    R, T, sigma_up, bands, kap = run_theory(C, cR, eps, rho, zeta0, sigma_up_opt, d)
    neighbors = wiring_neighbors_circulant(C, d=d)

    if mode == "unsat_hadamard":
        Phi, locks = schedule_unsat_hadamard(C, R, rho, zeta0, L=3, seed=seed)
    elif mode == "unsat":
        Phi = schedule_unsat_det(C, R, rho, zeta0, L=3, sigma_up=sigma_up, neighbor_atten=neighbor_atten, seed=seed, use_coupling=bool(couple))
    elif mode == "sat":
        Phi = schedule_sat_aligned(C, R, L=3)
    else:
        raise ValueError("mode must be one of: 'unsat_hadamard', 'unsat', 'sat'")

    G = gram_from_phases(Phi)
    mu, lam = mu_clause(G)
    conc_proxy = float(np.max(np.abs(G - G.mean())) / C)

    if mode == "unsat":
        print(f"Sanity (μ ≤ β?): μ={mu:.4f} vs β={bands['beta']:.4f}, conc_error_proxy={conc_proxy:.4f}")
    elif mode == "unsat_hadamard":
        print(f"UNSAT-Hadamard: μ={mu:.4f} (λmax={lam:.1f}), β(theory)={bands['beta']:.4f}, conc_error_proxy={conc_proxy:.4f}")
    elif mode == "sat":
        print(f"SAT-envelope: μ={mu:.4f}, α(theory)={bands['alpha']:.4f}, conc_error_proxy={conc_proxy:.4f}")

    if report_s2 == True:
        G_all = gram_from_phases(Phi)  # pro μ/λmax
        if mode == "unsat_hadamard":
            G_lock = gram_lock_only(Phi, locks)  # pro S2 empirical (lock-only)
        m_all = s2_metrics_from_G(G_all, neighbors)
        if mode == "unsat_hadamard":
            m_lock = s2_metrics_from_G(G_lock, neighbors)
        print("S2 empirical (all T): ")
        print(f"  |G_ij| over edges: max={m_all['max_edge']:.4f}, avg={m_all['avg_edge']:.4f}")
        print(f"  row-sum over neighbors: max={m_all['max_row_sum_neighbors']:.4f}, avg={m_all['avg_row_sum_neighbors']:.4f}")
        if mode == "unsat_hadamard":
            print("S2 empirical (lock-only): ")
            print(f"  |G_ij| over edges: max={m_lock['max_edge']:.4f}, avg={m_lock['avg_edge']:.4f}")
            print(f"  row-sum over neighbors: max={m_lock['max_row_sum_neighbors']:.4f}, avg={m_lock['avg_row_sum_neighbors']:.4f}")

        #m = s2_metrics_from_G(G, neighbors)
        #print("S2 empirical on neighbors:")
        #print(f"  |G_ij| over edges: max={m['max_edge']:.4f}, avg={m['avg_edge']:.4f}")
        #print(f"  row-sum over neighbors: max={m['max_row_sum_neighbors']:.4f}, avg={m['avg_row_sum_neighbors']:.4f}")


    return mu

def run_sweep(C_list, cR, eps, rho, zeta0, sigma_up_opt, d):
    print("C,R,T,sigma_up,alpha,beta,Delta_spec,gamma0,kappa_bound,d*kappa")
    for C in C_list:
        R, T, sig, bands, kap = run_theory(C, cR, eps, rho, zeta0, sigma_up_opt, d)
        d = len(wiring_neighbors_circulant(C, d=4)[0])
        print(f"{C},{R},{T},{sig:.5f},{bands['alpha']:.5f},"
              f"{bands['beta']:.5f},{bands['delta_spec']:.5f},"
              f"{bands['gamma0']:.5f},{kap:.5f},{d*kap:.5f}")

# ---------- CLI ----------
def main():

    #--sigma_up 0.045 --rho_lock 0.734296875 --zeta0 0.4 --neighbor_atten 0.9495

    ap = argparse.ArgumentParser(description="ECC v6 deterministic (A2/A3) + theory, samples")
    ap.add_argument("--mode", choices=["theory", "unsat", "unsat_hadamard", "sat", "sweep"], default="theory")
    ap.add_argument("--report_s2", type=bool, default=True, help='S2 report')
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--C_list", type=str, default="", help="comma-separated list for sweep")
    ap.add_argument("--cR", type=float, default=15.0)
    ap.add_argument("--eps_lock", type=float, default=0.01)
    ap.add_argument("--rho_lock", type=float, default=0.734296875)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--sigma_up", type=float, default=0.045, help="if omitted, computed via sigma_proxy (also used as noise std)")
    ap.add_argument("--neighbor_atten", type=float, default=0.9495, help="phase attenuation on lock-overlap for neighbors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--couple", type=int, default=1, help="1: enable neighbor attenuation; 0: disable")
    ap.add_argument("--d", type=int, default=4)

    args = ap.parse_args()

    if args.mode == "theory":
        run_theory(args.C, args.cR, args.eps_lock, args.rho_lock, args.zeta0, args.sigma_up, args.d)

    elif args.mode in ("unsat", "sat", "unsat_hadamard"):
        run_sample(args.mode, args.C, args.cR, args.eps_lock, args.rho_lock, args.zeta0,
                   args.sigma_up, args.neighbor_atten, args.seed, args.couple, args.report_s2, args.d)

    #elif args.mode == "report_s2":
    #    run_sample("unsat", args.C, args.cR, args.eps_lock, args.rho_lock, args.zeta0,
    #               args.sigma_up, args.neighbor_atten, args.seed, args.couple, report_s2=True)

    else:
        C_list = [int(x.strip()) for x in args.C_list.split(",") if x.strip()]
        if not C_list:
            C_list = [10, 50, 100, 200, 1000]  # Rozšířený rozsah pro test
        run_sweep(C_list, args.cR, args.eps_lock, args.rho_lock, args.zeta0, args.sigma_up, args.d)

if __name__ == "__main__":
    main()