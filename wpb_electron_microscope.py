#!/usr/bin/env python3
"""
WHO PAYS THE BILL? — ELECTRON-MICROSCOPE AUDIT

This file imports the executable sorted-arithmetic construction from
wpb_executable_proof.py and attacks the remaining FINITE proof obligations.

No Monte Carlo is used.

Checked here
------------
M1. The coupling exists for several independent (L,m) choices.
M2. Exact P^N and Q^N marginals.
M3. Symbolwise one-way causality.
M4. Every deterministic action word of every length <= N.
M5. The actual induced (B_j,A_{j+1}) law equals monotone quantile transport.
M6. The telescoping running-information-deficit identity holds pointwise.
M7. The actual bad-boundary probability is computed and compared with
    N_Q(L) p_*^L.
M8. The finite dyadic prefix reader is built explicitly.
M9. Its entropy is compared to the paper's running-deficit upper bound.
M10. All elementary o(sqrt(n)) bookkeeping terms are evaluated deterministically.

Anything machine-checkable that fails raises AssertionError.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wpb_executable_proof as wpb

mp.mp.dps = 80
TOL = mp.mpf("1e-45")


def block_info(block, law):
    return -mp.fsum(wpb.log2_mp(law[s]) for s in block)


def centered_block_info(block, law, L):
    return block_info(block, law) - mp.mpf("1.5") * L


def running_deficit_for_atom(x, w, L, m):
    A, B = [], []
    for j in range(m):
        xb = x[j*L:(j+1)*L]
        wb = w[j*L:(j+1)*L]
        A.append(centered_block_info(xb, wpb.P, L))
        B.append(centered_block_info(wb, wpb.Q, L))
    s = mp.mpf(0)
    mx = mp.mpf(0)
    Z = []
    for j in range(m-1):
        z = A[j+1] - B[j]
        Z.append(z)
        s += z
        mx = max(mx, s)
    return A, B, Z, mx


def check_telescoping(joint, L, m):
    max_err = mp.mpf(0)
    worst = None
    for (x, w), prob in joint.items():
        A, B, Z, _ = running_deficit_for_atom(x, w, L, m)
        for j in range(m):
            lhs = mp.mpf("1.5")*L + A[j]
            lhs += mp.fsum(A[i]-B[i] for i in range(j))
            rhs = mp.mpf("1.5")*L + A[0]
            rhs += mp.fsum(Z[i] for i in range(j))
            err = abs(lhs-rhs)
            if err > max_err:
                max_err = err
                worst = (x, w, j, lhs, rhs)
    assert max_err < TOL, worst
    return max_err


def exact_expected_running_deficit(joint, L, m):
    return mp.fsum(
        p * running_deficit_for_atom(x, w, L, m)[3]
        for (x, w), p in joint.items()
    )


def surprisal_level_boundaries(partition, base):
    atoms = partition.atoms
    vals = [block_info(a.label, base) for a in atoms]
    boundaries = []
    for i in range(len(atoms)-1):
        if abs(vals[i+1]-vals[i]) > mp.mpf("1e-60"):
            boundaries.append(atoms[i].right)
    return boundaries


def actual_bad_probability(L):
    ppart = wpb.sorted_partition(wpb.P, L)
    qpart = wpb.sorted_partition(wpb.Q, L)
    q_bounds = surprisal_level_boundaries(qpart, wpb.Q)
    bad_atoms = set()
    for b in q_bounds:
        for idx, a in enumerate(ppart.atoms):
            if a.left < b < a.right:
                bad_atoms.add(idx)
                break
    actual = mp.fsum(ppart.atoms[i].prob for i in bad_atoms)
    nlevels = len(q_bounds) + 1
    pstar = max(wpb.P)
    bound = mp.mpf(nlevels) * (pstar ** L)
    assert actual <= bound + mp.mpf("1e-60")
    return actual, bound, nlevels, len(bad_atoms)


def paper_prefix_upper_bound(joint, L, m):
    run = exact_expected_running_deficit(joint, L, m)
    g_bound = wpb.log2_mp(mp.mpf(2*m)) + 1/mp.log(2) + 1
    total = mp.mpf("1.5")*L + run + g_bound + 2
    return run, g_bound, total


def audit_case(L, m, prefix_depth=55):
    N = L*m
    print("\n" + "="*92)
    print(f"FINITE CASE L={L}, m={m}, N={N}")
    print("="*92)

    joint = wpb.build_joint_sorted_swap(L, m)

    errx = wpb.check_product_marginal(wpb.marginal_x(joint), wpb.P, TOL)
    errw = wpb.check_product_marginal(wpb.marginal_w(joint), wpb.Q, TOL)
    assert errx < TOL and errw < TOL

    errc, worstc = wpb.check_symbol_causality(joint)
    assert errc < TOL, worstc

    errtree, worsttree, nactions = wpb.check_all_action_words(joint)
    assert errtree < TOL, worsttree

    monotone_errors = []
    if m >= 2:
        qlevels = wpb.surprisal_level_law(wpb.Q, L, mp.mpf("1.5"))
        plevels = wpb.surprisal_level_law(wpb.P, L, mp.mpf("1.5"))
        theoretical = wpb.monotone_pair_law_mp(qlevels, plevels)
        for j in range(m-1):
            induced = wpb.induced_BA_law(joint, L=L, block_index=j)
            err = wpb.dict_l1_diff(induced, theoretical)
            monotone_errors.append(err)
            assert err < TOL

    telescoping_err = check_telescoping(joint, L, m)

    eprefix, unresolved = wpb.expected_prefix_bits_over_w(
        L=L, m=m, max_depth=prefix_depth
    )
    hxw, hxy, hw = wpb.conditional_entropy_x_given_w(joint)
    run, gbound, theorem_prefix_bound = paper_prefix_upper_bound(joint, L, m)
    assert hxw <= theorem_prefix_bound + mp.mpf("1e-40")

    print(f"nonzero K atoms                    : {len(joint)}")
    print(f"all deterministic action words    : {nactions}")
    print(f"max P^N marginal error            : {mp.nstr(errx, 6)}")
    print(f"max Q^N marginal error            : {mp.nstr(errw, 6)}")
    print(f"max causality error               : {mp.nstr(errc, 6)}")
    print(f"max path-law error                : {mp.nstr(errtree, 6)}")
    if monotone_errors:
        print("max induced-vs-quantile L1 error  :", mp.nstr(max(monotone_errors), 6))
    print(f"max telescoping identity error    : {mp.nstr(telescoping_err, 6)}")
    print(f"H(X^N|W^N)                        : {mp.nstr(hxw, 14)}")
    print(f"E running prefix deficit          : {mp.nstr(run, 14)}")
    print(f"paper E[G_m] upper bound          : {mp.nstr(gbound, 14)}")
    print(f"paper E[prefix bits] upper bound  : {mp.nstr(theorem_prefix_bound, 14)}")
    print(f"resolved dyadic E[length], d={prefix_depth}: {mp.nstr(eprefix, 14)}")
    print(f"unresolved dyadic mass            : {mp.nstr(unresolved, 6)}")
    print("PASS")



# ---------------------------------------------------------------------------
# M8b: rigorous dyadic prefix expectation with analytic tail bound
# ---------------------------------------------------------------------------

def prefix_survival_for_cells(cells, depth_max=30):
    """
    If B is the first dyadic depth at which the output cell is determined,
    E[B] = sum_{d>=0} P(B>d).

    We compute survival masses exactly through depth_max-1.  If the final
    partition has R internal boundaries, then at depth d at most R dyadic
    cells can still straddle a boundary, hence U_d <= R 2^{-d}.  The
    infinite tail is therefore <= R 2^(1-depth_max).
    """
    unresolved=[(mp.mpf(0),mp.mpf(1))]
    surv=mp.mpf(0)
    for depth in range(depth_max):
        keep=[]
        for lo,hi in unresolved:
            if wpb.node_label_if_determined(lo,hi,cells) is None:
                keep.append((lo,hi))
        U=mp.fsum(hi-lo for lo,hi in keep)
        surv += U
        split=[]
        for lo,hi in keep:
            mid=(lo+hi)/2
            split.extend([(lo,mid),(mid,hi)])
        unresolved=split
    R=max(0,len(cells)-1)
    tail=mp.mpf(R)*mp.power(2,1-depth_max)
    return surv,tail


def rigorous_expected_prefix_bits(L,m,depth_max=30):
    qpart=wpb.sorted_partition(wpb.Q,L)
    q_by_label={a.label:a for a in qpart.atoms}
    labels=[a.label for a in qpart.atoms]
    partial=mp.mpf(0); tail=mp.mpf(0)
    for wblocks in itertools.product(labels,repeat=m):
        pw=mp.mpf(1)
        for wb in wblocks:
            pw*=q_by_label[wb].prob
        cells=wpb.cells_for_wblocks(L,wblocks)
        a,b=prefix_survival_for_cells(cells,depth_max)
        partial+=pw*a; tail+=pw*b
    return partial,partial+tail,tail


# ---------------------------------------------------------------------------
# M11: finite-policy seed-profile converse on the actual exact seed
# ---------------------------------------------------------------------------

def threshold_policy_endpoint(x,w,z):
    N=len(x); zstream=tuple(reversed(w)); ip=iq=0
    M=mp.mpf(0); outs=[]; acts=[]
    threshold=mp.mpf(z)*mp.sqrt(N); h=mp.mpf('1.5')
    for _ in range(N):
        a=0 if M<threshold else 1
        acts.append(a)
        if a==0:
            y=x[ip]; ip+=1; law=wpb.P
        else:
            y=zstream[iq]; iq+=1; law=wpb.Q
        outs.append(y)
        M += -wpb.log2_mp(law[y])-h
    return N*h+M,M,tuple(outs),tuple(acts)


def seed_profile_attack(joint,zgrid):
    N=len(next(iter(joint))[0]); h=mp.mpf('1.5')
    worst=mp.inf; Emax=mp.mpf(0)
    for (x,w),gamma in joint.items():
        seed_info=-wpb.log2_mp(gamma); ms=[]
        for z in zgrid:
            I,M,y,actions=threshold_policy_endpoint(x,w,z)
            gap=seed_info-I
            worst=min(worst,gap)
            assert gap>-TOL,(x,w,z,gamma,I,gap,y,actions)
            ms.append(M)
        Emax += gamma*max(ms)
    HS=wpb.entropy_dict(joint); residual=HS-N*h
    assert residual+TOL>=Emax
    return worst,Emax,residual

def asymptotic_bookkeeping():
    print("\n" + "="*92)
    print("ASYMPTOTIC BOOKKEEPING FOR L_n = ceil((log2 n)^2)")
    print("="*92)
    print("No simulation: these are the elementary error scales in the proof.\n")
    print("        n       L      L/sqrt(n)    log2(n)/sqrt(n)      n(L+1)2^-L/sqrt(n)")
    for k in [16, 24, 32, 48, 64, 96, 128]:
        L = k*k
        sqrt_n = mp.power(2, mp.mpf(k)/2)
        term_start = mp.mpf(L) / sqrt_n
        term_log = mp.mpf(k) / sqrt_n
        boundary_scaled = mp.mpf(L+1) * mp.power(2, mp.mpf(k)/2 - L)
        print(
            f"2^{k:<3d}  {L:6d}"
            f"   {mp.nstr(term_start, 7):>12s}"
            f"   {mp.nstr(term_log, 7):>16s}"
            f"   {mp.nstr(boundary_scaled, 7):>24s}"
        )


def main():
    print("="*92)
    print("WHO PAYS THE BILL? — ELECTRON-MICROSCOPE EXECUTABLE AUDIT")
    print("NO MONTE CARLO")
    print("="*92)
    print("q =", mp.nstr(wpb.Q_Q, 32))
    print("H(P) =", mp.nstr(wpb.H_mp(wpb.P), 25))
    print("H(Q) =", mp.nstr(wpb.H_mp(wpb.Q), 25))

    for L, m in [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2)]:
        audit_case(L, m)

    print("\n" + "="*92)
    print("RIGOROUS PREFIX-CODE TAIL AUDIT")
    print("="*92)
    for L,m in [(1,4),(2,2)]:
        joint=wpb.build_joint_sorted_swap(L,m)
        hxw,_,_=wpb.conditional_entropy_x_given_w(joint)
        lo,hi,tail=rigorous_expected_prefix_bits(L,m,depth_max=28)
        run,gb,theorem=paper_prefix_upper_bound(joint,L,m)
        print(f"L={L}, m={m}: E[prefix] in [{mp.nstr(lo,14)}, {mp.nstr(hi,14)}]")
        print(f"           rigorous tail <= {mp.nstr(tail,8)}")
        print(f"           H(X|W)={mp.nstr(hxw,14)}, theorem bound={mp.nstr(theorem,14)}")
        assert lo+TOL>=hxw
        assert hi<=theorem+mp.mpf('1e-20')
        print("           PASS")

    print("\n" + "="*92)
    print("FINITE-POLICY SEED-PROFILE CONVERSE")
    print("="*92)
    joint=wpb.build_joint_sorted_swap(1,4)
    zgrid=[mp.mpf(x) for x in ['-1.0','-0.5','0.0','0.5','1.0']]
    worst,emax,residual=seed_profile_attack(joint,zgrid)
    print("threshold grid:",[str(z) for z in zgrid])
    print("min atomwise slack -log(gamma)-I_pi =",mp.nstr(worst,14))
    print("E max_pi M_pi,n =",mp.nstr(emax,14))
    print("H(S)-N h        =",mp.nstr(residual,14))
    print("PASS: pointwise and averaged seed-profile converse.")

    print("\n" + "="*92)
    print("ACTUAL BAD-BOUNDARY PROBABILITY")
    print("="*92)
    print("   L        actual bad mass          analytic bound       Q levels   bad P-atoms")
    for L in [1,2,3,4,5,6]:
        actual, bound, nlevels, nbad = actual_bad_probability(L)
        print(
            f"{L:4d}"
            f"   {mp.nstr(actual, 12):>20s}"
            f"   {mp.nstr(bound, 12):>20s}"
            f"   {nlevels:8d}"
            f"   {nbad:11d}"
        )

    asymptotic_bookkeeping()

    print("\n" + "="*92)
    print("AUDIT VERDICT")
    print("="*92)
    print("Finite construction: survived all executable attacks above.")
    print("Remaining analytic input: triangular-array FCLT + uniform integrability")
    print("of the running maximum. A finite Python run cannot quantify over all n.")


if __name__ == "__main__":
    main()
