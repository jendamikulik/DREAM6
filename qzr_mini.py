#!/usr/bin/env python3
"""qzr_mini.py — miniaturní standalone (jen stdlib) pro coarse QZR.

Co dělá
--------
1) Algebraický dyadický kořen R_n a E[R_n] = 1 + log2(n)  (Fraction).
2) Ověří R4_K4 strom (ZR), spočte fueled děti c=1/2, zkontroluje ZR R8.
3) Složení: outery → K7 (concat/shift/mixture lemma, zapsané jako claim).
4) Volitelně načte reference-DAG certifikát (dag16/dag32.json) a ověří
   podmínku (C) ve zlomcích → pak hlásí R8∈K8, R16∈K17, … podle targets.

Bez DAG souboru NEPROHLAŠUJE B_n ≤ 1+log2 n (jen algebraický mean).
S DAG (z tvého balíku) ano, pokud verify projde.

Použití
-------
  python qzr_mini.py
  python qzr_mini.py --r4 R4_K4_exact_M8.json
  python qzr_mini.py --dag certificates/dag32.json
  python qzr_mini.py --dag certificates/dag16.json --check-kernels
"""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys

P = [F(w, 64) for w in (1, 12, 38, 12, 1)]
Q = [F(w, 64) for w in (0, 16, 32, 16, 0)]
V = [F(1, 4), F(1, 2), F(1, 4)]  # (1,2,1)/4


# ---------- laws ----------

def trim(a):
    a = list(a)
    while a and a[-1] == 0:
        a.pop()
    return a or [F(0)]


def pad(a, n):
    a = list(a)
    return a + [F(0)] * max(0, n - len(a))


def atoms(a):
    return [[j, str(v)] for j, v in enumerate(a) if v]


def law(entries):
    """atoms [[j,'p'],...] or dict {j:p} → list"""
    if isinstance(entries, dict):
        items = [(int(j), F(str(v))) for j, v in entries.items()]
    else:
        items = [(int(j), F(str(v))) for j, v in entries]
    m = max((j for j, _ in items), default=0)
    a = [F(0)] * (m + 1)
    for j, v in items:
        a[j] += v
    s = sum(a)
    if s != 1:
        raise ValueError(f"not a probability (sum={s})")
    if min(a) < 0:
        raise ValueError("negative mass")
    return a


def mean(a):
    return sum(j * p for j, p in enumerate(a))


def conv(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if not ai:
            continue
        for j, bj in enumerate(b):
            if bj:
                out[i + j] += ai * bj
    return trim(out)


def U(a):
    return [F(0)] + list(a)


def D(a):
    """shift down by 1 (drop index 0; must be 0)."""
    if a[0] != 0:
        raise ValueError("D requires a[0]==0")
    return list(a[1:])


def mix(parts):
    L = max(len(x) for _, x in parts)
    out = [F(0)] * L
    for w, x in parts:
        for i, v in enumerate(x):
            out[i] += w * v
    return trim(out)


def CDF(a, k):
    if k < 0:
        return F(0)
    return sum(a[: min(len(a), k + 1)])


def conv_Q(a):
    out = [F(0)] * (len(a) + 4)
    for j, pj in enumerate(a):
        if not pj:
            continue
        for y, qy in enumerate(Q):
            if qy:
                out[j + y] += pj * qy
    return trim(out)


def zr_exact(parent, children):
    mu = conv_Q(parent)
    mx = max(len(ch) + x for x, ch in enumerate(children))
    rhs = [F(0)] * (mx + 1)
    for x, ch in enumerate(children):
        for j, p in enumerate(ch):
            if not p:
                continue
            if j + x >= len(rhs):
                rhs.extend([F(0)] * (j + x + 1 - len(rhs)))
            rhs[j + x] += P[x] * p
    L = max(len(mu), len(rhs))
    return pad(mu, L) == pad(rhs, L)


# ---------- algebraic dyadic root ----------

def algebraic_root(n: int):
    """R_1=δ_1, R_{2n}=½δ_1 + ½ U(R_n * R_n)."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two ≥1")
    R = [F(0), F(1)]  # δ_1
    m = 1
    while m < n:
        R = mix([(F(1, 2), [F(0), F(1)]), (F(1, 2), U(conv(R, R)))])
        m *= 2
    return R


def mean_table(max_n=32):
    rows = []
    n = 1
    while n <= max_n:
        R = algebraic_root(n)
        e = mean(R)
        claim = F(1) + F(n.bit_length() - 1)  # 1 + log2(n)
        rows.append((n, e, claim, e == claim))
        n *= 2
    return rows


# ---------- fueled fan (c=1/2) ----------

def fueled_children(R, A, c=F(1, 2)):
    C = U(R)
    A0, A1, A2, A3, A4 = A
    A2C, A3C = conv(A2, C), conv(A3, C)
    return [
        mix([(1 - c, conv(A0, C)), (c, U(conv(A1, C)))]),
        mix([(F(12 - 13 * c) / 12, conv(A1, C)), (F(13 * c) / 12, U(conv(A2, C)))]),
        mix([(F(32 * c) / 19, V), (1 - F(32 * c) / 19, conv(A2, C))]),
        mix([(F(13 * c) / 12, D(A2C)), (F(12 - 13 * c) / 12, A3C)]),
        mix([(c, D(A3C)), (1 - c, conv(A4, C))]),
    ]


# ---------- R4 nested tree ----------

def verify_r4_tree(data):
    root = law(data["root"])
    d1 = [law(a) for a in data["depth1"]]
    assert zr_exact(root, d1), "R4 root ZR"
    d2 = [[law(a) for a in row] for row in data["depth2"]]
    for x in range(5):
        assert zr_exact(d1[x], d2[x]), f"R4 d1[{x}] ZR"
    d3 = [[[law(a) for a in row] for row in block] for block in data["depth3"]]
    for x in range(5):
        for y in range(5):
            assert zr_exact(d2[x][y], d3[x][y]), f"R4 d2[{x}][{y}]"
    d4 = [
        [[[law(a) for a in row] for row in block] for block in cube]
        for cube in data["depth4"]
    ]
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert zr_exact(d3[x][y][z], d4[x][y][z]), f"R4 d3[{x}][{y}][{z}]"
    return root, d1


# ---------- reference-DAG verify (condition C) ----------

def parse_weights(raw, n_refs):
    """weights: 5 rows of [[index, 'frac'], ...] → dense 5 x n_refs"""
    W = [[F(0)] * n_refs for _ in range(5)]
    for x, row in enumerate(raw):
        for item in row:
            if isinstance(item[0], list) or (isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], int)):
                # already weird
                pass
            idx, val = int(item[0]), F(str(item[1]))
            W[x][idx] = val
        if sum(W[x]) != P[x]:
            raise ValueError(f"weights row {x} sum {sum(W[x])} != P[{x}]={P[x]}")
        if min(W[x]) < 0:
            raise ValueError(f"negative weight row {x}")
    return W


def stoch_dom_ok(root, weights, refs, slack_ok=True):
    """Q*root ⪰_st σ = sum w_{x,i} δ_x * refs[i]  via CDF inequalities."""
    M = max(len(root), max((len(r) for r in refs), default=1)) + 4
    for k in range(M + 1):
        src = sum(Q[y] * CDF(root, k - y) for y in range(5))
        dst = sum(
            weights[x][l] * CDF(refs[l], k - x)
            for x in range(5)
            for l in range(len(refs))
        )
        if src > dst:
            return False, k, src, dst
    return True, None, None, None


def verify_dag(data, check_all_entries=True):
    """Verify positive reference DAG. Returns targets depths dict."""
    assert data.get("kind") in (
        None,
        "positive_reference_dag",
    ) or "library" in data
    lib = data["library"]
    # depth 0
    assert lib[0]["depth"] == 0
    for e in lib[0]["entries"]:
        law(e["root"])  # valid prob

    verified_layers = {0: [law(e["root"]) for e in lib[0]["entries"]]}

    for layer in lib[1:]:
        d = layer["depth"]
        prev = verified_layers[d - 1]
        laws = []
        for e in layer["entries"]:
            root = law(e["root"])
            W = parse_weights(e["weights"], len(prev))
            ok, k, src, dst = stoch_dom_ok(root, W, prev)
            if not ok:
                raise AssertionError(
                    f"layer {d} root fails (C) at k={k}: {src} > {dst}"
                )
            laws.append(root)
        verified_layers[d] = laws
        if check_all_entries:
            print(f"  DAG layer {d}: {len(laws)} entries OK", flush=True)

    targets = {}
    for name, info in data.get("targets", {}).items():
        depth = int(info["depth"])
        cert = info["certificate"]
        root = law(cert["root"])
        prev = verified_layers[depth - 1]
        W = parse_weights(cert["weights"], len(prev))
        ok, k, src, dst = stoch_dom_ok(root, W, prev)
        if not ok:
            raise AssertionError(f"target {name} fails (C) at k={k}")
        # match algebraic root if name like R8
        if name.startswith("R") and name[1:].isdigit():
            n = int(name[1:])
            if trim(root) != trim(algebraic_root(n)):
                # allow equal as law with trailing zeros already trimmed
                if atoms(root) != atoms(algebraic_root(n)):
                    raise AssertionError(f"{name} root ≠ algebraic_root({n})")
        targets[name] = depth
        print(f"  TARGET {name} ∈ K_{depth} OK", flush=True)
    return targets


# ---------- main report ----------

def find_r4(path: Path | None) -> Path:
    cands = []
    if path:
        cands.append(path)
    here = Path(__file__).resolve().parent
    cands += [
        here / "R4_K4_exact_M8.json",
        here / "inputs" / "R4_K4_exact_M8.json",
        Path("/workspace/outputs/R4_K4_exact_M8.json"),
    ]
    for p in cands:
        if p and p.exists():
            return p
    raise FileNotFoundError("R4_K4_exact_M8.json not found (pass --r4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r4", type=Path, help="R4_K4_exact_M8.json")
    ap.add_argument("--dag", type=Path, help="certificates/dag16.json or dag32.json")
    ap.add_argument("--check-kernels", action="store_true", help="verbose per-layer DAG check")
    ap.add_argument("--max-n", type=int, default=32)
    args = ap.parse_args()

    print("=== algebraic mean (always) ===", flush=True)
    for n, e, claim, ok in mean_table(args.max_n):
        print(f"  n={n:4d}  E[R]={e}  1+log2(n)={claim}  match={ok}", flush=True)
    assert all(ok for *_, ok in mean_table(args.max_n))

    print("\n=== R4 tree + fueled 4→8 ZR ===", flush=True)
    r4_path = find_r4(args.r4)
    data = json.loads(r4_path.read_text(encoding="utf8"))
    R4, fan = verify_r4_tree(data)
    print(f"  R4 ∈ K4 re-verified from {r4_path.name}, E={mean(R4)}", flush=True)
    kids = fueled_children(R4, fan)
    R8 = algebraic_root(8)
    assert zr_exact(R8, kids), "R8 fueled ZR failed"
    assert mean(R8) == 4
    Z = kids[2]
    print(f"  fueled ZR OK; E[R8]=4; Z mean={mean(Z)} supp={max(j for j,p in enumerate(Z) if p)}", flush=True)
    print("  outer B0,B1,B3,B4: lemma claim → K7 (concat/shift/mixture)", flush=True)
    print("  center Z: needs DAG (or expanded) depth ≥7 for R8∈K8", flush=True)

    bound_n = {1, 2, 4}  # R1 trivial / R2 if wanted / R4 from tree
    print("\n=== membership so far (without DAG) ===", flush=True)
    print("  R1 ∈ K1 (δ_1, trivial protocol)", flush=True)
    print("  R4 ∈ K4 (expanded tree)", flush=True)
    print("  R8 ∈ K8?  NOT YET (need Z∈K7+)", flush=True)

    if args.dag:
        print(f"\n=== reference DAG: {args.dag} ===", flush=True)
        dag = json.loads(args.dag.read_text(encoding="utf8"))
        targets = verify_dag(dag, check_all_entries=args.check_kernels or True)
        # interpret
        for name, d in sorted(targets.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]) if kv[0][1:].isdigit() else 0)):
            if name.startswith("R") and name[1:].isdigit():
                n = int(name[1:])
                need = n
                ok = d >= need
                print(f"  {name}: certified K_{d}; need K_{need} for horizon → {'OK' if ok else 'SHORT'}", flush=True)
                if ok:
                    bound_n.add(n)
            elif name == "Z":
                print(f"  Z: certified K_{d}; need ≥7 for 4→8 → {'OK' if d >= 7 else 'SHORT'}", flush=True)
                if d >= 7:
                    bound_n.add(8)  # composition with outers K7
                    print("  ⇒ composition: all five children ≥K7 ⇒ R8∈K8", flush=True)

        print("\n=== BOUND (from verified DAG targets) ===", flush=True)
        ns = sorted(bound_n)
        print(f"  B_n(P,Q) ≤ 1+log2(n)  for n in {ns}", flush=True)
        if 8 in bound_n and 16 in bound_n and 32 in bound_n:
            print("  (matches construction.tex finite list through 32)", flush=True)
        missing = [n for n in (8, 16, 32) if n not in bound_n]
        if missing:
            print(f"  missing horizons: {missing}", flush=True)
            sys.exit(2)
        print("\nSTRICT finite log bound: YES (for listed n)", flush=True)
    else:
        print("\n=== BOUND ===", flush=True)
        print("  algebraic E[R_n]=1+log2(n) for all dyadic n (identity).", flush=True)
        print("  membership bound B_n≤E[R_n] for n∈{1,4} from local trees.", flush=True)
        print("  For n=8,16,32: pass --dag certificates/dag32.json from the QZR bundle.", flush=True)
        print("\nSTRICT finite log bound through 32: NOT YET (no DAG)", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
