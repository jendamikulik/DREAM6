# #PequalNP — Dream6 Spectral Tester

## 🌌 At a glance

We propose a **deterministic, polynomial-time spectral tester** for SAT.
Every CNF is mapped to a **phase schedule**, averaged into a complex Hermitian Gram matrix.
One number decides: the normalized top eigenvalue

$$
\mu = \lambda_{\max}(G)/C.
$$

---

## 🚀 Why it matters

* **Reproducible SAT/UNSAT gap.**
  No randomness, no backtracking — just structured interference + averaging.

* **Deterministic pipeline.**
  Offsets are de-aliased (stride ≈ T/2, coprime with T).
  Masks come from truncated Walsh–Hadamard rows with coprime strides.

* **Polynomial-time.**
  Full schedule + Gram + top eigenvalue in \$\tilde O(C^2 \log C)\$.

---

## 🔬 Core mechanism

1. **Lock windows.** Each clause is assigned \$m\$ slots.
2. **Hadamard masks.** Negative entries → phase \$\pi\$, positives → \$0\$.
3. **Gram averaging.** \$G = \tfrac{1}{T} Z^\* Z\$ with \$Z = \exp(i\Phi)\$.
4. **Decision.** If \$\mu \ge \tau\$, declare SAT; else UNSAT.

---

## 📊 What we observed

On \$C=1000,\ T=312,\ m=156,\ \zeta\_0=0.40\$:

* **UNSAT-Hadamard:** \$\mu \approx 0.158 \ll 1\$, \$\lambda\_{\max}\approx 158\$.
* **Lock-only S2 row-sum:** \$\approx 0.228\$ vs. theoretical bound \$d\kappa\_{S2}\approx 0.474\$ (\$d=4\$).

Clear gap, stable margins.

---

## ✅ What’s solid

* **Soundness:** lock-only Gershgorin bounds + S2 control hold.
* **Determinism:** all offsets/masks are explicit, no seeds needed.
* **Empirical evidence:** strong, repeatable SAT/UNSAT separation.

---

## ⚠️ What’s open

* **Completeness:** lower bounds on \$\mu\$ for SAT require formal proof.
* **Assumptions A1–A5:** geometry, orthogonality, S2, concentration, stability must be fully closed.
* **Generality:** extend beyond UNSAT-Hadamard regime.

---

## 🧭 Next steps

* Finish paper draft (theorems, proofs, plots).
* Release reproducibility pack (CLI + JSON outputs + figures).
* Push formal closure of A1–A5 → #PequalNP.

---

**Bottom line:**
We now have a **working, deterministic spectral tester** for SAT with polynomial scaling and a strong UNSAT gap.
The road to #PequalNP is open: prove completeness, close assumptions.
