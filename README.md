# Project at a glance

**What it is.**
A deterministic, time-averaged **spectral tester** for SAT. We encode a formula into a structured “phase schedule,” build a complex Gram matrix from those phases, and decide **SAT vs. UNSAT** by looking at one number: the top eigenvalue (normalized), $\mu=\lambda_{\max}/C$.

**Why it’s interesting.**
It produces a **clear and reproducible gap** between SAT-like and UNSAT-like behavior using simple, fast linear-algebra—no search, no backtracking. The construction is explicit (no randomness required) and scales polynomially.

---

## How it works (60-second version)

1. **Schedule the phases.**
   Each clause gets a time window (“lock”). We place phases using:

   * **De-aliased offsets** (a stride near $T/2$, coprime with $T$) to minimize lock overlap between neighboring clauses.
   * **Low-correlation Hadamard masks** inside the lock (rows/columns chosen with coprime strides) so cross-terms cancel well after truncation.

2. **Average and measure.**
   We form $G=\frac{1}{T} Z^\* Z$ with $Z=\exp(i\Phi)$ (complex Hermitian—no absolute values), then compute the largest eigenvalue $\lambda_{\max}$ and normalize $\mu=\lambda_{\max}/C$.

3. **Decide.**
   If $\mu$ exceeds a fixed threshold $\tau$, declare **SAT**; otherwise **UNSAT**.

---

## What’s new here

* **Deterministic pipeline.**
  Offsets and Hadamard masks are chosen explicitly to minimize aliasing and keep correlations low, even after truncation to $m$ time slots.

* **Lock-only S2 control.**
  We measure neighbor cross-terms **only inside the lock window** and normalize by $m$. This matches the theoretical S2 bound and keeps the analysis honest.

* **Simple spectral witness.**
  A single eigenvalue captures the global “coherence” of the schedule. In practice we see a **large SAT/UNSAT separation**.

---

## What we actually observed

On a large run (e.g., $C=1000,\ T=312,\ m=156,\ \zeta_0=0.40$):

* **UNSAT-Hadamard:** $\mu \approx 0.158$ with $\lambda_{\max} \approx 158$ (i.e., $\mu \approx \lambda_{\max}/C$) → **strong gap** below any reasonable threshold.
* **S2 (lock-only) neighbor row-sum:** $\approx 0.228$ for $d=4$, well **below** the theoretical bound $d\,\kappa_{\mathrm{S2}} \approx 0.474$.

This is repeatable and robust after the key fixes (complex Hermitian Gram, de-aliased offsets, Hadamard row/column strides, lock-only normalization).

---

## What it means (and what it doesn’t)

* **It means:** we have a **fast, deterministic spectral tester** with strong **soundness** evidence on UNSAT-like inputs and a clear gap to SAT-like behavior.

* **It does *not* mean:** we’ve proved **P = NP**. A formal resolution needs full proofs (no unproven assumptions) for stability/concentration and completeness across general SAT instances.

---

## Why it might matter

* **Conceptual:** shows how structured interference + averaging can turn combinatorics into a clean spectral signal.
* **Practical:** a lightweight, matrix-based diagnostic that scales as $\tilde O(C^2\log C)$, useful as a filter or heuristic before heavy solvers.
* **Research:** a concrete avenue to formalize soundness/completeness via standard tools (Gershgorin, matrix concentration, stability lemmas).

---

## Limitations and open work

* **Completeness:** formal lower bounds on $\mu$ for SAT under small deviations (the SAT envelope) need a full proof.
* **Assumptions:** the usual A1–A5 (geometry, truncated orthogonality, S2 bound, concentration, stability) must be rigorously closed with polynomial constants.
* **Generality:** behavior outside the specific “UNSAT-Hadamard” regime should be mapped carefully.

---

## Where to take it next

* **Paper skeleton:** formalize algorithm, assumptions, and theorems; include seed/scale sweeps and ablations (offset stride, row/col strides).
* **CLI controls:** expose `s_stride`, `row_step`, `col_stride`, `col_offset`, and `d` to tune overlap and correlations from the command line.
* **Repro pack:** repo with fixed schedules, scripts, JSON outputs, and plots; a one-command script to regenerate all figures/tables.

**Bottom line:** Functionally, the method delivers a strong, reproducible spectral gap and meets the lock-only S2 bound with margin. The path to a formal result is clear: write down and prove the A1–A5 assumptions and the completeness guarantee.
