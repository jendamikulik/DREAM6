import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BRUTAL CLOSED-LOOP ANCHORING DEMO
# ============================================================
# Model:
#   admissible sector     S_ad = [alpha, beta]
#   singular state        Omega_x = delta_x
#   Gaussian hammer       H_sigma(Omega_x) = G_sigma(. - x)
#   barycenter            mu(f) = \int t f(t) dt
#   backreaction          B(f) = x_star + lam * (mu(f) - x_star)
#   closed loop           Phi(x) = B(H_sigma(Omega_x))
#
# Theory says:
#   Phi(x) = x_star + lam * (x - x_star)
#   Lip(Phi) = lam < 1
#   unique fixed point = x_star
#   fixed-point readout r_* = G_sigma(. - x_star) != 0
# ============================================================


# -----------------------------
# Parameters
# -----------------------------
alpha = 1.0
beta = 3.0
x_star = 2.0
lam = 0.4
sigma = 0.3

assert 0 < alpha <= x_star <= beta
assert 0 < lam < 1
assert sigma > 0


# -----------------------------
# Gaussian hammer
# -----------------------------
def gaussian_kernel(t: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian centered at `center`."""
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -((t - center) ** 2) / (2.0 * sigma**2)
    )


def hammer_readout(x: float, t_grid: np.ndarray, sigma: float) -> np.ndarray:
    """H_sigma(Omega_x) = G_sigma(. - x)."""
    return gaussian_kernel(t_grid, x, sigma)


# -----------------------------
# Barycenter and backreaction
# -----------------------------
def barycenter(t_grid: np.ndarray, f: np.ndarray) -> float:
    """Numerical barycenter mu(f) = ∫ t f(t) dt / ∫ f(t) dt."""
    mass = np.trapezoid(f, t_grid)
    if mass <= 0:
        raise ValueError("Readout has nonpositive mass.")
    return float(np.trapezoid(t_grid * f, t_grid) / mass)


def backreaction(mu_f: float, x_star: float, lam: float) -> float:
    """B(f) = x_star + lam * (mu(f) - x_star)."""
    return x_star + lam * (mu_f - x_star)


def Phi_numeric(x: float, t_grid: np.ndarray, sigma: float, x_star: float, lam: float) -> float:
    """Closed loop computed numerically through the readout."""
    f = hammer_readout(x, t_grid, sigma)
    mu_f = barycenter(t_grid, f)
    return backreaction(mu_f, x_star, lam)


def Phi_exact(x: float, x_star: float, lam: float) -> float:
    """Exact closed-loop formula from the theorem."""
    return x_star + lam * (x - x_star)


# -----------------------------
# Grids
# -----------------------------
t_grid = np.linspace(alpha - 4 * sigma, beta + 4 * sigma, 4000)
x_grid = np.linspace(alpha, beta, 200)


# -----------------------------
# Check: barycenter identity
# -----------------------------
bary_errors = []
for x in x_grid:
    f = hammer_readout(x, t_grid, sigma)
    mu_f = barycenter(t_grid, f)
    bary_errors.append(abs(mu_f - x))

max_bary_error = max(bary_errors)


# -----------------------------
# Check: numeric Phi vs exact Phi
# -----------------------------
phi_errors = []
phi_vals_num = []
phi_vals_exact = []
for x in x_grid:
    p_num = Phi_numeric(x, t_grid, sigma, x_star, lam)
    p_ex = Phi_exact(x, x_star, lam)
    phi_vals_num.append(p_num)
    phi_vals_exact.append(p_ex)
    phi_errors.append(abs(p_num - p_ex))

max_phi_error = max(phi_errors)


# -----------------------------
# Numerical Lipschitz estimate
# -----------------------------
# For Phi(x) = x_star + lam (x - x_star), the exact Lipschitz constant is lam.
# Here we also estimate it numerically from neighboring grid points.
lip_est = 0.0
for i in range(len(x_grid) - 1):
    dx = abs(x_grid[i + 1] - x_grid[i])
    dy = abs(phi_vals_num[i + 1] - phi_vals_num[i])
    lip_est = max(lip_est, dy / dx)


# -----------------------------
# Iterate from several initial points
# -----------------------------
def iterate_closed_loop(x0: float, n_steps: int) -> np.ndarray:
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = Phi_exact(x, x_star, lam)
        xs.append(x)
    return np.array(xs)


initial_points = [1.1, 1.4, 2.8, 3.0]
n_steps = 10
all_iters = {x0: iterate_closed_loop(x0, n_steps) for x0 in initial_points}


# -----------------------------
# Fixed-point readout
# -----------------------------
r_star = hammer_readout(x_star, t_grid, sigma)
r_star_mass = np.trapezoid(r_star, t_grid)
r_star_L1 = np.trapezoid(np.abs(r_star), t_grid)
r_star_L2_sq = np.trapezoid(r_star**2, t_grid)
r_star_max = np.max(r_star)

assert r_star_L1 > 0, "Fixed-point readout unexpectedly vanished."
assert r_star_max > 0, "Fixed-point readout unexpectedly vanished."


# -----------------------------
# Console report
# -----------------------------
print("=" * 70)
print("BRUTAL CLOSED-LOOP ANCHORING REPORT")
print("=" * 70)
print(f"Parameters: alpha={alpha}, beta={beta}, x_star={x_star}, lam={lam}, sigma={sigma}")
print()

print("[1] Gaussian hammer barycenter test")
print(f"    max |mu(H_sigma(Omega_x)) - x| over grid = {max_bary_error:.3e}")
print()

print("[2] Closed-loop formula test")
print(f"    max |Phi_numeric(x) - Phi_exact(x)| over grid = {max_phi_error:.3e}")
print(f"    exact Phi(x) = {lam:.6f} x + {x_star * (1 - lam):.6f}")
print()

print("[3] Contraction test")
print(f"    exact Lipschitz constant   = {lam:.6f}")
print(f"    numerical Lipschitz estimate = {lip_est:.6f}")
print(f"    contraction? {'YES' if lip_est < 1.0 else 'NO'}")
print()

print("[4] Fixed-point test")
print(f"    fixed point x_* = {x_star:.6f}")
print("    Check Phi(x_*) =", Phi_exact(x_star, x_star, lam))
print()

print("[5] Zero-veto / nonzero readout test")
print(f"    mass(r_*)   = {r_star_mass:.6f}")
print(f"    L1(r_*)     = {r_star_L1:.6f}")
print(f"    L2^2(r_*)   = {r_star_L2_sq:.6f}")
print(f"    max(r_*)    = {r_star_max:.6f}")
print(f"    r_* != 0 ?  {'YES' if r_star_L1 > 0 else 'NO'}")
print()

print("[6] Iterates")
for x0, xs in all_iters.items():
    print(f"    start x0 = {x0:.6f}")
    for n, x in enumerate(xs):
        print(f"      x_{n:02d} = {x:.10f}   |x_n - x_*| = {abs(x - x_star):.3e}")
    print()

print("=" * 70)
print("Punchline:")
print("The hammer smooths. The barycenter returns. The loop contracts. Collapse is vetoed.")
print("=" * 70)


# -----------------------------
# Plots
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left plot: Phi vs identity
axes[0].plot(x_grid, x_grid, label="identity  y = x")
axes[0].plot(x_grid, phi_vals_num, label=r"$\Phi_\sigma(x)$  (numeric)")
axes[0].scatter([x_star], [x_star], s=60, label=r"fixed point $x_*$")
axes[0].set_title("Closed-loop map")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Phi(x)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Right plot: iterates converging to x_star
for x0, xs in all_iters.items():
    axes[1].plot(range(len(xs)), xs, marker="o", label=f"x0={x0}")
axes[1].axhline(x_star, linestyle="--", label=r"$x_*$")
axes[1].set_title("Iterates collapse to the anchor")
axes[1].set_xlabel("iteration n")
axes[1].set_ylabel(r"$x_n$")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()


# -----------------------------
# Optional third plot: fixed-point Gaussian profile
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(t_grid, r_star, label=r"$r_* = G_\sigma(\cdot - x_*)$")
plt.axvline(x_star, linestyle="--", label=r"$x_*$")
plt.title("Fixed-point readout is nonzero")
plt.xlabel("t")
plt.ylabel("r_*(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()