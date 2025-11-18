import mpmath as mp
import numpy as np

mp.mp.dps = 80  # high precision


# -------------------------------------------------------
# 1) Ramanujan-style integral vs. Gamma/phase formula
#    ∫_0^∞ x^(α-1) cos(ωx+φ) dx
#    = Γ(α) ω^(-α) cos(φ + π α / 2)
# -------------------------------------------------------

def ramanujan_numeric(alpha, omega, phi):
    f = lambda x: x**(alpha-1) * mp.cos(omega*x + phi)
    # tell mpmath the oscillation frequency:
    return mp.quadosc(f, [0, mp.inf], omega=omega)

def ramanujan_analytic(alpha, omega, phi):
    return mp.gamma(alpha) * (omega**(-alpha)) * mp.cos(phi + mp.pi*alpha/2)

print("=== Ramanujan-type integral check ===")
alpha = mp.mpf('2')/3
omega = mp.mpf('2')
phi   = mp.mpf('1')

I_num = ramanujan_numeric(alpha, omega, phi)
I_an  = ramanujan_analytic(alpha, omega, phi)

print("Numeric :", I_num)
print("Analytic:", I_an)
print("Abs error:", abs(I_num - I_an))
print()


# -------------------------------------------------------
# 2) MIT binary integral:
#    I = ∫_0^1 ( Σ_{n≥1} floor(2^n x) / 3^n )^2 dx = 27/32
#
# We check:
#   (a) Monte Carlo approximation of the integral
#   (b) v^T K v with v_n = 3^{-n}, K_{nn}=1/2, K_{nm}=1/4 (n≠m)
# -------------------------------------------------------

def f_binary(x, N=25):
    """Truncated version of f(x) = Σ floor(2^n x) / 3^n."""
    s = mp.mpf('0')
    two_pow   = mp.mpf('2')
    three_pow = mp.mpf('3')
    for n in range(1, N+1):
        s += mp.floor(two_pow * x) / three_pow
        two_pow   *= 2
        three_pow *= 3
    return s

def binary_integral_monte_carlo(samples=20000, N=25):
    rng = np.random.default_rng(12345)
    acc = mp.mpf('0')
    for _ in range(samples):
        x = mp.mpf(rng.random())
        fx = f_binary(x, N)
        acc += fx**2
    return acc / samples

print("=== Binary MIT integral check ===")
I_mc = binary_integral_monte_carlo(samples=20000, N=25)
I_exact = mp.mpf('27')/32
print("Monte Carlo approx :", I_mc)
print("Exact value        :", I_exact)
print("Abs error          :", abs(I_mc - I_exact))
print()

# (b) one-vector + kernel computation: I = 9 v^T K v,
#     with v_n = 3^{-n}, K_{nn}=1/2, K_{nm}=1/4.

def binary_vTKv(N=40):
    # vector v
    v = mp.matrix([mp.mpf(1)/ (3**n) for n in range(1, N+1)])
    # build K
    K = mp.matrix(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                K[i,j] = mp.mpf('1')/2
            else:
                K[i,j] = mp.mpf('1')/4
    return 9 * (v.T * K * v)[0]

I_vTKv = binary_vTKv(N=40)
print("One-vector kernel evaluation 9 v^T K v:", I_vTKv)
print("Exact value                         :", I_exact)
print("Abs error                           :", abs(I_vTKv - I_exact))
print()


# -------------------------------------------------------
# 3) Exponential limit:
#    lim_{x→∞} (e^x + e^{-x}) / (2 - 3 e^x) = -1/3
# -------------------------------------------------------

def expr(x):
    return (mp.e**x + mp.e**(-x)) / (2 - 3*mp.e**x)

print("=== Exponential limit check ===")
for X in [5, 10, 20, 40, 80]:
    val = expr(X)
    print(f"x = {X:>3}  ->  expr(x) = {val}")

print("Target limit =", mp.mpf('-1')/3)
print()


# -------------------------------------------------------
# 4) Basis invariance demo:
#    v^T K v invariant under orthonormal change of basis
# -------------------------------------------------------

print("=== Basis invariance demo (finite-dim) ===")
N = 6
# simple symmetric kernel
K = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        K[i,j] = 0.5 if i==j else 0.25

# random vector
rng = np.random.default_rng(123)
v = rng.normal(size=N)

# v^T K v in original basis
orig_scalar = v @ (K @ v)

# random orthonormal matrix (QR)
Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
v_new = Q.T @ v
K_new = Q.T @ K @ Q

new_scalar = v_new @ (K_new @ v_new)

print("v^T K v (original basis):", orig_scalar)
print("v'^T K' v' (rotated basis):", new_scalar)
print("Difference:", abs(orig_scalar - new_scalar))
print("\nAll checks done.")
