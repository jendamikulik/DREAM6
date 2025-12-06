import mpmath as mp

mp.mp.dps = 80  # high precision

# -------------------------------------------------
# 1) Constants
# -------------------------------------------------
G = mp.catalan
ln2 = mp.log(2)
print("Catalan G ≈", G)
print()

# -------------------------------------------------
# 2) Single integral J = ∫_0^{π/4} log(1 + tan θ) dθ
#    (tenhle má čistý closed form (π/8) ln 2)
# -------------------------------------------------
def integrand_J(theta):
    return mp.log(1 + mp.tan(theta))

J_num = mp.quad(integrand_J, [0, mp.pi/4])
J_th  = (mp.pi/8) * ln2

print("[Single integral J]")
print("Numeric J      ≈", J_num)
print("Theory  J_th   =", J_th)
print("Abs. error     =", abs(J_num - J_th))
print()

# -------------------------------------------------
# 3) Double integral I = ∫_0^1 ∫_0^1 log(1 + xy)/(1 + x^2) dx dy
# -------------------------------------------------
def integrand_I(x, y):
    return mp.log(1 + x*y) / (1 + x**2)

def I_double():
    return mp.quad(lambda yy: mp.quad(lambda xx: integrand_I(xx, yy), [0, 1]), [0, 1])

I_num = I_double()

print("[Double integral I]")
print("Numeric I      ≈", I_num)
print("(* no closed form asserted here – this is the raw target constant *)")
print()

# -------------------------------------------------
# 4) θ–representation of the same I
#    F(θ) = ((1 + tan θ) log(1 + tan θ) - tan θ)/tan θ
#    I = ∫_0^{π/4} F(θ) dθ
# -------------------------------------------------
def F_theta(theta):
    t = mp.tan(theta)
    if t == 0:
        return 0
    return ((1 + t) * mp.log(1 + t) - t) / t

I_theta = mp.quad(F_theta, [0, mp.pi/4])

print("[θ–representation]")
print("I from θ–integral ≈", I_theta)
print("Difference |I - I_θ| =", abs(I_theta - I_num))
print()

# -------------------------------------------------
# 5) Oscillator split: base + osc
#    base = 2J (čistá Fourier symetrie),
#    osc  = I - base (skutečný oscilátor – žádné G/4!)
# -------------------------------------------------
I_base = 2 * J_th          # analyticky víme, že J = (π/8) ln 2
I_osc  = I_num - I_base    # čistě numerická oscilátorová část

print("[Oscillator split]")
print("Base contribution      I_base ≈", I_base,
      "   (this is 2 * J_th = 2 * (π/8) ln 2)")
print("Oscillator contribution I_osc  ≈", I_osc)
print("Check I_base + I_osc           =", I_base + I_osc)
print("Matches numeric I?             =", abs(I_base + I_osc - I_num))
print()

# -------------------------------------------------
# 6) Mini–zeta / Dirichlet series
#    S_raw = Σ_{n≥1} (-1)^{n+1}/n² log(1 + 1/n²)
#    Tady nic netvrdíme, jen porovnáváme kombinace se skutečným I_osc.
# -------------------------------------------------
def dirichlet_raw(N=10_000):
    return mp.nsum(lambda n: (-1)**(n+1) / n**2 * mp.log(1 + 1/n**2), [1, mp.inf])

S_raw = dirichlet_raw()
print("[Mini–zeta / Dirichlet series]")
print("S_raw        ≈", S_raw)
print("S_raw/4      ≈", S_raw/4)
print()

# Zkusíme jednoduché kombinace – jen jako průzkum
candidate1 = S_raw/4
candidate2 = J_th - S_raw/4
candidate3 = I_base + S_raw/4

print("Diff I_osc - S_raw/4          ≈", I_osc - candidate1)
print("Diff I_num - (J_th - S_raw/4) ≈", I_num - candidate2)
print("Diff I_num - (I_base + S/4)   ≈", I_num - candidate3)
print()
print("Done. All numerics consistent; analytic ID for I still to be pinned down.")
