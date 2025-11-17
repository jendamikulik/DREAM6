import cmath
import numpy as np

# --- 1. Definice Funkcí (Stejné) ---

def D(z: complex) -> complex:
    """Denominator: D(z) = 1 + (z + tan z)^2."""
    return 1 + (z + cmath.tan(z)) ** 2

def D_prime(z: complex) -> complex:
    """Analytická derivace D'(z) = 2 * (z + tan z) * (1 + sec^2 z)."""
    try:
        return 2 * (z + cmath.tan(z)) * (1 + 1 / cmath.cos(z) ** 2)
    except ZeroDivisionError:
        return float('inf') + 0j

def newton_root(z0: complex, max_iter: int = 50, tol: float = 1e-14) -> complex:
    """Newtonova metoda pro hledání kořenů D(z) = 0."""
    z = z0
    for _ in range(max_iter):
        Dz = D(z)
        Dpz = D_prime(z)

        if abs(Dpz) < 1e-16:
            break

        z_new = z - Dz / Dpz
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    return z

def residue(z0: complex) -> complex:
    """Residuum f v jednoduchém pólu z0: Res = 1 / D'(z0)."""
    return 1 / D_prime(z0)

# --- 2. Hledání pólů v pásu 0 < Im z < π (N=100, optimalizováno) ---

poles_raw = []
N_final = 100
# Systématičtější prohledávání imaginární osy
imag_shifts = np.linspace(0.1, np.pi - 0.1, 10)

for k in range(-N_final, N_final + 1):
    x0 = (k + 0.5) * np.pi
    for s in imag_shifts:
        z0 = x0 + 1j * s
        z_root = newton_root(z0)
        # Filtrování pouze H+ pólů v pásu 0 < Im z < π
        if 0.0 < z_root.imag < np.pi:
            poles_raw.append(z_root)

# --- 3. Deduplikace a Součet ---

poles_unique: list[complex] = []
eps = 1e-10

for z in poles_raw:
    if not any(abs(z - w) < eps for w in poles_unique):
        poles_unique.append(z)

# Součet residuí
residues = [residue(z) for z in poles_unique]
sum_res = sum(residues)

I_est = 2 * np.pi * 1j * sum_res

# --- 4. Výstup ---

print(f"--- KOREKCE: Výsledky s N={N_final} a lepšími odhady (Total Poles: {len(poles_unique)}) ---")
print(f"Sum of residues (Σ Res):    {sum_res.real:.15f} + {sum_res.imag:.15f}j")
print(f"Integral I = 2πi ΣRes:      {I_est.real:.15f} + {I_est.imag:.15f}j")
print(" ")
print(f"Expected Sum of residues:   0.000000000000000 + -0.500000000000000j")
print(f"Expected Integral I:        {np.pi:.15f} + 0.000000000000000j")