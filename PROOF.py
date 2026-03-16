import numpy as np

alpha, beta = 1.0, 3.0
x_star = 2.0
lam = 0.4
sigma = 0.3

def Phi(x):
    return x_star + lam * (x - x_star)

x = 2.8
xs = [x]
for _ in range(8):
    x = Phi(x)
    xs.append(x)

print("Iterates:")
for n, val in enumerate(xs):
    print(f"x_{n} = {val:.8f}")

print("\nLipschitz constant q =", lam)
print("Fixed point x_* =", x_star)
print("Distance decay check:")
for n, val in enumerate(xs):
    print(f"|x_{n} - x_*| = {abs(val - x_star):.8e}")