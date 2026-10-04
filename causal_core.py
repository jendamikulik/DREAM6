#!/usr/bin/env python3
"""CITELNE JADRO: presny konecny priklad C_2 = 27/4 bitu.
Spusteni: python causal_core.py   (jen standardni knihovna)
P=(1/2, 16krat 1/32), Q=(2krat 1/4, 8krat 1/16), H(P)=H(Q)=3.
Seed se losuje jednou. Odpoved pak zavisi na seedu a cele historii.
Tento konecny priklad ilustruje obecnou konstrukci; sam nedokazuje limitu.
"""
from fractions import Fraction as F
from collections import Counter


def check(ok, why):
    if not ok:
        raise ValueError(why)


laws = [[F(1, 2)]+[F(1, 32)]*16, [F(1, 4)]*2+[F(1, 16)]*8]
# U kazdeho vrcholu si pamatujeme (pravdepodobnost historie, deti podle akce).
tree = []


def grow(mass, rounds):
    v = len(tree)
    tree.append([mass, []])
    if rounds:
        for law in laws:
            tree[v][1].append([grow(mass*p, rounds-1) for p in law])
    return v


grow(F(1), 2)
residual = [mass for mass, children in tree]
cards = []
while residual[0]:
    # Nejvetsi odebratelna karticka: pro KAZDOU akci vyber NEJLEPSI vystup.
    capacity = residual.copy()
    for v in reversed(range(len(tree))):
        children = tree[v][1]
        if children:
            capacity[v] = min(max(capacity[c] for c in row) for row in children)
    weight = capacity[0]
    check(weight > 0, 'Kladny tok bez odebratelne karticky')
    answers, visited = {}, []

    def choose(v):
        visited.append(v)
        answers[v] = []
        for row in tree[v][1]:
            y = max(range(len(row)), key=lambda y: capacity[row[y]])
            answers[v].append(y)
            choose(row[y])

    choose(0)
    for v in visited:
        residual[v] -= weight
    check(all(r >= 0 for r in residual), 'Zaporny tok')
    cards.append((weight, answers))

# Nezavisla rekonstrukce. Nepouziva residual ani capacity.
reconstructed = [F(0)]*len(tree)
for weight, answers in cards:
    def replay(v):
        reconstructed[v] += weight
        for action, row in enumerate(tree[v][1]):
            replay(row[answers[v][action]])
    replay(0)
check(reconstructed == [mass for mass, children in tree], 'Nesedi historie')
# Rovnost kazde historie zajisti spravnou podminenou pravdepodobnost
# pro kazdou akci. Plati proto pro VSECHNY adaptivni politiky.

# Presna dolni obalka: kazdy seed ma alespon informaci sveho transkriptu.
# Pocitame min_pi P(probabilita transkriptu >= prah), bez logaritmu.
levels = sorted({mass for mass, children in tree if not children}, reverse=True)
envelope, previous = [], F(0)
for threshold in levels:
    cdf = [mass if mass >= threshold else F(0) for mass, children in tree]
    for v in reversed(range(len(tree))):
        children = tree[v][1]
        if children:
            cdf[v] = min(sum(cdf[c] for c in row) for row in children)
    envelope.append((threshold, cdf[0]-previous))
    previous = cdf[0]
check(previous == 1, 'Obalka nema hmotu 1')


def information(p):
    # Zde vsechny atomy jsou 2^-k, takze -log2(p)=k PRESNE.
    check(p.numerator == 1 and p.denominator & (p.denominator-1) == 0,
          'Atom neni mocnina dvojky')
    return p.denominator.bit_length()-1


lower = sum(mass*information(p) for p, mass in envelope)
upper = sum(weight*information(weight) for weight, answers in cards)
check(lower == upper == F(27, 4), 'Dolni a horni mez se nesetkaly')
print('PRESNE overene historie:', len(tree))
print('Seed: vaha jedne karticky -> pocet karticek')
for weight, count in sorted(Counter(w for w, answers in cards).items(), reverse=True):
    print(f'  {str(weight):>6s} -> {count}')
print('Dolni mez pro kazdy seed:', lower, 'bitu')
print('Entropie sestrojeneho seedu:', upper, 'bitu')
print('Tedy C_2 = 27/4 = 6.75 bitu PRESNE. Zaklad 2h=6, priplatek=3/4.')
print('Zadny float, optimalizacni solver ani numericka tolerance.')
