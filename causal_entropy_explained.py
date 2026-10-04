#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAUZALNI ENTROPIE: konstrukce, nezavisla kontrola, Bellman, asymptotika.

Spusteni: python causal_entropy_explained.py
Pouze standardni knihovna Pythonu. Zadny solver, zadne nahodne pokusy.

CTETE NEJDRIVE JEN FUNKCE make_tree, make_seed, check_seed.
  1. Historie je posloupnost (pozadovana akce, vraceny symbol).
  2. Seed je karticka: pro kazdou historii a akci urcuje odpoved.
  3. Vylosujeme JEDNU karticku na zacatku. Dalsi nahoda uz neni.
  4. Kontrolor overi pravdepodobnost KAZDE historie, vcetne alternativnich
     akci. Tim overi spravnost pro kazdou adaptivni politiku.

MODEL: plny history-indexed kauzalni strom. Odpoved muze zaviset na cele
historii. Nejde o silnejsi synchronni model, v nemz musi byt odpoved pro
stejny cas a akci stejna pri ruznych historiich. Seed smi zaviset na n.

CO JE OBECNY DUKAZ A CO JE VYPOCET?
A. Obecny kombinatoricky argument je vysvetlen u make_seed a proof_map.
B. Program presne zkontroluje konkretni konecne stromy a certifikaty.
C. Bellmanova tabulka se pocita v celych cislech, bez diskretizace prostoru.
D. Limitu n -> nekonecno dokazuje analyticky argument v proof_map, nikoli
   to, ze konecna tabulka vypada presvedcive. Nejde o Lean certifikat.

Vypisovana desetinna entropie a normalizovana Bellmanova tabulka jsou
pouze prezentacni floaty. Rozhodujici kontroly pouzivaji int/Fraction.
Intervaly crash pair jsou presne racionalni a zaokrouhlene smerem ven.
"""
from fractions import Fraction as F
from itertools import product
from math import isqrt, log2, sqrt, pi
from pathlib import Path
import argparse
import copy
import json


def require(condition, message):
    """Na rozdil od assert se kontrola nevypne ani pri python -O."""
    if not condition:
        raise ValueError(message)


# =====================================================================
# 1. STROM. Kazda akce ma vlastni rozdeleni vystupnich symbolu.
# =====================================================================
def make_tree(laws, depth):
    """Vraci historie, jejich presne vahy a deti pro kazdou akci."""
    for law in laws:
        require(all(p > 0 for p in law) and sum(law) == 1, 'Neplatny zakon')
    tree = []

    def add(history, mass, remaining):
        node = len(tree)
        tree.append(dict(history=history, mass=mass, children=[]))
        if remaining:
            for action, law in enumerate(laws):
                children = [add(history + ((action, symbol),), mass*p,
                                remaining-1) for symbol, p in enumerate(law)]
                tree[node]['children'].append(children)
        return node

    add((), F(1), depth)
    return tree


def make_seed(tree):
    """Vyrobi karticky a jejich vahy. Toto je samotna konstrukce.

    INDUKCE: B(v) je nejvetsi vaha jedne karticky pod vrcholem v.
    Na listu muzeme odebrat jeho zbyvajici hmotu r(v).
    Pro kazdou akci si karticka smi vybrat vystup: proto max.
    Musi ale zvladnout VSECHNY mozne akce: proto min.

        B(list) = r(list)
        B(v)    = min_akce max_vystup B(dite)

    Odebereme w=B(koren) ze vsech historii kompatibilnich s kartickou.
    Kazda akce ma prave jedno vybrane dite, proto se zachova tok.
    Alespon jeden kladny list zmizi. Konecny strom => algoritmus skonci.
    Soucet odebranych karticek tedy rekonstruuje puvodni tok presne.
    """
    residual = [node['mass'] for node in tree]
    cards = []
    while residual[0] > 0:
        bottleneck = residual.copy()
        for v in reversed(range(len(tree))):
            children = tree[v]['children']
            if children:
                bottleneck[v] = min(max(bottleneck[c] for c in row)
                                    for row in children)
        weight = bottleneck[0]
        require(weight > 0, 'Kladny tok ma nulovy bottleneck')
        answers, reached = {}, []

        def choose(v):
            reached.append(v)
            if tree[v]['children']:
                answers[v] = []
                for row in tree[v]['children']:
                    y = max(range(len(row)), key=lambda y: bottleneck[row[y]])
                    answers[v].append(y)
                    choose(row[y])

        choose(0)
        require(any(not tree[v]['children'] and residual[v] == weight
                    for v in reached), 'Nezmizel zadny list')
        for v in reached:
            residual[v] -= weight
        require(all(r >= 0 for r in residual), 'Zaporny zbytkovy tok')
        for v, node in enumerate(tree):
            for row in node['children']:
                require(sum(residual[c] for c in row) == residual[v],
                        'Porusena rovnost toku pro nekterou akci')
        cards.append(dict(weight=weight, answers=answers))
    return cards


# =====================================================================
# 2. NEZAVISLY KONTROLOR. Nevola make_seed ani bottleneck.
# =====================================================================
def check_seed(laws, depth, cards):
    """Znovu postavi zadani. Z karticek rekonstruuje kazdou historii.

    Pokud hmoty sedi, pro kazdou historii v a akci a plati
      P(Y=y | historie=v, akce=a) = mass(vay)/mass(v) = laws[a][y].
    To je presne kauzalni pozadavek. Zadna politika se nemuze schovat.
    """
    tree = make_tree(laws, depth)
    require(all(card['weight'] > 0 for card in cards), 'Neplatna vaha')
    require(sum(card['weight'] for card in cards) == 1, 'Vahy nedavaji 1')
    actual = [F(0)] * len(tree)
    for card in cards:
        def visit(v):
            actual[v] += card['weight']
            rows = tree[v]['children']
            if rows:
                require(v in card['answers'], 'Chybi dostupna odpoved')
                reply = card['answers'][v]
                require(len(reply) == len(laws), 'Chybi nektera akce')
                for a, row in enumerate(rows):
                    y = reply[a]
                    require(isinstance(y, int) and 0 <= y < len(row),
                            'Neplatny symbol')
                    visit(row[y])
        visit(0)
    for v, node in enumerate(tree):
        require(actual[v] == node['mass'],
                f"Nesedi historie {node['history']}: {actual[v]} != {node['mass']}")
    return len(tree)


def aggregate(tree, leaf_values, output_operation, action_operation):
    """Backward induction; neobsahuje zadne logaritmy ani zaokrouhlovani."""
    values = list(leaf_values)
    for v in reversed(range(len(tree))):
        rows = tree[v]['children']
        if rows:
            values[v] = action_operation(output_operation(values[c] for c in row)
                                         for row in rows)
    return values[0]


def profile(tree, cards):
    """Presna CDF obalky a obe spektralni nerovnosti.

    Pracujeme s prahem pravdepodobnosti p misto t=-log2(p).
    List patri do udalosti I<=t prave kdyz mass(list)>=p.
    Obalka ma atom -log2(p) o hmotnosti 'jump'.
    """
    levels = sorted({v['mass'] for v in tree if not v['children']}, reverse=True)
    envelope, previous = [], F(0)
    for p in levels:
        cdf = aggregate(tree, [v['mass'] if v['mass'] >= p else F(0)
                               for v in tree], sum, min)
        require(cdf >= previous, 'CDF neni monotonna')
        if cdf > previous:
            envelope.append((p, cdf-previous))
        previous = cdf
    require(previous == 1, 'Obalka nema celkovou hmotu 1')

    # Converse: informace seedu stochasticky dominuje kazdemu transkriptu.
    for p in set(levels) | {card['weight'] for card in cards}:
        seed_cdf = sum(card['weight'] for card in cards if card['weight'] >= p)
        lower_cdf = sum(jump for level, jump in envelope if level >= p)
        require(seed_cdf <= lower_cdf, 'Porusena spektralni dolni mez')

    # Globalni horni mez: male atomy <= politika s orezanymi listy <= smoothing.
    # Staci uzly w: mezi nimi leva strana konstantni, prava neklesa.
    for delta in {card['weight'] for card in cards}:
        small_atoms = sum(card['weight'] for card in cards if card['weight'] <= delta)
        cut = aggregate(tree, [min(v['mass'], delta) for v in tree], sum, max)
        smoothing = sum(jump*min(F(1), delta/p) for p, jump in envelope)
        require(small_atoms <= cut <= smoothing, 'Porusena soft-tail nerovnost')
    return envelope


def all_policies(tree, v=0):
    """Vycerpavajici enumerace politik na MALÉM stromu; nikoli Bellman DP.

    Politika zvoli jednu akci a zvlastni pokracovani pro kazdy jeji vystup.
    Vraci seznam terminalnich historii pro kazdou politiku.
    """
    rows = tree[v]['children']
    if not rows:
        return [(v,)]
    result = []
    for row in rows:
        for subpolicies in product(*(all_policies(tree, c) for c in row)):
            result.append(tuple(leaf for sub in subpolicies for leaf in sub))
    return result


def check_all_policies(tree, cards, envelope):
    """Druha kontrola: skutecne projdeme kazdou malou adaptivni politiku."""
    policies = all_policies(tree)
    for leaves in policies:
        wanted = {tree[v]['history']: tree[v]['mass'] for v in leaves}
        action_at = {}
        for history in wanted:
            for k, (a, _) in enumerate(history):
                prefix = history[:k]
                require(prefix not in action_at or action_at[prefix] == a,
                        'Enumerator vratil nekonzistentni politiku')
                action_at[prefix] = a
        seen = {h: F(0) for h in wanted}
        for card in cards:
            v = 0
            while tree[v]['children']:
                a = action_at[tree[v]['history']]
                y = card['answers'][v][a]
                v = tree[v]['children'][a][y]
            seen[tree[v]['history']] += card['weight']
        require(seen == wanted, 'Nespravny transkript politiky')
    for p in {tree[v]['mass'] for leaves in policies for v in leaves}:
        brute = min(sum(tree[v]['mass'] for v in leaves if tree[v]['mass'] >= p)
                    for leaves in policies)
        dp = sum(jump for level, jump in envelope if level >= p)
        require(brute == dp, 'Bellman != vycerpavajici enumerace')
    return len(policies)


# =====================================================================
# 3. PRESNA BELLMANOVA TABULKA PRO SKUTECNOU DVOJICI ROZDELENI.
# =====================================================================
def exact_bellman(steps):
    """P=(1/2, 16krat 1/32); Q=(2krat 1/4, 8krat 1/16).

    H(P)=H(Q)=3 presne. Centrovaná informace:
      P: -2 a +2, kazda s pravdepodobnosti 1/2; sigma_P=2.
      Q: -1 a +1, kazda s pravdepodobnosti 1/2; sigma_Q=1.
    Proto je cela CDF v kroku n nasobkem 2^(-n) na cele mrizce.
    Nepouzivame normalni aproximaci k vypoctu teto tabulky.

    adaptive: MIN uvnitr kazdeho kroku = vsechny adaptivni politiky.
    fixed:    MIN az na konci = pouze dve konstantni politiky.
    """
    adaptive = fixed_p = fixed_q = [1]  # CDF bodove hmoty v nule
    denominator = 1
    records = []
    checkpoints = {1, 2, 10, 100, 400, 1000, steps}
    for n in range(1, steps+1):
        size = len(adaptive)+4
        a = [0]*4 + adaptive + [denominator]*4
        p = [0]*4 + fixed_p + [denominator]*4
        q = [0]*4 + fixed_q + [denominator]*4
        adaptive = [min(a[j]+a[j+4], a[j+1]+a[j+3]) for j in range(size)]
        fixed_p = [p[j]+p[j+4] for j in range(size)]
        fixed_q = [q[j+1]+q[j+3] for j in range(size)]
        denominator *= 2
        if n in checkpoints:
            fixed = [min(x, y) for x, y in zip(fixed_p, fixed_q)]
            require(all(0 <= x <= y <= denominator for x, y in zip(adaptive, fixed)),
                    'Adaptivni obalka neni pod pevnou obalkou')
            require(all(x <= y for x, y in zip(adaptive, adaptive[1:])), 'Neplatna CDF')
            # Pro celociselnou X na [L,U]: E X = U - sum_{k=L}^{U-1} F_X(k).
            ea = F(2*n*denominator-sum(adaptive[:-1]), denominator)
            ef = F(2*n*denominator-sum(fixed[:-1]), denominator)
            records.append(dict(n=n, adaptive_mean_exact=str(ea),
                fixed_mean_exact=str(ef), adaptive_over_sqrt_n=float(ea)/sqrt(n),
                fixed_over_sqrt_n=float(ef)/sqrt(n),
                optimum_upper_over_sqrt_n=float(ea)/sqrt(n)+log2(2.718281828459045)/sqrt(n)))
    return records


# =====================================================================
# 4. RACIONALNI INTERVALY: zadny float nerozhoduje o PASS/FAIL.
# =====================================================================
def ln_unit(x):
    require(1 <= x <= 2, 'Logaritmus mimo redukovany interval')
    z = (x-1)/(x+1)
    z2, power, partial = z*z, z, F(0)
    for j in range(48):
        partial += 2*power/(2*j+1)
        power *= z2
    return partial, partial + 2*power/(97*(1-z2))


LN2 = ln_unit(F(2))


def ln_interval(x):
    require(x > 0, 'Neplatny argument logaritmu')
    k = 0
    while x < 1:
        x *= 2
        k -= 1
    while x > 2:
        x /= 2
        k += 1
    lo, hi = ln_unit(x)
    return (lo+min(k*LN2[0], k*LN2[1]), hi+max(k*LN2[0], k*LN2[1]))


def log2_interval(x):
    # Pro mocniny dvojky je logaritmus cele cislo: zachovame presnou rovnost.
    if (x.numerator & (x.numerator-1) == 0 and
            x.denominator & (x.denominator-1) == 0):
        value = F(x.numerator.bit_length()-x.denominator.bit_length())
        return value, value
    lo, hi = ln_interval(x)
    candidates = [lo/LN2[0], lo/LN2[1], hi/LN2[0], hi/LN2[1]]
    return min(candidates), max(candidates)


def sqrt_interval(x):
    scale = 10**40
    k = isqrt(x.numerator*scale*scale//x.denominator)
    return F(k, scale), F(k+1, scale)


def outward(x, places=12, upper=False):
    scale = 10**places
    y = x*scale
    k = -((-y.numerator)//y.denominator) if upper else y.numerator//y.denominator
    sign = '-' if k < 0 else ''
    k = abs(k)
    return f'{sign}{k//scale}.{k%scale:0{places}d}'


def entropy_interval(law):
    lo = hi = F(0)
    for p in law:
        a, b = log2_interval(p)
        lo -= p*b
        hi -= p*a
    return lo, hi


def envelope_mean_interval(envelope):
    lo = hi = F(0)
    for p, mass in envelope:
        a, b = log2_interval(p)
        lo -= mass*b
        hi -= mass*a
    return lo, hi


def certify_crash():
    """Certifikuje aritmetiku analyticke nerovnosti, NE optimalni seedy.

    q je JEDINY koren H(q,q,1-2q)=3/2 v (1/3,1/2).
    H'(q)=2log2((1-2q)/q)<0; oba koncove znaky se overi presne.
    """
    qlo, qhi = F(410, 1000), F(411, 1000)
    for _ in range(65):
        q = (qlo+qhi)/2
        lo, hi = entropy_interval([q, q, 1-2*q])
        if lo > F(3, 2):
            qlo = q
        elif hi < F(3, 2):
            qhi = q
        else:
            raise ValueError('Nutna presnejsi logaritmicka rada')
    require(entropy_interval([qlo,qlo,1-2*qlo])[0] > F(3,2), 'Spatny levy znak')
    require(entropy_interval([qhi,qhi,1-2*qhi])[1] < F(3,2), 'Spatny pravy znak')
    llo = log2_interval(qlo/(1-2*qlo))[0]
    lhi = log2_interval(qhi/(1-2*qhi))[1]
    vlo = 2*qhi*(1-2*qhi)*llo*llo
    vhi = 2*qlo*(1-2*qlo)*lhi*lhi
    blo, bhi = sqrt_interval(vlo)[0], sqrt_interval(vhi)[1]

    def atan_interval(x):
        value = sum(((-1)**j*x**(2*j+1)/(2*j+1) for j in range(40)), F(0))
        return value, value+x**81/81

    a5, a239 = atan_interval(F(1,5)), atan_interval(F(1,239))
    pi_lo, pi_hi = 16*a5[0]-4*a239[1], 16*a5[1]-4*a239[0]
    clo = sqrt_interval(2/pi_hi)[0]*(F(1,2)-bhi)
    chi = sqrt_interval(2/pi_lo)[1]*(F(1,2)-blo)
    require(F(9,20) < blo < bhi < F(1,2), 'Nesedi variance')
    for p in [F(1,2),F(1,4),qlo,qhi,1-2*qlo,1-2*qhi]:
        lo, hi = log2_interval(p)
        require(max(abs(-lo-F(3,2)), abs(-hi-F(3,2))) < 1, 'B neni <1')
    require(pi_lo > 2 and LN2[0] > F(2,3), 'Nesedi jednoduche konstanty')
    require(F(20,19)*(F(1,4)/(3*F(9,20)**2)+F(1,2)) == F(4430,4617) < 1,
            'Nesedi konstanta Taylorovy chyby')
    log10 = ln_interval(F(10))
    rows = []
    for power in [16,24,32,48]:
        root_n, fourth_root = 10**(power//2), 10**(power//4)
        tlo, thi = sqrt_interval(power*log10[0])[0], sqrt_interval(power*log10[1])[1]
        error = F(17,2)*thi/fourth_root + F(3,root_n)/tlo + F(2,root_n)
        rows.append(dict(n=f'10^{power}', lower=outward(max(F(0),clo-error)),
                         upper=outward(chi+error, upper=True),
                         half_coefficient_excluded=clo-error > chi/2))
    return dict(q_interval=[outward(qlo,20),outward(qhi,20,True)],
                sigma_P_exact='1/2',
                sigma_Q_interval=[outward(blo,20),outward(bhi,20,True)],
                c_interval=[outward(clo,20),outward(chi,20,True)], rows=rows,
                scope='Rational arithmetic of analytic enclosures; not numerical optimization of C_n')


def proof_map():
    """Obecne kroky, ktere konecny beh programu nemuze nahradit.

    I. GLOBALNI REALIZACE (libovolny konecny strom)
    F_*(t)=min_pi P(I_pi<=t), Z~F_*. Kazdy seed ma I_S>=I_pi,
    proto H(S)>=E Z. Pro greedy seed S_G po odebrani vah >delta
    zbyva rezidualni tok s B(koren)<=delta. Politika volici MIN-akci
    v bottleneck rekurzi ma vsechny terminalni rezidualy <=delta.
    Proto
      sum_{w_j<=delta} w_j <= max_pi sum_list min(p_list,delta)
                           <= E min(1,delta*2^Z).
    Posledni nerovnost je stochastic dominance a rostouci funkce.
    Pri delta=2^-t je prava strana P(Z+E>=t), E~Exp(rate ln2).
    Integrace => E Z <= C <= H(S_G) <= E Z + 1/ln2.
    Dulezite: jeden globalni entropy gap, ne poplatek za kazdy blok.

    II. OSTRA ASYMPTOTIKA (stejna entropie, a>=b>0)
    T f(x)=min_A E f(x-D_A), G_n=T^n 1_[0,infty).
    K=sqrt(2/pi)/(a+b). Kandidat limitni CDF ma hustotu
      F'(x)=K exp(-x^2/(2b^2)) pro x<0,
      F'(x)=K exp(-x^2/(2a^2)) pro x>0.
    Jeji stredni hodnota je K(a^2-b^2)=sqrt(2/pi)(a-b).
    u_t(x)=F(x/sqrt(t)) splnuje u_t'=min(a^2 u_xx,b^2 u_xx)/2.
    F'' je globalne Lipschitz s konstantou <=2K/b^2, i pres nulu.
    Taylor + nulovy prvni moment dava pro t>=1
      ||T u_t-u_(t+1)||_infty <= A t^-3/2,
      A=K(m3/(3b^2)+a), m3=max E|D_A|^3.
    T je monotonne a neexpanzivni => chyba po n krocich <=3A/sqrt(tau).
    Posunute hladke CDF s odchylkou eta obklopi pocatecni schod.
    Odtud pro VSECHNA x,n,tau>=1,R>=0:
      F((x-R)/sqrt(n+tau))-d <= G_n(x)
                            <= F((x+R)/sqrt(n+tau))+d,
      d=max(F(-R/sqrt(tau)),1-F(R/sqrt(tau)))+3A/sqrt(tau).
    Nejdrive n->infty, potom tau a R: bodova konvergence k F.
    Azuma-Hoeffding pri |D|<=B dava oba ocasy <=exp(-t^2/(2B^2))
    po normalizaci sqrt(n), stejnomerne pres politiky.
    Integrace ocasu je tedy legitimni. Dostavame e_n/sqrt(n)->E F.
    Spojeni s I dava ostry koeficient pro C_n, ne jen pro jednu architekturu.

    III. KONKRETNI CRASH-PAIR ERROR, VSECHNA n>=16
    A<1, B<1, a=1/2, b>9/20; certify_crash overuje tyto vstupy.
    Volime tau=sqrt(n), eta0=n^-1/4,
      R=a sqrt(2 tau ln(2/eta0)), s=sqrt(1+n^-1/2), r=R/sqrt(n).
    Pak d<=4n^-1/4. Integrace CDF sandwich na [-T,T], T=sqrt(ln n),
    dava chybu <=r+2Td. Posun stoji r, protoze integral rozdilu
    dvou CDF posunutych o r je presne r.
    Ocasy Y=Z_n/sqrt(n) a sF dohromady <=3/sqrt(n ln n).
    r<=0.5 n^-1/4 sqrt(ln n), 2Td<=8 n^-1/4 sqrt(ln n).
    Zmena stredni hodnoty sF oproti F <=0.5/sqrt(n).
    Realizace stoji <=log2(e)/sqrt(n)<1.5/sqrt(n).
    Tedy |(C_n-1.5n)/sqrt(n)-c| <=
       (17/2)sqrt(ln n)/n^1/4 + 3/sqrt(n ln n) + 2/sqrt(n).
    Toto je analyticky dukaz pro vsechna n>=16; tabulka overuje aritmetiku.
    """
    return proof_map.__doc__


# =====================================================================
# 5. JEDEN SPUSTITELNY PRIBEH A PRENOSITELNY CERTIFIKAT.
# =====================================================================
def encode_certificate(laws, depth, cards):
    return dict(laws=[[str(p) for p in law] for law in laws], depth=depth,
                cards=[dict(weight=str(c['weight']),
                            answers={str(v): ys for v,ys in c['answers'].items()})
                       for c in cards])


def read_certificate(data):
    laws = [[F(p) for p in law] for law in data['laws']]
    cards = [dict(weight=F(c['weight']),
                  answers={int(v): ys for v,ys in c['answers'].items()}) for c in data['cards']]
    return laws, data['depth'], cards


def explain_case(name, laws, depth, enumerate_policies=False):
    tree = make_tree(laws, depth)
    cards = make_seed(tree)
    histories = check_seed(laws, depth, cards)
    envelope = profile(tree, cards)
    h_lo, h_hi = entropy_interval([c['weight'] for c in cards])
    e_lo, e_hi = envelope_mean_interval(envelope)
    require(h_lo >= e_hi, 'Intervaly necertifikuji H(seed)>=E(obalky)')
    require(h_hi-e_lo <= 1/LN2[1], 'Intervaly necertifikuji gap<=log2(e)')
    print(f'\n{name}: hloubka {depth}, {histories} historii, {len(cards)} karticek')
    print('  Vsechny historie: PRESNE OK. Spektralni nerovnosti: PRESNE OK.')
    print(f'  E(obalky) v [{outward(e_lo)}, {outward(e_hi,upper=True)}] bitu')
    print(f'  H(seedu)  v [{outward(h_lo)}, {outward(h_hi,upper=True)}] bitu')
    print('  0 <= H(seedu)-E(obalky) <= log2(e): RACIONALNE OVERENO.')
    policies = None
    if enumerate_policies:
        policies = check_all_policies(tree, cards, envelope)
        print(f'  Vsech {policies} adaptivnich politik: transkripty i obalka PRESNE OK.')
    # Kontrolor musi odmitnout pokazenou dostupnou odpoved, i kdyz vahy sedi.
    broken = copy.deepcopy(cards)
    last = max(broken[0]['answers'], key=lambda v: len(tree[v]['history']))
    broken[0]['answers'][last][0] = (broken[0]['answers'][last][0]+1) % len(laws[0])
    try:
        check_seed(laws, depth, broken)
    except ValueError:
        print('  Zamerne poskozena odpoved: kontrolor spravne ODMITL.')
    else:
        raise ValueError('Kontrolor prijal poskozeny certifikat')
    record = dict(name=name, depth=depth, histories=histories, cards=len(cards),
                  policies_exhausted=policies, exact_checks='PASS',
                  envelope_mean_interval=[outward(e_lo),outward(e_hi,upper=True)],
                  seed_entropy_interval=[outward(h_lo),outward(h_hi,upper=True)])
    return record, encode_certificate(laws, depth, cards), envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=1000, help='Bellmanova hloubka; default 1000')
    parser.add_argument('--out', type=Path, default=Path('causal_results'), help='Adresar vysledku')
    parser.add_argument('--check', type=Path, help='Pouze nezavisle overit zadany JSON certifikat')
    parser.add_argument('--proof', action='store_true', help='Vypsat obecny dukazovy most')
    args = parser.parse_args()
    if args.check:
        laws, depth, cards = read_certificate(json.loads(args.check.read_text()))
        count = check_seed(laws, depth, cards)
        print(f'PASS: presne rekonstruovano {count} historii; vahy seedu davaji 1.')
        print('Tento rezim kontroluje kauzalni realizaci, nikoli optimalitu entropie.')
        return
    require(2 <= args.steps <= 20000, '--steps musi byt mezi 2 a 20000')
    if args.proof:
        print(proof_map())
    print('KAUZALNI ENTROPIE: vsechny rozhodujici kontroly jsou presne.')
    print('Model: odpoved zavisi na cele historii, stejny seed pro vsechny politiky.')
    print('Vahy a transkripty: Fraction. Bellman: cela cisla. Zadny fit asymptoty.')
    toy = [[F(1,2),F(1,2)], [F(1,3),F(2,3)]]
    one, certificate, _ = explain_case('A. Malý citelny priklad (entropie se nerovnaji)', toy, 3, True)
    print('  Prvnich pet karticek: vaha; odpoved pri akci P; odpoved pri akci Q')
    for card in certificate['cards'][:5]:
        print('   ', card['weight'], card['answers']['0'])

    laws = [[F(1,2)]+[F(1,32)]*16, [F(1,4)]*2+[F(1,16)]*8]
    two, dyadic_certificate, dyadic_envelope = explain_case('B. H(P)=H(Q)=3, sigma=2 a 1', laws, 2)
    print('\nC. Presna Bellmanova rekurze; desetinne zobrazeni presnych vysledku')
    print('       n    adaptivni e_n/sqrt(n)    dve pevne politiky /sqrt(n)')
    bellman = exact_bellman(args.steps)
    for row in bellman:
        print(f"{row['n']:8d}    {row['adaptive_over_sqrt_n']:20.12f}    {row['fixed_over_sqrt_n']:20.12f}")
    # Presny most: obalka konkretniho stromu v B musi byt 3n + Bellman.
    tree_mean = F(0)
    for p, mass in dyadic_envelope:
        require(p.numerator == 1 and p.denominator & (p.denominator-1) == 0,
                'List nema dyadickou pravdepodobnost')
        tree_mean += mass*(p.denominator.bit_length()-1)
    n2 = next(row for row in bellman if row['n'] == 2)
    require(tree_mean == 6+F(n2['adaptive_mean_exact']), 'Strom a Bellman nesouhlasi')
    print('  Most konkretni strom <-> Bellman pro n=2: PRESNE OK.')
    print(f'  Analyticky limit: {sqrt(2/pi):.12f}; dve pevne politiky: {1/sqrt(2*pi):.12f}.')
    print('  Optimum C_n-3n lezi mezi e_n a e_n+log2(e); e_n neni samo C_n-3n.')

    print('\nD. Irracionalni crash pair: racionalni intervaly z konecneho odhadu', flush=True)
    crash = certify_crash()
    print('  sigma_P = 1/2; sigma_Q v', crash['sigma_Q_interval'])
    print('  c v', crash['c_interval'])
    for row in crash['rows']:
        print(f"  n={row['n']:>5s}: [{row['lower']}, {row['upper']}], polovina vyloucena: {row['half_coefficient_excluded']}")
    print('  Tato tabulka je analyticke ohraniceni C_n, ne jeho prime spocitani.')

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {'toy_seed.json':certificate, 'dyadic_seed.json':dyadic_certificate,
               'results.json':dict(finite_cases=[one,two], bellman=bellman, crash=crash,
                  status='PASS_EXACT_FINITE_CHECKS_AND_RATIONAL_BOUNDS',
                  universal_proof_assistant_certificate=False)}
    for name, data in outputs.items():
        (args.out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    (args.out/'proof_map.txt').write_text(proof_map(), encoding='utf-8')
    print('\nHOTOVO. Certifikaty a presne zlomky:', args.out)
    print('Nezavisla kontrola: python causal_entropy_explained.py --check '+str(args.out/'toy_seed.json'))
    print('Obecny dukazovy most: python causal_entropy_explained.py --proof --steps 2')


if __name__ == '__main__':
    main()
