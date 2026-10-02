from fractions import Fraction as F
from itertools import combinations_with_replacement

# Exact rational-interval verifier for the scalar/local inequalities in
# "The second connectivity tax".  No floating point is used.

R = F(231,500)
HALF_MINUS_R = F(19,500)

# ---------- rational interval arithmetic ----------
def add(a,b): return (a[0]+b[0], a[1]+b[1])
def neg(a): return (-a[1],-a[0])
def sub(a,b): return add(a,neg(b))
def scale(a,q):
    if q>=0: return (a[0]*q,a[1]*q)
    return (-a[1]*(-q),-a[0]*(-q))
def mul(a,b):
    z=[a[i]*b[j] for i in (0,1) for j in (0,1)]
    return (min(z),max(z))
def divpos(a,b):
    assert b[0]>0
    z=[a[i]/b[j] for i in (0,1) for j in (0,1)]
    return (min(z),max(z))

def logI(x,n=90):
    assert x>0
    y=(x-1)/(x+1)
    ay=abs(y)
    s=F(0)
    yp=y
    for k in range(n):
        s += yp/F(2*k+1)
        yp *= y*y
    s*=2
    rem = 2*(ay**(2*n+1))/F(2*n+1)/(1-ay*ay)
    return (s-rem,s+rem)

LN2=logI(F(2))
TWO_LN2=scale(LN2,F(2))

def over2ln2(log_interval): return divpos(log_interval,TWO_LN2)

def ell(d):
    # ell_d = [log((5/4)/(1+2^-d))]_+
    x=F(5,4)/(1+F(1,2**d))
    if x<=1: return (F(0),F(0))
    return logI(x)

def s_charge(d):
    # s_d = (1/2-r)(d-1) - ell_d/(2 log 2)
    return sub((HALF_MINUS_R*(d-1),)*2, over2ln2(ell(d)))

def positive_credit(d):
    s=s_charge(d)
    if s[0]>=0: return scale(s,F(1,d))
    if s[1]<=0: return (F(0),F(0))
    raise AssertionError(('undetermined sign',d,s))

def deficit(d):
    s=s_charge(d)
    if s[1]<=0: return (-s[1],-s[0])
    if s[0]>=0: return (F(0),F(0))
    raise AssertionError(('undetermined sign',d,s))

D4=deficit(4)
D3=deficit(3)
assert D4[0]>D3[1]>0
D=D4

# kappa = log(290/289)/(2log2)
KAPPA=over2ln2(logI(F(290,289)))

# Exact pair mutual information at cavity odds a=b=1/8:
# I=(128 log(290/289)+16 log(145/153)+log(145/81))/145.
mi_num=(F(0),F(0))
mi_num=add(mi_num,scale(logI(F(290,289)),F(128)))
mi_num=add(mi_num,scale(logI(F(145,153)),F(16)))
mi_num=add(mi_num,logI(F(145,81)))
MI=scale(mi_num,F(1,145))
IOTA=over2ln2(MI)
assert IOTA[0]>0

# collision contribution of a factor with child degrees ds.
def collision(ds):
    A=F(1); B=F(1)
    for d in ds:
        p=F(1,2**d+1)
        A*=1+p; B*=1-p
    x=(A+B)/2
    return over2ln2(logI(x))

# The only use of total correlation in the finite local audit is the
# q=2 deficit-deficit case; IOTA is a rigorous lower bound there.
def clean_slack(ds,parent):
    bud=collision(ds)
    for d in ds: bud=add(bud,positive_credit(d))
    if len(ds)==2 and all(d in (3,4) for d in ds):
        bud=add(bud,IOTA)
    need=(F(0),F(0))
    for j,d in enumerate(ds):
        if j!=parent: need=add(need,deficit(d))
    return sub(bud,need)

# contaminated factor: kR>=1 neighbours in R.  We use half of IOTA
# only in the unique critical case kR=1 and two deficit C-neighbours.
def contam_residual(ds,kR):
    bud=collision(ds)
    for d in ds: bud=add(bud,positive_credit(d))
    if kR==1 and len(ds)==2 and all(d in (3,4) for d in ds):
        bud=add(bud,scale(IOTA,F(1,2)))
    need=(F(0),F(0))
    for d in ds: need=add(need,deficit(d))
    return sub(need,bud)

# zeta = 2D4-kappa-iota/2, upper interval.
ZETA=sub(sub(scale(D,F(2)),KAPPA),scale(IOTA,F(1,2)))
assert ZETA[0]>0

# 1. Scalar signs.  d=3,4 are the only negative charges up to 80.
for d in range(1,81):
    s=s_charge(d)
    if d in (3,4): assert s[1]<0,(d,s)
    else: assert s[0]>=0,(d,s)
# Tail: for d>=81 ell_d < log(5/4), while (19/500)(d-1)
# is already much larger. Check the endpoint exactly; monotonic afterwards.
tail=sub((HALF_MINUS_R*80,)*2,over2ln2(logI(F(5,4))))
assert tail[0]>0

# d>=6 positive credit alone is at least D4.
for d in range(6,81):
    assert sub(positive_credit(d),D)[0]>=0,(d,positive_credit(d),D)

# 2. Clean local factor audit.  After stripping degree>=6 neighbours
# (each has credit >=D), the residual critical degrees are 1..5.
# q>=7 is covered by the analytic collision tail in the note; enumerate q<=6.
min_clean=None
arg_clean=None
for q in range(2,7):
    for ds in combinations_with_replacement(range(1,6),q):
        for parent in range(q):
            sl=clean_slack(ds,parent)
            if min_clean is None or sl[0]<min_clean:
                min_clean=sl[0]; arg_clean=(ds,parent,sl)
            assert sl[0]>=0,(ds,parent,sl)

# 3. Contaminated factor audit.  Residual per R-incidence <= ZETA.
min_margin=None; arg_cont=None
for kR in (1,2,3,4):
    for q in range(1,7):
        for ds in combinations_with_replacement(range(1,6),q):
            res=contam_residual(ds,kR)
            margin=sub(scale(ZETA,F(kR)),res)
            if not (kR==1 and ds==(4,4)) and (min_margin is None or margin[0]<min_margin):
                min_margin=margin[0];arg_cont=(kR,ds,res,margin)
            if kR==1 and ds==(4,4):
                # exact equality by the definition of ZETA; interval dependency
                # would otherwise create an artificial sign ambiguity.
                pass
            else:
                assert margin[0]>=0,(kR,ds,res,margin)

# 4. R-core budget.  For every nonisolated B-vertex, k=deg_K>=1.
# s_d + (19/1000) k >= zeta*d.  Check d<=80 and k=1 (worst);
# k>1 only increases the left side.
for d in range(1,81):
    lhs=add(s_charge(d),(F(19,1000),)*2)
    rhs=scale(ZETA,F(d))
    assert sub(lhs,rhs)[0]>=0,(d,lhs,rhs)
# Tail d>=81: s_d/d tends to 19/500 and is increasing past this point;
# endpoint check plus zeta<19/500 suffices.
assert sub((F(19,500),)*2,ZETA)[0]>0

# 5. Critical margins and target.
# clean (4,4): kappa+iota > D4
assert sub(add(KAPPA,IOTA),D)[0]>0
# contaminated (4,4), one R neighbour: residual <= zeta by definition
# and R-core d=4,k=1 has strict room.
r4=add(s_charge(4),(F(19,1000),)*2)
assert sub(r4,scale(ZETA,F(4)))[0]>0

def dec(q,places=18):
    from decimal import Decimal, getcontext
    getcontext().prec=places+8
    return Decimal(q.numerator)/Decimal(q.denominator)

print('PASS')
print('target r = 231/500 = 0.462 exactly')
print('only negative scalar degrees: 3,4')
print('critical clean case: (4,4), margin >', dec(min_clean,16))
print('critical contaminated case: kR=1,(4,4), exact equality in zeta definition')
print('smallest strict contaminated audit margin >', dec(min_margin,16), 'at', arg_cont[:2])
print('D4 ~', dec((D4[0]+D4[1])/2,18))
print('kappa ~', dec((KAPPA[0]+KAPPA[1])/2,18))
print('iota ~', dec((IOTA[0]+IOTA[1])/2,18))
print('zeta ~', dec((ZETA[0]+ZETA[1])/2,18))
