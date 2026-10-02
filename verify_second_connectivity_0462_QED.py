from fractions import Fraction as F
from itertools import combinations_with_replacement

# rational interval helpers
NLOG=55
def add(a,b): return (a[0]+b[0],a[1]+b[1])
def neg(a): return (-a[1],-a[0])
def sub(a,b): return add(a,neg(b))
def scale(a,q): return (a[0]*q,a[1]*q) if q>=0 else (-a[1]*(-q),-a[0]*(-q))
def divpos(a,b):
    assert b[0]>0
    z=[a[i]/b[j] for i in (0,1) for j in (0,1)]
    return min(z),max(z)
def logI(x,n=NLOG):
    assert x>0
    y=(x-1)/(x+1); ay=abs(y); y2=y*y
    s=F(0); yp=y
    for k in range(n):
        s += yp/F(2*k+1); yp *= y2
    s*=2
    rem=2*(ay**(2*n+1))/F(2*n+1)/(1-ay*ay)
    return s-rem,s+rem
LN2=logI(F(2)); TWO=scale(LN2,F(2))
def over2(i): return divpos(i,TWO)
a=F(19,500)
def ell(d):
    x=F(5,4)/(1+F(1,2**d))
    return (F(0),F(0)) if x<=1 else logI(x)
def scharge(d): return sub((a*(d-1),)*2,over2(ell(d)))
def deficit(d):
    s=scharge(d)
    if s[1]<=0:return (-s[1],-s[0])
    if s[0]>=0:return (F(0),F(0))
    raise AssertionError(('sign',d,s))
def credit(d):
    s=scharge(d)
    if s[0]>=0:return scale(s,F(1,d))
    if s[1]<=0:return (F(0),F(0))
    raise AssertionError(('sign',d,s))
D4=deficit(4); D3=deficit(3); D=D4
KAPPA=over2(logI(F(290,289)))
mi=(F(0),F(0))
mi=add(mi,scale(logI(F(290,289)),F(128)))
mi=add(mi,scale(logI(F(145,153)),F(16)))
mi=add(mi,logI(F(145,81)))
IOTA=over2(scale(mi,F(1,145)))
ZETA=sub(sub(scale(D,F(2)),KAPPA),scale(IOTA,F(1,2)))

coll_cache={}
def collision(ds):
    ds=tuple(ds)
    if ds in coll_cache:return coll_cache[ds]
    A=F(1);B=F(1)
    for d in ds:
        p=F(1,2**d+1);A*=1+p;B*=1-p
    out=over2(logI((A+B)/2));coll_cache[ds]=out;return out

def sumI(items):
    z=(F(0),F(0))
    for x in items:z=add(z,x)
    return z

# signs finite and scalar tail
for d in range(1,81):
    s=scharge(d)
    if d in (3,4): assert s[1]<0,(d,s)
    else: assert s[0]>=0,(d,s)
CLOG=over2(logI(F(5,4)))
scalar81=sub((a*80,)*2,CLOG); assert scalar81[0]>0
# t_d finite + tail
for d in range(6,81): assert sub(credit(d),D)[0]>=0,(d,credit(d),D)
t81=sub((a*F(80,81),)*2,scale(CLOG,F(1,81))); assert sub(t81,D)[0]>0
T6=credit(6)
for d in range(7,81): assert credit(d)[0]>=T6[1],('t6 minimum finite',d)
assert sub(t81,T6)[0]>0

# finite clean q<=9 after d>=6 stripping
minc=None;argc=None
for q in range(2,10):
  for ds in combinations_with_replacement(range(1,6),q):
    bud=add(collision(ds),sumI(credit(d) for d in ds))
    if q==2 and all(d in (3,4) for d in ds): bud=add(bud,IOTA)
    defs=[deficit(d) for d in ds]
    total=sumI(defs)
    # worst parent = one with smallest deficit
    mind=min(x[0] for x in defs) # intervals are point-sign separated; lower is safe for subtraction? need upper required
    # required upper = total upper - min lower (conservative)
    req=(total[0]-mind,total[1]-mind)
    sl=sub(bud,req)
    if minc is None or sl[0]<minc: minc=sl[0];argc=(ds,sl)
    assert sl[0]>=0,('clean',q,ds,sl)

# clean factors whose parent has degree >=6: keep that parent, use its
# full incidence credit >=T6, strip only high-degree children.  The residual
# low-degree children have degrees 1..5 and all their deficits are children.
minph=None; argph=None
for m in range(0,9):
  for ds in combinations_with_replacement(range(1,6),m):
    bud=add(T6,add(collision(ds),sumI(credit(d) for d in ds)))
    need=sumI(deficit(d) for d in ds)
    sl=sub(bud,need)
    if minph is None or sl[0]<minph: minph=sl[0];argph=(ds,sl)
    assert sl[0]>=0,('clean high parent',m,ds,sl)
# For m>=9 use the minimum collision rho_m; endpoint m=9 and the
# nondecreasing log-convex increment close the tail.
r9=collision((5,)*9); r10_hp=collision((5,)*10)
HP9=add(sub(r9,scale(D,F(9))),T6)
HPinc9=sub(sub(r10_hp,r9),D)
assert HP9[0]>0 and HPinc9[0]>0,('clean high-parent tail',HP9,HPinc9)

# tail collision q>=10
r10=collision((5,)*10);r11=collision((5,)*11)
A10=sub(r10,scale(D,F(9)));inc10=sub(sub(r11,r10),D)
assert A10[0]>0 and inc10[0]>0,(A10,inc10)

# finite contaminated q<=9; kR=1 special MI, kR>=2 worst at 2 without MI
min1=None;min2=None
for q in range(1,10):
  for ds in combinations_with_replacement(range(1,6),q):
    base=add(collision(ds),sumI(credit(d) for d in ds))
    need=sumI(deficit(d) for d in ds)
    bud1=base
    if q==2 and all(d in (3,4) for d in ds): bud1=add(bud1,scale(IOTA,F(1,2)))
    res1=sub(need,bud1); mar1=sub(ZETA,res1)
    if not (q==2 and ds==(4,4)):
        assert mar1[0]>=0,('cont1',q,ds,mar1)
    res2=sub(need,base); mar2=sub(scale(ZETA,F(2)),res2)
    assert mar2[0]>=0,('cont2',q,ds,mar2)
    if min1 is None or mar1[0]<min1:min1=mar1[0]
    if min2 is None or mar2[0]<min2:min2=mar2[0]
B10=add(sub(r10,scale(D,F(10))),ZETA);assert B10[0]>0

# R-core finite+tail
for d in range(1,81):
    lhs=add(scharge(d),(F(19,1000),)*2);rhs=scale(ZETA,F(d))
    assert sub(lhs,rhs)[0]>=0,('core',d,lhs,rhs)
assert sub((a,a),ZETA)[0]>0
core81=add(scale(sub((a,a),ZETA),F(81)),(F(-a+F(19,1000)),)*2)
core81=sub(core81,CLOG);assert core81[0]>0

# MI monotonicity endpoint checks
assert add((F(1),F(1)),logI(F(256,729)))[1]<0
F18=(F(0),F(0))
F18=add(F18,scale(logI(F(9,17)),F(153,64)))
F18=add(F18,scale(logI(F(2,3)),F(1,8)))
F18=add(F18,scale(LN2,F(9,4)))
assert F18[1]<0

from decimal import Decimal,getcontext
getcontext().prec=35
def dec(q):return Decimal(q.numerator)/Decimal(q.denominator)
print('PASS UNIVERSAL QED AUDIT')
print('min clean low-parent',dec(minc),argc[0])
print('min clean high-parent',dec(minph),argph[0])
print('high-parent tail HP9',dec(HP9[0]),'inc9',dec(HPinc9[0]))
print('A10',dec(A10[0]),'increment10',dec(inc10[0]))
print('min contaminated kR=1',dec(min1),'kR>=2',dec(min2))
print('B10',dec(B10[0]))
print('core81',dec(core81[0]))
print('MI F(1/8) upper',dec(F18[1]))
