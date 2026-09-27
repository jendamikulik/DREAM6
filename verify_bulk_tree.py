#!/usr/bin/env python3
"""Rational certificate for bulk_log_concavity.tex; standard library only.
Run this file directly. No floating-point values decide any assertion.
The JSON output contains exact rational mass bounds, not coefficient lists.
"""
from fractions import Fraction as F
from dataclasses import dataclass

@dataclass
class IV:
 lo:F
 hi:F
 def __init__(self,a,b=None):self.lo=F(a);self.hi=F(a if b is None else b);assert self.lo<=self.hi
 def __add__(s,t):
  if not isinstance(t,IV):t=IV(t)
  return IV(s.lo+t.lo,s.hi+t.hi)
 __radd__=__add__
 def __neg__(s):return IV(-s.hi,-s.lo)
 def __sub__(s,t):return s+-asiv(t)
 def __rsub__(s,t):return asiv(t)+-s
 def __mul__(s,t):
  t=asiv(t);v=[s.lo*t.lo,s.lo*t.hi,s.hi*t.lo,s.hi*t.hi];return IV(min(v),max(v))
 __rmul__=__mul__
 def __truediv__(s,t):
  t=asiv(t);assert t.lo>0
  return s*IV(1/t.hi,1/t.lo)
 def square(s):
  return IV(0 if s.lo<=0<=s.hi else min(s.lo*s.lo,s.hi*s.hi),max(s.lo*s.lo,s.hi*s.hi))
 def mid(s):return (s.lo+s.hi)/2

def asiv(x):return x if isinstance(x,IV) else IV(x)
def dist(c,mu):
 if mu.lo<=c<=mu.hi:return F(0)
 return min(abs(c-mu.lo),abs(c-mu.hi))

def run(H=10):
 b=64;delta=IV(F(1,2**b),F(2,2**b))
 assert delta.lo*(2-delta.lo)**b<1<delta.hi*(2-delta.hi)**b
 assert (b+1)*delta.hi<2 and delta.hi<=F(1,b*b)
 r=1-delta;q=r/(1+r)
 assert q.lo>=F(49,100) and b*q.lo-2>=F(7*b,15)
 assert F(225,49)*(F(5,4*b)+F(1,4))<F(5,4)
 mu0,mu1=r,IV(1);v0,v1=r*delta,IV(0);alpha0=alpha1=1;n=2
 print('All arithmetic below uses rational interval endpoints.')
 rows=[]
 for h in range(1,H+1):
  D=mu1-mu0
  mu0,mu1=b*((1-q)*mu0+q*mu1),1+b*mu0
  v0,v1=b*((1-q)*v0+q*v1+q*(1-q)*D.square()),b*v0
  alpha0,alpha1=b*max(alpha0,alpha1),1+b*alpha0
  alpha=max(alpha0,alpha1);n=1+b*n
  D=mu1-mu0;d=dist(0,D)
  if d<40:continue
  K=int(d//4)-2
  c0=int(mu0.mid()//1);c1=int(mu1.mid()//1);cm=int(((mu0.mid()+mu1.mid())/2)//1)
  e0=max(abs(c0-mu0.lo),abs(c0-mu0.hi));e1=max(abs(c1-mu1.lo),abs(c1-mu1.hi))
  assert K>e0 and K>e1
  flank0=(1-q.hi)*(1-v0.hi/(K-e0)**2)
  flank1=q.lo*(1-v1.hi/(K-e1)**2)
  z0=dist(cm,mu0)-K;z1=dist(cm,mu1)-K
  assert z0>0 and z1>0
  middle=max(v0.hi/z0**2,v1.hi/z1**2)
  ordered=sorted([c0,cm,c1]);assert ordered[0]+2*K<ordered[1] and ordered[1]+2*K<ordered[2]
  assert 0<=ordered[0]-K<ordered[-1]+K<=alpha
  valley=min(flank0,flank1)>middle
  below99=100*(ordered[-1]+K)<99*alpha
  row={'h':h,'vertices':n,'alpha':alpha,'hull':[ordered[0]-K,ordered[-1]+K],
       'left_mass_lower':str(flank0),'right_mass_lower':str(flank1),'middle_mass_upper':str(middle),
       'valley_certified':valley,'hull_below_99_percent':below99}
  if h==8:
   assert min(flank0,flank1)>F(37,100) and middle<F(26,100)
   assert valley and below99
  rows.append(row)
  print(h,n,alpha,row['hull'],'mass bounds',round(float(flank0),5),round(float(flank1),5),round(float(middle),5),'valley',valley,'below99',below99)
 assert any(row['valley_certified'] and row['hull_below_99_percent'] for row in rows)
 return rows
def polynomial_moment_check():
 # Independent polynomial calculations verify the moment recursions.
 def mul(a,b):
  c=[0]*(len(a)+len(b)-1)
  for i,x in enumerate(a):
   for j,y in enumerate(b):c[i+j]+=x*y
  return c
 def add(a,b):
  c=[0]*max(len(a),len(b))
  for i,x in enumerate(a):c[i]+=x
  for i,x in enumerate(b):c[i]+=x
  return c
 def moments(p,z):
  w=[v*z**k for k,v in enumerate(p)];Z=sum(w)
  mu=sum(k*v for k,v in enumerate(w))/Z
  va=sum(k*k*v for k,v in enumerate(w))/Z-mu*mu
  return Z,mu,va
 z=F(3,2);A=[1,1];B=[0,1]
 for h in range(4):
  za,ma,va=moments(A,z);zb,mb,vb=moments(B,z);q=zb/(za+zb)
  newA=mul(add(A,B),add(A,B));newB=[0]+mul(A,A)
  _,na,nva=moments(newA,z);_,nb,nvb=moments(newB,z)
  assert na==2*((1-q)*ma+q*mb) and nb==1+2*ma
  assert nva==2*((1-q)*va+q*vb+q*(1-q)*(mb-ma)**2)
  assert nvb==2*va
  A,B=newA,newB
 print('PASS: independent exact polynomial moment checks.')

if __name__=='__main__':
 import json
 from pathlib import Path
 polynomial_moment_check()
 rows=run()
 out=Path(__file__).with_name('bulk_certificate.json')
 out.write_text(json.dumps(rows,indent=2))
 print('PASS: rational interval certificate; wrote',out.name)
