"""Independent checks of exact-null identities, covariance and perturbation."""
import itertools, math
import numpy as np
from scipy.linalg import eigvals, solve
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
n=np.array([2,3,4]);N=n.sum();M=4;K=M/N
P=np.tile(n/N,(3,1));cov=np.zeros((3,3));mass=0
for b in itertools.product(*[range(int(x)+1) for x in n]):
    if sum(b)!=M:continue
    prob=math.prod(math.comb(int(ni),bi) for ni,bi in zip(n,b))/math.comb(int(N),M)
    r=np.array(b)/n;eps=r-K;cov+=prob*np.outer(eps,eps);mass+=prob
    assert abs(max(abs(eigvals(P*r)))-K)<1e-12
assert abs(mass-1)<1e-12
C=K*(1-K)/(N-1)*(N*np.diag(1/n)-np.ones((3,3)))
assert np.allclose(cov,C,atol=1e-14)
# Verify the second-order expansion against dense eigensolutions as noise shrinks.
P=np.array([[.7,.2,.1],[.1,.7,.2],[.3,.1,.6]])
pi=solve(np.eye(3)-P.T+np.ones((3,3)),np.ones(3));Pi=np.tile(pi,(3,1))
H=solve(np.eye(3)-P+Pi,P-Pi);e=np.array([.03,-.02,.01]);errs=[]
for t in [1,.5,.25]:
    d=t*e
    pred=K+pi@d+pi@(d*(H@d))/K
    exact=max(abs(eigvals(P*(K+d))))
    errs.append(abs(pred-exact))
assert errs[1]<errs[0]/5 and errs[2]<errs[1]/5
# Saved simulation mean intervals and per-draw approximation checks.
x=pd.read_csv(ROOT/'outputs/k_approximation/large_summary.csv')
assert (x.approximation_rmse<1e-8).all()
assert (abs(x.adjusted_bias)+1.645*x.se/(1-x.K)<.02).all()
toy=pd.read_csv(ROOT/'outputs/k_approximation/toy_summary.csv')
nondeg=toy[(toy.graph=='two_communities') & (toy.se>1e-10)]
assert (abs(nondeg['mean']-nondeg.exact)<4*nondeg.se).all()
print('PASS: complete mixing with unequal populations; exact allocation covariance; cubic perturbation remainder; saved Monte Carlo checks.')
