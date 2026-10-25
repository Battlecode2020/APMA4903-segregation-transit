"""Reproducible fixed-count allocation and random-matrix experiments for PSSI."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.linalg import solve, eigvals
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.stats import norm, kstest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/k_approximation'; OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(20261024)

def rho(P,r):
    n=len(r)
    if n<20: return float(np.max(np.abs(np.linalg.eigvals(P*r))))
    op=LinearOperator((n,n),matvec=lambda v:P@(r*v),dtype=float)
    w,v=eigs(op,k=1,which='LM',v0=np.ones(n),tol=1e-10)
    val=float(w[0].real)
    residual=float(np.max(np.abs(op@v[:,0]-w[0]*v[:,0])))
    assert abs(w[0].imag)<1e-8 and residual<1e-8
    return val

def draws(n,K,B):
    total=int(n.sum()); count=round(K*total)
    x=rng.multivariate_hypergeometric(n,count,size=B,method='marginals')
    assert np.all(x.sum(axis=1)==count)
    return x/n,count/total

def summary(values,K,**kw):
    sd=float(np.std(values,ddof=1)); mean=float(np.mean(values)); se=sd/np.sqrt(len(values))
    return dict(K=K,mean=mean,sd=sd,se=se,lo=mean-1.96*se,hi=mean+1.96*se,
                q025=float(np.quantile(values,.025)),q975=float(np.quantile(values,.975)),
                adjusted_bias=(mean-K)/(1-K),**kw)

def main():
    # Two connected weighted graphs on four locations, five residents each.
    n=np.full(4,5,dtype=int)
    Fs={'fully_mixed':np.ones((4,4)), 'two_communities':np.where(np.arange(4)[:,None]//2==np.arange(4)[None,:]//2,1.,.02)}
    rows=[]; raw=[]
    import itertools, math
    for name,F in Fs.items():
        P=F*n;P/=P.sum(axis=1,keepdims=True)
        np.savetxt(OUT/f'toy_{name}_F.csv',F,delimiter=',')
        for count in range(1,20):
            K=count/20; R,_=draws(n,K,10000)
            vals=np.array([rho(P,r) for r in R])
            exact=0.;mass=0.
            for b in itertools.product(range(6),repeat=4):
                if sum(b)!=count:continue
                prob=math.prod(math.comb(5,x) for x in b)/math.comb(20,count)
                exact+=prob*rho(P,np.array(b)/5);mass+=prob
            assert abs(mass-1)<1e-12
            if name=='fully_mixed':assert np.max(abs(vals-K))<1e-12
            rows.append(summary(vals,K,graph=name,exact=exact,B=len(vals)))
        print('toy',name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'toy_summary.csv',index=False)
    # Inventory selects largest primary PSSI matrix, excluding scenario duplicates.
    paths=list((ROOT/'outputs/multicity').glob('*/analysis/analysis_matrices.npz'))
    inv=[]
    for path in paths:
        with np.load(path) as z:inv.append((len(z['P']),path))
    _,path=max(inv)
    z=np.load(path);P=z['P'];n=z['population'].astype(np.int64);m=len(n);N=int(n.sum())
    assert np.allclose(P.sum(axis=1),1) and np.all(n>0)
    print('selected',path,m,N,flush=True)
    pi=solve(np.eye(m)-P.T+np.ones((m,m)),np.ones(m));pi/=pi.sum()
    assert np.max(abs(pi@P-pi))<1e-10 and pi.min()>0
    Pi=np.tile(pi,(m,1)); H=solve(np.eye(m)-P+Pi,P-Pi)
    assert np.max(abs((np.eye(m)-P+Pi)@H-(P-Pi)))<1e-9
    C0=(N*np.diag(1/n)-np.ones((m,m)))/(N-1)
    biascoef=float(np.sum(pi[:,None]*H*C0))
    varcoef=float(pi@C0@pi)
    spectrum=eigvals(P);others=np.delete(spectrum,np.argmin(abs(spectrum-1)))
    np.savez_compressed(OUT/'large_matrix_theory.npz',pi=pi,H=H,spectrum=spectrum,population=n)
    results=[];raw=[]
    for Ktarget in [.1,.3,.5,.7,.9]:
        R,K=draws(n,Ktarget,400)
        vals=np.array([rho(P,r) for r in R]); e=R-K
        approx=K+e@pi+np.sum((e@H.T)*e*pi,axis=1)/K
        pred=K+(1-K)*biascoef;sd=np.sqrt(K*(1-K)*varcoef)
        results.append(summary(vals,K,B=len(vals),predicted_mean=pred,predicted_sd=sd,
                               approximation_rmse=float(np.sqrt(np.mean((vals-approx)**2)))))
        raw.extend(dict(K=K,value=v,second_order=a) for v,a in zip(vals,approx))
        print(results[-1],flush=True)
    pd.DataFrame(results).to_csv(OUT/'large_summary.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'large_draws.csv',index=False)
    # Exponential ensemble: directed independent raw affinities, actual populations.
    # Independent graph+allocation per replicate, so standard errors are iid MC errors.
    expvals=[]
    for b in range(300):
        F=rng.exponential(size=(m,m));Q=F*n;Q/=Q.sum(axis=1,keepdims=True)
        R,K=draws(n,.5,1);expvals.append(rho(Q,R[0]))
    pd.DataFrame({'value':expvals}).to_csv(OUT/'exponential_draws.csv',index=False)
    expsum=summary(expvals,K,B=len(expvals))
    cost=np.load(path.parent/'cost_matrices.npz');print(cost.files,flush=True)
    F=np.exp(cost['log_F']);off=F[~np.eye(m,dtype=bool)];scaled=off/off.mean()
    # Descriptive discrepancy only: dependent entries invalidate iid KS p-values.
    quantiles=[.1,.5,.9,.99]
    diagnostics=dict(empirical_cv=float(off.std()/off.mean()),exponential_cv=1,
        empirical_quantiles=np.quantile(scaled,quantiles).tolist(),exponential_quantiles=(-np.log(1-np.array(quantiles))).tolist(),
        ks_descriptive=float(kstest(scaled,'expon').statistic),reciprocity=float(np.corrcoef(F.ravel(),F.T.ravel())[0,1]))
    meta=dict(seed=20261024,matrix=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dimension=m,N=N,min_population=int(n.min()),small_populations=int((n<100).sum()),
        second_eigenvalue_modulus=float(max(abs(others))),bias_coefficient=biascoef,variance_coefficient=varcoef,
        stationary_vs_population_l1=float(np.sum(abs(pi-n/N))),exponential=expsum,weight_diagnostics=diagnostics,
        inventory=[dict(dimension=s,path=str(p.relative_to(ROOT))) for s,p in sorted(inv)])
    (OUT/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':
    main()
