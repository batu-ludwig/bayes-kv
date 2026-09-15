import numpy as np
rng = np.random.default_rng(3)
nk, dk = 8192, 128
V = rng.standard_normal((nk, dk))
s = rng.standard_normal(nk) * 6.0
p = np.exp(s - s.max()); p /= p.sum()
mu = p @ V
v   = float((p*(V**2).sum(1)).sum() - (mu**2).sum())      # tr(Sigma)
vmax= float(np.linalg.norm(V - mu, axis=1).max())
nmu = float(np.linalg.norm(mu))
neff= float(np.exp(-(p*np.log(p+1e-300)).sum()))
print(f"realistic peaked head:  n_eff={neff:.1f}  ||mu||={nmu:.2f}  tr(S)={v:.1f}  V_max={vmax:.1f}\n")

def bound_S(eps, delta):
    """smallest S s.t. Bernstein bound gives relative error <= eps w.p. 1-delta"""
    L = np.log(2/delta); t = eps*nmu
    for S in range(1, 10**7):
        if np.sqrt(2*v*L/S) + 2*vmax*L/(3*S) <= t: return S
    return None

def empirical_S(eps, delta, trials=3000):
    """smallest S (systematic) where the eps-quantile is actually met"""
    cp = np.cumsum(p)
    for S in [16,32,64,128,256,512,1024,2048]:
        T = (np.arange(S)[None,:] + rng.random((trials,1)))/S
        idx = np.searchsorted(cp, T, side='right')
        err = np.linalg.norm(V[idx].mean(1)-mu, axis=1)/nmu
        if np.quantile(err, 1-delta) <= eps: return S
    return ">2048"

print(f"  {'target':>16} {'Bernstein says S >=':>22} {'actually needed':>17}")
for eps, delta in [(0.5,0.01),(0.25,0.01),(0.10,0.01),(0.10,0.0001)]:
    print(f"  eps={eps:<5} d={delta:<7} {bound_S(eps,delta):>22,} {str(empirical_S(eps,delta)):>17}")
