import numpy as np
rng = np.random.default_rng(3)
nk, dk = 8192, 128
V = rng.standard_normal((nk, dk))
s = rng.standard_normal(nk)*6.0
p = np.exp(s-s.max()); p /= p.sum()
mu = p@V; nmu = np.linalg.norm(mu)
v  = float((p*(V**2).sum(1)).sum() - (mu**2).sum())
vmax = float(np.linalg.norm(V-mu, axis=1).max())
cp = np.cumsum(p)
eps, delta = 0.10, 0.01
L = np.log(2/delta); t = eps*nmu

def smallest(pred):
    for S in range(1, 2*10**6):
        if pred(S): return S

# 1. full Bernstein
S_bern = smallest(lambda S: np.sqrt(2*v*L/S) + 2*vmax*L/(3*S) <= t)
# 2. Bernstein without the V_max jump term (isolates its cost)
S_novmax = smallest(lambda S: np.sqrt(2*v*L/S) <= t)
# 3. "typical error" / Markov-free reference: E||err|| ~ sqrt(v/S), no confidence factor
S_typ = smallest(lambda S: np.sqrt(v/S) <= t)

def emp(sampler, S, trials=20000):
    idx = sampler(S, trials)
    return np.linalg.norm(V[idx].mean(1)-mu, axis=1)/nmu
iid  = lambda S,T: rng.choice(nk, size=(T,S), p=p)
sysm = lambda S,T: np.searchsorted(cp, (np.arange(S)[None,:]+rng.random((T,1)))/S, side='right')

def emp_S(sampler):
    lo, hi = 1, 4096
    while lo < hi:
        mid = (lo+hi)//2
        if np.quantile(emp(sampler, mid, 6000), 1-delta) <= eps: hi = mid
        else: lo = mid+1
    return lo

S_iid, S_sys = emp_S(iid), emp_S(sysm)
print(f"target: 10% relative L2 error, 99% confidence   (n_eff=6.6, dk=128)\n")
print(f"  [A] Bernstein bound, as stated            S >= {S_bern:>6}")
print(f"  [B] Bernstein, V_max jump term deleted    S >= {S_novmax:>6}   <- jump term costs {S_bern/S_novmax:.2f}x")
print(f"  [C] typical error sqrt(v/S) = t           S >= {S_typ:>6}   <- confidence factor costs {S_novmax/S_typ:.1f}x")
print(f"  [D] measured, i.i.d. sampling (99th pct)  S  = {S_iid:>6}   <- remaining bound slack {S_typ/S_iid:.1f}x")
print(f"  [E] measured, systematic (99th pct)       S  = {S_sys:>6}   <- variance reduction {S_iid/S_sys:.1f}x")
print(f"\n  total gap [A]/[E] = {S_bern/S_sys:.0f}x")

print("\n  why the confidence factor [C] is illusory -- the norm self-averages over dk:")
print(f"    {'S':>6} {'mean err':>10} {'99th pct':>10} {'ratio':>7}   Bernstein assumes ratio = sqrt(2*log(2/d)) = 3.26")
for S in (64, 256, 1024):
    e = emp(iid, S); print(f"    {S:>6} {e.mean():>10.4f} {np.quantile(e,0.99):>10.4f} {np.quantile(e,0.99)/e.mean():>7.2f}")
