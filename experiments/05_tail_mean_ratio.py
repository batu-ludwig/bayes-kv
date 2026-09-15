import numpy as np
rng = np.random.default_rng(3)
nk, dk = 8192, 128
V = rng.standard_normal((nk, dk))

def setup(temp):
    s = rng.standard_normal(nk)*temp
    p = np.exp(s-s.max()); p /= p.sum()
    return p, p@V

print("ratio of 99th-percentile error to MEAN error  (i.i.d. sampling)")
print("Bernstein's implicit assumption for this ratio: sqrt(2*log(200)) = 3.26\n")
print(f"  {'attention':<12} {'n_eff':>7} {'S':>6} {'mean':>9} {'99th':>9} {'ratio':>7}")
for name, temp in [("peaked", 6.0), ("moderate", 3.0), ("diffuse", 1.0)]:
    p, mu = setup(temp); nmu = np.linalg.norm(mu)
    neff = np.exp(-(p*np.log(p+1e-300)).sum())
    for S in (64, 256):
        errs = []
        for _ in range(8):   # chunked to bound memory
            idx = rng.choice(nk, size=(1500, S), p=p)
            errs.append(np.linalg.norm(V[idx].mean(1)-mu, axis=1)/nmu)
        e = np.concatenate(errs)
        print(f"  {name:<12} {neff:>7.0f} {S:>6} {e.mean():>9.4f} {np.quantile(e,0.99):>9.4f} {np.quantile(e,0.99)/e.mean():>7.2f}")
