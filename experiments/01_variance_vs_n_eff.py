import numpy as np
rng = np.random.default_rng(0)

nk, dk = 8192, 128
V = rng.standard_normal((nk, dk))

def make_A(temp):
    """Attention distribution with tunable peakedness."""
    s = rng.standard_normal(nk) * temp
    e = np.exp(s - s.max())
    return e / e.sum()

def entropy_eff(p):
    """Perplexity of the attention distribution = effective # of attended keys."""
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))

def stats(p, S, trials=200):
    mu = p @ V
    tr_sigma = float((p * (V**2).sum(1)).sum() - (mu**2).sum())
    idx = rng.choice(nk, size=(trials, S), p=p)
    est = V[idx].mean(axis=1)
    err = np.linalg.norm(est - mu, axis=1) / np.linalg.norm(mu)
    return tr_sigma, np.linalg.norm(mu), err.mean()

print(f"{'regime':<12} {'eff.keys':>9} {'||mu||':>8} {'tr(Sigma)':>10}   relative L2 error at S=")
print(f"{'':<12} {'':>9} {'':>8} {'':>10}   " + "".join(f"{S:>9}" for S in (16,64,256,1024)))
for name, temp in [("peaked", 6.0), ("moderate", 3.0), ("diffuse", 1.0)]:
    p = make_A(temp)
    row = []
    for S in (16, 64, 256, 1024):
        tr, nmu, e = stats(p, S)
        row.append(e)
    print(f"{name:<12} {entropy_eff(p):>9.1f} {nmu:>8.2f} {tr:>10.1f}   " + "".join(f"{e:>9.3f}" for e in row))
