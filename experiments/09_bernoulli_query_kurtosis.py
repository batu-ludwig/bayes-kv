import numpy as np
rng = np.random.default_rng(11)
dk = 128

def access(q, B):
    p = np.abs(q)/np.abs(q).max()
    return np.mean(np.minimum(1.0, B*p))        # stratified-Bernoulli survival

def kurt(q):
    z = (q-q.mean())/q.std(); return float((z**4).mean())

fams = {
  "Gaussian (flat)":      lambda: rng.standard_normal(dk),
  "Laplace (heavier)":    lambda: rng.laplace(0,1,dk),
  "Student-t df=2":       lambda: rng.standard_t(2, dk),
  "1 massive outlier":    lambda: np.r_[rng.standard_normal(dk-1), 20*np.sign(rng.standard_normal())],
  "4 massive outliers":   lambda: np.r_[rng.standard_normal(dk-4), 20*rng.standard_normal(4)],
}
print(f"  {'query distribution':<22}{'kurtosis':>9}   K-feature access at B =")
print(f"  {'':<22}{'':>9}{'2':>8}{'4':>8}{'8':>8}{'16':>8}")
for name, f in fams.items():
    qs = [f() for _ in range(400)]
    k  = np.mean([kurt(q) for q in qs])
    row = [np.mean([access(q,B) for q in qs]) for B in (2,4,8,16)]
    print(f"  {name:<22}{k:>9.1f}" + "".join(f"{r:>8.1%}" for r in row))

print(f"\n  reference points from the paper (mean-group-query, B=8):")
print(f"    Llama 8B  72.8%   BitNet 2B  97.1%")
