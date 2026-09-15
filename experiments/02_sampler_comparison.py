import numpy as np
rng = np.random.default_rng(1)
nk, dk = 8192, 128
V = rng.standard_normal((nk, dk))

def make_A(temp):
    s = rng.standard_normal(nk) * temp
    e = np.exp(s - s.max()); return e / e.sum()

def multinomial(p, S):
    return rng.choice(nk, size=S, p=p)

def stratified(p, S):
    T = (np.arange(S) + rng.random(S)) / S           # S random numbers
    return np.searchsorted(np.cumsum(p), T, side='right')

def systematic(p, S):
    T = (np.arange(S) + rng.random()) / S            # ONE random number
    return np.searchsorted(np.cumsum(p), T, side='right')

print("relative L2 error (mean of 300 trials), nk=8192\n")
for name, temp in [("peaked", 6.0), ("moderate", 3.0)]:
    p = make_A(temp); mu = p @ V; nmu = np.linalg.norm(mu)
    print(f"  {name} attention (eff. keys = {np.exp(-(p*np.log(p+1e-300)).sum()):.0f})")
    print(f"    {'S':>6} {'multinomial':>12} {'stratified':>12} {'systematic':>12}")
    for S in (16, 64, 256):
        row = []
        for fn in (multinomial, stratified, systematic):
            errs = [np.linalg.norm(V[fn(p, S)].mean(0) - mu)/nmu for _ in range(300)]
            row.append(np.mean(errs))
        print(f"    {S:>6} {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f}"
              f"   ({row[0]/row[2]:.1f}x better)")
    print()

# heavy-hitter guarantee: every key with p_j >= 1/S must be selected by systematic
p = make_A(6.0)
print("heavy-hitter capture rate (keys with p_j >= 1/S), 200 trials:")
print(f"    {'S':>6} {'#heavy':>8} {'multinomial':>12} {'stratified':>12} {'systematic':>12}")
for S in (16, 64, 256):
    heavy = np.where(p >= 1.0/S)[0]
    if len(heavy) == 0: continue
    rates = []
    for fn in (multinomial, stratified, systematic):
        hits = [np.isin(heavy, fn(p, S)).mean() for _ in range(200)]
        rates.append(np.mean(hits))
    print(f"    {S:>6} {len(heavy):>8} {rates[0]:>11.1%} {rates[1]:>11.1%} {rates[2]:>11.1%}")
