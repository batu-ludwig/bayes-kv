import numpy as np
rng = np.random.default_rng(5)
nk, dk = 2048, 128
Q = rng.standard_normal((nk,dk)); K = rng.standard_normal((nk,dk))
s = Q@K.T/np.sqrt(dk)
s[np.triu_indices(nk,1)] = -np.inf          # causal
P = np.exp(s - s.max(1,keepdims=True)); P /= P.sum(1,keepdims=True)

print(f"prefill union coverage, causal, nk={nk}, Gaussian inputs\n")
print(f"  {'S':>4} {'decode (last query)':>21} {'prefill (union, all queries)':>30}")
for S in (1,4,8,16,64):
    touched = np.zeros(nk, bool)
    for i in range(nk):
        touched[rng.choice(nk, size=S, p=P[i])] = True
    dec = min(S, nk)/nk
    print(f"  {S:>4} {dec:>20.2%} {touched.mean():>29.1%}")
print("\n  paper Table 9 (same setup):  S=1 -> 49.9%,  S=4 -> 80.0%,  S=16 -> 94.1%")
