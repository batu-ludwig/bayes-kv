H, Hkv, dk, G, B, Btile = 32, 8, 128, 4, 2, 128
LAYERS, BW = 32, 960e9          # RTX 6000 Ada ~960 GB/s
MB, GB = 1024**2, 1024**3

def report(nk, S_prop=128, S_eff_per_tile=8):
    T = nk//Btile
    K   = nk*Hkv*dk*B                       # always read in full
    Vd  = nk*Hkv*dk*B
    stash = 2*nk*H*B                        # write + read
    Vp  = min(nk, S_prop*G)*Hkv*dk*B        # prop: budget stays constant
    S_flash = S_eff_per_tile*T              # flash: budget must scale with tile count!
    Vf  = min(nk, S_flash*G)*Hkv*dk*B
    return dict(nk=nk, T=T, dense=(K+Vd), prop=(K+stash+Vp), flash=(K+Vf),
                S_flash=S_flash, Vp_frac=Vp/Vd, Vf_frac=Vf/Vd)

print(f"{'context':>9} {'tiles':>7} {'S_prop':>7} {'S_flash':>9} {'prop V%':>8} {'flash V%':>9}"
      f" {'prop x':>7} {'flash x':>8}")
for nk in (4096, 8192, 32768, 131072, 1048576):
    r = report(nk)
    print(f"{r['nk']:>9,} {r['T']:>7,} {128:>7} {r['S_flash']:>9,}"
          f" {r['Vp_frac']:>7.2%} {r['Vf_frac']:>8.2%}"
          f" {r['dense']/r['prop']:>7.2f} {r['dense']/r['flash']:>8.2f}")

print(f"\nabsolute wall-clock floor, ALL {LAYERS} layers, one token (bandwidth only):")
print(f"{'context':>9} {'dense':>12} {'prop':>12} {'tok/s dense':>13} {'tok/s prop':>12}")
for nk in (32768, 131072, 1048576):
    r = report(nk)
    d, p = r['dense']*LAYERS, r['prop']*LAYERS
    print(f"{r['nk']:>9,} {d/GB:>9.2f} GB {p/GB:>9.2f} GB {BW/d:>13.1f} {BW/p:>12.1f}")
