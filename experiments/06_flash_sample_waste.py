import numpy as np
rng = np.random.default_rng(7)
nk, Btile = 32768, 128
T = nk // Btile

s = rng.standard_normal(nk)*6.0                 # realistic peaked attention
p = np.exp(s - s.max()); p /= p.sum()
tile_mass = p.reshape(T, Btile).sum(1)
order = np.argsort(tile_mass)[::-1]

print(f"32k context, {T} tiles of {Btile}.  Attention mass is concentrated:")
print(f"  top tile holds {tile_mass[order[0]]:.1%} of mass;"
      f" top 8 tiles hold {tile_mass[order[:8]].sum():.1%}")
print(f"  {(tile_mass < 1e-6).sum()} of {T} tiles hold <0.0001% each\n")

print(f"  {'budget S':>9} {'scheme':>6} {'samples landing in top-8 tiles':>32} {'V rows read':>13}")
for S in (128, 512, 2048):
    prop_eff = S * tile_mass[order[:8]].sum()               # proportional allocation
    prop_tiles = (np.floor(S*tile_mass) > 0).sum()
    flash_per  = max(1, round(S/T))                          # uniform per-tile budget
    flash_eff  = flash_per * 8
    print(f"  {S:>9} {'prop':>6} {prop_eff:>32.0f} {min(S, prop_tiles*Btile):>13,}")
    print(f"  {'':>9} {'flash':>6} {flash_eff:>32.0f} {flash_per*T:>13,}")
print(f"\n  -> flash needs S ~ {T}x larger to put the same number of samples")
print(f"     on the tiles that actually matter. Paper measures 2048/128 = 16x.")
