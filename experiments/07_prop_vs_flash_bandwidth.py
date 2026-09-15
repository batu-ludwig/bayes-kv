nk, H, Hkv, dk, G, B = 32768, 32, 8, 128, 4, 2   # Llama-3.1-8B decode, bf16
MB = 1024**2

K_read = nk*Hkv*dk*B / MB          # full key cache, both kernels read this
V_full = nk*Hkv*dk*B / MB
print(f"per layer, one decode step, 32k context:")
print(f"  K cache read (unavoidable, both kernels)  {K_read:7.1f} MB")
print(f"  V cache read (dense SDPA)                 {V_full:7.1f} MB")
print(f"  total dense                               {K_read+V_full:7.1f} MB\n")

def v_traffic(S):
    rows = min(nk, S*G)                     # GQA: 4 query heads/group sample independently
    return rows*Hkv*dk*B/MB, rows/nk

stash = nk*H*B/MB                            # exp-scores, one per QUERY head
print(f"  {'kernel':<22}{'S':>6}{'V rows':>9}{'V traffic':>11}{'stash r/w':>11}{'total':>9}{'vs dense':>10}")
rows_p, frac_p = v_traffic(128)
rows_f, frac_f = v_traffic(2048)
for name, S, (vt, fr), st in [("S2ANTA-prop", 128, v_traffic(128), 2*stash),
                              ("S2ANTA-flash", 2048, v_traffic(2048), 0.0)]:
    tot = K_read + vt + st
    print(f"  {name:<22}{S:>6}{int(fr*nk):>9,}{vt:>10.2f} MB{st:>10.2f} MB{tot:>8.1f}{(K_read+V_full)/tot:>9.2f}x")
print(f"  {'SDPA (dense)':<22}{'-':>6}{nk:>9,}{V_full:>10.2f} MB{0.0:>10.2f} MB{K_read+V_full:>8.1f}{1.0:>9.2f}x")

print(f"\n  flash reads {v_traffic(2048)[0]-v_traffic(128)[0]:.1f} MB more V than prop")
print(f"  prop pays   {2*stash:.1f} MB for the score stash (write+read)")
print(f"  net difference: {v_traffic(2048)[0]-v_traffic(128)[0] - 2*stash:.1f} MB out of {K_read+V_full:.0f} MB"
      f"  = {(v_traffic(2048)[0]-v_traffic(128)[0]-2*stash)/(K_read+V_full)*100:.1f}% of dense traffic")
print(f"\n  stash cost as a fraction of the V read it replaces: {2*stash/V_full*100:.1f}%")
print(f"  (= 2*H/(Hkv*dk) = 2*{H}/({Hkv}*{dk}) = 1/{Hkv*dk//(2*H)})")
