# Session handoff

Written so a fresh session (or a human) can pick this up without the originating
conversation. The work so far was a methodology study of the SANTA paper, not an
implementation. Nothing in `experiments/` is production code — each script is a
small numpy probe that tests one claim.

**Context:** the target hardware makes Bayesian sampling and accumulation cheap.
Throughout, the question has been *which parts of the paper are fundamental and
which are GPU workarounds that such hardware makes irrelevant.*

---

## Repository state

```
literature/
  Lee_et_al_2026_Stochastic_Sparse_Attention_for_Memory-Bound_Inference.pdf
  survey_sampling_methods_for_transformers.md
experiments/
  01..10_*.py          numpy probes, fixed seeds, self-contained
HANDOFF.md
```

Branch: `claude/paper-methodology-tutoring-1mkgoh`. No implementation yet.

Run everything with `python3 experiments/<file>` — numpy is the only dependency.
All numbers quoted below were reproduced on 2026-09-15.

---

## The paper in brief

**SANTA** treats the post-softmax attention row `A` as a categorical distribution
over KV positions, samples `S << n_k` indices, and averages the gathered `V` rows:

```
AV_hat = (1/S) * sum_s V_{i_s},    i_s ~ Categorical(A)
```

Unbiased for the post-softmax value aggregation, and the value stage becomes
gather-and-add with no multiplies.

**S2ANTA** adds variance reduction:

- *stratified* — S equal-mass CDF intervals, one jittered sample each. S random
  numbers. Provable variance reduction via law of total covariance.
- *systematic* — one offset `U ~ Unif[0, 1/S)`, then a fixed-stride comb. **One**
  random number total. No variance theorem, but empirically the best of the three.

Equivalent count-vector form `A_hat = c/S` is a single streaming pass of stochastic
rounding: `c_j = floor(a0 + acc + S*p_j) - floor(a0 + acc)`.

**Bernoulli qK^T** is a separate mechanism for the *score* stage. It sparsifies the
**feature** axis `d_k` (orthogonal to SANTA's **token** axis `n_k`) using ternary
queries `q_hat_i = b_i * sign(q_i)` with `b_i ~ Bernoulli(|q_i|)`.

Two kernels ship:

| | kernels | budget | S at 32k | waste |
|---|---|---|---|---|
| `S2ANTA-prop` | 3 (score stash → global allocation → gather) | per-tile, largest-remainder | 128 | none; skips zero-budget tiles |
| `S2ANTA-flash` | 2 (FlashDecoding-style merge) | uniform `S/T` per tile | 2048 | 16x; hard sparsity floor of `T` rows |

---

## Results derived here that are NOT in the paper

**1. Systematic sampling has a deterministic heavy-hitter guarantee.**
Because the comb has fixed spacing `1/S`, any key with `p_j >= 1/S` spans at least
one lattice point and is selected *with probability 1*. Stratified only guarantees
this at `p_j >= 2/S`. Confirmed empirically (`02_sampler_comparison.py`): at S=16,
heavy-hitter capture is **100.0% systematic / 94.2% stratified / 88.5% multinomial**.

This matters for the hardware question. Systematic needs one random number per
attention row instead of S, and buys a *stronger* guarantee. If sampling is cheap,
that trade stops mattering — but the guarantee still does.

**2. The sample budget tracks `n_eff`, not `n_k`.**
Relative error goes as `sqrt(n_eff / S)` where `n_eff = 1 / sum_j p_j^2` is the
effective number of attended keys. From `01_variance_vs_n_eff.py`:

| regime | n_eff | \|\|mu\|\| | rel. L2 err @ S=16 | @ S=1024 |
|---|---|---|---|---|
| peaked | 2.4 | 7.40 | 0.283 | 0.036 |
| moderate | 91.4 | 1.04 | 2.673 | 0.332 |
| diffuse | 2680.8 | 0.23 | 12.441 | 1.551 |

> **Corrected 2026-09-15.** The `n_eff` column previously read 5.3 / 309.8 / 4842.2.
> Those were Shannon perplexity `exp(-sum p log p)`, not the `1 / sum_j p_j^2` the
> text states — `01_variance_vs_n_eff.py` had a helper whose name and docstring said
> "effective # of attended keys" while computing the other functional. The prose was
> right; the code was wrong. The error columns are unaffected.
>
> The distinction is not cosmetic, because `sqrt(n_eff / S)` is only predictive under
> the stated definition:
>
> | regime | S | observed | `sqrt(n_IPR/S)` | `sqrt(n_Shannon/S)` |
> |---|---|---|---|---|
> | peaked | 16 | 0.283 | 0.387 | 0.574 |
> | moderate | 16 | 2.673 | 2.390 | 4.401 |
> | diffuse | 16 | 12.441 | 12.944 | 17.397 |
>
> Shannon perplexity overpredicts required `S` by ~2-4x. Anything built on the old
> numbers — in particular the per-layer profiling in "Profiling to run first" — would
> have over-provisioned the budget by that factor.

Note what drives the blow-up: `tr(Sigma)` is roughly constant across regimes
(78 → 123 → 128). The error explodes because **`||mu||` collapses**, not because
noise grows. Diffuse attention has a small signal, not a noisy estimate.

Consequence: required `S` should be roughly context-independent as long as
attention stays peaked. This is the single highest-value thing to measure on real
models, and the paper does not test past 32k.

**3. Bernoulli qK^T sparsity is governed by query kurtosis.**
From `09_bernoulli_query_kurtosis.py`, K-feature access at B=8:

| query distribution | kurtosis | access |
|---|---|---|
| Gaussian | 2.9 | 86.1% |
| Laplace | 5.5 | 73.0% |
| Student-t df=2 | 24.5 | 51.2% |
| one massive outlier | 73.4 | 32.2% |

The paper's own measurements (mean-group-query, B=8) are **Llama 8B 72.8%**
(≈ Laplace) and **BitNet 2B 97.1%** (flatter than Gaussian). So Bernoulli qK^T is
worth almost nothing on BitNet and modest on Llama — and the mechanism is entirely
explained by how heavy-tailed the query vectors are. Measure kurtosis first; it
predicts the payoff without implementing anything.

---

## The theory-practice gap, decomposed

The paper's vector-Bernstein bound (Appendix B, dimension-free in `d_k`) is far
looser than observed. `03_bernstein_vs_empirical.py` on a realistic peaked head
(n_eff=6.6):

| target | Bernstein says S >= | actually needed |
|---|---|---|
| eps=0.5, d=0.01 | 115 | 16 |
| eps=0.25, d=0.01 | 398 | 32 |
| eps=0.1, d=0.01 | 2,243 | **64** |

35x apparent looseness. But that is not all conservatism. `04_theory_practice_gap.py`
splits it:

```
[A] Bernstein bound            2243
[D] measured, i.i.d. SANTA      731     <- Bernstein slack:       3.1x
[E] measured, systematic         54     <- estimator mismatch:   13.5x
```

**The dominant term is not bound slack — it is that the theorem analyses i.i.d.
SANTA while the deployed kernel uses systematic sampling.** The paper proves a
bound for an estimator it does not ship.

`05_tail_mean_ratio.py` quantifies the remaining 3.1x: Bernstein implicitly assumes
a 99th-percentile-to-mean error ratio of `sqrt(2 log 200) = 3.26`; measured ratios
are 1.99 (peaked), 1.28 (moderate), 1.15 (diffuse). Attention error concentrates
much harder than the worst case.

---

## prop vs flash: correcting two intuitions

Both of these came up and both are wrong.

**"prop is slower because it waits for the whole probability vector."** No — prop
is 1.50x vs flash 1.51x in the paper, essentially tied, and prop reads *less*.
From `07_prop_vs_flash_bandwidth.py` (per layer, one decode step, 32k):

| kernel | S | V traffic | stash r/w | total | vs dense |
|---|---|---|---|---|---|
| prop | 128 | 1.00 MB | 4.00 MB | 69.0 MB | 1.86x |
| flash | 2048 | 16.00 MB | 0.00 MB | 80.0 MB | 1.60x |
| dense | — | 64.00 MB | — | 128.0 MB | 1.00x |

The score stash costs `2H/(H_kv*d_k) = 1/16` = 6.2% of the V read it replaces.
Cheap.

**"prop degrades linearly with context."** The opposite. From
`08_scaling_to_1M.py`:

| context | S_prop | S_flash | prop speedup | flash speedup |
|---|---|---|---|---|
| 32,768 | 128 | 2,048 | 1.86x | 1.60x |
| 131,072 | 128 | 8,192 | 1.88x | 1.60x |
| 1,048,576 | 128 | 65,536 | **1.88x** | 1.60x |

prop's budget is **constant**; flash's grows linearly with tile count because its
uniform per-tile budget forces it to. flash is pinned at 1.60x forever and its S
reaches 65,536 at 1M. prop asymptotes upward. The ceiling at 1.88x is the K cache,
which neither kernel touches — that is what Bernoulli qK^T is for.

`06_flash_sample_waste.py`: at 32k, the top tile holds 47.6% of attention mass and
the top 8 tiles hold 89.7%. flash needs S ~256x larger to land the same number of
samples on tiles that matter; the paper measures 16x.

---

## Prefill vs decode

`10_prefill_union_coverage.py` independently reproduces the paper's Table 9. Union
coverage over all queries in causal prefill vs a single decode query:

| S | decode (last query) | prefill (union) | paper |
|---|---|---|---|
| 1 | 0.05% | 51.2% | 49.9% |
| 4 | 0.20% | 80.4% | 80.0% |
| 16 | 0.78% | 94.3% | 94.1% |

Sampling saves nothing in prefill — the union of per-query samples covers almost
everything, so you read the whole cache anyway. SANTA is a decode-only technique.

---

## Open questions, ranked

1. **Does `S` need to grow past 32k?** Untested in the paper. Result (2) predicts
   no, provided `n_eff` stays flat. Highest-value unknown.
2. **prop's load imbalance at very long context.** 8,192 tiles at 1M with most
   budgets zero — needs tile compaction, which the paper does not implement.
3. **Non-determinism.** Breaks speculative decoding (draft/target distributions
   must match) and reproducible serving. No treatment in the paper.
4. **Gumbel-top-k as a barrier-free alternative — narrowed, see below.** An exact
   without-replacement sample using a top-k kernel, with a Horvitz-Thompson
   correction for unbiasedness. The claim that there is "no published instance for
   KV caches" was **wrong** and has been retracted from the survey (§6, obs. 3);
   three papers occupy the space:
   - [Nexus Sampling, 2606.23961](https://arxiv.org/abs/2606.23961) — priority keys
     `pi_j = u_j^(1/w_j)`, proved equivalent to the exponential-race form of
     Gumbel-top-k, with Prop. 4.3 giving exactly the HT unbiasedness result and a
     concentration bound. Training-free; 80% eviction within 1% of dense on LongBench.
   - [Neural Garbage Collection, 2604.18002](https://arxiv.org/abs/2604.18002) —
     Gumbel-top-k for RL-learned eviction, citing Kool et al. by name; the WOR
     log-probability feeds policy gradients, and eval reverts to deterministic top-k.
   - [Keyformer, 2403.09054](https://arxiv.org/abs/2403.09054) (MLSys 2024) —
     Gumbel-perturbed top-k KV selection since 2024, but as a Gumbel-Softmax
     regularizer: no exact-WOR claim, no HT correction.

   **What survives, and it is still worth doing:** none of the three applies an
   inclusion-probability-corrected estimator to the *attention output itself*. Nexus
   and NGC both use the machinery for **which tokens to keep**; the open move is
   replacing SANTA's with-replacement categorical draw with an exact WOR draw
   carrying HT weights, i.e. estimating `AV` rather than choosing an eviction set.
   That also interacts with open question 3 above — a WOR draw has lower variance
   than the with-replacement one at equal S, which changes the non-determinism budget.

## Profiling to run first (one forward pass gets the first three)

1. Per-layer `n_eff` profile — tells you `S` directly.
2. `V_max / ||mu||` per layer — the Bernstein constant that actually binds.
3. Query kurtosis per layer — predicts Bernoulli qK^T payoff before implementing it.
4. prop vs flash at equal `S` — isolates prop's structural penalty from its budget advantage.
5. End-to-end Amdahl fraction — how much of decode is attention at your context length.

---

## Note for a session in the `research` environment

`literature/survey_sampling_methods_for_transformers.md` §8 lists what could not be
verified against primary sources, because the originating session ran under a
network policy that blocked arxiv.org, openreview.net, huggingface.co,
semanticscholar.org, aclanthology.org and proceedings.mlr.press. Everything there
came from search-index snippets of abstracts.

With those sources reachable, the work is: confirm the flagged numbers, read the
proofs that abstracts omit (HyperAttention's two hardness parameters, vAttention's
(eps, delta) construction, LARA's importance-sampling derivation of RFA bias), and
update the survey in place.
