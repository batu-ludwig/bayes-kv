# Verification Log: Sampling-Methods Survey

Companion to `survey_sampling_methods_for_transformers.md`. Records what was checked, what
changed, and the evidence for each change.

**Date:** 2026-09-15 · **Scope:** all 48 arXiv IDs cited by the survey as of commit `3eb5178`,
plus 11 papers added during verification (59 total).

---

## 1. Why this pass was necessary

The original survey was assembled under an egress policy that returned 403 at the proxy for
`arxiv.org`, `openreview.net`, `huggingface.co`, `semanticscholar.org` and `ar5iv`. Its own §8
recorded the consequence: **no paper PDF or abstract page was ever fetched.** Every arXiv ID was
confirmed only by a search index returning a listing with a matching bracketed title, and every
quantitative claim came from search-result snippet text.

That policy no longer applies. Reachability at the time of this pass:

| Host | Status |
|---|---|
| `arxiv.org/abs/…` | 200 |
| `openreview.net` | 200 |
| `semanticscholar.org` | 200 |
| `huggingface.co` | 200 |
| `ar5iv.labs.arxiv.org` | 200 |
| `export.arxiv.org/api` | 429 — rate-limited, not blocked; unusable for bulk metadata |

## 2. Method

1. **ID and title resolution.** All 48 IDs fetched from `arxiv.org/abs/`, throttled to one request
   per ~3.2 s. Extracted `citation_title`, `citation_author`, `citation_date`, the comments and
   journal-ref fields, and the abstract. All 48 resolved; all titles matched their attribution.
2. **Venue establishment.** From arXiv comments and journal-refs where present; otherwise
   cross-checked against PMLR, OpenReview, ACL Anthology, NeurIPS proceedings, JMLR and the ACM DL.
   Several arXiv comment fields are stale (TOVA still says "preprint" while the paper is EMNLP 2024
   Main) and several are empty for published papers (QuaRot, SnapKV, H2O).
3. **Claim verification.** Five parallel readers, one per cluster (KV sampling estimators;
   deterministic top-k; speculative decoding; quantization/Bayesian; 2026 preprints). Each worked
   from full text — PDF or ar5iv HTML — not abstracts. Each claim received one of CONFIRMED / WRONG /
   IMPRECISE / UNSUPPORTED plus a locating quote.
4. **Independent re-check.** The findings that refute the survey's own novelty claim (§5 below) were
   re-verified directly from the source PDFs rather than accepted on a single reader's report.

**A caution that emerged from the method itself:** four papers here have abstracts that contradict
their own bodies (Quest, SampleAttention, and both via stale arXiv metadata). Abstract-level
verification would have propagated two of the errors this pass caught. Full text or nothing.

---

## 3. Outright errors

### 3.1 The Gumbel-top-k novelty claim was refuted

The survey's §6 observation 3 proposed adding Gumbel noise to logits before top-k to obtain an exact
without-replacement KV sample, then applying a Horvitz–Thompson or priority-sampling correction —
flagged as "an unclaimed unbiasedness retrofit," with the search noted as non-exhaustive. It was not
exhaustive. Three papers occupy the space.

**[Nexus Sampling, 2606.23961](https://arxiv.org/abs/2606.23961)** — Duong, Le, Xie, Shrivastava, Xu,
22 Jun 2026. A direct hit. Selection is by priority key `π_j = u_j^(1/w_j)`; the paper proves this is
PPS-without-replacement via *"The priority rule of Efraimidis and Spirakis is equivalent to drawing
independent exponential clocks E_j = −log(u_j)/w_j with rates w_j and taking the first K arrivals"* —
Gumbel-top-k in exponential-race form. Proposition 4.3 supplies the proposed correction outright:

> *"The Horvitz–Thompson estimator Ẑ_HT = Σ_j I_j z_j / p_j is unbiased for Z = Σ_j z_j, and if
> p_j ≥ p_min > 0 and 0 ≤ z_j ≤ Z_max, then with probability ≥ 1−δ,
> |Ẑ_HT − Z| ≤ Z_max √(K N_k / (δ p_min))."*

Results: 80% KV eviction, within 1% of dense attention on LongBench, up to 10× smaller per-sequence
cache, training-free.

**[Neural Garbage Collection, 2604.18002](https://arxiv.org/abs/2604.18002)** — Li, Hamid, Fox,
Goodman (Stanford), 20 Apr 2026. §4.3 is titled *"Efficiently sampling eviction actions with
Gumbel-top-k"* and cites Kool et al. by name: *"We formulate this task as sequential sampling without
replacement using the Gumbel-top-k trick [Kool et al., 2019, Vieira, 2014]."* Purpose differs — the
closed-form WOR log-probability feeds unbiased **policy gradients** for RL-learned eviction, and at
eval time it reverts to deterministic top-k.

> **Trap:** NGC's text contains four "Horvitz" hits. They are *Eric* Horvitz, cited on bounded
> rationality. Not Horvitz–Thompson. Do not read this as a second HT paper.

**[Keyformer, 2403.09054](https://arxiv.org/abs/2403.09054)** — MLSys 2024. Adds Gumbel noise to
attention logits before top-k KV selection, via `f_θ(x_i) = e^((x_i+ζ_i)/τ) / Σ_j e^((x_j+ζ_j)/τ)`.
But it is framed as a Gumbel-Softmax **regularizer** justified by an entropy argument — no Kool
citation, no exact-WOR claim, no HT correction. The *mechanic* has existed since early 2024; the
principled WOR/HT framing arrives with Nexus.

**Narrowed claim that survives:** Gumbel-top-k with an inclusion-probability-corrected estimator **of
the attention output itself** — replacing SANTA's with-replacement categorical draw with an exact WOR
draw carrying HT weights. Eviction policy and RL gradients are taken; estimation of the attention
aggregate is not.

### 3.2 The EAGLE row conflated three papers

The survey cited only `2503.01840` but labelled the row "EAGLE / EAGLE-3, ICML 2024 / 2025."

| Paper | arXiv | Venue | Headline |
|---|---|---|---|
| EAGLE | 2401.15077 | ICML 2024 | 2.7–3.5× latency, ~2× throughput |
| EAGLE-2 | 2406.16858 | EMNLP 2024 | 3.05–4.26× |
| EAGLE-3 | 2503.01840 | **NeurIPS 2025** | up to 6.5×; 1.38× throughput @ bs64 SGLang |

Both quoted numbers belong to EAGLE-3 alone: *"EAGLE-3 achieves a speedup ratio up to 6.5x… In the
SGLang framework, EAGLE-3 achieves a 1.38x throughput improvement at a batch size of 64."* EAGLE-1
reports neither and has no SGLang experiment. EAGLE-3's venue is NeurIPS 2025, not ICML.

### 3.3 BinaryConnect's storage saving is 16×, not 32×

Conclusion, verbatim: *"reducing by a factor of **at least 16 (from 16 bits single-float precision to
single bit precision)** the memory requirement."* The survey doubled it by assuming a 32-bit baseline.

### 3.4 Quartet's SR-vs-RTN comparison ran backwards

The survey said stochastic rounding is *"unbiased but higher variance than well-scaled
round-to-nearest, and RTN can beat it on forward-pass alignment."* Quartet
([2505.14669](https://arxiv.org/abs/2505.14669), NeurIPS 2025) says the opposite about alignment:
*"SR trades higher error for perfect alignment."* Table 2 (MXFP4): SR has projection-magnitude
misalignment **0** where RTN AbsMax has 9.3e-3; RTN wins on **MSE** (1.37e-2 vs 2.77e-2) and on fitted
parameter efficiency (effN 0.59 vs 0.42). The paper also says *error/MSE* throughout, never
*variance* — defensible shorthand for an unbiased estimator, but RTN is biased, so its MSE is not a
variance and the two should not be compared as though they were.

---

## 4. Materially imprecise claims

Ordered by how much they affect the survey's argument.

**HashAttention's retrieval is not multiply-free** — and §6 observation 2 rested on it. Only the
*matching* step is (XOR + popcount over packed integers). Signature generation is a 3-layer MLP
(128×128-128×128-128×32), and the paper reports it dominates: *"The matrix multiplication in mapping
functions used to obtain bit-signatures dominates the latency up to 8K context length in GPT-FAST and
up to 65K in Flash Decode."*

**SANTA's multiply-free property is not realized on the measured kernels.** The authors' own
disclaimer: *"While S²ANTA is multiplier-free in principle, current GPUs are highly optimized for
dense fused multiply-accumulate instructions; we therefore view multiplier-free arithmetic as an
additional energy-oriented benefit that may become more pronounced on future hardware."* The measured
1.5× is bandwidth, not arithmetic. Further: the `1/S` normalization is a bit-shift only if S is a
power of two and the model is fixed-point; the property covers the post-softmax stage only, as the
score stage still multiplies unless Bernoulli qKᵀ is layered on.

**SANTA's variance-reduction guarantee covers stratified only.** *"Only S²ANTA-strat has a
theoretical guarantee of variance reduction compared to default SANTA."* Remark A.5 on the systematic
variant: *"does not provide tractable variance guarantees; empirically it performs comparably."*

**MagicPIG's "4× lower estimation error" belongs to a different method.** The 4× is *oracle sampling*,
which the paper defines as assuming *"the exact attention vector w is known, which is not true for
sparse attention approximations."* Unbiasedness is asserted (Thm 3.2) for oracle sampling alone;
MagicPIG's own SNIS estimator gets only a.s. consistency. The survey's verdict that MagicPIG is not
unbiased was right — the reason is stronger than stated.

**The 90–99% weight-bandwidth citation was inappropriate.** The figure appears verbatim in
[2607.08407](https://arxiv.org/abs/2607.08407) §3.1, but (a) that paper cites it secondhand from
[DeepSpeed Inference, 2207.00032](https://arxiv.org/abs/2207.00032); (b) it is scoped to *"small to
medium batch sizes"* with no context length or named model, and the example is FP16 not BF16; (c) the
paper is *"Who Needs DRAM? We Have Fiber"* — an unrefereed Uppsala proposal for recirculating optical
delay-line memory over multi-core fiber; and (d) it deploys the statistic to argue the KV cache is the
**negligible** 1–10%, the opposite of this survey's purpose.

**SOCKET is not a sampling estimator** despite "EsTimator" in its name. The paper draws the line:
*"while MagicPig is fundamentally a sampling-based estimator, SOCKET is a retrieval-based approach
centered on accurate top-k selection."* Its Theorem 3 is an error decomposition, not an ordering
guarantee. Moved to the deterministic table.

**Smaller corrections:**

| Claim | Correction |
|---|---|
| Quest "100% passkey at 1% budget" | 99% @10k/64 and 96% @100k/1024. Paper says "nearly perfect" |
| H2O "29× vs DeepSpeed Zero-Inference" | 29× applies to DeepSpeed **and** HF Accelerate; only **3×** vs FlexGen — the strong baseline H2O is built on |
| HashAttention "16–32×" | 16× on generic data; 32× requires task-specific fine-tuning |
| SnapKV "8.2× memory efficiency" | Max context before OOM (16k→131k), not a compression ratio |
| TOVA 1/8 cache, 4.8×, 88% | One operating point (512 of 4096), not three independent bests. Conclusion says 1/8–1/4 typical |
| KDEformer 18× / 4× | FLOP ratios, not wall-clock. Only wall-clock figure is ~8× on LRA-Text |
| HyperAttention 5.4× / 54× | 5.4× is **with** causal masking, 54× **without** — not forward vs backward. Single-layer microbenchmark |
| Performer "competitive on LRA" | LRA results are reproduced from Tay et al. in App. D.5, not run by the Performer authors |
| MCA "11× on GLUE" | CoLA only at α=0.2; GLUE **mean** is 4.64×. No wall-clock anywhere. MCA also rejects the DKM-optimal probabilities as impractical |
| Reformer "64k on one accelerator" | A motivating calculation in §1; experiments ran on 8 devices |
| Loki 45% | Attention-only, not end-to-end; Loki explicitly does not reduce KV memory |
| Deja Vu ">2×" | Measured §5.1 figure is 1.8–2× at 75% sparsity |
| vAttention "+4.5 pp" | Over HashAttention at 10% sparsity, not over full attention. And every experiment uses a **denominator-only** relaxation of the (ε,δ) guarantee |
| vAttention "wall-clock not verified" | Fig. 5 does report it — but CPU-offloaded KV only, unoptimized PyTorch indexing |
| SMC-SD "2.36× / within 3%" | 2.36× is 4×H100, 1B→70B; single-GPU iso-accuracy range is 1.1–2.5×. "3%" means 3 percentage points of **SD's** accuracy at one operating point. Fig. 1's own bars give 2.43× |
| Bayesian Compression 771× | k-means+codebook "maximum compression"; bit-precision-only is 419×. Bit-width is per **matrix** from the **mean** posterior variance, not per weight |
| Bayesian Bits 69.16% | Pre-fine-tuning row; post-FT is 69.39% at the same BOPs |
| Sparse VD 280× / 68× | 280× = LeNet-5-Caffe; VGG is a CIFAR "VGG-like" 13conv+2FC net, not ImageNet VGG-16 |
| Vogel et al. "1-bit" | The 5.81% CIFAR-10 result used **ternary** projections |
| PSB "33%" | Measured against the authors' own psb16 baseline, not float32 |
| BinaryConnect `P(w=+1)=σ(w)` | σ is the **hard** sigmoid `clip((w+1)/2, 0, 1)`, not logistic. Multiply-free at deploy holds only for the deterministic variant |
| Shazeer "training-time only" | The noise's stated purpose is differentiable load balancing, but the paper never says it is disabled at inference and Eq. 3–4 include it unconditionally |
| Medusa "temperature-derived" | Entropy-adaptive `min(ε, δ·exp(−H(p)))`. Losslessness applies to Medusa-1 only; Medusa-2 fine-tunes the backbone |
| BitNet STE/latent weights | That machinery is in BitNet 2310.11453, not b1.58 |
| Deja Vu "25%" | Correct as an operating point (75% sparsity); the 85%-sparsity figure is a separate existence result |

---

## 5. Version drift

Six papers' headline numbers changed between the version the survey drew on and the current one.

| Paper | Earlier | Current |
|---|---|---|
| MagicPIG | 1.9–3.9× throughput, 110 ms (v1/v2) | up to 5×, 54 ms (v3/v4 + ICLR camera-ready) |
| SANTA | kernel-level only (v1 — the PDF in this repo) | v2 adds 1.25× end-to-end; abstract changes to "stratified *and* systematic" |
| QTIP | 2-bit 70B ppl 3.78 (v1) | 3.70 (NeurIPS camera-ready) |
| QuaRot | ≤0.29 ppl, 3.39× memory (v1) | ≤0.47 ppl, 3.89× (v2 / NeurIPS) |
| SampleAttention | 2.42× TTFT (v1 — still on the abs page) | 5.29× @1M (v3); typical range 1.24–2.36× |
| Quartet | SR on the backward pass (v1) | RTN on the backward pass (v4, Jan 2026) |

**Stale abstracts.** Quest and SampleAttention both have arXiv abs pages contradicting their own full
texts. For Quest the error is load-bearing: both the abs page and the official PMLR v235 abstract read
*"2.23× self-attention speedup, which reduces inference latency by 7.03×"*, while the v2 body and
conclusion have 7.03× self-attention → 2.23× end-to-end. Only the body can be right — a component
speedup cannot be smaller than the end-to-end speedup it produces. This explains *why* the survey's
original observation that the numbers are "frequently quoted backwards" is correct.

---

## 6. Venue corrections

| Paper | Survey said | Correct |
|---|---|---|
| EAGLE-3 (2503.01840) | ICML 2024 / 2025 | **NeurIPS 2025** |
| HashAttention (2412.14468) | 2024/25 | **ICML 2025** |
| TOVA (2401.06104) | 2024 | **EMNLP 2024 (Main), pp. 18724–18741** |
| SnapKV (2404.14469) | 2024 | **NeurIPS 2024** |
| SampleAttention (2406.15486) | — | **No venue** — preprint only (v3, Sep 2025) |
| Speculative sampling (2302.01318) | — | **No venue** — DeepMind tech report |
| Switch Transformer (2101.03961) | — | **JMLR 23 (2022)** |
| Shazeer et al. (1701.06538) | — | **ICLR 2017** |

Confirmed correct as originally stated: MagicPIG (ICLR 2025 Spotlight), vAttention (ICLR 2026),
LARA (ICML 2022), KDEformer (ICML 2023), HyperAttention (ICLR 2024), Performer (ICLR 2021 Oral),
MCA (AAAI 2022), Reformer (ICLR 2020), Quest (ICML 2024), SparQ (ICML 2024), H2O (NeurIPS 2023),
Loki (NeurIPS 2024), SANTA (ICML 2026), PSB (EMC²@NeurIPS 2019), DQT (ACML 2025), and the whole
Bayesian/quantization block.

**One suspicion that proved wrong.** YOSO's arXiv comment claims "Proceedings of the 38th ICML (2021)"
despite a 2021-11-18 posting date, months after ICML 2021 convened. This looked like an author error.
It is not: the paper is genuinely in PMLR 139:12321–12332. The late posting is just a late preprint
upload. Do not "correct" it.

---

## 7. Load-bearing claims that held

The survey's sharpest arguments survived contact with the sources.

- **SampleAttention contains no sampling.** A full-text search of v3 for *random / stochastic /
  probabilit\* / Monte Carlo / unbiased / estimator* found no sampling distribution, no variance
  analysis, no unbiasedness argument. The paper explicitly rejects randomness: *"Compared to random
  sampling or bottom sampling, this equidistant sampling technique is low-overhead and more stable."*
  Its one theoretical object, CRA, is deterministic.
- **H2O's guarantee is on the eviction objective, not output error.** Theorem 4.4 bounds
  `f(S̃ᵢ) ≥ (1−1/e)(1−α)optᵢ − β` where `f` = F_score, the summed attention mass of the retained set.
  No bound on output error, logit error, or generation quality appears anywhere.
- **BitNet does not use stochastic rounding.** The word "stochastic" occurs **zero** times in
  2402.17764. Quantization is `RoundClip(W/(γ+ε), −1, 1)` with `γ = mean|W_ij|` — absmean scaling,
  round to nearest.
- **SpinQuant's 13 points, exact.** *"the best random rotation matrix outperforming the worst by 13
  points"* — LLaMA-2 7B, W4A4, 100 randomized trials. Bonus for the argument: even random *Hadamard*
  rotations vary *"as large as 6 points."*
- **Speculative sampling really does claim to beat the bandwidth ceiling.** *"the mean tokens per
  second with SpS often exceeds the idealised ceiling on auto-regressive sampling speed imposed by the
  memory bandwidth."*
- **LARA's framing, verbatim.** *"recasts RFAs as self-normalized importance samplers"*; RA is *"the
  first kernel linearization estimator that approximates the whole softmax attention … in an unbiased
  manner."* One addition: LARA itself is self-normalized MIS and therefore **not** unbiased — only the
  quadratic RA is.
- **YOSO never asserts unbiasedness.** The string *"unbias"* does not occur. The estimand is YOSO-E,
  the expectation over hash functions, which only *resembles* softmax. The survey's "plausible by
  construction but not asserted" was exactly right.
- **QuIP's spectral-bound admission.** Thm 4 confirmed. Two precisions: it concerns **LDLQ**, QuIP's
  adaptive-rounding subroutine rather than QuIP end-to-end, and it fails to separate LDLQ from
  round-to-nearest **and** stochastic rounding, not SR alone.
- **HashAttention's learned hash.** Table 8: random-projection LSH *"fails to achieve comparable
  performance even with more than 1000 bits"* against 32 learned bits.
- **Deja Vu is deterministic by design, having tried the alternative:** *"random selection fails to
  identify the accurate contextual sparsity, resulting in drastic model degradation."*
- **THOR samples experts at inference**, not just training, and the +2.0 BLEU (22.4→24.4) and 18×
  (5.5B Switch matched by 300M THOR) figures are exact.
- **Switch routing is deterministic top-1 argmax** — with the caveat that it applies multiplicative
  input jitter during training.
- **ScatterBrain**, previously cited with no arXiv ID at all, is
  [2110.15343](https://arxiv.org/abs/2110.15343) (NeurIPS 2021), and the claim attributed to it is
  verbatim: *"sparse and low-rank approximations excel in different regimes, determined by the softmax
  temperature in attention."*

---

## 8. Claims that are the survey's own inference

Marked inline in the survey. Each is defensible reasoning; none is what the cited paper says.

| Claim | Status |
|---|---|
| Vogel et al. re-read FP shadow weights per sample, so bandwidth worsens | The paper never mentions memory bandwidth. The inference is structurally sound — Fig. 1 shows one FP network feeding N stochastic projections, and the conclusion refers to "the high-precision shadow weights" — but it is ours |
| Bayes by Backprop costs "N× traffic" | No bandwidth analysis exists in the paper. The "2× storage" half **is** verbatim: *"may have twice as many weights"* |
| Nyströmformer's guarantee fails because softmax attention is not symmetric PSD | The paper has **no error bound at all** for the attention approximation. The compromise it names is ordering, not symmetry: *"instead of subsampling the matrix after the softmax operation — as one should do in principle"* |
| MagicPIG's SNIS has O(1/n) finite-sample bias | Correct textbook fact, but not stated in the paper |

---

## 9. Unresolved

**`n_eff` — RESOLVED, and a bug found in passing.** Survey §6.1 appealed to "the `n_eff` scaling
law" with no citation. It is not literature: it is this project's own construct, defined in
`HANDOFF.md` as `n_eff = 1 / Σ_j p_j²` with error scaling `√(n_eff / S)`. The survey now says so.

While confirming this, `experiments/01_variance_vs_n_eff.py` turned out to contradict its own
documentation: a helper named `entropy_eff`, docstringed *"effective # of attended keys"*, computed
Shannon perplexity `exp(−Σ p log p)` instead of the inverse participation ratio the prose specifies.
The HANDOFF table therefore reported 5.3 / 309.8 / 4842.2 where the stated definition gives
2.4 / 91.4 / 2680.8.

The prose was right and the code was wrong — confirmed by checking which functional actually predicts
the measured error:

| regime | S | observed | `√(n_IPR/S)` | `√(n_Shannon/S)` |
|---|---|---|---|---|
| peaked | 16 | 0.283 | 0.387 | 0.574 |
| moderate | 16 | 2.673 | 2.390 | 4.401 |
| diffuse | 16 | 12.441 | 12.944 | 17.397 |

IPR tracks; Shannon overshoots by 1.4–2× consistently. Fixed in both the script and the HANDOFF
table. Consequence worth noting: the per-layer `n_eff` profile listed under "Profiling to run first"
is described as telling you `S` directly — run with the old helper it would have over-provisioned the
budget by 2–4×.

**§5 "The Gap" has not been adversarially searched.** The central claim — that no published method
stochastically samples weights at inference to form an unbiased estimator of a weight matvec — was
carried forward from the original survey and *not* pressure-tested in this pass, which was scoped to
verifying existing claims and the 2026 preprints. Given that §6's other novelty claim turned out to be
occupied by three papers, §5 warrants the same treatment before it is relied on.

---

## 10. Summary

| Category | Count |
|---|---|
| arXiv IDs checked | 48 (all resolved, all titles matched) |
| IDs added during verification | 11 |
| Outright errors | 4 |
| Materially imprecise claims | ~25 |
| Venue corrections | 8 |
| Papers with version drift | 6 |
| Papers with abstracts contradicting their own body | 2 |
| Novelty claims refuted | 1 |
| Load-bearing claims confirmed | 13 |
