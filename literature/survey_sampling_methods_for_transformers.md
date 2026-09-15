# Sampling-Based Methods for Transformer Inference: A Comparative Survey

**Scope.** Methods that use sampling / Monte Carlo / Bayesian estimation to reduce memory
traffic in transformer inference, segmented by what they attack: **weight transfer** or
**activations (KV cache)**.

**Why the segmentation matters.** In batch-1 long-context decode both are bandwidth-bound,
but the two halves of the literature are in completely different states of maturity. The
KV side has genuine unbiased estimators with concentration guarantees. The weight side has
almost none — see [The Gap](#5-the-gap).

---

## 1. Taxonomy: three things people call "stochastic"

Most confusion in this literature comes from collapsing these. Keep them apart:

| Class | Definition | Does variance average out at inference? |
|---|---|---|
| **(A) Estimator** | Draws a random subset at runtime with known probabilities; forms an estimator of the *full* operation. Unbiasedness / concentration is claimed and proven. | **Yes** — this is the point |
| **(B) Train-time stochastic** | Randomness used during training or calibration; the deployed artifact is a fixed byte string. | No — error is baked in |
| **(C) Randomized, not probabilistic** | Random rotations/hashes chosen once offline as a *preconditioner*; deterministic at runtime. | No |

Class (C) is not a weaker estimator — it is not an estimator at all. SpinQuant's own finding
that different random rotations differ by **up to 13 points** of downstream accuracy is the
cleanest proof: a real estimator's variance would average away, not lock in 13 points.

A fourth category is worth naming because it dominates deployed systems: **deterministic
top-k heuristics** (Quest, H2O, SnapKV, Loki, TOVA, SparQ, HashAttention). These select the
argmax-scoring tokens and renormalize softmax over the kept set. That drops mass from **both
the numerator and the denominator**, so surviving tokens get inflated weight — a *biased*
estimator whose bias grows exactly when attention is flat, i.e. when the sparsity assumption
they rely on fails. This is the statistical opening that MagicPIG, vAttention and SANTA exploit.

---

## 2. Track A — Activations / KV cache

### 2a. Genuine sampling estimators

| Method | Venue | What is sampled | Estimator & guarantee | Reduces | Headline result | Key limitation |
|---|---|---|---|---|---|---|
| **[SANTA](https://arxiv.org/abs/2605.01910)** (Lee, Delacour, …, Çamsarı) | ICML 2026 | KV **token** axis, drawn from the **post-softmax attention distribution** itself; plus Bernoulli qKᵀ on the key-**feature** axis | **Unbiased** for post-softmax value aggregation; variance reduction by stratified/systematic sampling | V-cache read bandwidth + value-stage arithmetic | **1.5×** decode attention-kernel speedup vs FlashInfer/FlashDecoding, RTX 6000 Ada, 32k ctx, baseline accuracy retained (Llama-3.1-8B-Instruct) | Modest speedup vs top-k systems claiming 3–7×; kernel-level, not end-to-end; needs scores before it can sample |
| **[MagicPIG](https://arxiv.org/abs/2410.16179)** (Chen et al.) | ICLR 2025 Spotlight | KV tokens, proposal from **LSH (SimHash)** tables | **Self-normalized importance sampling** — consistent, asymptotically unbiased, *biased at finite S* (O(1/n)). Claim is lower error than top-k, not unbiasedness | KV read bandwidth; relocates hash tables to CPU | **1.9–3.9×** serving throughput; **110 ms** decode latency, Llama-3.1-8B-Instruct @96k on one RTX 4090; up to **4× lower estimation error** than top-k | Requires CPU-offload co-design; SimHash is cosine-similarity while attention needs inner product |
| **[vAttention](https://arxiv.org/abs/2510.05688)** (Desai et al.) | ICLR 2026 | KV tokens: deterministic top-k head **+ random sample of the tail**, unified | **First practical (ε, δ) guarantee** on attention-output approximation, verified at runtime | KV bandwidth + FLOPs | **+4.5 pp** on RULER-HARD (Llama-3.1-8B-Inst, DeepSeek-R1-Distill-Llama-8B); matches full quality at **10–20× sparsity** | (ε,δ) bookkeeping per head per step; budget is data-dependent → unpredictable serving latency |
| **[LARA / RA](https://arxiv.org/abs/2204.04667)** (Zheng, Wang, Kong) | ICML 2022 | Random-feature axis, with **query-specific proposals** | Recasts random-feature attention **as self-normalized importance sampling** — this is *why* RFA is biased — then derives RA, an **unbiased** estimator of full softmax attention | FLOPs, memory | Substantially lower approximation error than RFA on image classification + MT | Prefill/training-era; RA is quadratic, LARA is the linear compromise |
| **[KDEformer](https://arxiv.org/abs/2302.02451)** (Zandieh et al.) | ICML 2023 | Key tokens / matrix columns, via kernel density estimation | **Spectral-norm** error bound — explicitly stronger than the entry-wise bounds of all prior work | FLOPs | **>18×** speedup, **<0.5%** accuracy drop (T2T-ViT, ImageNet); beats exact on BigGAN with >4× speedup | Vision/generative only; no causal-decode story; heavy constants |
| **[HyperAttention](https://arxiv.org/abs/2310.05869)** (Han et al.) | ICLR 2024 | Key columns: sortLSH for heavy entries + column subsampling for the residual | Spectral guarantee, but **conditional** on two hardness parameters (max column norm; row-norm ratio after removing large entries) | FLOPs, near-linear in n | **5×** on a single attention layer @131k ctx with causal masking vs FlashAttention | The 5× is single-layer prefill, not end-to-end decode; guarantee is routinely overstated |
| **[Performer / FAVOR+](https://arxiv.org/abs/2009.14794)** (Choromanski et al.) | ICLR 2021 | **Random feature dimensions** (positive orthogonal random features) — nothing on the token axis | Unbiased / nearly-unbiased attention matrix estimate, uniform convergence, low variance | FLOPs + memory, O(L) not O(L²) | Competitive on Long Range Arena and protein modelling; retrofittable with light finetuning | Replaces the KV cache with a fixed state — no KV-bandwidth story; weak exactly where attention is spiky |
| **[YOSO](https://arxiv.org/abs/2111.09714)** (Zeng et al.) | ICML 2021 | Key tokens via **Bernoulli** variables, LSH-parameterized | Bernoulli sampling estimator; unbiasedness plausible by construction but not asserted | FLOPs + memory (quadratic→linear) | Matches softmax attention on LRA with sizable speed/memory savings | Encoder/BERT-era; no causal decode |
| **[MCA](https://arxiv.org/abs/2201.12854)** (Kim & Ko) | AAAI 2022 | **Inner contraction dimension** — textbook Drineas–Kannan–Mahoney approximate matmul, with per-token error budgets | DKM: unbiased with Frobenius-norm concentration; optimal probabilities ∝ ‖A⁽ⁱ⁾‖·‖B₍ᵢ₎‖ | FLOPs only | **Up to 11×** FLOP reduction on GLUE without accuracy loss | FLOPs-only metric, no wall-clock; BERT-scale; per-token precision is tensor-core-hostile |
| **[Reformer](https://arxiv.org/abs/2001.04451)** (Kitaev et al.) | ICLR 2020 | Random projections defining the hash; tokens then bucketed deterministically | LSH collision probability, *not* an output-error bound; accuracy rises with hash rounds (≈exact at 8) | FLOPs + memory, O(L log L) | 64k-length sequences on one accelerator; 1.05 bits/dim on enwik8 | Requires **shared Q/K** — a real architectural constraint; GPU-awkward sorting |
| **[SOCKET](https://arxiv.org/abs/2602.06283)** | 2026 preprint | Soft LSH: graded collision evidence aggregated across tables instead of hard bucket matches | Probabilistic, similarity-aware scoring kernel; preserves relative ordering of true top-k | KV scoring cost | Elevates LSH from candidate generation to a scoring kernel | Preprint; numbers unverified here |
| **[VaSE](https://arxiv.org/abs/2606.03928)** (Chang et al.) | 2026 preprint | **Stochasticity injected into KV eviction decisions** | None formal — empirical: diversity in the evicted set helps | KV memory footprint | Protects large-magnitude value states; prevents repetitive-reasoning collapse | Eviction, not estimation; training-free recipe |

### 2b. The deterministic contrast (what most deployed systems actually are)

None of these is a sampling estimator. None is unbiased. All incur the numerator+denominator
bias above. All require multiplies. Listed because they set the performance bar.

| Method | Venue | Selection rule | Headline result |
|---|---|---|---|
| **[Quest](https://arxiv.org/abs/2406.10774)** | ICML 2024 | Per-**page** min/max key summaries upper-bound q·k; load top-k pages | **7.03×** self-attention speedup, **2.23×** inference latency reduction; 100% passkey retrieval at 1% budget |
| **[SparQ](https://arxiv.org/abs/2312.04985)** | ICML 2024 | Top-r query components → approximate scores → top-k full K/V gather | Up to **8×** attention data-transfer savings (Llama 2/3, Mistral, Gemma, Pythia) |
| **[H2O](https://arxiv.org/abs/2306.14048)** | NeurIPS 2023 | Evict by accumulated attention score + recency | Up to **29×** throughput vs DeepSpeed Zero-Inference at 20% KV budget. *Has a guarantee* — dynamic-submodular greedy — but on the **eviction objective**, not on output error |
| **[SnapKV](https://arxiv.org/abs/2404.14469)** | 2024 | Score prompt KV by attention from a trailing observation window | **3.6×** generation speed, **8.2×** memory efficiency at 16k input; ~92% compression |
| **[Loki](https://arxiv.org/abs/2406.02542)** | NeurIPS 2024 | PCA keys offline; rank tokens in low-d subspace | ~**45%** speedup @3072 prompt / 512 gen |
| **[TOVA](https://arxiv.org/abs/2401.06104)** | 2024 | Drop the token with lowest attention from the most recent query | Near-full quality at 1/8 cache; **4.8×** throughput; up to 88% memory reduction |
| **[HashAttention](https://arxiv.org/abs/2412.14468)** | 2024/25 | **Learned** hash into Hamming space; bitwise top-k | **16–32×** token reduction at 32 bits/token aux memory; −4.3× attention latency in GPT-FAST. **Retrieval is multiply-free (XOR/popcount)** |

> ⚠️ **`SampleAttention` ([2406.15486](https://arxiv.org/abs/2406.15486)) contains no sampling.**
> Despite the name it is adaptive structured sparsity with deterministic two-stage query-guided
> KV filtering. Do not cite it in the Monte Carlo family.

---

## 3. Track B — Weight transfer

Framing: in typical LLM decode, **weights are 90–99% of memory-bandwidth consumption**
(cf. [2607.08407](https://arxiv.org/abs/2607.08407)); a 70B model at BF16, batch 1, reads
~140 GB per token.

### 3a. (A) — genuinely stochastic at inference

| Method | Venue | What is sampled | Guarantee | Reduces weight bytes/token | Headline result | Limitation |
|---|---|---|---|---|---|---|
| **[Speculative decoding](https://arxiv.org/abs/2211.17192)** (Leviathan et al.) | ICML 2023 | Draft tokens from `q`, accepted w.p. `min(1, p/q)`; on rejection resample from `norm(max(0, p−q))` | **Exactly distribution-preserving**, for *any* draft `q` — the accept branch contributes `min(p,q)`, the residual branch contributes the deficit, and they telescope to `p` | **Yes — by amortization.** `bytes/token ≈ (W_target + γ·W_draft) / (E[accepted]+1)` | **2–3×** on T5-XXL with identical outputs | Needs an aligned draft; gains shrink at large batch; accept rate is task-dependent |
| **[Speculative sampling](https://arxiv.org/abs/2302.01318)** (Chen et al.) | 2023 | same | Exact within hardware numerics | Yes | **2–2.5×** on Chinchilla 70B. Explicitly exceeds the *idealized memory-bandwidth ceiling* on autoregressive sampling | same |
| **[SpecInfer](https://arxiv.org/abs/2305.09781)** | ASPLOS 2024 | Token **tree**, verified in one parallel pass | Exact | Yes | **2.6–3.5×** for offloading-based inference (the bandwidth-starved regime) | Tree construction overhead |
| **[EAGLE / EAGLE-3](https://arxiv.org/abs/2503.01840)** | ICML 2024 / 2025 | Drafts at feature level (EAGLE) / direct token + multi-layer fusion (EAGLE-3) | **Lossless** | Yes | **6.5×** speedup; **1.38×** throughput at **batch 64** in SGLang — amortization survives into the batched regime | Draft head must be trained per target |
| **[SMC-SD](https://arxiv.org/abs/2604.15672)** | 2026 preprint | **Importance-weighted resampling over a population of draft particles** instead of token-level rejection | **Inverts the guarantee**: speedup becomes deterministic (fixed-size vectorized verification, no rollback), approximation error becomes stochastic but bounded | Yes | **2.36×** over speculative decoding, **5.2×** over autoregressive, within 3% of target accuracy | Approximate, not exact |
| **[THOR](https://arxiv.org/abs/2110.04260)** | ICLR 2022 | **Experts sampled at random per input, at training *and* inference**, with no gating network | No formal claim; consistency regularization (symmetric KL) makes experts interchangeable | Active weights only — but lets you sample the expert *already resident* | **+2 BLEU** over Switch Transformer; matches a SOTA MoE 18× larger | Doesn't shrink total weight storage |
| **[Progressive Stochastic Binarization](https://arxiv.org/abs/1904.02205)** (Hartmann & Wand) | EMC2@NeurIPS 2019 | Scalar products approximated by **progressive sampling of stochastic shifts** (random choice between adjacent powers of two), at inference, sample count tunable per operation | **Explicitly unbiased**; converges to continuous computation as samples grow | Partly — 1-bit + shifts | Up to **33%** inference cost reduction on ImageNet pre-pruning. **Multiply-free: additions of small integers and fixed shifts only** | CNN-era, workshop scale; saves *arithmetic*, not HBM bytes |
| **[Bitwise stochastic inference](https://arxiv.org/abs/1611.06539)** (Vogel et al.) | NIPS 2016 wkshp | Multiple 1-bit networks sampled at inference and ensembled | Ensemble of unbiased samples; **surpasses the high-precision base model** | **No — re-reads weights per sample** | 5.81% error on CIFAR-10 | Trades bandwidth *for* accuracy — the wrong direction |

### 3b. (B) — randomness at training only, deterministic at inference

| Method | Venue | Role of randomness | Reduces weight bytes/token | Headline result |
|---|---|---|---|---|
| **[Stochastic rounding](https://arxiv.org/abs/1502.02551)** (Gupta et al.) | ICML 2015 | Unbiased rounding direction during training. `E[SR(x)] = x` | No | 16-bit fixed-point training with negligible degradation |
| **[BinaryConnect](https://arxiv.org/abs/1511.00363)** | NIPS 2015 | Weights binarized stochastically per forward pass, `P(w=+1)=σ(w)`; shadow FP weights for gradients | Yes (32× storage) | Conceptual ancestor of sample-and-accumulate; multiply-free at deploy |
| **[BitNet b1.58](https://arxiv.org/abs/2402.17764)** | 2024 | **None — absmean round-to-nearest + STE.** See correction below | **Yes**, ~1.58 bits/weight | 3B matches FP LLaMA perplexity at **2.71×** faster, **3.55×** less GPU memory; 70B **4.1×** faster |
| **[DQT](https://arxiv.org/abs/2412.04787)** | ACML 2025 | **This** is where SR enters the BitNet line — SR lets you drop the high-precision shadow weights entirely | Yes (ternary deploy) | LLaMA-structured models train at ternary weights |
| **[QSGD](https://arxiv.org/abs/1610.02132)** | NIPS 2017 | Unbiased stochastic quantizer with an explicit **bits-vs-variance trade-off and convergence guarantee** | n/a (gradients) | **The cleanest formal template** for a weight-side unbiased quantizer |
| **[Bayesian Bits](https://arxiv.org/abs/2005.07093)** | NeurIPS 2020 | Learnable stochastic gates over a 2→4→8→16 residual bit-width decomposition; 0-bit = pruning. Gates frozen at inference | Yes | ResNet18 **69.16% vs 69.68%** FP at ≈**1.93%** of FP BOPs |
| **[Bayesian Compression](https://arxiv.org/abs/1705.08665)** | NIPS 2017 | Hierarchical sparsity priors; **posterior variance of a weight determines its bit-width** | Yes | **771×** compression on LeNet-5-Caffe at 1.0% error |
| **[Sparse Variational Dropout](https://arxiv.org/abs/1701.05369)** | ICML 2017 | Per-weight multiplicative Gaussian noise during training; α→∞ weights removed | Yes (unstructured) | **280×** on LeNet, **68×** on VGG |
| **[Bayes by Backprop](https://arxiv.org/abs/1505.05424)** | ICML 2015 | Weights sampled `w ~ q(w|θ)` — at inference too | **No — 2× storage, N× traffic** | The canonical Bayesian weight method makes bandwidth *worse*. Included as the counterexample |
| **[Deja Vu](https://arxiv.org/abs/2310.17157)** | ICML 2023 | Learned predictor of needed heads/neurons — **deterministic thresholding**, not sampling | **Yes, ~25% of weights loaded** | **>2×** lower latency on OPT-175B vs FasterTransformer |
| **[LLM in a flash](https://arxiv.org/abs/2312.11514)** | ACL 2024 | Low-rank predictor of surviving ReLU neurons; windowing + row-column bundling | **Yes, from flash** | Runs models **2× available DRAM**; 20–25× GPU speedup over naive loading |

### 3c. (C) — randomized, not probabilistic

[QuIP](https://arxiv.org/abs/2307.13304) · [QuIP#](https://arxiv.org/abs/2402.04396) ·
[QTIP](https://arxiv.org/abs/2406.11235) · [QuaRot](https://arxiv.org/abs/2404.00456) ·
[SpinQuant](https://arxiv.org/abs/2405.16406) · [Hash Layers](https://arxiv.org/abs/2106.04426)

Random orthogonal/Hadamard matrices used **once, offline**, as a preconditioner making weights
approximately i.i.d. sub-Gaussian so a *deterministic* rounding rule works better. No sampling
at inference, no unbiased estimator, no variance that averages over tokens.

Best results: QTIP Llama-2-70B **2-bit, WikiText-2 ppl 3.78** (QuIP# 3.91); QuaRot
LLaMA-2-70B **W4A4KV4** at ≤0.29 ppl loss, 99% of zero-shot retained, 3.39× memory saving.

Note QuIP's own framing: *without* incoherence processing, the best spectral bound **cannot
distinguish their method from stochastic rounding** — SR is the baseline these beat.

---

## 4. Direct comparison of the two tracks

| | Weights | Activations / KV |
|---|---|---|
| Share of decode bandwidth | 90–99% at short context | Dominates at long context (32k+) |
| Unbiased runtime estimators exist? | **Essentially no** (only speculative decoding, and by amortization not estimation) | **Yes** — SANTA, vAttention, LARA, KDEformer |
| Multiply-free operator exists? | Ternary quantization (deterministic selection) | **SANTA only** — gather-and-add |
| Guarantee quality | Exactness (spec. decoding) or none | Unbiasedness, (ε,δ), spectral-norm |
| Best measured decode gain | 2–6.5× (speculative decoding family) | 1.5× kernel (SANTA), 1.9–3.9× system (MagicPIG) |
| Maturity | Deployed everywhere | Research |

---

## 5. The Gap

**No published method stochastically samples weights at inference to form an unbiased estimator
of a weight matvec in order to reduce weight bytes read per token in a transformer.**

The space is occupied at the edges:

- **Hartmann & Wand (2019)** — unbiased, multiply-free, sampled at inference, but CNN-scale and
  saves *arithmetic* rather than HBM bytes.
- **QSGD** — unbiased quantizer with a formal bits-vs-variance trade-off, but for gradient
  *communication*, not weight *reads*.
- **SANTA** — the one modern unbiased sample-and-accumulate operator at LLM scale, aimed at the
  **KV cache**, not at weights.
- **Speculative decoding** — exact, but **amortizes** weight reads over accepted tokens rather
  than shrinking the per-read footprint.

Two objections any weight-sampling proposal must answer, both already recorded in the literature:

1. **Stochastic rounding at PTQ time empirically does not help and can hurt accuracy.**
2. **Quartet's finding**: SR is unbiased but *higher variance* than well-scaled round-to-nearest,
   and RTN can beat it on forward-pass alignment.

So the question is not "is sampling unbiased" (it is) but **"where does the variance go?"** SANTA's
answer is that it averages over S samples within a single value aggregation. A weight-side analogue
needs an equivalent averaging structure — over the accumulation dimension, over the residual stream,
or over tokens.

---

## 6. Three observations

**1. The sparsity assumption is the fault line, and the field converged on it from three directions.**
Top-k is optimal when attention mass concentrates; sampling is optimal when it is diffuse. Stated
independently by ScatterBrain (2021, via softmax temperature),
MagicPIG (2024, *"attention is not always as sparse as expected"*), and vAttention (2025, as its
founding insight). This is the same quantity as the `n_eff` scaling law — required sample budget
tracks the *effective* number of attended keys, not `n_k`.

**2. Sample-and-accumulate is nearly unexploited.** Of everything surveyed, only **SANTA**
eliminates multiplies from the value stage, and only **HashAttention** makes retrieval multiply-free
(XOR/popcount). Nobody has combined them.

**3. Gumbel-top-k is an unclaimed unbiasedness retrofit.** Every deployed system in §2b already ships
a top-k kernel. Adding Gumbel noise to the logits before top-k yields an *exact* without-replacement
sample of k KV positions ([Kool et al., ICML 2019](https://arxiv.org/abs/1903.06059)); a
Horvitz–Thompson or priority-sampling correction then makes the aggregate estimator approximately
unbiased at near-zero marginal cost. No paper was found doing this for KV caches. Flagged as a gap,
not as established prior art — the search was not exhaustive.

---

## 7. Corrections to common claims

- **BitNet does not use stochastic rounding.** BitLinear scales by **absmean** and rounds to the
  nearest integer in {−1,0,+1}, trained with a straight-through estimator over full-precision latent
  weights. SR enters the BitNet line only via [DQT](https://arxiv.org/abs/2412.04787), where it
  removes the shadow weights.
- **Quest's headline numbers are frequently quoted backwards.** It is **7.03× self-attention
  speedup** and **2.23× inference latency reduction**, not the reverse.
- **MagicPIG is not unbiased.** Self-normalized importance sampling is consistent and asymptotically
  unbiased, with O(1/n) finite-sample bias. The paper claims lower *error* than top-k, not unbiasedness.
- **Medusa is not distribution-preserving by default.** Its "typical acceptance" scheme uses a
  temperature-derived threshold and explicitly does not insist on exact correspondence. It is lossless
  only if run with standard rejection sampling.
- **Switch Transformer routing is deterministic** top-1 argmax. The noisy-top-k gating of
  [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538) is training-time load balancing only.
- **Nyströmformer does not sample.** It uses deterministic segment-means landmarks, and the Nyström
  low-rank guarantee does not transfer cleanly because softmax attention is not symmetric PSD.

---

## 8. Verification status

This survey was assembled under an egress policy that blocked `arxiv.org`, `openreview.net`,
`huggingface.co`, `semanticscholar.org` and `ar5iv` (403 at the proxy). **No paper PDF or abstract
page was fetched directly.** Sources were: web-search result text quoting abstracts, fetched GitHub
READMEs (reachable), and the SANTA reference list read from the local PDF.

Consequently:

- **arXiv IDs** were confirmed by the search index returning the arXiv listing with that exact
  bracketed title. This is index-level verification, not a page fetch.
- **Quantitative claims** come from retrieved abstract text where available.
- **Not verified in-session:** per-benchmark accuracy tables for SANTA, MagicPIG, Loki; Performer LRA
  numerics; vAttention wall-clock speedups; venues for SnapKV, TOVA, HashAttention, PowerInfer, QuIP#,
  Medusa, EAGLE; the canonical citation for Drineas–Kannan–Mahoney (SIAM J. Comput. 36(1):132–157,
  2006 — not arXiv-primary).
- **2026 arXiv IDs** (26xx.*) had titles and IDs confirmed via search listings only; every quantitative
  claim from them should be re-checked against the paper before being relied upon.
