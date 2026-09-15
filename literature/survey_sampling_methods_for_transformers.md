# Sampling-Based Methods for Transformer Inference: A Comparative Survey

**Scope.** Methods that use sampling / Monte Carlo / Bayesian estimation to reduce memory
traffic in transformer inference, segmented by what they attack: **weight transfer** or
**activations (KV cache)**.

**Why the segmentation matters.** In batch-1 long-context decode both are bandwidth-bound,
but the two halves of the literature are in completely different states of maturity. The
KV side has genuine unbiased estimators with concentration guarantees. The weight side has
almost none — see [The Gap](#5-the-gap).

> **Verification status.** Every arXiv ID, venue and quantitative claim below was checked
> against the primary source. See [§9](#9-sources-and-verification) for method, per-claim
> provenance, and the list of claims that changed on verification.

---

## 1. Taxonomy: three things people call "stochastic"

Most confusion in this literature comes from collapsing these. Keep them apart:

| Class | Definition | Does variance average out at inference? |
|---|---|---|
| **(A) Estimator** | Draws a random subset at runtime with known probabilities; forms an estimator of the *full* operation. Unbiasedness / concentration is claimed and proven. | **Yes** — this is the point |
| **(B) Train-time stochastic** | Randomness used during training or calibration; the deployed artifact is a fixed byte string. | No — error is baked in |
| **(C) Randomized, not probabilistic** | Random rotations/hashes chosen once offline as a *preconditioner*; deterministic at runtime. | No |

Class (C) is not a weaker estimator — it is not an estimator at all. SpinQuant's own finding
is the cleanest proof: testing LLaMA-2 7B at W4A4 over **100 randomized trials**, *"the best
random rotation matrix outperform[ed] the worst by 13 points"* of downstream zero-shot
accuracy. A real estimator's variance would average away, not lock in 13 points. (Even random
*Hadamard* rotations vary by *"as large as 6 points"*.)

A fourth category is worth naming because it dominates deployed systems: **deterministic
top-k heuristics** (Quest, H2O, SnapKV, Loki, TOVA, SparQ, HashAttention, SOCKET). These select
the argmax-scoring tokens and renormalize softmax over the kept set. That drops mass from
**both the numerator and the denominator**, so surviving tokens get inflated weight — a *biased*
estimator whose bias grows exactly when attention is flat, i.e. when the sparsity assumption
they rely on fails. This is the statistical opening that MagicPIG, vAttention and SANTA exploit.

---

## 2. Track A — Activations / KV cache

### 2a. Genuine sampling estimators

| Method | Venue | What is sampled | Estimator & guarantee | Reduces | Headline result | Key limitation |
|---|---|---|---|---|---|---|
| **[SANTA](https://arxiv.org/abs/2605.01910)** (Lee, Delacour, Callahan-Coray, Jiang, Yaras, Oymak, Srimani, Çamsarı) | ICML 2026 | KV **token** axis, drawn i.i.d. with replacement from the **post-softmax attention distribution** itself; plus Bernoulli qKᵀ on the key-**feature** axis | **Unbiased** (Prop. A.2, `E[ÂV] = AV`; Cov = Σ_q/S by Prop. A.3; vector-Bernstein tail bound, Thm. B.1). Variance reduction proved for **stratified** only (Thm. A.6) | V-cache read bandwidth + value-stage arithmetic | **1.50×** (S²ANTA-prop) / **1.51×** (S²ANTA-flash) decode attention-**kernel** speedup vs FlashInfer/FlashDecoding, RTX 6000 Ada, bs 1, 32k ctx, Llama-3.1-8B-Instruct tensor shapes. v2 adds **1.25× end-to-end** | Accuracy holds only at the stated operating points (S=128 prop, S=2048 flash); flash at S=256 collapses (FWE 66.20 vs 95.60). Multiply-free is **not realized** on the measured kernels — see §6.2 |
| **[MagicPIG](https://arxiv.org/abs/2410.16179)** (Chen et al.) | ICLR 2025 Spotlight | KV tokens, proposal from **LSH (SimHash)** tables | **Self-normalized importance sampling** — the paper asserts only a.s. **consistency**, never unbiasedness for its own estimator | KV read bandwidth; relocates hash tables to CPU | **Up to 5×** decoding throughput; **54 ms** decode latency, Llama-3.1-8B-Instruct @96k on one RTX 4090 | Requires CPU-offload co-design; SimHash is cosine-similarity while attention needs inner product. The "4× lower error" result belongs to *oracle* sampling, not to MagicPIG — see §7 |
| **[vAttention](https://arxiv.org/abs/2510.05688)** (Desai et al.) | ICLR 2026 (Poster) | KV tokens: deterministic head (sinks ∪ local window ∪ predicted top-k) **+ uniform random sample of the residual tail** | **First practical (ε, δ) guarantee** on attention-output approximation, estimated at runtime from a base sample | KV bandwidth + FLOPs | **+4.6 pp** (Llama-3.1-8B-Inst) / **+4.3 pp** (DeepSeek-R1-Distill-Llama-8B) on RULER-HARD **over HashAttention at 10% sparsity**; matches full quality at **10–20× sparsity** | Every experiment uses a **denominator-only** relaxation of the guarantee, not the full output bound. Wall-clock (Fig. 5) is CPU-offloaded KV only, unoptimized PyTorch indexing; no GPU-resident kernel |
| **[LARA / RA](https://arxiv.org/abs/2204.04667)** (Zheng, Wang, Kong) | ICML 2022 (PMLR 162:27011–27041) | Random-feature axis, with **query-specific proposals** | Recasts random-feature attention **as self-normalized importance sampling** — this is *why* RFA is biased — then derives RA, *"the first kernel linearization estimator that approximates the whole softmax attention … in an **unbiased** manner"* | FLOPs, memory | Substantially lower approximation error than RFA | Prefill/training-era; RA is quadratic, LARA is the linear compromise. **LARA itself is self-normalized MIS and is therefore *not* unbiased** — only quadratic RA is |
| **[KDEformer](https://arxiv.org/abs/2302.02451)** (Zandieh et al.) | ICML 2023 (PMLR 202:40605–40623) | Key tokens / matrix columns, via kernel density estimation | **Spectral-norm** error bound — *"all prior results merely provide entry-wise error bounds"* | FLOPs | **18.30× fewer FLOPs**, 0.47 pp accuracy drop (T2T-ViT, ImageNet: 82.55→82.08); BigGAN FID 31.41 vs exact 32.17 at **4.14×** fewer FLOPs | Vision/generative only; no causal-decode story. **The 18×/4× are FLOP ratios, not wall-clock** — the only wall-clock figure is ~8× on LRA-Text end-to-end training |
| **[HyperAttention](https://arxiv.org/abs/2310.05869)** (Han et al.) | ICLR 2024 (poster) | Key columns: sortLSH for heavy entries + column subsampling for the residual | Spectral guarantee, but **conditional** on two hardness parameters (max column norm of the normalized attention matrix; row-norm ratio after removing large entries) | FLOPs, near-linear in n | **5.4×** with causal masking / **54×** without, single attention layer @131k, fwd and fwd+bwd | The 5.4× is a **single-layer microbenchmark**, not end-to-end decode. End-to-end figure is different: ChatGLM2 50% faster @32k, ppl 5.6→6.3 |
| **[Performer / FAVOR+](https://arxiv.org/abs/2009.14794)** (Choromanski et al.) | ICLR 2021 (Oral) | **Random feature dimensions** (positive orthogonal random features) — nothing on the token axis | *"unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance"* (the hedge is the paper's own; uniform convergence is Thm. 4) | FLOPs + memory, O(L) not O(L²) | Protein modelling (TrEMBL, 36 layers), ImageNet64, PG-19; retrofittable with light finetuning | Replaces the KV cache with a fixed state — no KV-bandwidth story; weak exactly where attention is spiky. **LRA results are reproduced from Tay et al. in App. D.5, not run by the authors** |
| **[YOSO](https://arxiv.org/abs/2111.09714)** (Zeng et al.) | ICML 2021 (PMLR 139:12321–12332) | Key tokens via **Bernoulli** variables, LSH-parameterized (success prob. = collision prob. of qᵢ with kⱼ) | Bernoulli sampling estimator. The string *"unbias"* does not occur in the paper; the estimand is **YOSO-E** (expectation over hash functions), which only *resembles* softmax | FLOPs + memory (quadratic→linear) | Matches softmax attention on LRA with sizable speed/memory savings | Encoder/BERT-era; no causal decode |
| **[MCA](https://arxiv.org/abs/2201.12854)** (Kim & Ko) | AAAI 2022 (36(7):7185–7193) | **Inner contraction dimension** — the feature dim of XW, *not* tokens/keys — textbook Drineas–Kannan–Mahoney | DKM: unbiased with Frobenius-norm concentration; optimal probabilities ∝ ‖A⁽ⁱ⁾‖·‖B₍ᵢ₎‖ | FLOPs only | **11.44×** FLOP reduction on **CoLA** at α=0.2 with no accuracy loss (MCC 53.74 vs 53.74). **GLUE mean at the same α is 4.64×** | FLOPs-only metric, **no wall-clock anywhere**; BERT-scale. MCA **does not use** the DKM-optimal probabilities — it calls them impractical and substitutes WᵀW |
| **[Reformer](https://arxiv.org/abs/2001.04451)** (Kitaev et al.) | ICLR 2020 | Random projections defining the hash; tokens then bucketed deterministically | LSH collision probability, *not* an output-error bound; *"At n_rounds = 8, it already almost matches full attention"* | FLOPs + memory, O(L log L) | 1.05 bits/dim on enwik8 | Requires **shared Q/K** — a real architectural constraint; GPU-awkward sorting. The "64k on a single accelerator" line is a **motivating calculation**; experiments ran on 8 devices |
| **[VaSE](https://arxiv.org/abs/2606.03928)** (Chang, Fu, Fu, Yang, Thomason, Jia) | 2026 preprint (no venue) | **Stochasticity injected into KV eviction decisions** — ATTNV samples from the SnapKV attention distribution; DKV resamples a random projection each step | None formal — **zero theorems in the paper**. Empirical + a quantization-error analogy | KV memory footprint | Qwen3-4B avg **59.09** (VASE-ATTNV) vs R-KV 54.69, SnapKV 49.15, full 65.04; Qwen3-14B **65.81** vs R-KV 60.90, full 72.30. Throughput 411 vs 133 tok/s (**3.1×**) | Eviction, not estimation; training-free recipe. The protected statistic is **Range(vᵢ)** (max−min), not ℓ₂ magnitude. Abstract's "+4%" means percentage *points* |

### 2b. The deterministic contrast (what most deployed systems actually are)

None of these is a sampling estimator. None is unbiased. All incur the numerator+denominator
bias above. All require multiplies. Listed because they set the performance bar.

| Method | Venue | Selection rule | Headline result |
|---|---|---|---|
| **[Quest](https://arxiv.org/abs/2406.10774)** | ICML 2024 | Per-**page** min/max key summaries upper-bound q·k; load top-k pages | **7.03×** self-attention speedup → **2.23×** end-to-end decode latency reduction (32k ctx, 2048 budget, 4-bit weights). "Nearly perfect" passkey at ~1% budget: **99%** @10k, **96%** @100k |
| **[SparQ](https://arxiv.org/abs/2312.04985)** | ICML 2024 | Top-r query components → approximate scores → top-k full K/V gather | Up to **8×** attention data-transfer savings (Llama 2/3, Mistral, Gemma, Pythia) |
| **[H2O](https://arxiv.org/abs/2306.14048)** | NeurIPS 2023 | Evict by accumulated attention score + recency | Up to **29×** throughput vs DeepSpeed Zero-Inference **and vs HF Accelerate**, but only **3×** vs FlexGen — the strong baseline H2O is built on. *Has a guarantee* — Thm. 4.4, (1−1/e) submodular — but on the **eviction objective** F_score, with no bound on output error anywhere |
| **[SnapKV](https://arxiv.org/abs/2404.14469)** | NeurIPS 2024 | Score prompt KV by attention from a trailing observation window, **max-pooled** before top-k | **3.6×** generation speed at 16k; **8.2×** = max context before OOM (16k→131k), not a compression ratio; **92%** average compression at budget 1024 |
| **[Loki](https://arxiv.org/abs/2406.02542)** | NeurIPS 2024 | PCA keys offline on a calibration set; rank in low-d subspace, then revert to full d | ~**45%** speedup @3072 prompt / 512 gen (Llama2-13B, bs 16, **attention-only**, not end-to-end). Explicitly does **not** reduce KV-cache memory |
| **[TOVA](https://arxiv.org/abs/2401.06104)** | EMNLP 2024 (Main, pp. 18724–18741) | Drop the token with lowest attention from the most recent query, **averaged across heads in a layer** | One operating point (512 of 4096): **1/8 cache ↔ 4.8× throughput ↔ 87.2% memory reduction**. Conclusion's general claim is 1/8–1/4 |
| **[HashAttention](https://arxiv.org/abs/2412.14468)** | ICML 2025 | **Learned** MLP hash into Hamming space; bitwise top-k | **16×** token reduction on generic data (**32×** only with task-specific fine-tuning) at 32 bits/token; **−4.3×** attention latency in GPT-FAST at 32× sparsity. Random-projection LSH *"fails to achieve comparable performance even with more than 1000 bits"* |
| **[SOCKET](https://arxiv.org/abs/2602.06283)** (Joshi, Chowdhury, …, Desai, Shrivastava) | 2026 preprint (no venue) | Soft LSH: graded collision evidence aggregated across tables instead of hard bucket matches, then deterministic top-k + exact attention | **1.84×** vs FlashAttention2 @140k (H200); **1.53×** @72k (A100). Best-in-class at high sparsity: RULER-HARD-32K @50× — SOCKET **71.9** vs PQCache 69.5, Quest 64.1, MagicPIG 19.45. Matches hard LSH at **4.34× less memory / 4.20× less time** |

> ⚠️ **`SampleAttention` ([2406.15486](https://arxiv.org/abs/2406.15486)) contains no sampling.**
> Despite the name it is adaptive structured sparsity with deterministic two-stage query-guided
> KV filtering. The paper *explicitly rejects* randomness: *"Compared to **random sampling** or
> bottom sampling, this equidistant sampling technique is low-overhead and more stable."* A full-text
> search of v3 for *random / stochastic / probabilit\* / Monte Carlo / unbiased / estimator* finds no
> sampling distribution, no variance analysis, and no unbiasedness argument. Its one theoretical
> object, CRA, is deterministic. Do not cite it in the Monte Carlo family. (Preprint only, no venue.
> Its abs page still shows v1's 2.42×; v3 claims 5.29× TTFT @1M, with a typical range of 1.24–2.36×.)

> ⚠️ **SOCKET is not a sampling estimator either**, despite the "EsTimator" in its name. The paper
> draws the line itself: *"while MagicPig is fundamentally a **sampling-based estimator**, SOCKET is
> a **retrieval-based** approach centered on accurate top-k selection."* Its Theorem 3 is an error
> decomposition, not an ordering guarantee; "preserves top-k ordering" is an empirical claim.

---

## 3. Track B — Weight transfer

Framing: in typical LLM decode at **small-to-medium batch**, weights dominate memory-bandwidth
consumption — commonly quoted as **90–99%**, originating with
[DeepSpeed Inference](https://arxiv.org/abs/2207.00032) (Aminabadi et al., 2022). A 70B model at
FP16, batch 1, must read ~140 GB of parameters per token, against activations and KV cache totalling
under ~1.4 GB. Note the scope condition: this ratio inverts as batch size and context length grow,
which is precisely the regime Track A targets.

### 3a. (A) — genuinely stochastic at inference

| Method | Venue | What is sampled | Guarantee | Reduces weight bytes/token | Headline result | Limitation |
|---|---|---|---|---|---|---|
| **[Speculative decoding](https://arxiv.org/abs/2211.17192)** (Leviathan et al.) | ICML 2023 (Oral) | Draft tokens from `q`, accepted w.p. `min(1, p/q)`; on rejection resample from `norm(max(0, p−q))` | **Exactly distribution-preserving**, *"for any distributions p(x) and q(x)"* — the paper states it **guarantees an identical output distribution for any choice of approximation model, without restriction** | **Yes — by amortization.** `bytes/token ≈ (W_target + γ·W_draft) / (E[accepted]+1)` | **2×–3×** on T5-XXL (11B) with identical outputs; measured 2.3–3.4× | Needs an aligned draft; gains shrink at large batch; accept rate is task-dependent |
| **[Speculative sampling](https://arxiv.org/abs/2302.01318)** (Chen et al.) | arXiv / DeepMind tech report (**no conference venue**) | same | Exact within hardware numerics | Yes | **2–2.5×** on Chinchilla 70B (measured 1.92–2.46×). *"the mean tokens per second with SpS often exceeds the **idealised ceiling** on auto-regressive sampling speed imposed by the memory bandwidth"* | same |
| **[SpecInfer](https://arxiv.org/abs/2305.09781)** | ASPLOS 2024 | Token **tree**, verified in one parallel pass | Exact | Yes | **2.6–3.5×** for offloading-based inference vs FlexGen (the bandwidth-starved regime); 1.5–2.8× distributed | Tree construction overhead |
| **[EAGLE](https://arxiv.org/abs/2401.15077)** | ICML 2024 | Drafts at **feature** level | **Lossless** | Yes | **2.7–3.5×** latency, ~2× throughput (LLaMA2-Chat 70B) | Draft head must be trained per target |
| **[EAGLE-3](https://arxiv.org/abs/2503.01840)** | **NeurIPS 2025** | Direct token prediction + multi-layer feature fusion | **Lossless** — *"uses strict speculative sampling acceptance conditions, ensuring no loss in performance"* | Yes | Up to **6.5×** speedup; **1.38× throughput at batch 64** in SGLang — amortization survives into the batched regime | Draft head must be trained per target |
| **[SMC-SD](https://arxiv.org/abs/2604.15672)** (Emara, Barba da Costa, …, Abdelfattah) | 2026 preprint, under review | **Importance-weighted resampling over a population of draft particles** instead of token-level rejection | **Reverses the guarantee** (the paper's word): *"approximation quality is stochastic and the speed-up factor is deterministic."* Thm. 3.1 bounds L₂ bias/MSE decaying linearly in particle count N — but **per round only** | Yes | **2.36×** over tree-based SD and **5.2×** over autoregressive — but that is **4× H100, Llama-3.2-1B → Llama-3-70B**. Single-GPU iso-accuracy range is **1.1–2.5×** | Approximate. "Within 3%" means within 3 percentage *points of SD's accuracy* at one operating point, not a uniform envelope. Fig. 1's own bars give 2.43×, not 2.36× |
| **[THOR](https://arxiv.org/abs/2110.04260)** | ICLR 2022 | **Experts sampled at random per input, at training *and* inference**, with no gating network | No formal claim; symmetric-KL consistency regularization makes experts interchangeable | Active weights only — but lets you sample the expert *already resident* | **+2.0 BLEU** over Switch (22.4→24.4); matches a SOTA MoE **18×** larger (5.5B Switch = 300M THOR) | Doesn't shrink total weight storage |
| **[Progressive Stochastic Binarization](https://arxiv.org/abs/1904.02205)** (Hartmann & Wand) | EMC²@NeurIPS 2019 | Scalar products approximated by **progressive sampling of stochastic shifts** (random choice between adjacent powers of two), at inference, sample count tunable per operation | **Explicitly unbiased**: *"Our representation is unbiased — it approaches continuous computation with increasing sample size"* (`E[w̄] = s·|w| = w`) | Partly — 1-bit + shifts | Up to **33%** inference-cost reduction on ImageNet **relative to their own psb16 baseline**, not to float32. **Multiply-free: additions of small integers and fixed shifts only** | CNN-era, workshop scale; saves *arithmetic*, not HBM bytes |
| **[Bitwise stochastic inference](https://arxiv.org/abs/1611.06539)** (Vogel et al.) | NIPS 2016 wkshp (EMDNN) | Multiple stochastically projected **binary/ternary** networks sampled at inference and ensembled | Ensemble of unbiased samples; **surpasses the high-precision base model** (5.81% vs 10.74% FP on CIFAR-10) | **No — re-reads FP shadow weights per sample** *(our inference; the paper never discusses bandwidth)* | 5.81% error on CIFAR-10, ensemble of 29, **ternary** test-time projections | Trades bandwidth *for* accuracy — the wrong direction |

### 3b. (B) — randomness at training only, deterministic at inference

| Method | Venue | Role of randomness | Reduces weight bytes/token | Headline result |
|---|---|---|---|---|
| **[Stochastic rounding](https://arxiv.org/abs/1502.02551)** (Gupta et al.) | ICML 2015 | Unbiased rounding direction during training. *"E(Round(x, ⟨IL,FL⟩)) = x"* | No | 16-bit fixed-point training with little to no degradation (MNIST, CIFAR-10) |
| **[BinaryConnect](https://arxiv.org/abs/1511.00363)** | NIPS 2015 | Weights binarized stochastically per forward pass, `P(w=+1) = σ(w)` where **σ is the *hard* sigmoid** `clip((w+1)/2, 0, 1)`; shadow FP weights for gradients | Yes — **16×** (16-bit float → 1 bit), not 32× | Conceptual ancestor of sample-and-accumulate. Multiply-free at deploy **only for the deterministic variant**; the stochastic variant used real-valued weights at test time |
| **[BitNet b1.58](https://arxiv.org/abs/2402.17764)** | arXiv preprint ("work in progress") | **None — absmean round-to-nearest + STE.** See correction below | **Yes**, ~1.58 bits/weight | **3B**: matches FP LLaMA ppl at **2.71×** faster, **3.55×** less GPU memory. **70B**: **4.1×** faster (7.16× memory) |
| **[DQT](https://arxiv.org/abs/2412.04787)** (Zhao et al.) | ACML 2025 (PMLR 304) | **This** is where SR enters the BitNet line — SR lets you drop the high-precision shadow weights entirely, *"without relying on straight-through estimation"* | Yes (ternary deploy) | 8-bit DQT matches BitNet b1.58; ternary "feasible" but weaker. Separate group from BitNet's authors |
| **[QSGD](https://arxiv.org/abs/1610.02132)** | NIPS 2017 | Unbiased stochastic quantizer, Lemma 3.1: `E[Q_s(v)] = v` and `E‖Q_s(v)−v‖² ≤ min(n/s², √n/s)·‖v‖²` | n/a (gradients) | **The cleanest formal template** for a weight-side unbiased quantizer |
| **[Bayesian Bits](https://arxiv.org/abs/2005.07093)** | NeurIPS 2020 | Learnable stochastic gates over a 2→4→8→16 residual bit-width decomposition; 0-bit = pruning. Gates thresholded to deterministic {0,1} at inference | Yes | ResNet18 **69.16% vs 69.68% FP** at ≈**1.93%** of FP BOPs *(pre-fine-tuning; 69.39% after FT at the same BOPs)* |
| **[Bayesian Compression](https://arxiv.org/abs/1705.08665)** | NIPS 2017 | Hierarchical sparsity priors; the **mean posterior variance across a weight matrix** sets that matrix's bit-width | Yes | **771×** on LeNet-5-Caffe at 1.0% error — but that is the k-means+codebook "maximum compression" scenario the authors call *"fairly unpractical at test time"*; bit-precision-only is **419×** |
| **[Sparse Variational Dropout](https://arxiv.org/abs/1701.05369)** | ICML 2017 | Per-weight multiplicative Gaussian noise during training; α→∞ weights removed | Yes (unstructured) | **280×** on LeNet-5-Caffe, **68×** on a CIFAR "VGG-like" net (13 conv + 2 FC, *not* ImageNet VGG-16); body text says "over 65×" |
| **[Bayes by Backprop](https://arxiv.org/abs/1505.05424)** | ICML 2015 | Weights sampled `w ~ q(w|θ)` — at inference too, explicitly in the bandit and regression experiments | **No — 2× storage** (*"may have twice as many weights"*), and N× traffic *(our inference; the paper has no bandwidth analysis)* | The canonical Bayesian weight method makes bandwidth *worse*. Included as the counterexample |
| **[Deja Vu](https://arxiv.org/abs/2310.17157)** | ICML 2023 | Learned two-layer classifier for needed heads/neurons — **deterministic**; the paper tried sampling and reports *"random selection fails … resulting in drastic model degradation"* | **Yes, ~25% of weights loaded** (75% sparsity operating point) | **1.8–2×** lower latency on OPT-175B vs FasterTransformer (4.8–6× vs HF) |
| **[LLM in a flash](https://arxiv.org/abs/2312.11514)** | ACL 2024 | Low-rank predictor of surviving ReLU neurons; windowing + row-column bundling | **Yes, from flash** | Runs models **2× available DRAM**; **4–5× CPU / 20–25× GPU** speedup over naive loading |

### 3c. (C) — randomized, not probabilistic

[QuIP](https://arxiv.org/abs/2307.13304) (NeurIPS 2023) ·
[QuIP#](https://arxiv.org/abs/2402.04396) (ICML 2024) ·
[QTIP](https://arxiv.org/abs/2406.11235) (NeurIPS 2024 Spotlight) ·
[QuaRot](https://arxiv.org/abs/2404.00456) (NeurIPS 2024) ·
[SpinQuant](https://arxiv.org/abs/2405.16406) (ICLR 2025) ·
[Hash Layers](https://arxiv.org/abs/2106.04426)

Random orthogonal/Hadamard matrices used **once, offline**, as a preconditioner making weights
approximately i.i.d. sub-Gaussian so a *deterministic* rounding rule works better. No sampling
at inference, no unbiased estimator, no variance that averages over tokens.

Best results (current versions): QTIP Llama-2-70B **2-bit, WikiText-2 ppl 3.70** (v1 reported 3.78;
QuIP# 3.91; FP16 3.12); QuaRot LLaMA-2-70B **W4A4KV4** at ≤**0.47** ppl loss with **3.89×** decode
memory saving (v1 reported 0.29 / 3.39×), 99% of zero-shot retained.

Note QuIP's own framing (Thm. 4): *"without incoherence, the best spectral bound for **LDLQ** cannot
differentiate it from the **nearest and stochastic rounding** baselines."* Two precisions: the claim is
about LDLQ, QuIP's adaptive-rounding subroutine rather than QuIP end-to-end, and it fails to separate
LDLQ from round-to-nearest *and* SR, not SR alone.

---

## 4. Direct comparison of the two tracks

| | Weights | Activations / KV |
|---|---|---|
| Share of decode bandwidth | 90–99% at small-to-medium batch | Dominates at long context (32k+) |
| Unbiased runtime estimators exist? | **Essentially no** (only speculative decoding, and by amortization not estimation) | **Yes** — SANTA, vAttention, RA, KDEformer |
| Multiply-free operator exists? | Ternary quantization (deterministic selection) | **SANTA only** — gather-and-add, but *in principle*, not on current GPUs |
| Guarantee quality | Exactness (spec. decoding) or none | Unbiasedness, (ε,δ), spectral-norm |
| Best measured decode gain | 2–6.5× (speculative decoding family) | 1.5× kernel / 1.25× end-to-end (SANTA), up to 5× (MagicPIG) |
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
2. **Quartet's finding** ([2505.14669](https://arxiv.org/abs/2505.14669), NeurIPS 2025): SR is unbiased
   but has **higher MSE** than well-scaled round-to-nearest, and *"the results align with the analysis
   of Chmiel et al. that determined deterministic RTN to always be preferable to stochastic rounding
   for the forward pass."* Note the trade direction: *"SR trades higher error for perfect alignment"* —
   SR achieves **zero** projection-magnitude misalignment where RTN has 9.3e-3; RTN wins anyway, on
   MSE (1.37e-2 vs 2.77e-2) and on fitted parameter efficiency.

So the question is not "is sampling unbiased" (it is) but **"where does the variance go?"** SANTA's
answer is that it averages over S samples within a single value aggregation. A weight-side analogue
needs an equivalent averaging structure — over the accumulation dimension, over the residual stream,
or over tokens.

---

## 6. Three observations

**1. The sparsity assumption is the fault line, and the field converged on it from three directions.**
Top-k is optimal when attention mass concentrates; sampling is optimal when it is diffuse. Stated
independently by [ScatterBrain](https://arxiv.org/abs/2110.15343) (Chen, Dao, Winsor, Song, Rudra, Ré,
NeurIPS 2021 — *"sparse and low-rank approximations excel in different regimes, determined by the
softmax temperature in attention"*), MagicPIG (2024 — *"attention is not always as sparse as
expected"*), and vAttention (2025, as its founding insight).

> **On `n_eff`.** Earlier drafts appealed to "the `n_eff` scaling law" as though it were established
> literature. It is not — no such result was found, and none is cited here. It is **this project's own
> construct**, defined and measured in `HANDOFF.md` and `experiments/01_variance_vs_n_eff.py`:
> `n_eff = 1 / Σ_j p_j²`, the inverse participation ratio of the attention distribution, with relative
> estimator error scaling as `√(n_eff / S)`. Stated that way the connection to the sparsity fault line
> is real — required sample budget tracks the effective number of attended keys rather than `n_k` — but
> it is our result, not a citable one, and should be presented as such.

**2. Sample-and-accumulate is nearly unexploited — but no one has actually shipped it.** Of everything
surveyed, only **SANTA** eliminates multiplies from the value stage, and only **HashAttention** makes
the key-matching step multiply-free. Both claims need qualifying, and the qualifications are the
interesting part:

- SANTA's aggregation `ÂV = (1/S)·Σ V_{i_s}` is genuinely gather-and-add, and the `1/S` is a bit-shift
  *if* S is a power of two and the model is fixed-point. But it covers the **post-softmax stage only** —
  the score stage still multiplies unless Bernoulli qKᵀ is layered on — and the authors concede the
  win is not realized: *"While S²ANTA is multiplier-free in principle, current GPUs are highly optimized
  for dense fused multiply-accumulate instructions; we therefore view multiplier-free arithmetic as an
  additional energy-oriented benefit that may become more pronounced on future hardware."* The measured
  1.5× is **bandwidth, not arithmetic**. The arithmetic case is an energy argument: ~5× lower value-stage
  energy at k=S on Horowitz numbers.
- HashAttention is multiply-free only in the **comparison** (XOR + popcount over packed integers).
  Producing the signatures is a 3-layer MLP, and the paper reports it dominates: *"The matrix
  multiplication in mapping functions used to obtain bit-signatures dominates the latency up to 8K
  context length in GPT-FAST and up to 65K in Flash Decode."*

So the honest statement is that a fully multiply-free attention path — multiply-free *selection* and
multiply-free *aggregation*, realized in a kernel that beats FMA hardware — does not yet exist. That is
a sharper and more defensible gap than "nobody has combined them."

**3. ~~Gumbel-top-k is an unclaimed unbiasedness retrofit.~~ — RETRACTED.** An earlier draft flagged
this as an unexploited opening while noting the search was not exhaustive. It was not: three papers
occupy this space.

- **[Nexus Sampling](https://arxiv.org/abs/2606.23961)** (Duong, Le, Xie, Shrivastava, Xu, Jun 2026) is
  a direct hit. It selects KV blocks by priority keys `π_j = u_j^(1/w_j)`, proved equivalent to
  *"drawing independent exponential clocks E_j = −log(u_j)/w_j with rates w_j and taking the first K
  arrivals"* — Gumbel-top-k in its exponential-race form — yielding a probability-proportional-to-size
  sample without replacement. Proposition 4.3 then supplies exactly the correction proposed here:
  *"The **Horvitz–Thompson** estimator Ẑ_HT = Σ_j I_j z_j / p_j is **unbiased** for Z = Σ_j z_j,"* with a
  concentration bound. Results: at 80% eviction, within 1% of dense on LongBench, up to 10× smaller
  per-sequence cache.
- **[Neural Garbage Collection](https://arxiv.org/abs/2604.18002)** (Li, Hamid, Fox, Goodman, Stanford,
  Apr 2026) has a section titled *"Efficiently sampling eviction actions with Gumbel-top-k"* citing
  Kool et al. by name. Its purpose differs: the closed-form without-replacement log-probability feeds
  **unbiased policy gradients** for RL-learned eviction, and at eval time it reverts to deterministic
  top-k. *(Caution: this paper contains four "Horvitz" hits that are* Eric *Horvitz on bounded
  rationality — not Horvitz–Thompson.)*
- **[Keyformer](https://arxiv.org/abs/2403.09054)** (MLSys 2024) has been adding Gumbel noise to
  attention logits before top-k KV selection since early 2024 — but as a Gumbel-Softmax *regularizer*,
  with no Kool citation, no exact-WOR claim and no HT correction.

**What survives is narrower and worth stating precisely:** applying Gumbel-top-k with an
inclusion-probability-corrected (Horvitz–Thompson) estimator **of the attention output itself** — i.e.
replacing SANTA's with-replacement categorical draw with an exact without-replacement draw carrying HT
weights. Eviction policy (Nexus) and RL gradients (NGC) are taken; estimation of the attention
aggregate is not. [Kool et al.](https://arxiv.org/abs/1903.06059) (ICML 2019) supplies the exactness
result either way — Theorem 1 gives the exact sequential without-replacement law for the top-k of
Gumbel-perturbed log-probabilities.

---

## 7. Corrections to common claims

- **BitNet does not use stochastic rounding.** BitLinear scales by **absmean** and rounds to the
  nearest integer in {−1,0,+1}: `RoundClip(W/(γ+ε), −1, 1)`, `γ = mean|W_ij|`. The word "stochastic"
  appears **zero times** in the b1.58 paper. STE over full-precision latent weights comes from the
  original [BitNet](https://arxiv.org/abs/2310.11453), not b1.58. SR enters the line only via
  [DQT](https://arxiv.org/abs/2412.04787), where it removes the shadow weights.
- **Quest's headline numbers are frequently quoted backwards — and the paper's own abstract is why.**
  The correct reading is **7.03× self-attention speedup → 2.23× end-to-end latency reduction**. But both
  the arXiv abs page and the official PMLR v235 abstract state the reverse ("2.23× self-attention
  speedup, which reduces inference latency by 7.03×"); only the v2 full text, body and conclusion have
  it right. Physically only the body can be correct — a component speedup cannot be smaller than the
  end-to-end speedup it produces.
- **MagicPIG is not unbiased, and its "4× lower error" is not its own.** Self-normalized importance
  sampling is consistent; the paper claims only a.s. consistency for its estimator and never asserts
  unbiasedness. The 4× error reduction belongs to **oracle sampling**, which the paper defines as
  assuming *"the exact attention vector w is known, which is not true for sparse attention
  approximations."* Unbiasedness is asserted (Thm. 3.2) for oracle sampling alone.
- **Medusa** ([2401.10774](https://arxiv.org/abs/2401.10774), ICML 2024) **is not distribution-preserving
  by default.** *"We ascertain that it is typically unnecessary to match the distribution of the original
  model."* Its threshold is entropy-adaptive, `min(ε, δ·exp(−H(p)))`. It is lossless only with standard
  rejection sampling — and only for **Medusa-1**; Medusa-2 fine-tunes the backbone.
- **Switch Transformer routing is deterministic** top-1 argmax ([2101.03961](https://arxiv.org/abs/2101.03961),
  JMLR 2022), though it applies multiplicative input jitter during training. The noisy-top-k gating of
  [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538) (ICLR 2017) is introduced for **differentiable
  load balancing during training**; the paper never presents it as inference-time stochastic routing,
  but neither does it state the noise is switched off at test time.
- **Nyströmformer** ([2102.03902](https://arxiv.org/abs/2102.03902), AAAI 2021) **does not sample.** It
  uses deterministic Segment-means landmarks. The paper contains **no error bound at all** for the
  attention approximation; the compromise it actually names is ordering, not symmetry: *"instead of
  subsampling the matrix **after the softmax operation — as one should do in principle** — … landmarks
  [are] selected before softmax."* The "softmax attention is not symmetric PSD" argument is this
  survey's own gloss, not the paper's.
- **The 90–99% weight-bandwidth figure should be sourced to DeepSpeed Inference**
  ([2207.00032](https://arxiv.org/abs/2207.00032)), not to *"Who Needs DRAM? We Have Fiber"*
  ([2607.08407](https://arxiv.org/abs/2607.08407)), which quotes it secondhand in a single motivating
  sentence. That paper is an unrefereed proposal for recirculating **optical delay-line memory** over
  multi-core fiber, and it deploys the statistic to argue the KV cache is the *negligible* 1–10% — the
  opposite of this survey's purpose.
- **BinaryConnect's storage saving is 16×, not 32×** — *"at least 16 (from 16 bits single-float
  precision to single bit precision)."*

---

## 8. Known version drift

Several papers' headline numbers changed between the version this survey originally drew on and the
current one. Where they differ, the current version is quoted above.

| Paper | Earlier | Current |
|---|---|---|
| MagicPIG (2410.16179) | 1.9–3.9× throughput, 110 ms latency (v1/v2) | **up to 5×, 54 ms** (v3/v4 + ICLR camera-ready) |
| SANTA (2605.01910) | kernel-level only (v1 — the local PDF) | **v2 adds 1.25× end-to-end**; abstract changes to "stratified *and* systematic" |
| QTIP (2406.11235) | 2-bit 70B ppl 3.78 (v1) | **3.70** (NeurIPS camera-ready) |
| QuaRot (2404.00456) | ≤0.29 ppl, 3.39× memory (v1) | **≤0.47 ppl, 3.89×** (v2 / NeurIPS) |
| SampleAttention (2406.15486) | 2.42× TTFT (v1, still on the abs page) | **5.29× @1M** (v3); typical range 1.24–2.36× |
| Quartet (2505.14669) | SR on the backward pass (v1) | **RTN on the backward pass** (v4, Jan 2026) |

Quest and SampleAttention additionally have **stale abstracts on their arXiv abs pages** that
contradict their own full texts. Prefer the body.

---

## 9. Sources and verification

**Method.** All 48 arXiv IDs cited were fetched directly from `arxiv.org/abs/` and confirmed to
resolve to papers matching their attributed titles. Venues were established from arXiv comments and
journal-refs, cross-checked against PMLR, OpenReview, ACL Anthology, NeurIPS proceedings and ASPLOS
where the arXiv metadata was empty or stale. Every quantitative claim was verified against the paper's
full text (PDF or ar5iv HTML), not against search-result snippets or abstracts — a distinction that
mattered, since four papers here have abstracts contradicting their own bodies. The SANTA PDF in this
repository was read directly.

**Confidence tiers.**

- **Verified against full text** — every number in §2, §3, §5, §7 and §8, each traced to a specific
  abstract line, section, theorem or table.
- **Verified but version-dependent** — everything in §8. Pin the version when quoting.
- **This survey's own inference, not a paper claim** — explicitly marked inline: the bandwidth argument
  against Vogel et al.; the "N× traffic" figure for Bayes by Backprop; the symmetric-PSD gloss on
  Nyströmformer.
- **Unsupported, flagged for resolution** — the `n_eff` scaling law (§6.1). No citation exists in this
  document and none was found.

**What changed on verification.** Four claims were outright wrong: the EAGLE row conflated three
separate papers and mis-stated the venue; BinaryConnect's storage saving was doubled; the Quartet
alignment comparison ran backwards; and §6's Gumbel-top-k novelty claim was refuted by three papers.
Roughly twenty more were imprecise — stale version numbers, best-case figures presented as typical,
FLOP ratios presented as wall-clock, single-task results presented as benchmark averages, and
attribution of a result to the wrong method. Eight venues were wrong or missing. Two papers
(SampleAttention, SOCKET) were confirmed to be misfiled by name rather than by method — one of which
this survey had already caught, the other of which it had not.
