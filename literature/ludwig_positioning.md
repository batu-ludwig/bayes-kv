# Where Ludwig Computing Sits in the Literature

A positioning map built from primary sources: the company's own publication record, the
surrounding academic corpus, the competitive field, and the specific intersection the company
is aiming at.

> **Read §1 first.** The most consequential finding is a correction to an assumption that the
> contents of this repository invite.

---

## 1. The map is not the one the repository implies

`literature/` contains the SANTA paper and a methodology study of it. That invites the reading
that SANTA is a Ludwig paper and that Ludwig sits inside Kerem Çamsarı's OPUS Lab orbit at UCSB.
**It is not, and it does not.**

Verified by full-text search of the PDFs:

| Paper | "Ludwig" occurrences | Printed affiliations |
|---|---|---|
| [SANTA, 2605.01910](https://arxiv.org/abs/2605.01910) | **0** | UCSB ECE, Michigan ECE, CMU ECE |
| [2109.14801](https://arxiv.org/abs/2109.14801) | 1 | Purdue ECE + **Ludwig Computing** |
| [2507.07763](https://arxiv.org/abs/2507.07763) | 2 | Purdue Elmore ECE, *"Also at Ludwig Computing"* |
| [2503.10302](https://arxiv.org/abs/2503.10302), [2606.25313](https://arxiv.org/abs/2606.25313), [2108.09836](https://arxiv.org/abs/2108.09836) | 0 | — |

**The actual structure is two companies from one academic lineage.** Supriyo Datta's group at
Purdue originated the p-bit. It produced both Behtash Behin-Aein (who founded Ludwig) and Kerem
Çamsarı (who did his PhD there in 2015, went to UCSB, and — per his own OPUS Lab bio — is
**co-founder and CEO of Flucta**, a separate startup commercializing probabilistic computing
hardware for AI inference).

```
              Supriyo Datta / Purdue  — origin of the p-bit
                        |
        +---------------+----------------+
        |                                |
  Behin-Aein, Kaiser,              Çamsarı → UCSB (OPUS Lab)
  Ghantasala, Jaiswal                      |
        |                                  |
   LUDWIG COMPUTING                     FLUCTA
   (Purdue axis)                        (UCSB axis)
   p-circuits silicon,                  SANTA, million-p-bit machine,
   sampling-based DNNs                  Ising/optimization, IsingFormer
```

So SANTA is a **Flucta-adjacent** artifact, not a Ludwig one. That does not make it less worth
studying — it is the best existing statement of the algorithmic case for sampling attention —
but it changes what it is evidence *of*. It is a competitor lineage building the algorithmic
demand for hardware very similar to Ludwig's.

### Company record

| | |
|---|---|
| Founded | **2021**, not 2024 — [2109.14801](https://arxiv.org/abs/2109.14801) (submitted 30 Sep 2021) already prints "Ludwig Computing, 516 Northwestern Ave, West Lafayette, IN 47906", the Purdue Research Park address |
| Founder | **Behtash Behin-Aein** (PhD Purdue, Datta group; all-spin logic, embedded MRAM) |
| Co-founder | **Jan Kaiser** (probable — Cyclotron Road profile) |
| CTO | **Lakshmi Anirudh Ghantasala** |
| Scientific advisor | Unnamed on the site as "inventor of p-bits"; the competing-interests declaration in 2507.07763 names **SD = Supriyo Datta** as holding a financial interest |
| Incubation | Cyclotron Road / Activate at LBNL; **DOE CRADA FP15578 (AWD6701)**, OSTI 2478023, via the DOE Advanced Manufacturing Office |
| Seed | 17 Nov 2025. Named investors: Climate Capital, Impact Science Ventures, Activate Global, Berkeley SkyDeck, SVBIG |
| Size | ~9–10 people, Mill Valley CA plus space at LBNL |
| Name | After **Ludwig Boltzmann** |

**Publication footprint: two arXiv papers plus two chip-conference papers, all on the Purdue axis.**

- [**2109.14801**](https://arxiv.org/abs/2109.14801) — *Benchmarking a Probabilistic Coprocessor*. Kaiser, Jaiswal, Behin-Aein (Ludwig), Datta.
- [**2507.07763**](https://arxiv.org/abs/2507.07763) — *Improving deep neural network performance through sampling*, **npj Unconventional Computing**, DOI 10.1038/s44335-026-00063-7. Funded by **ONR-MURI N000142312708 "OptNet"** — *not* an NSF STTR. **This is the load-bearing paper; see §4.**
- **ISSCC 2025**, *p-Circuits: Neither Digital Nor Analog* (DOI 10.1109/ISSCC49661.2025.10904553)
- **IEDM 2025**, *Building Block For P-Circuits* (DOI 10.1109/IEDM50572.2025.11353676)

The ISSCC/IEDM pair matters disproportionately: getting p-circuits into the two flagship
*circuits and devices* conferences is a different and harder credential than an arXiv preprint,
and it is where a hardware company's claims are actually adjudicated.

---

## 2. The taxonomy that decides everything

"Probabilistic hardware" names at least five unrelated things. Sorting by **what the hardware
outputs** separates the field far better than device physics does.

| Class | What it does | Output | Who |
|---|---|---|---|
| **A. True samplers** | Hardware's stationary distribution *is* the specified distribution | a **stream of samples** | **p-bits (Purdue, UCSB, Tohoku)**, Extropic TSU, Normal SPU (Gaussian only) |
| **B. Annealers** | Noise escapes local minima; a schedule *deliberately distorts* the target so it collapses onto the mode | **one answer** (argmin) | D-Wave, NTT CIM, Toshiba SBM, Fujitsu DA |
| **C. Stochastic computing** | Randomness is a *number format*; arithmetic is deterministic | **a number** | Gaines lineage (1967), SC-NNs, **Normal's CN101** |
| **D. Uncertainty propagation** | Deterministic arithmetic on distribution objects; no runtime randomness | **a distribution** | Signaloid |
| **E. Noise-free analog/optical/reversible** | No probabilistic content at all | **a number** | Mythic, Lightmatter, Vaire, Rain |

**Ludwig is class A.** That is the whole position, and three traps follow from it.

**Trap 1 — an annealer is not a sampler.** An annealer that sampled correctly would be a *bad*
annealer; the temperature schedule exists precisely to stop sampling the true distribution. So
D-Wave, coherent Ising machines, Toshiba's SBM and Fujitsu's Digital Annealer are **not
competitors for generative modeling or Bayesian inference**, however much the shared Ising
formalism suggests otherwise. Conversely, a good sampler is a mediocre optimizer until you add
a schedule.

**Trap 2 — "thermodynamic" does not imply analog, or even novel.** Normal Computing's CN101 is,
by its own abstract ([2608.00754](https://arxiv.org/abs/2608.00754)), *"a **digital**
thermodynamic computer"* implementing *"discrete accumulator dynamics on standard CMOS using
**stochastic computing principles**."* The best-funded thermodynamic chip in the field is, at
the transistor level, a 1967 stochastic computer wearing a new label. That is a usable line, and
it also means the thermodynamic-computing brand is currently substrate-agnostic marketing rather
than a claim about physics.

**Trap 3 — sampler *correctness* is the unsolved problem nobody markets.** A class-A device is
only useful if the hardware distribution matches the specified one. Physical-noise p-bits (sMTJ,
subthreshold shot noise) are tiny and cheap but carry device-to-device variation, drift and
bias, so the sampled distribution is wrong in uncontrolled ways. Digital p-bits (LFSR or ring
oscillator on FPGA — which is what nearly all state-of-the-art p-bit *systems* actually are) are
exactly specified but cost area. **That trade is the real technical axis of the field.** The
adjacent memristor-Bayesian literature has now named the same problem precisely: conductance
state (the mean) and stochastic noise (the variance) are physically coupled, so μ and σ cannot
be set independently — a 2026 *Nature Communications* result (DOI 10.1038/s41467-026-74898-w)
is notable specifically for decoupling them. Querlioz's group gave up on sampling hardware
altogether over this, shipping deterministic log-domain arithmetic instead
([2406.03492](https://arxiv.org/abs/2406.03492)). **Calibration and variation tolerance, not
fJ per sample, is where a p-bit company lives or dies.**

---

## 3. The competitive field

### The only true head-on competitor is Extropic

CMOS subthreshold shot noise; a Thermodynamic Sampling Unit that block-Gibbs-samples a
programmable discrete energy-based model. Genuine class A. Algorithm: Denoising Thermodynamic
Models, [2510.23972](https://arxiv.org/abs/2510.23972), published in *npj Unconventional
Computing* 2026 (DOI 10.1038/s44335-026-00075-3).

**The 10,000× claim, stated precisely.** The abstract says a *"**system-level analysis**
indicates that devices based on our architecture **could** achieve **performance parity** with
GPUs on a **simple image benchmark** using approximately 10,000 times less energy."* Unpacked:
the benchmark is **binarized Fashion-MNIST** scored by FID against generic VAE/GAN/DDPM
baselines; **no GPU is named**; and the paper states its experiments *"were done using
simulations of thermodynamic hardware, running on a classical computer."* The number rests on an
assumed E_cell ≈ 2 fJ. It is a projection against an un-optimized baseline, and the paper is
candid about that.

**Extropic has just pivoted onto NVIDIA's turf.** Z1T (5 Sep 2026) is a family of transformer-like
sparse models co-designed to Z1's fixed p-bit connectivity, run in a disaggregated pipeline with
FPGAs. Reported ~3 orders of magnitude for ops assigned to Z1, ~2 orders (~140×) overall — but
described as *"best estimates anchored to prior pbit experiments,"* not measurements, and the
sparse model reportedly needs **~10× more training FLOPs than GPT-2** to reach comparable loss.
Funding: $14M seed (Dec 2023), plus a **$75M Letter of Intent with the US Dept. of Commerce
CHIPS R&D Office** (announced 29–30 Jul 2026) within an $874M seven-company NIST package.

Two strategic consequences. First, **the generative-sampling lane is emptier this month than it
was last year** — Extropic has walked out of it toward LLM inference, at a large training-compute
tax. Second, expect their framing in the room: [2510.23972](https://arxiv.org/abs/2510.23972)'s
abstract dismisses prior stochastic computers as relying on *"exotic, unscalable hardware"* — a
direct swipe at MTJ p-bits. It is rebuttable, because most state-of-the-art p-bit work is digital
logic on commodity FPGAs and CMOS.

### Normal Computing is drifting out of the competitive set

Gen-1 SPU was 8 RLC cells on a PCB doing genuine analog Gaussian sampling and thermodynamic
linear algebra ([2312.04836](https://arxiv.org/abs/2312.04836), *Nature Communications* 2025).
CN101 is digital CMOS (see Trap 2). Its "up to 1000×" figure appears only in press releases —
the paper gives no process node, no die scale and no measured energy. The decisive signal is the
**$50M round led by Samsung Catalyst (Mar 2026), framed as "AI-native semiconductor design"**:
EDA software is now the primary business, with thermodynamic hardware secondary.

### Everyone else is a different class

| Player | Substrate | Class | Status |
|---|---|---|---|
| Signaloid | RISC-V extension, arithmetic on distributions | D | C0-ASIC taped out with TSMC, May 2026 |
| Vaire | 22nm adiabatic/reversible | E | "Ice River" test chip, net energy recovery |
| Mythic | Flash analog compute-in-memory | E | Shipping; recapitalized after 2022 near-death |
| Lightmatter | Silicon photonics | E | S-1 filed Apr 2026 |
| D-Wave | Superconducting flux qubits | B | Public; FY2025 revenue $24.6M |

**No other venture-funded p-bit-native startup was found besides Extropic and Flucta.** And no
company at all was found building a general-purpose Gaussian/Bernoulli sampling accelerator for
ML as a product — Extropic's TSU samples discrete EBMs only. That lane is commercially open.

### Do not pitch combinatorial optimization

After roughly fifteen years and more than $1B across superconducting, optical and ASIC
substrates, **no Ising machine holds a durable, independently replicated advantage over
well-tuned classical heuristics.** Toshiba's Simulated Bifurcation Machine — *an algorithm on
commodity hardware* — is state of the art on several benchmarks. An independent 2025 benchmark
([2507.22117](https://arxiv.org/abs/2507.22117)) concludes Fujitsu's Digital Annealer is merely
*"competitive."* D-Wave's revenue is largely access and services, not demonstrated optimization
advantage. The canonical survey is [2204.00276](https://arxiv.org/abs/2204.00276) (Mohseni,
McMahon, Byrnes, *Nature Reviews Physics* 2022). This is a graveyard; the strongest version of a
p-bit pitch walks around it.

---

## 4. The energy argument has to be relocated — and Ludwig's own paper already does it right

This is the most actionable finding in this document.

### The arithmetic-energy argument does not survive contact with the numbers

Horowitz's ISSCC 2014 figures (45nm), from the original slide:

| Operation | 8-bit | 32-bit |
|---|---|---|
| Integer add | 0.03 pJ | 0.1 pJ |
| Integer multiply | **0.2 pJ** | 3 pJ |
| FP add | 0.4 pJ (fp16) | 0.9 pJ |
| FP multiply | 1 pJ (fp16) | **4 pJ** |
| SRAM read (64-bit) | 8KB 10 pJ · 32KB 20 pJ · 1MB 100 pJ | |
| **DRAM** | **1.3–2.6 nJ** | |

The derived ratios decide the argument. Multiply ÷ add is **6.7× at int8** and 2.5× at fp16 — so
eliminating multiplies saves at most ~85% *of arithmetic alone*. Meanwhile **DRAM ÷ int8 multiply
≈ 6,500× per 64-bit access**, and even a 32KB SRAM read is 100× an int8 multiply.

And the gap is **widening**. Lee & Bruck, *Data Gravity and the Energy Limits of Computation*
([2603.26053](https://arxiv.org/abs/2603.26053), Mar 2026), gives 7nm figures and a
data-movement-to-compute ratio G_d:

| Regime | G_d |
|---|---|
| 45nm cache | 2.5–25 |
| 45nm DRAM | 325–650 |
| **7nm HBM** | **≈190–366** |
| **7nm DRAM** | **≈992** |
| Biology | <1 |

Their key observation: **logic improved ~3× from 45nm to 7nm while memory access energy stayed
essentially flat at ~1300 pJ.** The multiply-versus-memory ratio has gotten *worse*.

**The multiply-free literature's own results confirm this.** ShiftAddLLM
([2406.05981](https://arxiv.org/abs/2406.05981), NeurIPS 2024) models **87.2% energy reduction**
on OPT-66B using an Eyeriss-style model on Horowitz constants — 0.024 pJ per shift against
**3.7 pJ per multiply**. Its **measured** latency improvement on a real A100-80GB is
**6.5–60.1%**. That gap, inside a single paper, *is* the arithmetic-versus-memory gap. The same
critique lands on AdderNet, whose energy headlines come from arithmetic-only analytic models —
and note that every AdderNet accelerator that actually got built had to add memory optimization
to realize its wins. The ternary/BitNet accelerator line is the same story: its gains are
overwhelmingly from 16× weight-memory reduction, not from arithmetic.

> **This critique lands on SANTA's energy claim specifically.** SANTA estimates ~5× value-stage
> energy reduction from **3.7 pJ per FP32 multiply versus 0.9 pJ per add**. For a quantized LLM
> the relevant figure is **int8 multiply = 0.2 pJ** — an 18× inflation of the very quantity being
> eliminated. The paper is careful to call these "approximate … hardware-agnostic" estimates, and
> its abstract leads with bandwidth, not arithmetic: *"Autoregressive decoding becomes
> bandwidth-limited at long contexts."* The bandwidth claim is the strong one. The arithmetic
> claim should be the kicker, never the pitch.

### Ludwig's own paper makes the argument the correct way

[**2507.07763**](https://arxiv.org/abs/2507.07763) (*npj Unconventional Computing*, 2026) asks
exactly the right question: is it cheaper to take *T* samples from a 1-bit probabilistic network,
or one sample from a *b*-bit deterministic one? Its energy expression is

```
ε_EO = n·b_w·ε_wM  +  T[(n+1)·b_a·ε_aM + ε_S + ε_N]
```

**The weight-loading term is not multiplied by T.** Extra samples are therefore nearly free,
because the dominant cost — moving weights — is paid once. Results: **2 samples of a 1-bit
probabilistic network match the deterministic baseline; 10 samples approach 3-bit deterministic;
FPGA validation shows 2.55× energy improvement.**

Its stated reason sampling wins is that **DNNs are memory dominated**, so additional sampling
compute is negligible end-to-end. It does *not* argue that multiplies are expensive.

**That is the stronger argument, it is peer-reviewed, it is the company's own, and it is
currently buried.** Everything in the energy literature above independently corroborates it. The
positioning writes itself: *sampling is free because memory dominates, not because multipliers
are expensive.*

---

## 5. The open niche, stated precisely

**The framing is not novel.** "Sampling is cheap, therefore sampling-based ML becomes viable" is
the explicit thesis of Extropic, of Normal, of the Purdue p-bit line, and of the Bayesian-CIM
silicon community. It now has a survey.

**The application is completely open.** Across every search formulation tried, **zero results
connect probabilistic sampling hardware to transformer attention or the KV cache.** The two camps
have never been introduced:

- The **attention-hardware** camp (HiKV, gain-cell attention, FCDC, ASCEND, Xpikeformer) uses
  randomness only as a *number representation*, or fights it as *noise*.
- The **probabilistic-hardware** camp uses randomness as *computation* but has never looked at
  attention.

One theoretical bridge exists and nobody has built on it: attention weights ∝ exp(−βE_j) form a
**Gibbs distribution**, so an attention head is natively a Boltzmann sampler. Boltzmann Attention
([2606.12478](https://arxiv.org/abs/2606.12478)) makes the Ising connection algorithmically and
gestures at Ising samplers, but builds nothing.

### Sampling really has become nearly free in silicon

This is the measured evidence that the premise holds — and none of it targets transformers:

| Design | Energy per sample | Status |
|---|---|---|
| Naive 8-bit Box–Muller, 28nm | 119–229 pJ | reference |
| Generic CMOS RNG | ~2 pJ/bit | reference |
| [2501.04577](https://arxiv.org/abs/2501.04577) — 65nm BNN accelerator, in-word GRNG | **360 fJ** | **taped out** |
| [2606.07439](https://arxiv.org/abs/2606.07439) — 65nm multi-modal Bayesian engine, calibration-free GRNG | **16.3 fJ** | **taped out** |
| [2606.10822](https://arxiv.org/abs/2606.10822) — write-free FeFET GRNG | **640 aJ** | reported |

An MTJ p-bit is roughly 10× lower energy per random bit and ~300× smaller than a 32-bit LFSR.

### The benchmark to beat is deterministic, and it is in real silicon

**HiKV** ([2607.22389](https://arxiv.org/abs/2607.22389), IEEE TCAS-I 2026, KU Leuven) is RTL
synthesized and **post-layout in TSMC 16nm at 300 MHz** — the most credible silicon-grade KV
accelerator found. Two-stage KV sparsification with a reconfigurable importance sorter:
**7.95× attention speedup, 90% energy reduction, 7.17× fewer external memory accesses**, <1%
accuracy loss, and 1.82–4.87× fewer memory accesses than H2O at iso-accuracy. **Selection is
deterministic top-k via min-heap and chunk sorting — explicitly not sampling.**

HiKV is simultaneously the validation and the challenge. It proves the gather-sparse-V datapath
wins in silicon. And it spends real area on **a sorter that a p-bit sampler would replace**.

> **The sharp, winnable, currently unmade claim:** *sampling replaces the sorter more cheaply
> than the sorter costs.* That is an argument about **selection hardware** — not about
> multipliers, and not about bandwidth, both of which HiKV already captures. It is the one
> argument a p-bit company can make that neither HiKV nor Extropic nor SANTA has made.

### The closest near-miss worth reading

**FCDC** ([2605.28208](https://arxiv.org/abs/2605.28208)) finds that analog-input fragility
**localizes to the value projection** — precisely the stage SANTA targets — and fixes it by
**deliberately injecting dither at the periphery**, recovering worst-case collapse to near
baseline without retraining. That is noise added on purpose, in the V stage, and it works. It is
dithering rather than Monte Carlo, but it is the strongest existing evidence that V-aggregation
tolerates, and can benefit from, stochasticity. Its honest numbers are also instructive:
18–35× versus a single-user GPU, narrowing to **1.4–4.7× versus optimized serving baselines**.

---

## 6. The four hardest questions

**1. Why isn't a million digital p-bits already enough?**
[2606.25313](https://arxiv.org/abs/2606.25313) (Jun 2026) runs **1,000,000 p-bits across 18
FPGAs at ~10¹² flips/s**. Every headline p-bit *systems* result — this, the 5,000–10,000-p-bit
sparse Ising machine, the 4,264-p-bit Boltzmann trainer — is **digital emulation**. Physical-device
work has not yet produced a system that beats them. The answer must therefore be **energy per
sample and areal density at scale**, never capability. Note also that this flagship is calibrated
as GPU *parity*: its decay exponent is 0.2820 against the GPU's 0.2836, i.e. *"statistically
indistinguishable."*

**2. Where does the measured energy story actually stand?**
Best *measured* silicon in the p-bit corpus is **1.2 pJ/update** (28nm, 27,648 spins,
[2609.07907](https://arxiv.org/abs/2609.07907)). The famous "2 to 5 orders of magnitude" and
0.02 pJ/flip figures are **projections for monolithic CMOS+sMTJ chips that do not exist** — and
the paper making them says so: *"the large-scale monolithic integration of CMOS + sMTJ remains
to be seen."* Ludwig's own FPGA-validated 2.55× energy improvement is modest by comparison but
has the virtue of being real.

**3. Does the score stage shrink?**
Sampling from the post-softmax distribution requires the scores, which requires reading K. The
V-stage saving is real; the K-stage saving depends entirely on Bernoulli qKᵀ holding up. Roughly
half the KV traffic may remain. This is independently confirmed by the repository's own analysis
in `HANDOFF.md`: the ceiling at 1.88× *is* the K cache, which neither SANTA kernel touches.

**4. Is discrete-device p-bit hardware about to be commoditized?**
Two 2026 academic results point that way. Tohoku and **NIST** reported (IEEE EDL, 26 May 2026)
the first integrated spintronic p-bit fabricated on silicon in a standard semiconductor process
— an academic/government pair, not a company. And a *Nature Communications* 2026 result
(s41467-026-71906-x) cointegrates charge-trap FETs reconfigurable as **both p-bit and synapse**
on one wafer in a standard CMOS flow. If the device becomes a standard-process library element,
the defensible layer moves up the stack — to calibration, architecture and the application.

---

## 7. Where Ludwig sits, in one page

**By lineage:** the Purdue/Datta branch of the p-bit family, the branch that invented the
primitive. The UCSB branch (Çamsarı/Flucta) has more papers, larger demonstrated systems and
the transformer work; the Purdue branch has the origin, the ISSCC/IEDM circuits credentials, and
the sampling-for-DNNs argument.

**By class:** a **true sampler** (class A) — the same class as Extropic, a different class from
every annealer, and a different class from Normal's shipping chip, which is digital stochastic
computing.

**By workload:** the field's silicon is overwhelmingly in **optimization**, which is a graveyard.
**Generative modeling** just emptied out as Extropic left for LLM inference. **Inference
acceleration** is where Ludwig's own published argument (2507.07763) and this repository's
SANTA study both point, and it is the crowded lane — but the *specific* intersection of
probabilistic sampling hardware with attention and the KV cache is **entirely unoccupied**.

**The strongest defensible position**, given everything above:

1. Lead with **bandwidth**, because memory dominates arithmetic by 190–1000× and the ratio is
   widening. SANTA's abstract leads this way; Ludwig's own paper proves it formally with the
   T-independent weight-loading term.
2. Make the **selection-hardware** argument, because HiKV has already proven the datapath in
   16nm silicon and spends real area on a sorter that a sampler could replace. This is the one
   claim nobody has made.
3. Treat **multiplier-free** as a kicker, never the pitch, and avoid the 3.7 pJ FP32 figure.
4. Treat **calibration and variation tolerance** as the technical moat, because that is where
   physical-noise samplers actually fail and where the adjacent memristor community has already
   conceded the difficulty.
5. Stay out of combinatorial optimization.

---

## 8. Sources and verification

**Method.** Four parallel research tracks: the company record, the surrounding academic corpus,
the competitive landscape, and the hardware/attention intersection. Every arXiv ID cited here was
resolved against `arxiv.org`. The three affiliation claims in §1 were verified by me personally,
by extracting each PDF and searching its full text — not taken on report.

**Egress constraints (material to confidence).** This session's network policy blocked, among
others: **`ludwigcomputing.com`**, `opus.ece.ucsb.edu`, `ce.ucsb.edu`, `nsf.gov`, `sbir.gov`,
`osti.gov`, `patents.google.com`, all USPTO hosts, `linkedin.com`, `crunchbase`-class
aggregators, `scholar.google`, `openalex`, `crossref`, `web.archive.org`, `extropic.ai`,
`spectrum.ieee.org`. Reachable: `arxiv.org`, the Semantic Scholar API, `github.com`.

Consequences to keep in view:

- **The company's own website was never read.** Everything in §1 comes from primary PDFs,
  the DOE CRADA record, and search-engine summaries of pages that could not be opened directly.
- **The NSF STTR could not be verified at all** — no award number, amount, dates, PI or abstract.
  The only grant visible in a primary document is **ONR-MURI N000142312708**. If the STTR is real
  it should be confirmed from an unrestricted network.
- **No patent search was possible.** Absence of findings is not absence of patents. An assignee
  search plus Purdue Research Foundation licensing records is the obvious next step, since core
  IP may be licensed from Purdue rather than held directly.
- Team roles carry mixed confidence, flagged inline in §1. Behin-Aein as founder is high
  confidence; exact titles are not.
- Extropic's Z1/Z1T hardware timeline and Normal's "1000×" come from secondary coverage only.
- The Tohoku/NIST silicon p-bit (IEEE EDL, May 2026) is corroborated only by secondary sources
  and should be verified directly — it is the single most important device milestone for a
  spintronic thesis.

**What changed on investigation.** The working assumption that SANTA was a Ludwig paper and that
Ludwig sat in the OPUS Lab orbit was wrong, and the correction propagates through the whole
analysis: the company's real publication axis is Purdue, its real flagship argument is
2507.07763 rather than SANTA, and Çamsarı is a principal at a separate company. The energy
framing also inverted — the arithmetic case that SANTA's §4 and much of the multiply-free
literature rest on does not survive the Horowitz and post-Horowitz numbers, while the bandwidth
case does, and Ludwig's own paper already argues it correctly.
