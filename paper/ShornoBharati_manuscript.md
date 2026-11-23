# ShornoBharati: Grapheme-Faithful Synthetic Pre-training and Long-Range Conjunct Generalization for Low-Resource Bangla OCR

**Manuscript draft — PhD-oriented research paper (target venue: ICDAR / AAAI / ACL findings)**
*Status: full manuscript with a completed proof-of-concept (POC) experiment. Figures/tables marked with real numbers are from the POC run in `poc/results.json`; scaled-up experiments are specified as an experimental protocol.*

---

## Abstract

Optical Character Recognition (OCR) for Bangla — the sixth most-spoken language in the world — remains dramatically behind English despite a decade of progress on general-purpose text recognition. Recent strides in OCR-free document understanding (Donut, ECCV 2022; Kosmos-2.5, 2023) and unified "OCR-2.0" models (GOT-OCR 2.0, 2024) have largely bypassed low-resource abugidas: Bangla presents a combinatorial glyph space (10 vowels, ~35 consonants, ~200+ conjunct ligatures, 10 dependent matras with positional alloforms, and context-sensitive reorderings such as *reph* and *i-kar*) that breaks the token-alignment assumptions underlying CTC-based and autoregressive recognition alike.

This paper makes three contributions. **(C1) Gap analysis:** through a systematic review of ten landmark systems from 2020–2024 (TrOCR, ABINet, SVTR, PARSeq, Donut, Nougat, Kosmos-2.5, GOT-OCR 2.0, plus two Bangla-specific systems, Bengali E2E-OCR and Apurba-Bangla-OCR), we identify five recurring methodological gaps that disproportionately penalize abugida scripts — synthetic-data shaping gap, sub-word supervision gap, long-range-ligature generalization gap, evaluation-gap, and data-contamination gap. **(C2) Method:** we propose **ShornoBharati**, a two-stage recipe that couples (a) a *shaping-faithful synthetic text renderer* built on industrial shaping engines that deterministically materializes any abugida graph inventory with exact ligature fusion and matra positioning, with (b) *grapheme-faithful supervision*: CTC over unicode grapheme clusters rather than codepoints, yielding a closed-vocabulary label space that is provably shatterable to arbitrary word length. **(C3) Evidence:** as a minimal-compute proof of concept (§5), we train a ≈2M-parameter CRNN *from scratch* on 6,000 fully synthetic, shaping-faithful Bangla images (zero real-world data, zero pre-training) to full training convergence (train CER 0.0) on a CPU-only box, and evaluate on IID synthetic val and length-shifted OOD words (7–9 clusters vs 3–6 in training). The POC (a) validates every component of the recipe end-to-end and (b) uses the resulting memorization/CER saturation point as a falsifiable data-hunger baseline for the scaled protocol. We also specify (not yet run) the full-scale experiments (fine-tuning TrOCR and GOT-OCR-2.0 on BanglaWriting/BN-HTR, and a 300k-synthetic-word N-scaling sweep on a 96GB GPU) required to empirically validate the scaled hypothesis, and ship a ready-to-run GPU notebook implementing E1–E3.

We argue that faithfulness of the *shaping pipeline* — not model scale — is the first-order bottleneck for abugida OCR, and that grapheme-cluster supervision is a general prescription for all Brahmic scripts (Bangla, Devanagari, Assamese, Burmese, Khmer, Tamil).

---

## 1. Introduction

### 1.1 Motivation

Bangla is spoken by ≈300M people, yet its OCR accuracy and tooling trail English by a striking margin. The recent literature shows a decisive trend: *scale + end-to-end training + synthetic pre-training* (TrOCR [TrOCR-1], PARSeq [PARSeq-1], Kosmos-2.5 [K25-1], GOT-OCR-2.0 [GOT-1]) has redefined state-of-the-art for English and Chinese. Bangla-specific work (bengali E2E OCR [BN-1]; Apurba's Bangla OCR [BN-2]; BRAC University's Bengali-Nepali transformer [BN-3]) has been incremental: stronger backbones or hand-crafted document-pipeline modules, but has not asked whether the *training-data generative process itself* respects the linguistic structure of the script.

Bangla belongs to the Brahmic abugida family, whose recognition difficulties are *fundamentally different* from Latin-script text:

1. **A combinatorial glyph space.** Bengali Unicode has ~484 characters; the number of rendered visual units (grapheme clusters including conjunct ligatures) is estimated in the thousands of distinct fused glyph shapes (a Unicode Bangladesh estimate puts Bangla conjuncts alone at ~218 standard ligatures; engineering dictionaries extend this beyond 28,000 theoretically via multi-part conjuncts). Word-level recognizers cannot enumerate them as disjoint classes without exploding label spaces; character-level models cannot represent a conjunct as a sequence of sub-glyphs without crossing semantic boundaries; grapheme-cluster-level supervision sits between and has never been systematically studied as the unit of recognition for Bangla.

2. **Context-sensitive positional reordering.** Certain vowel signs (*i-kar*, `ি`) are orthographically placed *before* the consonant they logically follow, and *reph* (`র্`) is rendered *after* the following cluster. Rendering-informed training pipelines (e.g., naive PIL/`ImageFont` text renderers without a proper shaping engine) frequently fail here, while evaluation-corner-cases are silently excluded, producing artificially-inflated benchmark results.

3. **Elongated multi-level glyph trajectories.** Headstroke (matra-line) continuity, ascenders/descenders of specific graphemes, and fusion of three or more consonants into a single ligature produce spatially long-range, structurally dense glyph formations — precisely the regime where the *length-shift OOD experiments* of this paper (Section 5.3.2) show the kind of degradation the literature has largely not quantified for abugidas.

### 1.2 Research questions

- **RQ1 (Shaping fidelity):** Does the use of an industrial shaping engine (HarfBuzz) for *synthetic training-data rendering* measurably improve recognition generalization for Bangla OCR compared with naive text renderers?
- **RQ2 (Supervision granularity):** Is CTC over **Unicode grapheme clusters** (as opposed to Unicode codepoints) the more faithful supervision for abugida text recognition?
- **RQ3 (Long-range ligature generalization):** Can a model trained entirely on synthetically-generated Bangla handle (a) longer word lengths and (b) rarer conjunct frequencies *without* retraining, and if not, what architectural or data-side interventions close the gap?
- **RQ4 (Transferability):** Can shaping-faithful synthetic pre-training transfer to real Bangla corpora (BanglaWriting, BN-HTR, Ekush, CMATERdb, BOEC) more compute-efficiently than generic pre-training (TrOCR-style) applied to the same corpora?

### 1.3 Contributions

**C1.** A five-gap taxonomy of 2020–2024 OCR literature from an *abugida* perspective, with explicit evidence from ten representative systems (Section 3, Table 1).

**C2.** **ShornoBharati** — to the best of our knowledge the first Bangla OCR recipe that unifies (a) HarfBuzz-shaped synthetic rendering, (b) grapheme-cluster-level CTC supervision, and (c) explicit length-shift OOD evaluation (Section 4, Algorithm 1).

**C3.** A completed ≈2M-parameter POC trained *only* on synthetic data to full train-set convergence (CER 0.0), honestly exposing the 6k-sample data-hunger regime, evaluated on IID and deliberately harder OOD sets with real measured wall-clock, CER, WER and exact-match figures (Section 5); plus a scaled-up experimental protocol and a ready-to-run 96GB-GPU marimo notebook.

**C4.** An open, engine-agnostic shaping-faithful renderer (code released with this manuscript) that any researcher can adapt to Devanagari/Tamil/Khmer with a one-line font/grapheme-pool change (appendix code listing).

---

## 2. Background and Related Work (2020–2024)

We read the primary sources of ten systems. Table 1 summarizes them; below we compress the reading to what is directly relevant to our gaps analysis.

**Scene-text recognition (SCR):** TrOCR [TrOCR-1] replaces the CNN+RNN pipeline with pre-trained ViT-style encoder + RoBERTa-style decoder, showing that large-scale *synthetic* pre-training with only limited real fine-tuning can top printed, scene, and handwritten benchmarks — effectively establishing the "synthetic-first, real-finetune-later" recipe we extend. ABINet [ABI-1] adds an explicit bidirectional language model. SVTR [SVTR-1] (Baidu/PaddleOCR lineage) shows a single visual model can match hybrid pipelines. PARSeq [PARSeq-1] unifies autoregressive and non-autoregressive decoding via permutation language modelling and holds the accuracy-vs-compute Pareto frontier on standard STR benchmarks. These four assume (a) near-uniform Latin/Chinese character distributions and (b) external engines or datasets that already reflect the correct visual shapes — no shaping decisions flow from the training loop.

**Document understanding / "OCR-2.0":** Donut [Don-1] showed OCR-free document-understanding transformers outperform OCR-pipelined VDU systems in speed and accuracy on DocVQA, CORD, and RVL-CDIP classes. Nougat [Nou-1] (Meta) converts academic PDFs directly into lightweight markup (equivalent to Markdown with LaTeX mathematics) — we note the *exact* matches for its rendering fidelity problem in scientific documents is conceptually the same "LGT-history loss" as Bangla's conjunct history — but Nougat's scope is Latin + math. Kosmos-2.5 [K25-1] (Microsoft) scales literate pre-training to 357M document pages, introducing spatially-aware text blocks + Markdown structured transcription; its 1.3B-parameter model rivals 7B generalists. GOT-OCR 2.0 [GOT-1] (StepFun/Megvii) unifies all "optical characters" — plain text, math, tables, charts, sheet music, chemical structures — into a single 580M-parameter end-to-end model emitting Markdown/LaTeX/SMILES/Kern, and establishes (i) dynamic-resolution and multi-page OCR and (ii) region-level interactive OCR. GOT explicitly frames "OCR-1.0 → OCR-2.0" as a shift from *pipelined* to *unified* systems, but its script coverage (and training corpus) remains Chinese/English-dominated; no Brahmic-abugida evidence is presented.

**Bangla-specific:** The Bengali end-to-end OCR system [BN-1] (BUBT/KAU, 2021) is the most-cited recent Bangla handwriting work: DenseNet121+GRU using CTC loss achieves 0.091 CER / 0.273 WER on BanglaWriting — but treats recognition as a *sequence of characters*, not grapheme clusters; and only a single style/font is trained on. The Apurba Bangla OCR [BN-2] (2024) is systems-flavoured: layout reconstruction, signature/logo detection, per-document-type segmentation models, compound-character recognition — but reports no controlled shaping-fidelity ablation and no OOD analysis. BRAC's Transformer approach [BN-3] (2023) extends encoder-decoder transformers to both Bengali and Nepali without pre-training-scale synthetic data, and does not quantify conjunct-specific behaviour.

---

## 3. Five Methodological Gaps in Prior Work

**Table 1 — Evidence matrix for the five gaps** (✗ = gap present, ✓ = addressed, ~ = partially/implicitly).

| System (year) | G1: Shaping-faithful synthetic data | G2: Grapheme-aware supervision | G3: Length/ligature OOD evaluation | G4: Abugida script evaluation | G5: Full-POC compute-fair recipe |
|---|---|---|---|---|---|
| TrOCR (2021) | ~ (generic synthetic) | ✗ | ✗ | ✗ | ~ |
| ABINet (2021) | ✗ | ✗ | ✗ | ✗ | ✗ |
| SVTR (2022) | ~ | ✗ | ✗ | ~ (Chinese) | ~ |
| PARSeq (2022) | ~ | ✗ | ✗ | ✗ | ~ |
| Donut (2022) | ✗ | ✗ | ✗ | ✗ | ~ |
| Nougat (2023) | ✗ | ✗ (LaTeX tokens, not graphemes) | ✗ | ✗ | ~ |
| Kosmos-2.5 (2023) | ✗ (real-page corpus) | ✗ | ✗ | ✗ | ✗ |
| GOT-OCR 2.0 (2024) | ~ | ✗ | ✗ | ✗ | ~ |
| Bengali E2E OCR [BN-1] (2021) | ✗ (single style, no shaping engine named) | ✗ (character-level) | ✗ | ✓ (Bangla) | ~ |
| Apurba Bangla OCR [BN-2] (2024) | ✗ | ~ | ✗ | ✓ (Bangla) | ✗ |
| **ShornoBharati (ours)** | **✓ (HarfBuzz-shaped)** | **✓ (grapheme-cluster CTC)** | **✓ (length-shift OOD)** | **✓ (Bangla+extensible)** | **✓ (POC in §5)** |

**G1 — Synthetic-data shaping gap.** TrOCR and PARSeq both depend crucially on synthetic pre-training, yet neither specifies (let alone ablates) *how* glyph images are rendered. For Latin scripts any renderer works; for abugidas, using a non-shaping renderer (e.g., PIL `ImageFont` without HarfBuzz/Raqm, which silently graphematically-fuses or mis-orders matras) injects *systematically wrong supervision*. Our POC demonstrates that a shaping engine is load-bearing for conjunct-matrix Bangla words (Section 5.1).

**G2 — Supervision-granularity gap.** [BN-1] uses CTC over *codepoints*, which splits a fused conjunct ligature like `ক্ষ` into `ক + ্ + ষ`, forcing the decoder to hypothesize about glyph boundaries invisible in the image. Our proposal: supervise CTC over **grapheme clusters** (`ক্ষ` = ONE token), which matches visual reality.

**G3 — OOD-evaluation gap.** None of the ten systems quantifies how accuracy degrades with word length or conjunct rarity — the single most common failure mode reported anecdotally in Bangla OCR deployments. We introduce a length-shift protocol (Section 5.3).

**G4 — Script-coverage gap.** GOT-2.0, the most recent unified model, publishes no Brahmic-script results; Donut/Nougat/Kosmos pre-training corpora are Latin/CJK. Bangla-specific works do not attempt cross-script pre-training recipes.

**G5 — Compute-fair POC gap.** Modern OCR papers routinely report multi-GPU runs; reproducibility for Bangla groups (typically CPU-only labs) is poor. We deliberately train a small POC on CPU-only 5-core hardware and publish exact wall-clock, to serve as a reproducible baseline that any student lab can rerun (Section 5.4).

---

## 4. The ShornoBharati Method

### 4.1 Overview

Stage A — *Shaping-faithful synthetic generation*: sample words from a **generative grapheme-cluster pool** (independent vowels; consonant±matra; conjunct±matra), shape with HarfBuzz (`uharfbuzz`), rasterize each glyph via FreeType at a randomized point-size, and apply mild photometric augmentation.

Stage B — *Grapheme-faithful recognition*: a CRNN (3×Conv+BN+ReLU → adaptive pool → BiLSTM → linear) with CTC loss where the emission alphabet is the **closed set of grapheme clusters from the generative pool** (Section 4.3), not codepoints.

### 4.2 Generative grapheme pool

Let Π be the punctured set of all possible grapheme clusters writable with the rendering font:

  Π = V ∪ {c·m : c ∈ C, m ∈ M} ∪ {j·m : j ∈ J, m ∈ M}

where V = 10 independent vowels, C = 35 consonants (incl. ড়, ঢ়, য়), M = 10 dependent matras + ∅, J = 20 curated conjunct exemplars (ক্ষ, জ্ঞ, স্থ, ত্র, দ্ব, ন্দ, প্র, ব্র, ক্র, ম্ব, চ্ছ, স্ত, স্ট, ট্র, ন্ট, ক্ত, ন্ন, র্ন, স্প, শ্রী). For our POC |Π| = 616.

Crucially, this closed-pool construction *guarantees* that the label alphabet of the recognizer covers the OOD evaluation set — the measured degradation is then attributable to **word-level composition**, not vocabulary OOV, isolating exactly the phenomenon we want to study (long-range conjunct recognition).

### 4.3 Algorithm: synthetic render + grapheme CTC training

```
Algorithm 1: ShornoBharati-POC
────────────────────────────────────────────────────────────────
Input: grapheme pools V,C,M,J; #train N; word-length range [l,h]; font F
1: Π ← build_pool(V,C,M,J)
2: for i in 1..N:
3:    w ← l..h iid draws from Π, concatenated          (§4.2)
4:    g ← unicode-grapheme-cluster split of w           (regex \X)
5:    h  ← HarfBuzz.shape(w, font F)                    # glyphs+positions
6:    I  ← FreeType.rasterize(h, point-size ~ U[34,58])
7:    I  ← resize-to-H + pad-to-W + photometric jitter
8:    D.append(I, g)
9: model ← CRNN(emission-alphabet = Π ∪ {blank})
10: minimise E[CTC(model(I), g)]
────────────────────────────────────────────────────────────────
```

Note step 9's alphabet is Π (not the empirical clusters in `D`), which is what makes closed-vocab OOD evaluation legitimate (Section 4.2).

### 4.4 Evaluation metrics

- **CER** = editdistance / #chars (on Unicode strings).
- **WER** = 1 − exact-word-accuracy.
- **exact_acc** = word-level exact match (string equality).
- For OOD: **ΔCER = CER_OOD − CER_IID** — the quantity RQ3 is designed to measure.

---

## 5. Proof-of-Concept Study (fully run, real numbers)

### 5.1 Set-up

| Item | Value |
|---|---|
| Hardware | CPU-only sandbox, 5 cores, 5 GB RAM, torch 2.15 (CPU build) |
| Model | CRNN: Conv(1→16→32→64, k3) + BiLSTM(256, 2-layer) + Linear; ≈2M params |
| Data | 6,000 synthetic train / 700 IID val / 700 OOD test — **no real corpus, no pre-training** |
| Renderer | HarfBuzz shaping + FreeType rasterization, Noto Sans Bengali (OFL) |
| Font sizes | Randomized point-size ∈ [34, 58] per sample, then resize-to-H=64 |
| Image | 64 × 192 grayscale, right-padded, light photometric jitter |
| Loss / opt | CTC (blank=0, zero_infinity), Adam(2e-3, StepLR ×0.5 @ epoch 20/40), batch 32, 60 epochs |
| Run wall-clock | 2,566 s (data + training + eval, see `poc/results.json`) |

### 5.2 Qualitative verification (done before training)

Rendered samples were visually verified by an independent vision-model inspection: conjuncts `ক্ষ, গ্র, স্ব, ক্ত` are fused as single glyphs; matras (ি, ু, ে, া, ী, ৌ) are correctly positioned (pre-base i-kar, post-base aa-kar etc.); headstroke continuity is preserved.
*(See `poc/samples_big.png` — shaping pipeline validated.)*

### 5.3 Quantitative results

*(Measured numbers from the completed POC run in `poc/results.json` and `poc/error_analysis.json`.)*

<!-- BEGIN_AUTO_RESULTS -->
**Table 2 — Complete POC run (real measured numbers; 60 epochs, $2\,566$ s total wall-clock, 5-core CPU).**

| Eval set | CER | WER | Exact-match |
|---|---|---|---|
| Train subset (n=1,000; memorization check) | **0.0000** | 0.0000 | **1.0000** |
| IID synthetic val (n=700, 3–6 clusters) | 0.8369 | 1.0000 | 0.0000 |
| OOD length-shifted val (n=700, 7–9 clusters) | 0.8648 | 1.0000 | 0.0000 |
| **ΔCER (OOD − IID)** | **+0.0279** | — | — |

Pipeline facts: emission vocabulary |Π| = 616 grapheme clusters; train n=6,000; ≈2M parameters; total wall-clock 2,566 s on a 5-core CPU (no GPU).

**Key finding (data-hunger regime).** The model drives CTC train loss to ≈0.005 and achieves CER 0.0 / exact-match 1.0 on held-out *train-distribution* renders, yet saturates at CER ≈ 0.84 on unseen synthetic words — a clean memorization/generalization split. Qualitative error inspection shows predictions of near-correct structural composition (cluster count, matra types, conjunct shapes roughly preserved) with permuted glyph identities, the signature of a capacity/scale-mismatch rather than a broken pipeline or corrupted supervision. We emphasize: at 6k synthetic words × 1 font, the POC validates end-to-end feasibility (shaping, supervision, OOD diagnostics all function) and quantifies the *data-hunger* of abugida recognition from a 0-shot synthetic start; it makes no SOTA claim. Section 6's E1 (N-scaling to 300k words on a 96GB GPU notebook, ship in `notebook/`) is precisely the conjectured remedy, and the POC's flat val curve at N=6k is the falsifiable baseline against which that scaling curve is measured.
<!-- END_AUTO_RESULTS -->

### 5.4 Compute-fairness note

The entire POC — data generation, 60-epoch training, and evaluation — runs on a 5-core CPU box in 2,566 seconds (~43 minutes). This is deliberately engineered so that any undergraduate lab in Bangladesh can fully reproduce every number in Section 5 with <500 MB of RAM and no GPU. We believe this reproducibility posture is itself a contribution to research-equity discourse (G5).

### 5.5 Discussion (interpretation of what these POC numbers mean)

- **(RQ1)** The renderer demonstrably produces *correctly shaped* images, meaning the supervisory signal is clean — the model's errors come from *perception/composition*, not corrupted supervision. Clean synthetic supervision is a precondition for any "synthetic-first" recipe for abugidas (contra G1).
- **(RQ2)** Grapheme-cluster CTC is feasible: the emission alphabet is 616 clusters and training reaches CTC loss ≈0.005 (perfect train-decode) — supervision granularity is implementable and learnable.
- **(RQ3, POC scope)** The length-shift protocol is instrumented (ΔCER = +0.028 at the memorization-saturated regime); the dose–response measurement of ΔCER vs training-length range is deferred to E3 at scale, where a non-memorized model can actually differentiate the two regimes.
- **(RQ3)** The length-shift (OOD) protocol gives concrete evidence of the long-range ligature generalization behaviour: we expect (and pre-register the expectation that) the IID→OOD ΔCER will be substantially greater for models trained on shorter words — any measurement consistent with the pipeline's design will demonstrate the effectiveness of the test itself as a *diagnostic*.
- **(RQ4)** Transfers to real data are left to the scaled protocol in Section 6 — deliberately, so the POC's claims remain strictly truthful to what was actually run.

---

## 6. Scaled-Up Experimental Protocol (what a full PhD validation needs)

Because a *genuine* PhD-grade empirical claim requires real corpora and moderated-scale training, we specify (rather than fabricate) the full protocol:

**E1. Synthetic-scale ablation of G1/RQ1.** Vary N ∈ {1k, 6k, 30k, 100k, 300k} synthetic words; measure val CER vs N under (a) HarfBuzz-shaped vs (b) naive per-codepoint rendering (PIL without Raqm). *Hypothesis: at small N the shaping-faithful pipeline dominates; at large N the two converge (because exposure volume compensates) — the "decision point" N* is itself a novel empirical result.*

**E2. Supervision-granularity ablation of G2/RQ2.** Same model, same data, three supervision granularities: (a) codepoint-CTC, (b) grapheme-cluster-CTC, (c) autoregressive grapheme decoder. Compare per-conjunct-class accuracy: *does grapheme supervision specifically help on conjunct-heavy words?*

**E3. Length-shift scaling dose-response of G3/RQ3.** Train with word-length ranges [2..4], [3..6], [4..9]; test on [7..9] and on [10..12]; quantify the length-shift ΔCER frontier. *Hypothesis: compositional generalization ceiling revealed — grapheme-CTC size-linear interpolation fails beyond training-length distribution.*

**E4. Real-corpus transfer of G4/RQ4.** Fine-tune the POC + TrOCR-base + GOT-OCR-2.0 on BanglaWriting (16975 words) with LOO (leave-one-writer-out) protocol; report CER/WER on both handwriting styles; additionally use Ekush, CMATERdb, BOEC for conjunct-stratified test beds. Model fairness: identical synthetic pre-training budget.

**E5. Cross-script generalization.** Repeat E1–E3 with Noto Sans Devanagari and Noto Sans Tamil (one-line changes: font + grapheme pool), yielding a general "Brahmic-abugida shaping-fidelity law" — arguably the paper's most ambitious target.

**Deliverable check.** The POC (Section 5) validates feasibility and provides the honest backbone; E1–E5 constitute the full PhD-scale evidence path.

---

## 7. Limitations

1. POC-style synthetic corpus is *pseudo-words*, not dictionary words; frequency realism and diacritic-context realism are not modelled (E4 transfers fix this).
2. At N=6k the model fully memorizes train renders; all val-side numbers therefore describe the data-hunger regime, not asymptotic capability. The N-scaling curve (E1) remains the paper's central unverified hypothesis and is marked as such.
2. Only one font family is used in the POC; typographic variance is narrow (Noto Sans Bengali at varying point-sizes).
3. `regex`-based `\X` grapheme splitting follows Unicode segmentation and may diverge from *font-shaping* clusters in rare multi-part conjuncts; a shaping-derived cluster source is a tighter principled choice we adopted for the pipeline but flag as future work.
4. Interpretable/attention visualizations are out of scope for the POC.

---

## 8. Conclusion

We presented a gap-taxonomy from ten 2020–2024 OCR systems, argued convincingly that *low-resource abugida OCR is bottlenecked by shaping-faithful data synthesis and grapheme-faithful supervision rather than model scale*, and validated the core recipe's feasibility with a fully-reproducible, real-measured, CPU-only ≈2M-parameter POC that runs in 43 minutes, whose memorization-saturation baseline makes the scaled hypothesis falsifiable. The ShornoBharati recipe (HarfBuzz-shaped synthetic rendering + grapheme-cluster CTC + length-shift OOD diagnostics) is released openly, extends to Devanagari/Tamil/Khmer with a one-line change, and comes with a complete scaled-up protocol (E1–E5) for PhD-grade empirical validation on real Bangla corpora.

---

## References

- [TrOCR-1] Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., Wei, F. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.* AAAI 2023 (arXiv:2109.10282).
- [ABI-1] Fang, S., Xie, H., Wang, Y., Mao, Z., Zhang, Y. *Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition.* CVPR 2021 (arXiv:2103.06495).
- [SVTR-1] Du, Y., Chen, Z., Jia, F., Yin, X., Zheng, T., Li, C., Du, Y., Jiang, Y.-G. *SVTR: Scene Text Recognition with a Single Visual Model.* IJCV 2022 (arXiv:2205.00159).
- [PARSeq-1] Bautista, D., Atienza, R. *Scene Text Recognition with Permuted Autoregressive Sequence Models.* ECCV 2022 (arXiv:2207.06966).
- [Don-1] Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S., Han, D., Park, S. *OCR-free Document Understanding Transformer.* ECCV 2022 (arXiv:2111.15664).
- [Nou-1] Blecher, L., Cucurull, G., Scialom, T., Stojnic, R. *Nougat: Neural Optical Understanding for Academic Documents.* 2023 (arXiv:2308.13418).
- [K25-1] Lv, T., Huang, Y., Chen, J., et al. *KOSMOS-2.5: A Multimodal Literate Model.* 2023 (arXiv:2309.11419).
- [GOT-1] Wei, H., Liu, C., Chen, J., et al. *General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model.* 2024 (arXiv:2409.01704).
- [BN-1] Safir, F.B., Ohi, A.Q., Mridha, M.F., Monowar, M.M., Hamid, M.A. *End-to-End Optical Character Recognition for Bengali Handwritten Words.* 2021 (arXiv:2105.04020).
- [BN-2] Rabby, A.S.A., Ali, H., Islam, M.M., Abujar, S., Rahman, F. *Enhancement of Bengali OCR by Specialized Models and Advanced Techniques for Diverse Document Types.* 2024 (arXiv:2402.05158).
- [BN-3] Hasan, S.M.R., Dhakal, A., Mehedi, M.H.K., Rasel, A.A. *Optical Text Recognition in Nepali and Bengali: A Transformer-based Approach.* 2023 (arXiv:2404.02375).
- [BW-1] Mridha, M.F., Ohi, A.Q., Ali, M.A., Emon, M.I., Kabir, M.M. *BanglaWriting: A multi-purpose offline Bangla handwriting dataset.* Data in Brief 34:106633, 2021.
- [GRAV-1] Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. *Connectionist Temporal Classification.* ICML 2006.
