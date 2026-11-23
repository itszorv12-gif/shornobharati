# ShornoBharati — Project Bundle

PhD-level research project: Bangla OCR via shaping-faithful synthetic data + grapheme-cluster CTC.

## Contents

```
paper/    ShornoBharati_manuscript.md   ← full paper draft (gap analysis §3, method §4, POC §5, protocol §6)
poc/      render.py, train.py, samples_big.png, results.json (sandbox CPU POC)
notebook/ shornobharati_train.py        ← marimo notebook for RTX Pro 6000 Blackwell 96GB
papers/   10 downloaded arXiv PDFs + extracted text (txt/)
```

## Notebook quickstart

### Option A — molab (marimo's hosted GPU cloud; zero setup)
1. Open [molab.marimo.ai](https://molab.marimo.ai) → New notebook → paste `notebook/shornobharati_train.py` (Upload → local .py also works).
2. Start a **GPU session** (e.g. RTX Pro 6000 Blackwell 96GB).
3. Run top-to-bottom: the first cells install missing packages and auto-download Noto Sans Bengali (OFL).
4. Pick a preset → **▶ RUN PIPELINE**.

### Option B — your machine (Blackwell GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell sm_120 needs cu128 wheel
pip install marimo uharfbuzz freetype-py regex editdistance
cp notebook/shornobharati_train.py .
marimo edit shornobharati_train.py   # font auto-downloads on first run
```

Pick a preset (`quick-smoke`, `E1: shaping ablation`, `E2: supervision granularity`, `E3: length shift`) → **▶ RUN PIPELINE**. Results land in `results.json` + `crnn_shornobharati.pt`.

## Workflow

1. Run `quick-smoke` to sanity-check the pipeline end to end (~2 min).
2. Run E1 twice (`shaping` ON / OFF) with the same seed → Table for G1 (shaping gap).
3. Run E2 (`vocab_mode` grapheme vs codepoint) → Table for G2.
4. Run E3 (length ranges) → ΔCER dose–response for G3 (rename `results.json` per run).
5. E4/E5 need real corpora (BanglaWriting/Ekush/BOEC) and other Noto fonts — instructions in the notebook's last cell.
