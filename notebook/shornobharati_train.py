import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import random
    import time
    from pathlib import Path

    import numpy as np
    import regex as re
    import torch
    import torch.nn as nn
    import uharfbuzz as hb
    import freetype
    from PIL import Image as PILImage

    return (
        PILImage,
        Path,
        json,
        nn,
        np,
        random,
        re,
        time,
        torch,
        freetype,
        hb,
    )


@app.cell
def _(torch):
    import marimo as mo

    mo.md(
        f"""
        # ShornoBharati — GPU Training Notebook (RTX Pro 6000 Blackwell 96GB)

        **Recipe:** HarfBuzz-shaped synthetic Bangla rendering → grapheme-cluster CTC → length-shift OOD diagnostics.

        | | |
        |---|---|
        | Device detected | `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA)"}` |
        | VRAM | `{round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1) if torch.cuda.is_available() else 0} GB` |

        ⚠️ **Blackwell (sm_120) note:** use a CUDA 12.8 wheel:
        `pip install torch --index-url https://download.pytorch.org/whl/cu128`
        (torch ≥ 2.7). Also: `pip install uharfbuzz freetype-py regex marimo`.

        **Protocol coverage:** E1 (shaping ablation), E2 (supervision granularity), E3 (length-shift dose–response). E4/E5 need real corpora/fonts — see the notes cell at the bottom.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md("## 1. Controls\n\nConfigure, then click **RUN PIPELINE**. Changing any control re-triggers nothing until you click run again.")
    return


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        options=["custom", "E1: shaping ablation", "E2: supervision granularity", "E3: length shift", "quick-smoke"],
        value="quick-smoke",
        label="experiment preset",
    )
    preset
    return (preset,)


@app.cell
def _(mo, preset):

    defaults = {
        "quick-smoke": dict(n_train=6000, n_val=700, n_ood=700, batch=256, epochs=10, lr=1e-3, hidden=256, lstm_layers=1, wlen_min=3, wlen_max=6, ood_min=7, ood_max=9, shaping=True, vocab_mode="grapheme"),
        "E1: shaping ablation": dict(n_train=200000, n_val=2000, n_ood=2000, batch=1024, epochs=10, lr=1e-3, hidden=256, lstm_layers=2, wlen_min=3, wlen_max=6, ood_min=7, ood_max=9, shaping=True, vocab_mode="grapheme"),
        "E2: supervision granularity": dict(n_train=200000, n_val=2000, n_ood=2000, batch=1024, epochs=10, lr=1e-3, hidden=256, lstm_layers=2, wlen_min=3, wlen_max=6, ood_min=7, ood_max=9, shaping=True, vocab_mode="codepoint"),
        "E3: length shift": dict(n_train=200000, n_val=2000, n_ood=3000, batch=1024, epochs=10, lr=1e-3, hidden=256, lstm_layers=2, wlen_min=4, wlen_max=9, ood_min=10, ood_max=12, shaping=True, vocab_mode="grapheme"),
        "custom": dict(n_train=60000, n_val=1000, n_ood=1000, batch=512, epochs=15, lr=1e-3, hidden=256, lstm_layers=2, wlen_min=3, wlen_max=6, ood_min=7, ood_max=9, shaping=True, vocab_mode="grapheme"),
    }
    d = defaults[preset.value]

    n_train = mo.ui.slider(1000, 500000, value=d["n_train"], step=1000, label="n_train (synthetic words)")
    n_val = mo.ui.slider(100, 20000, value=d["n_val"], step=100, label="n_val (IID)")
    n_ood = mo.ui.slider(100, 20000, value=d["n_ood"], step=100, label="n_ood (length-shift)")
    batch = mo.ui.slider(32, 4096, value=d["batch"], step=32, label="batch size (96GB → 1024+ ok)")
    epochs = mo.ui.slider(1, 200, value=d["epochs"], label="epochs")
    lr = mo.ui.number(start=1e-5, stop=1e-1, value=d["lr"], step=1e-4, label="learning rate")
    hidden = mo.ui.dropdown([128, 256, 512], value=d["hidden"], label="BiLSTM hidden")
    lstm_layers = mo.ui.slider(1, 4, value=d["lstm_layers"], label="LSTM layers")
    wlen_min = mo.ui.slider(2, 12, value=d["wlen_min"], label="train min clusters/word")
    wlen_max = mo.ui.slider(2, 12, value=d["wlen_max"], label="train max clusters/word")
    ood_min = mo.ui.slider(2, 12, value=d["ood_min"], label="OOD min clusters/word")
    ood_max = mo.ui.slider(2, 12, value=d["ood_max"], label="OOD max clusters/word")
    shaping = mo.ui.switch(value=d["shaping"], label="HarfBuzz shaping (OFF = naive renderer = G1 ablation)")
    vocab_mode = mo.ui.dropdown(["grapheme", "codepoint"], value=d["vocab_mode"], label="CTC supervision granularity (E2)")
    run_btn = mo.ui.run_button(label="▶ RUN PIPELINE", kind="success")

    mo.hstack(
        [mo.vstack([n_train, n_val, n_ood, batch, epochs, lr]), mo.vstack([hidden, lstm_layers, wlen_min, wlen_max, ood_min, ood_max]), mo.vstack([preset, shaping, vocab_mode, run_btn])],
        justify="start",
    )
    return (
        batch,
        epochs,
        hidden,
        lr,
        lstm_layers,
        n_ood,
        n_train,
        n_val,
        ood_max,
        ood_min,
        run_btn,
        shaping,
        vocab_mode,
        wlen_max,
        wlen_min,
    )


@app.cell
def _(freetype, hb):
    import io as _io

    _FONT_BYTES = open("fonts/NotoSansBengali.ttf", "rb").read()
    _HB_BLOB = hb.Blob(_FONT_BYTES)
    _HB_FACE = hb.Face(_HB_BLOB)
    _HB_FONT = hb.Font(_HB_FACE)

    _INDEPENDENT_VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ"]
    _CONSONANTS = ["ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ", "ড়", "ঢ়", "য়"]
    _MATRAS = ["", "", "", "া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ"]
    _CONJUNCTS = ["ক্ষ", "জ্ঞ", "স্থ", "ত্র", "দ্ব", "ন্দ", "প্র", "ব্র", "ক্র", "ম্ব", "চ্ছ", "স্ত", "স্ট", "ট্র", "ন্ট", "ক্ত", "ন্ন", "র্ন", "স্প", "শ্রী"]

    def grapheme_split(s):
        return re.findall(r"\X", s)

    def make_word(rng, min_cl=3, max_cl=6):
        n = rng.randint(min_cl, max_cl)
        parts = []
        for i in range(n):
            r = rng.random()
            if i == 0 and r < 0.15:
                parts.append(rng.choice(_INDEPENDENT_VOWELS))
            elif r < 0.10 and i > 0:
                parts.append(rng.choice(_CONJUNCTS) + rng.choice(_MATRAS))
            elif r < 0.16 and i > 0:
                parts.append(rng.choice(_CONJUNCTS))
            else:
                parts.append(rng.choice(_CONSONANTS) + rng.choice(_MATRAS))
        return "".join(parts)

    def build_cluster_pool():
        pool = set(_INDEPENDENT_VOWELS)
        for m in _MATRAS:
            pool |= {c + m for c in _CONSONANTS}
            pool |= {cj + m for cj in _CONJUNCTS}
        return sorted(pool)

    class _BaseRenderer:
        def __init__(self, canvas_h=64):
            self.canvas_h = canvas_h
            self._ft = freetype.Face(_io.BytesIO(_FONT_BYTES))

        def _set_px(self, px):
            self._ft.set_pixel_sizes(0, px)

        def _draw(self, glyphs, px, pad=8):
            self._set_px(px)
            scale = px / _HB_FACE.upem
            adv = sum(p.x_advance for _, p in glyphs) * scale
            w = int(adv) + 2 * pad + 2
            asc = int(self._ft.size.ascender / 64.0)
            desc = int(-self._ft.size.descender / 64.0)
            h = asc + desc + 2 * pad
            canvas = np.full((h, w), 255, dtype=np.uint8)
            pen_x = float(pad)
            for gid, pos in glyphs:
                self._ft.load_glyph(gid, freetype.FT_LOAD_RENDER)
                g = self._ft.glyph
                bmp = g.bitmap
                if bmp.width == 0 or bmp.rows == 0:
                    pen_x += pos.x_advance * scale
                    continue
                arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)
                gx = int(round(pen_x + pos.x_offset * scale)) + g.bitmap_left
                gy = asc + pad - g.bitmap_top - int(round(pos.y_offset * scale))
                x0, y0 = max(gx, 0), max(gy, 0)
                x1, y1 = min(gx + arr.shape[1], w), min(gy + arr.shape[0], h)
                if x1 > x0 and y1 > y0:
                    canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], arr[y0 - gy:y1 - gy, x0 - gx:x1 - gx])
                pen_x += pos.x_advance * scale
            return canvas

    class ShapingRenderer(_BaseRenderer):
        def render(self, text, px=40):
            buf = hb.Buffer()
            buf.add_str(text)
            buf.guess_segment_properties()
            hb.shape(_HB_FONT, buf, {"kern": True, "liga": True})
            return self._draw(list(zip(buf.glyph_infos, buf.glyph_positions)), px)

    class NaiveRenderer(_BaseRenderer):
        def render(self, text, px=40):
            glyphs = []
            scale = px / hb.Face(hb.Blob(_FONT_BYTES)).upem
            for ch in text:
                self._ft.load_char(ch, freetype.FT_LOAD_RENDER)
                gid = self._ft.glyph.index
                adv_px = self._ft.glyph.advance.x / 64.0
                glyphs.append((gid, type("P", (), {"x_advance": adv_px / scale, "x_offset": 0, "y_offset": 0})()))
            return self._draw(glyphs, px)

    return (
        NaiveRenderer,
        ShapingRenderer,
        build_cluster_pool,
        grapheme_split,
        make_word,
    )


@app.cell
def _(HEIGHT, WIDTH):
    def augment(img, rng):
        img = img.astype(np.float32)
        img = img * rng.uniform(0.85, 1.15) + rng.uniform(-20, 20)
        if rng.random() < 0.5:
            img += rng.uniform(0, 12) * rng.choice([1, -1]) * np.random.rand(*img.shape)
        return np.clip(img, 0, 255)

    def render_resized(renderer, word, rng):
        px = rng.randint(34, 58)
        img = renderer.render(word, px=px)
        scale = HEIGHT / img.shape[0]
        nw = max(8, min(WIDTH, int(img.shape[1] * scale)))
        im = np.array(PILImage.fromarray(img.astype(np.uint8)).resize((nw, HEIGHT)))
        canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
        x0 = rng.randint(0, max(0, WIDTH - nw))
        canvas[:, x0:x0 + nw] = im
        return canvas

    def build_split(renderer, n, mincl, maxcl, seed):
        rng = random.Random(seed)
        xs, ys = [], []
        for _ in range(n):
            w = make_word(rng, min_cl=mincl, max_cl=maxcl)
            xs.append(augment(render_resized(renderer, w, rng), rng))
            ys.append(grapheme_split(w))
        return xs, ys

    return (build_split,)


@app.cell
def _(nn):
    HEIGHT, WIDTH = 64, 192

    def make_crnn(vocab_size, hidden=256, lstm_layers=2):
        return _CRNN(vocab_size, hidden, lstm_layers)

    class _CRNN(nn.Module):
        def __init__(self, vocab_size, hidden, lstm_layers):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, None)),
            )
            self.rnn = nn.LSTM(64, hidden, num_layers=lstm_layers, bidirectional=True)
            self.out = nn.Linear(2 * hidden, vocab_size)
            self.hidden, self.lstm_layers = hidden, lstm_layers

        def forward(self, x):
            f = self.cnn(x).squeeze(2).permute(2, 0, 1)
            r, _ = self.rnn(f)
            return self.out(r)

    return (make_crnn,)


@app.cell
def _(HEIGHT, WIDTH, nn, np, torch):
    def ctc_decode(logits, blank=0):
        pred = logits.argmax(-1).permute(1, 0)
        out = []
        for p in pred.tolist():
            s, last = [], blank
            for v in p:
                if v != last and v != blank:
                    s.append(v)
                last = v
            out.append(s)
        return out

    class Vocab:
        def __init__(self, clusters):
            uniq = sorted(clusters)
            self.itos = ["<blank>"] + uniq
            self.stoi = {c: i for i, c in enumerate(self.itos)}
            self.size = len(self.itos)

        def enc(self, clusters):
            return [self.stoi[c] for c in clusters]

        def dec_ids(self, ids):
            return "".join(self.itos[v] for v in ids)

    def evaluate(model, vocab, xs, ys, device):
        import editdistance

        model.eval()
        tot_cer, tot_wer, tot_exact, n = 0.0, 0.0, 0.0, len(xs)
        with torch.no_grad():
            for i in range(0, n, 256):
                xb = torch.tensor(np.stack(xs[i:i + 256]) / 255.0).unsqueeze(1).to(device)
                logits = model(xb)
                dec = ctc_decode(logits)
                for d, gold in zip(dec, ys[i:i + 256]):
                    p = vocab.dec_ids(d)
                    g = "".join(gold)
                    tot_cer += editdistance.eval(p, g) / max(1, len(g))
                    tot_wer += 1.0 if p != g else 0.0
                    tot_exact += 1.0 if p == g else 0.0
        return {"CER": tot_cer / n, "WER": tot_wer / n, "exact_acc": tot_exact / n}

    return (Vocab, evaluate)


@app.cell
def _(HEIGHT, mo, run_btn, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.md(
        "Click the **▶ RUN PIPELINE** button above to launch the configured experiment."
        if not run_btn.value
        else f"### Running on `{device}` …"
    )
    return (device,)


@app.cell
def _(
    HEIGHT,
    WIDTH,
    Vocab,
    batch,
    build_cluster_pool,
    build_split,
    device,
    epochs,
    evaluate,
    hidden,
    json,
    lr,
    lstm_layers,
    make_crnn,
    mo,
    n_ood,
    n_train,
    n_val,
    nn,
    np,
    ood_max,
    ood_min,
    random,
    run_btn,
    shaping,
    time,
    torch,
    vocab_mode,
    wlen_max,
    wlen_min,
    NaiveRenderer,
    ShapingRenderer,
):
    output = mo.md("Idle — click **▶ RUN PIPELINE**.")
    if run_btn.value:
        t0 = time.time()
        renderer = ShapingRenderer(HEIGHT) if shaping.value else NaiveRenderer(HEIGHT)

        mo.md(f"**[1/4] generating {n_train.value + n_val.value + n_ood.value} images on `{device}`…**")
        xtr, ytr = build_split(renderer, n_train.value, wlen_min.value, wlen_max.value, 11)
        xva, yva = build_split(renderer, n_val.value, wlen_min.value, wlen_max.value, 22)
        xoo, yoo = build_split(renderer, n_ood.value, ood_min.value, ood_max.value, 33)

        if vocab_mode.value == "grapheme":
            clusters = build_cluster_pool()
        else:
            clusters = sorted({ch for w in (ytr + yva + yoo) for c in w for ch in c})
        vocab = Vocab(clusters)

        model = make_crnn(vocab.size, hidden.value, lstm_layers.value).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr.value)
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        def enc_seqs(ys_b):
            if vocab_mode.value == "grapheme":
                return [vocab.enc(y) for y in ys_b]
            return [[vocab.stoi[ch] for ch in "".join(y)] for y in ys_b]

        order = list(range(len(xtr)))
        history = []
        for ep in range(epochs.value):
            random.Random(ep).shuffle(order)
            model.train()
            tot, nb = 0.0, 0
            for i in range(0, len(order), batch.value):
                idx = order[i:i + batch.value]
                ys_b = [ytr[j] for j in idx]
                xb = torch.tensor(np.stack([xtr[j] for j in idx]) / 255.0).unsqueeze(1).to(device)
                logits = model(xb)
                T, B, _ = logits.shape
                input_lens = torch.full((B,), T, dtype=torch.long, device=device)
                seqs = enc_seqs(ys_b)
                tgt = torch.tensor([v for s in seqs for v in s], dtype=torch.long, device=device)
                tgt_lens = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
                if tgt_lens.max().item() > T:
                    continue
                loss = ctc(logits.log_softmax(2), tgt, input_lens, tgt_lens)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            m = evaluate(model, vocab, xva, yva, device)
            history.append({"epoch": ep + 1, "loss": tot / max(1, nb), "val_CER": m["CER"]})
            print(f"epoch {ep+1}/{epochs.value}: loss {tot/max(1,nb):.3f} val CER {m['CER']:.4f}")

        iid = evaluate(model, vocab, xva, yva, device)
        ood = evaluate(model, vocab, xoo, yoo, device)
        res = {
            "device": str(device),
            "shaping": shaping.value,
            "vocab_mode": vocab_mode.value,
            "n_train": n_train.value, "n_val": n_val.value, "n_ood": n_ood.value,
            "wlen": [wlen_min.value, wlen_max.value], "ood_len": [ood_min.value, ood_max.value],
            "vocab_size": vocab.size,
            "iid_val": iid, "ood_long": ood,
            "d_CER": ood["CER"] - iid["CER"],
            "history": history,
            "wall_time_s": round(time.time() - t0, 1),
        }
        with open("results.json", "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        torch.save(model.state_dict(), "crnn_shornobharati.pt")

        output = mo.vstack(
            [
                mo.md(
                    f"""
                    ## 📊 Results ({res['wall_time_s']}s total)

                    | Eval | CER | WER | Exact | |
                    |---|---|---|---|---|
                    | IID val | **{iid['CER']:.4f}** | {iid['WER']:.4f} | {iid['exact_acc']:.4f} | n={n_val.value} |
                    | OOD len-shift | **{ood['CER']:.4f}** | {ood['WER']:.4f} | {ood['exact_acc']:.4f} | n={n_ood.value} |
                    | **ΔCER (OOD−IID)** | **{ood['CER'] - iid['CER']:+.4f}** | | | |

                    Emission alphabet |Π| = {vocab.size} ({vocab_mode.value}); renderer = {'HarfBuzz-shaped' if shaping.value else 'naive (C1 ablation)'}; saved to `results.json` + `crnn_shornobharati.pt`.
                    """
                ),
            ]
        )
    output
    return (output,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Notes — running the full protocol (E4/E5)

        **E4 (real-corpus transfer):** download [BanglaWriting](https://github.com/QuwsarOhi/bengali_word_ocr) / Ekush / BOEC, replace `build_split`'s synthetic images with a `Dataset` reading real word crops, keep the same `Vocab`/CRNN/CTC plumbing. Fine-tune from the synthetic checkpoint (load `crnn_shornobharati.pt`) vs from scratch — that comparison *is* the RQ4 result.

        **E5 (cross-script):** drop any Noto font in `fonts/` (Devanagari, Tamil, Khmer), swap the grapheme pools at the top of the renderer cell, and rerun. One-line change, as claimed in the manuscript.

        **E1/E2/E3 (ablations):** use the presets — E1 sets `shaping=OFF` (naive codepoint rendering) vs ON; E2 flips `vocab_mode` grapheme↔codepoint; E3 widens train/OOD word-length ranges. Record `results.json` after each run (it overwrites — rename per experiment, e.g. `results_E1_on.json`).

        **Blackwell tip:** with 96GB you can raise batch to 2048–4096 and epochs ×10 for the 300k-word regime. Keep `torch.set_float32_matmul_precision("high")` if you add AMP; on CPU it is pointless so it is not included by default.
        """
    )
    return


if __name__ == "__main__":
    app.run()
