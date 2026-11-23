"""POC: grapheme-aware CRNN trained purely on HarfBuzz-rendered synthetic Bangla."""
import json
import random
import time

import editdistance
import numpy as np
from PIL import Image as PILImage
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from render import (
    ShapingRenderer,
    _CONJUNCTS,
    _CONSONANTS,
    _INDEPENDENT_VOWELS,
    _MATRAS,
    grapheme_split,
    make_word,
)

HEIGHT = 64
WIDTH = 192
N_TRAIN, N_VAL, N_OOD = 6000, 700, 700
EPOCHS, BATCH = 60, 32
SEED = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(5)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.astype(np.float32)
    a = rng.uniform(0.85, 1.15)
    b = rng.uniform(-20, 20)
    img = img * a + b
    if rng.random() < 0.5:
        img += rng.uniform(0, 12) * rng.choice([1, -1]) * np.random.rand(*img.shape)
    return np.clip(img, 0, 255)


def render_resized(r: ShapingRenderer, word: str, rng: random.Random) -> np.ndarray:
    px = rng.randint(34, 58)
    img = r.render(word, px=px)
    scale = HEIGHT / img.shape[0]
    nw = max(8, min(WIDTH, int(img.shape[1] * scale)))
    im = np.array(PILImage.fromarray(img.astype(np.uint8)).resize((nw, HEIGHT)))
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
    x0 = rng.randint(0, max(0, WIDTH - nw))
    canvas[:, x0:x0 + nw] = im
    return canvas


def build_cluster_pool():
    pool = set(_INDEPENDENT_VOWELS)
    for m in _MATRAS:
        pool |= {c + m for c in _CONSONANTS}
        pool |= {cj + m for cj in _CONJUNCTS}
    return sorted(pool)


def build_split(n, mincl, maxcl, seed):
    r = ShapingRenderer(HEIGHT)
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        w = make_word(rng, min_cl=mincl, max_cl=maxcl)
        xs.append(augment(render_resized(r, w, rng), rng))
        ys.append(grapheme_split(w))
    return xs, ys


class Vocab:
    def __init__(self, clusters=None):
        seqs = clusters if clusters is not None else build_cluster_pool()
        uniq = sorted(seqs)
        self.itos = ["<blank>"] + uniq
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.size = len(self.itos)

    def enc(self, clusters):
        return [self.stoi[c] for c in clusters]


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(64, 256, num_layers=2, bidirectional=True)
        self.out = nn.Linear(512, vocab_size)

    def forward(self, x):
        f = self.cnn(x)              # (B, 64, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # (T, B, 64)
        r, _ = self.rnn(f)
        return self.out(r)           # (T, B, V)


def ctc_decode(logits, blank=0):
    pred = logits.argmax(-1).permute(1, 0).tolist()
    out = []
    for p in pred:
        s, last = [], blank
        for v in p:
            if v != last and v != blank:
                s.append(v)
            last = v
        out.append(s)
    return out


def evaluate(model, vocab, xs, ys, pad_token):
    model.eval()
    tot_cer, tot_wer, tot_exact, n = 0.0, 0.0, 0, len(xs)
    with torch.no_grad():
        for i in range(0, n, 64):
            xb = torch.tensor(np.stack(xs[i:i + 64]) / 255.0).unsqueeze(1)
            logits = model(xb)
            lens = logits.shape[0]
            targets = [vocab.enc(y) for y in ys[i:i + 64]]
            dec = ctc_decode(logits)
            for d, t in zip(dec, ys[i:i + 64]):
                pred = "".join(vocab.itos[v] for v in d)
                gold = "".join(t)
                tot_cer += editdistance.eval(pred, gold) / max(1, len(gold))
                tot_wer += 1.0 if pred != gold else 0.0
                tot_exact += 1.0 if pred == gold else 0.0
    return {"CER": tot_cer / n, "WER": tot_wer / n, "exact_acc": tot_exact / n}


def main():
    t0 = time.time()
    print("generating data...", flush=True)
    xtr, ytr = build_split(N_TRAIN, 3, 6, 11)
    xva, yva = build_split(N_VAL, 3, 6, 22)
    xoo, yoo = build_split(N_OOD, 7, 9, 33)
    vocab = Vocab()
    print(f"vocab size {vocab.size}, data gen {time.time()-t0:.1f}s", flush=True)

    model = CRNN(vocab.size)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    order = list(range(N_TRAIN))
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        model.train()
        tot, nb = 0.0, 0
        tt = time.time()
        for i in range(0, N_TRAIN, BATCH):
            idx = order[i:i + BATCH]
            ys_b = [ytr[j] for j in idx]
            xb = torch.tensor(np.stack([xtr[j] for j in idx]) / 255.0).unsqueeze(1)
            logits = model(xb)
            T, B, _ = logits.shape
            input_lens = torch.full((B,), T, dtype=torch.long)
            tgt_seq = [v for y in ys_b for v in vocab.enc(y)]
            tgt = torch.tensor(tgt_seq, dtype=torch.long)
            tgt_lens = torch.tensor([len(y) for y in ys_b], dtype=torch.long)
            loss = ctc(logits.log_softmax(2), tgt, input_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        m = evaluate(model, vocab, xva, yva, 255.0)
        print(f"epoch {ep+1}: loss {tot/nb:.3f}  val CER {m['CER']:.4f} exact {m['exact_acc']:.4f} ({time.time()-tt:.0f}s)", flush=True)

    res = {
        "vocab_size": vocab.size,
        "n_train": N_TRAIN, "n_val": N_VAL, "n_ood": N_OOD,
        "iid_val": evaluate(model, vocab, xva, yva, 255.0),
        "ood_long": evaluate(model, vocab, xoo, yoo, 255.0),
        "wall_time_s": time.time() - t0,
    }
    torch.save(model.state_dict(), "crnn_poc.pt")
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
