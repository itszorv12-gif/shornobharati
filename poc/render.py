"""HarfBuzz-shaped synthetic Bangla text image generator (POC)."""
import io
import random

import freetype
import numpy as np
import regex as re
import uharfbuzz as hb

FONT_PATH = "fonts/NotoSansBengali.ttf"

with open(FONT_PATH, "rb") as f:
    _FONT_BYTES = f.read()

_INDEPENDENT_VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ"]
_CONSONANTS = [
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ",
    "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়",
]
_MATRAS = ["", "", "", "া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ"]
_CONJUNCTS = [
    "ক্ষ", "জ্ঞ", "স্থ", "ত্র", "দ্ব", "ন্দ", "প্র", "ব্র", "ক্র", "ম্ব",
    "চ্ছ", "স্ত", "স্ট", "ট্র", "ন্ট", "ক্ত", "ন্ন", "র্ন", "স্প", "শ্রী",
]
_FINAL_CONS = ["ক", "র", "ল", "ম", "ন", "স", "স্য", "র্", "ত্"]


def grapheme_split(s: str):
    return re.findall(r"\X", s)


def make_word(rng: random.Random, min_cl: int = 3, max_cl: int = 6) -> str:
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
    w = "".join(parts)
    if rng.random() < 0.12:
        w = w + rng.choice(["।", ",", "ি?".replace("?", "")]) if False else w
    return w


class ShapingRenderer:
    def __init__(self, canvas_h: int = 48):
        self.canvas_h = canvas_h
        self._hb_blob = hb.Blob(_FONT_BYTES)
        self._hb_face = hb.Face(self._hb_blob)
        self._hb_font = hb.Font(self._hb_face)
        self._ft = freetype.Face(io.BytesIO(_FONT_BYTES))

    def patch_size(self, px: int):
        self._ft.set_pixel_sizes(0, px)

    def render(self, text: str, px: int = 40, pad: int = 8) -> np.ndarray:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self._hb_font, buf, {"kern": True, "liga": True})
        self.patch_size(px)
        scale = px / self._hb_face.upem

        infos = buf.glyph_infos
        poss = buf.glyph_positions
        adv = sum(p.x_advance for p in poss) * scale
        w = int(adv) + 2 * pad + 2
        asc = int(self._ft.size.ascender / 64.0)
        desc = int(-self._ft.size.descender / 64.0)
        h = asc + desc + 2 * pad
        canvas = np.full((h, w), 255, dtype=np.uint8)

        pen_x = float(pad)
        for info, pos in zip(infos, poss):
            gid = info.codepoint
            self._ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
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
