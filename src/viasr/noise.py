"""Sinh lỗi kiểu ASR cho văn bản tiếng Việt (chỉ BỔ TRỢ cho giả thuyết ASR thật).

Mô phỏng: nhầm dấu thanh (hỏi~ngã, sắc~nặng), nhầm phụ âm/vần (s/x, tr/ch, l/n, an/ang...),
nuốt âm, dính âm tiết, phiên âm sai từ tiếng Anh chêm vào."""
from __future__ import annotations

import random
import unicodedata

from .lexicon import Lexicon
from .text_norm import nfc

_TONES = ["\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]
_TONE_CONFUSION = {"\u0309": ["\u0303", "\u0301"], "\u0303": ["\u0309", "\u0301"], "\u0301": ["\u0323", "\u0309"],
                   "\u0323": ["\u0301", ""], "\u0300": ["", "\u0309"], "": ["\u0300", "\u0301"]}
_VOWELS = set("aăâeêioôơuưy")
_MOD_VOWELS = set("ăâêôơư")


def _split_tone(syl: str) -> tuple[str, str]:
    d = unicodedata.normalize("NFD", nfc(syl))
    tone = next((c for c in d if c in _TONES), "")
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONES)), tone


def place_tone(base: str, tone: str) -> str:
    """Đặt dấu thanh theo quy tắc chính tả (kiểu mới, xấp xỉ)."""
    if not tone:
        return base
    low = base.lower()
    start = 2 if low.startswith("qu") or (low.startswith("gi") and len(low) > 2 and low[2] in _VOWELS) else 0
    idx = [i for i in range(start, len(low)) if low[i] in _VOWELS]
    if not idx:
        return base
    grp = [idx[0]]
    for i in idx[1:]:
        if i == grp[-1] + 1:
            grp.append(i)
        else:
            break
    has_coda = grp[-1] + 1 < len(low)
    mods = [i for i in grp if low[i] in _MOD_VOWELS]
    if mods:
        pos = mods[-1]
    elif len(grp) == 1:
        pos = grp[0]
    elif has_coda:
        pos = grp[-1]
    elif len(grp) == 3:
        pos = grp[1]
    elif low[grp[0]:grp[0] + 2] in ("oa", "oe", "uy"):
        pos = grp[1]
    else:
        pos = grp[0]
    return base[:pos] + unicodedata.normalize("NFC", base[pos] + tone) + base[pos + 1:]


def change_tone(syl: str, rng: random.Random) -> str:
    base, tone = _split_tone(syl)
    if not any(c.lower() in _VOWELS for c in base):
        return syl
    new = rng.choice(_TONE_CONFUSION.get(tone, [""]))
    return syl if new == tone else place_tone(base, new)


def swap_onset_rhyme(syl: str, pairs: list[list[str]], rng: random.Random) -> str:
    base, tone = _split_tone(syl)
    low, cand = base.lower(), []
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            if x[0] not in _VOWELS:
                if low.startswith(x) and not (len(low) > len(x) and low[len(x)] in "hgr" and x in ("c", "t", "n", "g")):
                    cand.append(("onset", x, y))
            elif low.endswith(x) and len(low) > len(x):
                cand.append(("rhyme", x, y))
    if not cand:
        return syl
    kind, x, y = rng.choice(cand)
    nb = (y + base[len(x):]) if kind == "onset" else (base[: len(base) - len(x)] + y)
    if base[:1].isupper():
        nb = nb[:1].upper() + nb[1:]
    return place_tone(nb, tone)


class AsrNoiser:
    def __init__(self, lexicon: Lexicon | None = None, seed: int = 0, p_tone=0.06, p_phone=0.05,
                 p_drop=0.02, p_merge=0.02, p_codeswitch=0.8):
        self.lex = lexicon or Lexicon.load()
        self.rng = random.Random(seed)
        self.p = dict(tone=p_tone, phone=p_phone, drop=p_drop, merge=p_merge, cs=p_codeswitch)
        self._cs = {k.lower(): v for k, v in self.lex.code_switch.items()}

    def __call__(self, text: str) -> str:
        rng, p, toks, out, i = self.rng, self.p, nfc(text).split(), [], 0
        while i < len(toks):
            t = toks[i]
            core = t.strip(".,!?;:\"'()")
            if core.lower() in self._cs and rng.random() < p["cs"]:
                out.append(rng.choice(self._cs[core.lower()]) + t[len(core):]); i += 1; continue
            r = rng.random()
            if r < p["drop"] and len(toks) > 3:
                i += 1; continue
            if r < p["drop"] + p["merge"] and i + 1 < len(toks):
                out.append(t + toks[i + 1].lower()); i += 2; continue
            if rng.random() < p["tone"]:
                t = change_tone(t, rng)
            if rng.random() < p["phone"]:
                t = swap_onset_rhyme(t, self.lex.confusable_pairs, rng)
            out.append(t); i += 1
        return " ".join(out)
