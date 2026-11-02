"""Chuẩn hoá văn bản tiếng Việt (Unicode NFC, dấu thanh, âm tiết)."""
from __future__ import annotations

import re
import unicodedata

TONE_MARKS = {"\u0300": "huyen", "\u0301": "sac", "\u0309": "hoi", "\u0303": "nga", "\u0323": "nang"}
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def normalize_for_eval(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    """NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng. KHÔNG bỏ dấu thanh/dấu chữ."""
    t = nfc(text)
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = _PUNCT_RE.sub(" ", t)
    return _SPACE_RE.sub(" ", t).strip()


def syllables(text: str) -> list[str]:
    t = normalize_for_eval(text)
    return t.split() if t else []


def strip_tones(text: str) -> str:
    """Bỏ riêng dấu thanh, giữ dấu chữ (ă â ê ô ơ ư đ). Dùng để phân tích lỗi."""
    d = unicodedata.normalize("NFD", nfc(text))
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONE_MARKS))


def strip_diacritics(text: str) -> str:
    """Bỏ mọi dấu (đ->d). Chỉ dùng cho heuristic, KHÔNG dùng để chấm điểm."""
    d = unicodedata.normalize("NFD", nfc(text))
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return d.replace("đ", "d").replace("Đ", "D")


def get_tone(syllable: str) -> str | None:
    for c in unicodedata.normalize("NFD", nfc(syllable)):
        if c in TONE_MARKS:
            return TONE_MARKS[c]
    return None


# ------------------------------------------------------------------ số -> chữ tiếng Việt
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_NUM_TOKEN = re.compile(r"\d{1,3}(?:\.\d{3})+(?!\d)|\d+")


def _read_3(n: int, full: bool, le: str = "lẻ") -> list[str]:
    """Đọc số 0..999. full=True: đọc đủ 'không trăm' khi đứng sau nhóm lớn hơn."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    out: list[str] = []
    if h or full:
        out += [_DIGITS[h], "trăm"]
    if t == 0:
        if u and (h or full):
            out += [le, _DIGITS[u]]
        elif u:
            out.append(_DIGITS[u])
    elif t == 1:
        out.append("mười")
        if u:
            out.append("lăm" if u == 5 else _DIGITS[u])
    else:
        out += [_DIGITS[t], "mươi"]
        if u == 1:
            out.append("mốt")
        elif u == 5:
            out.append("lăm")
        elif u:
            out.append(_DIGITS[u])
    return out


def number_to_words(n: int, thousand: str = "nghìn", le: str = "lẻ") -> str:
    """Đọc số nguyên không âm theo cách nói phổ biến: 21 -> 'hai mươi mốt', 105 -> 'một trăm lẻ năm'."""
    if n == 0:
        return "không"
    if n >= 10 ** 12:
        return " ".join(_DIGITS[int(c)] for c in str(n))
    units = ["", thousand, "triệu", "tỷ"]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    words: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        words += _read_3(g, full=bool(words), le=le)
        if units[i]:
            words.append(units[i])
    return " ".join(words)


def digits_to_words(text: str, thousand: str = "nghìn") -> str:
    """Thay các số nguyên trong câu bằng chữ (vd '11 giờ' -> 'mười một giờ'). Số dài > 12 chữ số đọc từng chữ số."""
    def rep(m):
        tok = m.group().replace(".", "")
        if len(tok) > 1 and tok.startswith("0"):  # số điện thoại/mã: đọc từng chữ số
            return " ".join(_DIGITS[int(c)] for c in tok)
        return number_to_words(int(tok), thousand)
    return _NUM_TOKEN.sub(rep, text)
