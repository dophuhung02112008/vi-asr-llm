"""Thước đo ASR tiếng Việt: CER, WER theo âm tiết, WER theo từ (nếu có bộ tách từ),
và thống kê tác động của hậu xử lý (sửa đúng / sửa hỏng / phá câu vốn đúng)."""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

from .text_norm import normalize_for_eval


@dataclass
class EditStats:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else float(self.errors > 0)

    def __add__(self, o: "EditStats") -> "EditStats":
        return EditStats(self.sub + o.sub, self.dele + o.dele, self.ins + o.ins, self.ref_len + o.ref_len)


def edit_ops(ref: Sequence, hyp: Sequence) -> EditStats:
    """Levenshtein có đếm thay thế / xoá / chèn."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            d0 = prev[j - 1]
            best = d0 if ref[i - 1] == hyp[j - 1] else (d0[0] + 1, d0[1] + 1, d0[2], d0[3])
            d1 = prev[j]
            if d1[0] + 1 < best[0]:
                best = (d1[0] + 1, d1[1], d1[2] + 1, d1[3])
            d2 = cur[j - 1]
            if d2[0] + 1 < best[0]:
                best = (d2[0] + 1, d2[1], d2[2], d2[3] + 1)
            cur.append(best)
        prev = cur
    c = prev[m]
    return EditStats(c[1], c[2], c[3], n)


def syllable_stats(ref: str, hyp: str) -> EditStats:
    """WER theo âm tiết (tách theo khoảng trắng) — KHÔNG phải word-WER ngôn ngữ học."""
    return edit_ops(normalize_for_eval(ref).split(), normalize_for_eval(hyp).split())


def char_stats(ref: str, hyp: str, ignore_space: bool = True) -> EditStats:
    r, h = normalize_for_eval(ref), normalize_for_eval(hyp)
    if ignore_space:
        r, h = r.replace(" ", ""), h.replace(" ", "")
    return edit_ops(list(r), list(h))


def get_word_segmenter() -> Callable[[str], list[str]] | None:
    try:
        from pyvi import ViTokenizer  # type: ignore
        return lambda s: ViTokenizer.tokenize(s).split()
    except Exception:
        pass
    try:
        from underthesea import word_tokenize  # type: ignore
        return lambda s: [w.replace(" ", "_") for w in word_tokenize(s)]
    except Exception:
        return None


def word_stats(ref: str, hyp: str, seg: Callable[[str], list[str]]) -> EditStats:
    return edit_ops(seg(normalize_for_eval(ref)), seg(normalize_for_eval(hyp)))


@dataclass
class CorpusReport:
    n_utts: int
    syl_wer_before: float
    syl_wer_after: float
    cer_before: float
    cer_after: float
    word_wer_before: float | None
    word_wer_after: float | None
    improved: int
    harmed: int
    unchanged: int
    broke_correct: int
    fixed_fully: int
    insertions_before: int
    insertions_after: int
    edited_utts: int
    ci95_delta_syl_wer: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def corpus_report(rows: Iterable[dict], ref_key="ref", before_key="hyp", after_key="corrected",
                  group_key: str | None = "speaker", n_boot: int = 1000, seed: int = 0) -> CorpusReport:
    rows = list(rows)
    seg = get_word_segmenter()
    sb = sa = cb = ca = wb = wa = EditStats()
    improved = harmed = unchanged = broke = fixed = edited = 0
    per = []
    for r in rows:
        ref, b, a = r[ref_key], r[before_key], r.get(after_key, r[before_key])
        s1, s2 = syllable_stats(ref, b), syllable_stats(ref, a)
        sb, sa = sb + s1, sa + s2
        cb, ca = cb + char_stats(ref, b), ca + char_stats(ref, a)
        if seg:
            wb, wa = wb + word_stats(ref, b, seg), wa + word_stats(ref, a, seg)
        if normalize_for_eval(a) != normalize_for_eval(b):
            edited += 1
        if s2.errors < s1.errors:
            improved += 1
        elif s2.errors > s1.errors:
            harmed += 1
        else:
            unchanged += 1
        broke += s1.errors == 0 and s2.errors > 0
        fixed += s1.errors > 0 and s2.errors == 0
        per.append((r.get(group_key) if group_key else None, s1, s2))
    ci = _bootstrap_delta(per, n_boot, seed) if rows and n_boot else None
    return CorpusReport(len(rows), sb.rate, sa.rate, cb.rate, ca.rate,
                        wb.rate if seg else None, wa.rate if seg else None,
                        improved, harmed, unchanged, int(broke), int(fixed), sb.ins, sa.ins, edited, ci)


def _bootstrap_delta(per, n_boot, seed):
    """Bootstrap theo nhóm người nói cho ΔWER = sau - trước."""
    groups: dict = {}
    for i, (g, _, _) in enumerate(per):
        groups.setdefault(g if g is not None else f"_utt{i}", []).append(i)
    keys, rng, deltas = list(groups), random.Random(seed), []
    for _ in range(n_boot):
        e1 = e2 = n = 0
        for k in (rng.choice(keys) for _ in keys):
            for i in groups[k]:
                e1 += per[i][1].errors; e2 += per[i][2].errors; n += per[i][1].ref_len
        if n:
            deltas.append((e2 - e1) / n)
    if not deltas:
        return None
    deltas.sort()
    return (deltas[int(0.025 * (len(deltas) - 1))], deltas[int(0.975 * (len(deltas) - 1))])


def report_markdown(rep: CorpusReport, title="Báo cáo đánh giá") -> str:
    f = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"
    lines = [f"# {title}", "", f"Số câu: {rep.n_utts}", "",
             "| Thước đo | Trước hậu xử lý | Sau hậu xử lý |", "|---|---|---|",
             f"| WER theo âm tiết | {f(rep.syl_wer_before)} | {f(rep.syl_wer_after)} |",
             f"| CER (không tính khoảng trắng) | {f(rep.cer_before)} | {f(rep.cer_after)} |",
             f"| WER theo từ (tách từ pyvi) | {f(rep.word_wer_before)} | {f(rep.word_wer_after)} |",
             f"| Số lỗi chèn (âm tiết) | {rep.insertions_before} | {rep.insertions_after} |", "",
             f"- Câu bị sửa: {rep.edited_utts}",
             f"- Tốt hơn / tệ hơn / như cũ: {rep.improved} / {rep.harmed} / {rep.unchanged}",
             f"- Câu vốn đúng bị sửa hỏng: **{rep.broke_correct}**",
             f"- Câu sai được sửa đúng hoàn toàn: {rep.fixed_fully}"]
    if rep.ci95_delta_syl_wer:
        lo, hi = rep.ci95_delta_syl_wer
        lines.append(f"- KTC 95% (bootstrap theo người nói) của ΔWER: [{100*lo:+.2f}, {100*hi:+.2f}] điểm %")
    return "\n".join(lines) + "\n"
