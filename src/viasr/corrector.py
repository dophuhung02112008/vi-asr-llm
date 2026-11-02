"""Hậu xử lý bảo thủ: LLM đề xuất bản sửa → hàng rào kiểm tra → mới chấp nhận, nếu không giữ bản nháp ASR.

Hàng rào: tỷ lệ chỉnh sửa theo âm tiết, xoá âm tiết, lệch độ dài, đổi số, 'chuẩn hoá' từ địa phương/lóng.
Phiên âm tiếng Anh đã biết trong từ điển (vd "mít tinh"→"meeting") được coi là chỉnh sửa có căn cứ.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import Lexicon
from .llm_backends import ChatBackend
from .metrics import edit_ops
from .prompts import SELECT_PROMPT, build_select_prompt, chat_messages
from .text_norm import nfc, normalize_for_eval

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_PREFIX_RE = re.compile(r"^(bản (đã )?sửa|kết quả|đáp án|output|corrected)\s*[:：]\s*", re.I)


@dataclass
class CorrectionResult:
    hypothesis: str
    corrected: str
    raw_llm: str | None = None
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    edit_ratio: float = 0.0
    hints: list[str] = field(default_factory=list)


@dataclass
class GuardConfig:
    max_edit_ratio: float = 0.35
    max_len_ratio: float = 1.4
    min_len_ratio: float = 0.7
    protect_numbers: bool = True
    protect_lexicon: bool = True
    min_syllables_for_ratio: int = 4
    max_abs_edits_short: int = 2
    max_deletions: int = 0   # LLM xoá âm tiết thường là đang viết lại/làm mất ý


def guard_off() -> GuardConfig:
    return GuardConfig(9, 99, 0, False, False, 4, 999, 999)


def guard_preset(name: str) -> GuardConfig:
    """Các mức hàng rào dùng khi so sánh: off | loose | default | strict."""
    return {"off": guard_off(),
            "loose": GuardConfig(max_edit_ratio=0.6, max_len_ratio=1.6, min_len_ratio=0.6, max_deletions=2,
                                 max_abs_edits_short=3, protect_numbers=False),
            "default": GuardConfig(),
            "strict": GuardConfig(max_edit_ratio=0.2, max_abs_edits_short=1)}[name]


def clean_llm_output(text: str) -> str:
    t = nfc(text or "").strip().split("\n")[0].strip()
    t = _PREFIX_RE.sub("", t)
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”«»`":
        t = t[1:-1].strip()
    return re.sub(r"\s+", " ", t.strip("“”\"`")).strip()


class ConservativeCorrector:
    def __init__(self, backend: ChatBackend, lexicon: Lexicon | None = None,
                 guard: GuardConfig | None = None, use_hints: bool = True):
        self.backend, self.lex = backend, lexicon or Lexicon.load()
        self.guard, self.use_hints = guard or GuardConfig(), use_hints
        self._norm_map = self._build_norm_map()
        self._phon = self.lex.phonetic_index()

    def _build_norm_map(self) -> dict[str, list[str]]:
        mp: dict[str, set[str]] = {}
        srcs = list(self.lex.dialect.values()) + [self.lex.slang, self.lex.abbrev]
        for d in srcs:
            for w, std in d.items():
                t, s = normalize_for_eval(w), normalize_for_eval(std)
                if t and s and t != s:
                    mp.setdefault(t, set()).add(s)
        return {k: sorted(v) for k, v in mp.items()}

    def project_codeswitch(self, hyp: str, cand: str) -> str:
        h, c = f" {normalize_for_eval(hyp)} ", f" {normalize_for_eval(cand)} "
        for phon, en in sorted(self._phon.items(), key=lambda kv: -len(kv[0])):
            e = normalize_for_eval(en)
            if f" {phon} " in h and f" {e} " in c:
                h = h.replace(f" {phon} ", f" {e} ")
        return h.strip()

    def correct(self, hypothesis: str, context: str | None = None) -> CorrectionResult:
        hyp = nfc(hypothesis).strip()
        res = CorrectionResult(hypothesis=hyp, corrected=hyp)
        if not hyp:
            res.reasons.append("empty_input"); return res
        res.hints = self.lex.find_hints(hyp) if self.use_hints else []
        try:
            raw = self.backend.chat(chat_messages(hyp, context, res.hints))
        except Exception as e:
            res.reasons.append(f"backend_error:{type(e).__name__}"); return res
        res.raw_llm = raw
        cand = clean_llm_output(raw)
        ok, res.reasons, res.edit_ratio = self.check(hyp, cand)
        if ok:
            res.corrected, res.accepted = cand, True
        return res

    def correct_batch(self, hyps: list[str], contexts: list[str | None] | None = None,
                      batch_size: int = 16) -> list[CorrectionResult]:
        """Như correct() nhưng gọi LLM theo lô (nhanh hơn nhiều trên GPU)."""
        contexts = contexts or [None] * len(hyps)
        results = [CorrectionResult(hypothesis=nfc(h).strip(), corrected=nfc(h).strip()) for h in hyps]
        todo = [i for i, r in enumerate(results) if r.hypothesis]
        for i in range(len(results)):
            if not results[i].hypothesis:
                results[i].reasons.append("empty_input")
        has_batch = hasattr(self.backend, "chat_batch")
        for s0 in range(0, len(todo), batch_size):
            idx = todo[s0:s0 + batch_size]
            msgs = []
            for i in idx:
                r = results[i]
                r.hints = self.lex.find_hints(r.hypothesis) if self.use_hints else []
                msgs.append(chat_messages(r.hypothesis, contexts[i], r.hints))
            try:
                raws = self.backend.chat_batch(msgs) if has_batch else [self.backend.chat(m) for m in msgs]
            except Exception as e:
                for i in idx:
                    results[i].reasons.append(f"backend_error:{type(e).__name__}")
                continue
            for i, raw in zip(idx, raws):
                r = results[i]
                r.raw_llm = raw
                cand = clean_llm_output(raw)
                ok, r.reasons, r.edit_ratio = self.check(r.hypothesis, cand)
                if ok:
                    r.corrected, r.accepted = cand, True
        return results

    def check(self, hyp: str, cand: str) -> tuple[bool, list[str], float]:
        g, reasons = self.guard, []
        if not cand:
            return False, ["empty_output"], 1.0
        if normalize_for_eval(hyp) == normalize_for_eval(cand):
            return True, ["no_change"], 0.0
        h_syl, c_syl = normalize_for_eval(hyp).split(), normalize_for_eval(cand).split()
        st = edit_ops(self.project_codeswitch(hyp, cand).split(), c_syl)
        ratio = st.errors / max(1, len(h_syl))
        if len(h_syl) >= g.min_syllables_for_ratio:
            if ratio > g.max_edit_ratio:
                reasons.append(f"edit_ratio>{g.max_edit_ratio}")
        elif st.errors > g.max_abs_edits_short:
            reasons.append("too_many_edits_short")
        if st.dele > g.max_deletions:
            reasons.append(f"deleted_syllables:{st.dele}")
        lr = len(c_syl) / max(1, len(h_syl))
        if lr > g.max_len_ratio or lr < g.min_len_ratio:
            reasons.append("length_ratio")
        if g.protect_numbers and sorted(_NUM_RE.findall(hyp)) != sorted(_NUM_RE.findall(cand)):
            reasons.append("numbers_changed")
        if g.protect_lexicon:
            hn, cn = f" {normalize_for_eval(hyp)} ", f" {normalize_for_eval(cand)} "
            for term, stds in self._norm_map.items():
                if f" {term} " in hn and f" {term} " not in cn and any(
                        f" {s} " in cn and f" {s} " not in hn for s in stds):
                    reasons.append(f"normalized_protected_term:{term}"); break
        return (not reasons), reasons or ["ok"], ratio

    def select(self, candidates: list[str], context: str | None = None) -> tuple[int, str]:
        """Chọn 1 trong N-best; đầu ra luôn là nguyên văn 1 giả thuyết (không thể ảo giác)."""
        cands = [nfc(c).strip() for c in candidates if c and c.strip()]
        if not cands:
            return -1, ""
        if len(cands) == 1:
            return 0, cands[0]
        msgs = [{"role": "system", "content": SELECT_PROMPT},
                {"role": "user", "content": build_select_prompt(cands, context)}]
        try:
            m = re.search(r"\d+", self.backend.chat(msgs, max_new_tokens=8))
            idx = int(m.group()) - 1 if m else 0
        except Exception:
            idx = 0
        idx = idx if 0 <= idx < len(cands) else 0
        return idx, cands[idx]

    def correct_nbest(self, candidates: list[str], context: str | None = None) -> CorrectionResult:
        idx, best = self.select(candidates, context)
        res = self.correct(best, context)
        res.reasons.insert(0, f"selected_hyp:{idx + 1}")
        return res
