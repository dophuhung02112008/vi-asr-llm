"""Tạo dữ liệu SFT cho Qwen hậu xử lý.
- real      : giả thuyết ASR thật + bản chép người kiểm tra (QUAN TRỌNG NHẤT)
- synthetic : lỗi giả lập từ văn bản sạch (bổ trợ, mặc định ≤50% khi có dữ liệu thật)
- identity  : câu đúng → giữ nguyên (dạy KHÔNG sửa bừa)
- protect   : câu có từ địa phương/lóng/tiếng Anh đúng → giữ nguyên
- wrong_ctx : kèm ngữ cảnh sai chủ đề, đích vẫn là bản chuẩn (chống bị ngữ cảnh dắt mũi)"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

from .lexicon import Lexicon
from .noise import AsrNoiser
from .prompts import chat_messages
from .text_norm import nfc, normalize_for_eval

UNRELATED_CONTEXTS = ["Cuộc gọi tư vấn bảo hiểm xe máy.", "Bản tin thời tiết miền Bắc.",
                      "Buổi họp kỹ thuật về server.", "Livestream bán quần áo.", "Bài giảng hoá học lớp 10.",
                      "Tổng đài đặt món ăn.", "Trò chuyện gia đình ở miền Trung.", "Phỏng vấn tuyển dụng."]


def make_example(hyp, ref, kind, lex, context=None, meta=None) -> dict:
    return {"messages": chat_messages(hyp, context, lex.find_hints(hyp), target=nfc(ref)),
            "kind": kind, "hyp": hyp, "ref": nfc(ref), **(meta or {})}


def build_sft(real_pairs: Iterable[dict], clean_texts: Iterable[str] = (), lex: Lexicon | None = None,
              seed: int = 0, identity_ratio: float = 0.3, max_synth_ratio: float = 0.5,
              wrong_ctx_ratio: float = 0.1, synth_per_text: int = 1) -> list[dict]:
    """Chỉ truyền dữ liệu TRAIN vào đây."""
    lex = lex or Lexicon.load()
    rng, noiser, out = random.Random(seed), AsrNoiser(lex, seed=seed), []
    real = [p for p in real_pairs if p.get("hyp") is not None and p.get("ref")]
    for p in real:
        same = normalize_for_eval(p["hyp"]) == normalize_for_eval(p["ref"])
        out.append(make_example(p["hyp"], p["ref"], "identity_real" if same else "real", lex,
                                p.get("context") or None, {"speaker": p.get("speaker")}))
    synth = []
    for t in clean_texts:
        t = nfc(t).strip()
        for _ in range(synth_per_text if t else 0):
            noisy = noiser(t)
            synth.append(make_example(noisy, t, "identity_synth" if normalize_for_eval(noisy) == normalize_for_eval(t)
                                      else "synthetic", lex))
    if real and max_synth_ratio < 1:
        rng.shuffle(synth)
        synth = synth[: int(len(real) * max_synth_ratio / (1 - max_synth_ratio))]
    out += synth
    refs = [p["ref"] for p in real] + [e["ref"] for e in synth]
    n_id = sum(e["kind"].startswith("identity") for e in out)
    target = int(identity_ratio * len(out) / max(1e-9, 1 - identity_ratio))
    rng.shuffle(refs)
    for r in refs[: max(0, target - n_id)]:
        out.append(make_example(r, r, "identity", lex))
    for r in [r for r in refs if lex.find_hints(r)][: max(1, len(out) // 20)]:
        out.append(make_example(r, r, "protect", lex))
    non_id = [e for e in out if e["kind"] in ("real", "synthetic")]
    rng.shuffle(non_id)
    for e in non_id[: int(len(out) * wrong_ctx_ratio)]:
        out.append(make_example(e["hyp"], e["ref"], "wrong_ctx", lex, rng.choice(UNRELATED_CONTEXTS)))
    rng.shuffle(out)
    return out


def summarize(rows: list[dict]) -> dict:
    return dict(Counter(r["kind"] for r in rows))
