#!/usr/bin/env python
"""Tạo dữ liệu fine-tune Qwen từ giả thuyết ASR thật (+ văn bản sạch để sinh lỗi bổ trợ).
  python scripts/build_sft_data.py --hyps data/asr_hyps.jsonl --texts data/text_corpus.txt --out-dir data/sft
Chỉ split=train tạo dữ liệu huấn luyện; split=dev để chọn checkpoint. KHÔNG dùng test.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.data import read_jsonl, write_jsonl
from viasr.sft_data import build_sft, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--hyps"); ap.add_argument("--texts", action="append", default=[])
ap.add_argument("--out-dir", default="data/sft"); ap.add_argument("--identity-ratio", type=float, default=0.3)
ap.add_argument("--max-synth-ratio", type=float, default=0.5); ap.add_argument("--synth-per-text", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
pairs = read_jsonl(a.hyps) if a.hyps else []
texts = [l.strip() for t in a.texts for l in open(t, encoding="utf-8") if l.strip()]
for split in ("train", "dev"):
    sp = [p for p in pairs if p.get("split") == split]
    tx = texts if split == "train" else []
    if not sp and not tx: continue
    rows = build_sft(sp, tx, seed=a.seed, identity_ratio=a.identity_ratio,
                     max_synth_ratio=a.max_synth_ratio if sp else 1.0, synth_per_text=a.synth_per_text)
    print(split, write_jsonl(rows, Path(a.out_dir) / f"{split}.jsonl"), summarize(rows))
