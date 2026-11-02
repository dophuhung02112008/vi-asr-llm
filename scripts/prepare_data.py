#!/usr/bin/env python
"""Gộp dữ liệu về manifest chung + chia tập THEO NGƯỜI NÓI (trước mọi bước sinh lỗi).
  python scripts/prepare_data.py --vivos data/raw/vivos --custom data/raw/dia_phuong/meta.csv --out data/manifest.jsonl
"""
import argparse
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.data import load_vivos, load_common_voice, load_custom_csv, speaker_split, check_leakage, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--vivos"); ap.add_argument("--common-voice"); ap.add_argument("--custom", action="append", default=[])
ap.add_argument("--dev", type=float, default=0.1); ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--out", default="data/manifest.jsonl")
a = ap.parse_args()
rows = []
if a.vivos: rows += list(load_vivos(a.vivos))
if a.common_voice: rows += list(load_common_voice(a.common_voice))
for c in a.custom: rows += list(load_custom_csv(c))
if not rows: sys.exit("Chưa có dữ liệu. Xem docs/HUONG_DAN.md mục Dữ liệu")
rows = speaker_split(rows, a.dev, a.test)
leak = check_leakage(rows)
print(f"Đã ghi {write_jsonl(rows, a.out)} câu -> {a.out}", dict(Counter(r['split'] for r in rows)))
print("Người nói bị rò rỉ giữa các tập:", len(leak["speakers_in_multiple_splits"]),
      "| Câu trùng nội dung giữa các tập:", leak["texts_shared_across_splits"])
