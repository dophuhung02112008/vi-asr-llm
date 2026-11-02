#!/usr/bin/env python
"""Gộp nhiều file kết quả thành một bảng so sánh (runs/SUMMARY.md + SUMMARY.json).
  python scripts/summarize_runs.py --runs "Zero-shot, không rào=runs/zs_off.jsonl" "LoRA, rào mặc định=runs/ft_default.jsonl"
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.data import read_jsonl
from viasr.metrics import corpus_report

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help='"Tên=đường/dẫn.jsonl"')
ap.add_argument("--out", default="runs/SUMMARY.md"); ap.add_argument("--meta", default="")
a = ap.parse_args()
pct = lambda x: "n/a" if x is None else f"{100*x:.2f}%"
lines = ["# Tóm tắt kết quả", ""]
if a.meta:
    lines += [a.meta, ""]
lines += ["| Cấu hình | Số câu | WER âm tiết | CER | WER theo từ | ΔWER (KTC 95%) | Câu bị sửa | Tốt/Tệ | Câu đúng bị phá |",
          "|---|---|---|---|---|---|---|---|---|"]
base_done, js = False, {}
for spec in a.runs:
    name, path = spec.split("=", 1)
    if not Path(path).exists():
        lines.append(f"| {name} | (thiếu file) | | | | | | | |"); continue
    rows = read_jsonl(path)
    rep = corpus_report(rows)
    js[name] = rep.to_dict()
    if not base_done:
        lines.append(f"| ASR gốc (không LLM) | {rep.n_utts} | {pct(rep.syl_wer_before)} | {pct(rep.cer_before)} | "
                     f"{pct(rep.word_wer_before)} | – | – | – | – |")
        base_done = True
    ci = rep.ci95_delta_syl_wer
    cis = f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]" if ci else "n/a"
    lines.append(f"| {name} | {rep.n_utts} | {pct(rep.syl_wer_after)} | {pct(rep.cer_after)} | {pct(rep.word_wer_after)} | "
                 f"{100*(rep.syl_wer_after-rep.syl_wer_before):+.2f} {cis} | {rep.edited_utts} | {rep.improved}/{rep.harmed} | {rep.broke_correct} |")
lines += ["", "ΔWER âm = tốt hơn. KTC 95% bootstrap theo người nói; nếu khoảng chứa 0 thì chưa đủ bằng chứng là có cải thiện."]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(a.out).with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n".join(lines))
