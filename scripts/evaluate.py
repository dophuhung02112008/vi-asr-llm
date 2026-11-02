#!/usr/bin/env python
"""So sánh trước/sau hậu xử lý: WER âm tiết, CER, WER theo từ, câu bị sửa hỏng, KTC bootstrap theo người nói.
  python scripts/evaluate.py --in runs/corrected.jsonl --report runs/report.md [--by region]
"""
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.data import read_jsonl
from viasr.metrics import corpus_report, report_markdown, syllable_stats

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--report")
ap.add_argument("--title", default="Đánh giá ASR + Qwen hậu xử lý"); ap.add_argument("--by", default="")
ap.add_argument("--examples", type=int, default=10)
a = ap.parse_args()
rows = read_jsonl(a.inp)
rep = corpus_report(rows)
md = report_markdown(rep, a.title)
if a.by:
    g = {}
    for r in rows: g.setdefault(r.get(a.by) or "?", []).append(r)
    md += f"\n## Theo `{a.by}`\n\n| Nhóm | Số câu | WER trước | WER sau |\n|---|---|---|---|\n"
    for k, rs in sorted(g.items()):
        x = corpus_report(rs, n_boot=0)
        md += f"| {k} | {x.n_utts} | {100*x.syl_wer_before:.2f}% | {100*x.syl_wer_after:.2f}% |\n"
d = lambda r: syllable_stats(r["ref"], r.get("corrected", r["hyp"])).errors - syllable_stats(r["ref"], r["hyp"]).errors
for title, sel in (("Ví dụ sửa đúng", sorted([r for r in rows if d(r) < 0], key=d)),
                   ("Ví dụ sửa hỏng", sorted([r for r in rows if d(r) > 0], key=d, reverse=True)),
                   ("Ví dụ bị hàng rào chặn", [r for r in rows if r.get("accepted") is False and r.get("raw_llm")])):
    if sel and a.examples:
        md += f"\n## {title}\n\n"
        for r in sel[: a.examples]:
            md += (f"- REF: {r['ref']}\n  - ASR: {r['hyp']}\n  - LLM: {r.get('raw_llm') or ''}\n"
                   f"  - CUỐI: {r.get('corrected')}  ({', '.join(r.get('reasons') or [])})\n")
print(md)
if a.report:
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(md, encoding="utf-8")
    Path(a.report).with_suffix(".json").write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
