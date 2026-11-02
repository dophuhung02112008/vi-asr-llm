#!/usr/bin/env python
"""Áp lại một mức hàng rào lên đầu ra LLM đã có (trường raw_llm), không cần chạy lại mô hình.
  python scripts/apply_guard.py --in runs/zs.jsonl --guard off --out runs/zs_off.jsonl
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.corrector import ConservativeCorrector, clean_llm_output, guard_preset
from viasr.data import read_jsonl, write_jsonl
from viasr.llm_backends import DictBackend

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--guard", default="default", choices=["off", "loose", "default", "strict"])
a = ap.parse_args()
cor = ConservativeCorrector(DictBackend(), guard=guard_preset(a.guard))
out = []
for r in read_jsonl(a.inp):
    raw = r.get("raw_llm")
    if raw is None:
        out.append({**r, "corrected": r["hyp"], "accepted": False, "reasons": ["no_llm_output"], "guard": a.guard}); continue
    cand = clean_llm_output(raw)
    ok, reasons, ratio = cor.check(r["hyp"], cand)
    out.append({**r, "corrected": cand if ok else r["hyp"], "accepted": ok, "reasons": reasons,
                "edit_ratio": ratio, "guard": a.guard})
write_jsonl(out, a.out); print("->", a.out, f"({sum(o['accepted'] for o in out)}/{len(out)} chấp nhận)")
