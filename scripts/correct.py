#!/usr/bin/env python
"""Chạy Qwen hậu xử lý trên file giả thuyết ASR (JSONL có hyp[,nbest,context]). Đầu ra lưu cả raw_llm để
có thể áp lại các mức hàng rào khác bằng scripts/apply_guard.py mà không phải chạy lại LLM.

  python scripts/correct.py --in data/asr_hyps.jsonl --out runs/zs.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
  python scripts/correct.py --in ... --out runs/ft.jsonl --model ... --adapter runs/qwen-lora/best
  python scripts/correct.py --in ... --backend ollama --model qwen2.5:3b
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.corrector import ConservativeCorrector, guard_preset
from viasr.data import read_jsonl, write_jsonl
from viasr.llm_backends import make_backend

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="hf"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter"); ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--4bit", dest="four_bit", action="store_true")
ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--split", default="test"); ap.add_argument("--guard", default="default",
                choices=["off", "loose", "default", "strict"])
ap.add_argument("--no-hints", action="store_true"); ap.add_argument("--nbest", action="store_true")
ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
be = (make_backend("hf", model_name=a.model, adapter=a.adapter, load_in_4bit=a.four_bit, dtype=a.dtype,
                   max_new_tokens=a.max_new_tokens) if a.backend == "hf"
      else make_backend("openai", base_url=a.base_url, model=a.model))
cor = ConservativeCorrector(be, guard=guard_preset(a.guard), use_hints=not a.no_hints)
rows = [r for r in read_jsonl(a.inp) if not a.split or r.get("split", a.split) == a.split]
rows = rows[: a.limit] if a.limit else rows
t0 = time.time()
if a.nbest:
    res = [cor.correct_nbest(r["nbest"], r.get("context")) if r.get("nbest") else cor.correct(r["hyp"], r.get("context"))
           for r in rows]
else:
    res = []
    for s in range(0, len(rows), a.batch_size * 8):
        chunk = rows[s:s + a.batch_size * 8]
        res += cor.correct_batch([r["hyp"] for r in chunk], [r.get("context") for r in chunk], a.batch_size)
        print(f"{len(res)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
out = [{**r, "corrected": x.corrected, "accepted": x.accepted, "reasons": x.reasons, "raw_llm": x.raw_llm,
        "edit_ratio": x.edit_ratio, "guard": a.guard} for r, x in zip(rows, res)]
write_jsonl(out, a.out)
print(f"Xong {len(out)} câu, {time.time()-t0:.1f}s ({(time.time()-t0)/max(1,len(out)):.2f}s/câu) -> {a.out}")
