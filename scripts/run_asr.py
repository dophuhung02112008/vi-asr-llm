#!/usr/bin/env python
"""Chạy ASR trên manifest để lấy giả thuyết THẬT (dùng cho huấn luyện lẫn đánh giá). Có thể chạy tiếp khi bị ngắt.

  python scripts/run_asr.py --manifest data/manifest.jsonl --model small --out data/asr_hyps.jsonl
  python scripts/run_asr.py --manifest ... --engine phowhisper --model vinai/PhoWhisper-small --out ...
"""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.asr import make_asr
from viasr.data import read_jsonl
from viasr.text_norm import normalize_for_eval, digits_to_words

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--engine", default="faster-whisper"); ap.add_argument("--model", default="small")
ap.add_argument("--device", default="auto"); ap.add_argument("--compute-type", default="default")
ap.add_argument("--splits", default="train,dev,test"); ap.add_argument("--nbest", type=int, default=0)
ap.add_argument("--max-per-split", type=int, default=0, help="giới hạn số câu mỗi split (0 = tất cả)")
ap.add_argument("--no-vad", action="store_true", help="tắt VAD (nên tắt với câu ngắn đã cắt sẵn như VIVOS)")
ap.add_argument("--normalize", action="store_true",
                help="đưa giả thuyết về chữ thường, bỏ dấu câu (khớp kiểu bản chép VIVOS); bản gốc lưu ở hyp_raw")
ap.add_argument("--num2words", action="store_true", help="đổi số sang chữ (VIVOS ghi số bằng chữ)")
ap.add_argument("--thousand", default="nghìn", help="'nghìn' hoặc 'ngàn'")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = [r for r in read_jsonl(a.manifest) if r.get("split", "test") in a.splits.split(",")]
if a.max_per_split:
    rng, picked = random.Random(a.seed), []
    for sp in a.splits.split(","):
        part = [r for r in rows if r.get("split") == sp]
        rng.shuffle(part)
        picked += part[: a.max_per_split]
    rows = picked
done = set()
if Path(a.out).exists():  # chạy tiếp
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
todo = [r for r in rows if r["id"] not in done]
print(f"Tổng {len(rows)} câu, đã có {len(done)}, cần chạy {len(todo)}", flush=True)


def norm(t: str) -> str:
    if a.num2words:
        t = digits_to_words(t, a.thousand)
    return normalize_for_eval(t) if a.normalize else t


kw = (dict(model_size=a.model, device=a.device, compute_type=a.compute_type) if a.engine == "faster-whisper"
      else dict(model_name=a.model))
asr = make_asr(a.engine, **kw)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with open(a.out, "a", encoding="utf-8") as f:
    for i, r in enumerate(todo):
        try:
            raw = (asr.transcribe(r["audio"], vad=not a.no_vad).text if a.engine == "faster-whisper"
                   else asr.transcribe(r["audio"]).text)
        except Exception as e:
            print("Lỗi", r["id"], type(e).__name__, e, flush=True); continue
        rec = {k: v for k, v in r.items() if k != "audio_array"}
        rec.update(ref=r["text"], hyp_raw=raw, hyp=norm(raw),
                   asr_model=f"{a.engine}:{a.model}")
        if a.nbest > 1:
            nb = asr.nbest(r["audio"], n=a.nbest)
            rec["nbest"] = [norm(x) for x in nb]
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 200 == 0:
            f.flush(); el = time.time() - t0
            print(f"{i+1}/{len(todo)}  {el:.0f}s  (~{el/(i+1)*(len(todo)-i-1)/60:.1f} phút còn lại)", flush=True)
print("Xong ->", a.out)
