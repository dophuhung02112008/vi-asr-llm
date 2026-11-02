#!/usr/bin/env python
"""Nhận dạng 1 file: ASR rồi Qwen sửa lỗi.
  python scripts/transcribe.py audio.wav --asr-model small --llm Qwen/Qwen2.5-1.5B-Instruct
  python scripts/transcribe.py audio.wav --llm-backend ollama --llm qwen2.5:3b --context "họp nhóm dự án"
"""
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from viasr.asr import make_asr
from viasr.corrector import ConservativeCorrector
from viasr.llm_backends import make_backend
from viasr.pipeline import SpeechPipeline

ap = argparse.ArgumentParser()
ap.add_argument("audio"); ap.add_argument("--asr", default="faster-whisper"); ap.add_argument("--asr-model", default="small")
ap.add_argument("--llm-backend", default="hf"); ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter"); ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--context"); ap.add_argument("--nbest", type=int, default=0); ap.add_argument("--no-llm", action="store_true")
a = ap.parse_args()
asr = make_asr(a.asr, **({"model_size": a.asr_model} if a.asr == "faster-whisper" else {"model_name": a.asr_model}))
cor = None
if not a.no_llm:
    be = (make_backend("hf", model_name=a.llm, adapter=a.adapter) if a.llm_backend == "hf"
          else make_backend("openai", base_url=a.base_url, model=a.llm))
    cor = ConservativeCorrector(be)
out = SpeechPipeline(asr, cor, use_nbest=a.nbest > 1, n=a.nbest).run(a.audio, a.context)
print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2))
