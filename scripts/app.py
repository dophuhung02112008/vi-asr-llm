#!/usr/bin/env python
"""Giao diện web (Gradio): ghi âm/tải file → ASR → Qwen sửa lỗi, hiển thị trước/sau.
  pip install gradio && python scripts/app.py --asr-model small --llm Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gradio as gr
from viasr.asr import make_asr
from viasr.corrector import ConservativeCorrector
from viasr.llm_backends import make_backend
from viasr.pipeline import SpeechPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--asr-model", default="small"); ap.add_argument("--llm-backend", default="hf")
ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--adapter")
ap.add_argument("--base-url", default="http://localhost:11434/v1"); ap.add_argument("--share", action="store_true")
a = ap.parse_args()
be = (make_backend("hf", model_name=a.llm, adapter=a.adapter) if a.llm_backend == "hf"
      else make_backend("openai", base_url=a.base_url, model=a.llm))
pipe = SpeechPipeline(make_asr("faster-whisper", model_size=a.asr_model), ConservativeCorrector(be))


def run(audio, context):
    if not audio:
        return "", "", ""
    o = pipe.run(audio, context or None)
    info = (f"Chấp nhận bản sửa: {o.accepted} | Lý do: {', '.join(o.reasons)}\n"
            f"ASR: {o.asr_seconds:.2f}s | LLM: {o.llm_seconds:.2f}s\n" + "\n".join(o.hints))
    return o.asr_text, o.final_text, info


with gr.Blocks(title="ASR tiếng Việt + Qwen") as demo:
    gr.Markdown("## Nhận dạng giọng nói tiếng Việt + LLM sửa lỗi ngữ cảnh")
    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Âm thanh")
        ctx = gr.Textbox(label="Ngữ cảnh (tuỳ chọn)", placeholder="vd: họp nhóm dự án phần mềm")
    btn = gr.Button("Nhận dạng")
    raw, fin = gr.Textbox(label="ASR thô"), gr.Textbox(label="Sau khi Qwen sửa")
    info = gr.Textbox(label="Chi tiết", lines=6)
    btn.click(run, [audio, ctx], [raw, fin, info])
demo.launch(share=a.share)
