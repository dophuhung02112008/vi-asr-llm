"""Backend LLM: transformers (Qwen + LoRA), máy chủ tương thích OpenAI (Ollama/vLLM/llama.cpp/LM Studio),
và backend tra bảng để kiểm thử."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


class ChatBackend(Protocol):
    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str: ...


def pick_dtype(torch, name: str = "auto"):
    """auto: bf16 nếu GPU hỗ trợ thật (Ampere trở lên), fp16 cho GPU cũ như T4, fp32 cho CPU."""
    if name != "auto":
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def limit_cpu_threads(torch) -> None:
    """Trên CPU trong container bị giới hạn lõi, torch mặc định tạo quá nhiều luồng và chạy rất chậm."""
    if torch.cuda.is_available():
        return
    import os
    n = len(os.sched_getaffinity(0))
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(q) // int(per)))
    except Exception:
        pass
    torch.set_num_threads(n)


class TransformersBackend:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter: str | None = None,
                 load_in_4bit: bool = False, device_map: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        limit_cpu_threads(torch)
        self.max_new_tokens = max_new_tokens
        kw: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=pick_dtype(torch, dtype))
        else:
            kw["dtype"] = pick_dtype(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self._torch = torch

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def chat_batch(self, batch: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        inputs = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        L = inputs["input_ids"].shape[1]
        return [self.tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]


class OpenAICompatBackend:
    """/v1/chat/completions — Ollama: base_url http://localhost:11434/v1, model 'qwen2.5:3b'."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:3b",
                 api_key: str | None = None, timeout: int = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout = model, timeout
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "none")

    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": max_new_tokens}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()


class DictBackend:
    """Backend giả lập: tra bảng {bản nháp: đầu ra}. Dùng cho test/demo không cần GPU."""

    def __init__(self, table: dict[str, str] | None = None, default: str | None = None):
        self.table, self.default = table or {}, default

    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        user = messages[-1]["content"]
        hyp = user.split("Bản nháp ASR:")[-1].split("\n")[0].strip() if "Bản nháp ASR:" in user else user
        return self.table.get(hyp, self.default if self.default is not None else hyp)


def make_backend(kind: str, **kw) -> ChatBackend:
    kind = kind.lower()
    if kind in ("hf", "transformers"):
        return TransformersBackend(**kw)
    if kind in ("openai", "ollama", "vllm", "server"):
        return OpenAICompatBackend(**kw)
    if kind in ("dict", "mock", "none"):
        return DictBackend(**kw)
    raise ValueError(f"Backend không hỗ trợ: {kind}")
