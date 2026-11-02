#!/usr/bin/env python
"""Fine-tune Qwen2.5-Instruct bằng LoRA/QLoRA cho hậu xử lý ASR tiếng Việt (loss chỉ trên phần trả lời).

  # GPU 8-12 GB, QLoRA 4-bit:
  python scripts/train_qwen_lora.py --model Qwen/Qwen2.5-3B-Instruct --train data/sft/train.jsonl \
      --dev data/sft/dev.jsonl --out runs/qwen3b-lora --4bit
  # CPU / thử nhanh:
  python scripts/train_qwen_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 30 --bf16 --out runs/smoke

License: Qwen2.5-3B = Qwen Research License (phi thương mại); 0.5B/1.5B/7B = Apache-2.0.
"""
import argparse, json, math, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train", default="data/sft/train.jsonl"); ap.add_argument("--dev")
ap.add_argument("--out", default="runs/qwen-lora")
ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=16); ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--4bit", dest="four_bit", action="store_true"); ap.add_argument("--bf16", action="store_true")
ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=200); ap.add_argument("--max-dev", type=int, default=300)
ap.add_argument("--threads", type=int, default=0, help="số luồng CPU (0 = tự động theo hạn mức container)")
a = ap.parse_args()

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.manual_seed(a.seed)
if not torch.cuda.is_available():  # tránh tạo quá nhiều luồng trong container bị giới hạn CPU
    import os
    n = a.threads or len(os.sched_getaffinity(0))
    if not a.threads:
        try:
            q, per = open("/sys/fs/cgroup/cpu.max").read().split()
            n = min(n, max(1, int(q) // int(per))) if q != "max" else n
        except Exception:
            pass
    torch.set_num_threads(n)
tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.pad_token or tok.eos_token
cuda = torch.cuda.is_available()
if a.dtype != "auto":
    cdt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
elif cuda:
    cdt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4 -> fp16
else:
    cdt = torch.bfloat16 if a.bf16 else torch.float32
use_scaler = cuda and cdt == torch.float16
print(f"dtype tính toán: {cdt}, GradScaler: {use_scaler}", flush=True)
if a.four_bit:
    from transformers import BitsAndBytesConfig
    kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=cdt), "device_map": "auto"}
else:
    kw = {"dtype": cdt}
model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
if a.four_bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    if cuda: model.to("cuda")
model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
dev = next(model.parameters()).device


def encode(ex):
    m = ex["messages"]
    prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(m[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"][: a.max_len // 2]
    if len(p_ids) + len(t_ids) > a.max_len:  # cắt prompt từ bên trái, luôn giữ phần đích
        p_ids = p_ids[-(a.max_len - len(t_ids)):]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


def load(path):
    rows = [encode(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if any(x != -100 for x in r[1])]
    if not rows:
        raise SystemExit(f"Không có mẫu hợp lệ trong {path}")
    print(f"{path}: {len(rows)} mẫu, dài nhất {max(len(r[0]) for r in rows)} token", flush=True)
    return rows


def collate(b):
    m = max(len(x[0]) for x in b)
    ids = torch.full((len(b), m), tok.pad_token_id); lab = torch.full((len(b), m), -100)
    att = torch.zeros((len(b), m), dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


tr = DataLoader(load(a.train), batch_size=a.batch, shuffle=True, collate_fn=collate)
dv = DataLoader(load(a.dev)[: a.max_dev], batch_size=a.batch, collate_fn=collate) if a.dev and Path(a.dev).exists() else None
steps = a.max_steps or math.ceil(len(tr) * a.epochs / a.grad_accum)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
sch = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
for p_ in model.parameters():  # tham số LoRA giữ fp32 để cập nhật ổn định
    if p_.requires_grad:
        p_.data = p_.data.float()
scaler = torch.amp.GradScaler("cuda") if use_scaler else None
print(f"Số bước tối ưu: {steps}", flush=True)


@torch.no_grad()
def evaluate():
    model.eval(); tot = n = 0
    for b in dv:
        with torch.autocast("cuda", dtype=cdt, enabled=cuda):
            tot += model(**{k: v.to(dev) for k, v in b.items()}).loss.item()
        n += 1
    model.train(); return tot / max(1, n)


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model.train(); step = micro = 0; best = float("inf"); t0 = time.time(); log = []; acc = 0.0
while step < steps:
    for b in tr:
        with torch.autocast("cuda", dtype=cdt, enabled=cuda):
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss / a.grad_accum
        (scaler.scale(loss) if scaler else loss).backward(); micro += 1; acc += loss.item()
        if micro % a.grad_accum:
            continue
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        sch.step(); opt.zero_grad(); step += 1
        rec = {"step": step, "loss": round(acc, 4), "lr": sch.get_last_lr()[0], "sec": round(time.time() - t0, 1)}; acc = 0.0
        if dv and (step % a.eval_every == 0 or step == steps):
            rec["dev_loss"] = evaluate()
            if rec["dev_loss"] < best:
                best = rec["dev_loss"]; model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
        log.append(rec); print(json.dumps(rec), flush=True)
        if step >= steps:
            break
model.save_pretrained(out / "last"); tok.save_pretrained(out / "last")
json.dump({"args": vars(a), "log": log, "best_dev_loss": best}, open(out / "train_log.json", "w"), indent=2)
print("Đã lưu adapter vào", out)
