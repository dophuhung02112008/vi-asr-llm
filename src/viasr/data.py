"""Nạp dữ liệu về manifest JSONL chung và chia tập THEO NGƯỜI NÓI.

Dòng manifest: {"id","audio","text","speaker","source","split"[,"region","context"]}"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from .text_norm import nfc


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows: Iterable[dict], path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n


def load_vivos(root) -> Iterator[dict]:
    """VIVOS (AILab, ĐH KHTN TP.HCM): root/{train,test}/prompts.txt + waves/<SPK>/<ID>.wav."""
    root = Path(root)
    for split in ("train", "test"):
        p = root / split / "prompts.txt"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            spk = uid.split("_")[0]
            yield {"id": uid, "audio": str(root / split / "waves" / spk / f"{uid}.wav"), "text": nfc(text.strip()).lower(),
                   "speaker": f"vivos_{spk}", "source": "vivos", "orig_split": split}


def load_common_voice(root, tsv: str = "validated.tsv") -> Iterator[dict]:
    """Mozilla Common Voice tiếng Việt: root/validated.tsv + clips/*.mp3; speaker = client_id."""
    root = Path(root)
    with open(root / tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            yield {"id": r["path"].rsplit(".", 1)[0], "audio": str(root / "clips" / r["path"]),
                   "text": nfc(r["sentence"]), "speaker": f"cv_{r['client_id'][:16]}", "source": "common_voice"}


def load_hf_dataset(name="google/fleurs", config="vi_vn", split="train", text_field="transcription",
                    speaker_field: str | None = None) -> Iterator[dict]:
    """Hugging Face datasets (vd FLEURS vi_vn). Không có speaker id thì mỗi câu là 1 nhóm."""
    from datasets import load_dataset
    for i, ex in enumerate(load_dataset(name, config, split=split)):
        a = ex["audio"]
        spk = str(ex.get(speaker_field)) if speaker_field else str(ex.get("id", i))
        tag = name.split("/")[-1]
        yield {"id": f"{tag}_{split}_{i}", "audio": a.get("path"), "audio_array": a["array"],
               "sampling_rate": a["sampling_rate"], "text": nfc(ex[text_field]), "speaker": f"{tag}_{spk}",
               "source": name, "orig_split": split}


def load_custom_csv(path) -> Iterator[dict]:
    """CSV cột: audio,text,speaker[,region,context] — dữ liệu giọng địa phương tự thu."""
    base = Path(path).parent
    with open(path, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            a = r["audio"]
            yield {"id": r.get("id") or f"custom_{i}", "audio": a if Path(a).is_absolute() else str(base / a),
                   "text": nfc(r["text"]), "speaker": r.get("speaker") or f"custom_spk{i}",
                   "region": r.get("region", ""), "context": r.get("context", ""), "source": "custom"}


def speaker_split(rows: list[dict], dev: float = 0.1, test: float = 0.1, key: str = "speaker",
                  salt: str = "viasr") -> list[dict]:
    """Chia theo hash(người nói): 1 người nói chỉ ở 1 tập. Chạy TRƯỚC mọi bước sinh lỗi/augment."""
    out = []
    for r in rows:
        r = dict(r)
        if r.get("orig_split") == "test":
            r["split"] = "test"
        else:
            h = int(hashlib.sha1(f"{salt}:{r.get(key, r['id'])}".encode()).hexdigest(), 16) % 10000 / 10000
            r["split"] = "test" if h < test else ("dev" if h < test + dev else "train")
        out.append(r)
    return out


def check_leakage(rows: list[dict], key: str = "speaker") -> dict:
    spk, txt = {}, {}
    for r in rows:
        spk.setdefault(r.get(key), set()).add(r["split"])
        txt.setdefault(r["text"], set()).add(r["split"])
    return {"speakers_in_multiple_splits": {k: sorted(v) for k, v in spk.items() if len(v) > 1},
            "texts_shared_across_splits": sum(len(v) > 1 for v in txt.values())}
