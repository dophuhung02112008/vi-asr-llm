"""Lớp ASR: faster-whisper (CTranslate2) hoặc checkpoint HF như vinai/PhoWhisper.

Lưu ý: `beam_size=5` của faster-whisper.transcribe() KHÔNG trả 5 giả thuyết; dùng `nbest()`."""
from __future__ import annotations

from dataclasses import dataclass, field

SAMPLE_RATE = 16000


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class AsrResult:
    text: str
    segments: list[AsrSegment] = field(default_factory=list)
    language: str | None = "vi"


def load_audio(path: str, sr: int = SAMPLE_RATE):
    try:
        from faster_whisper.audio import decode_audio
        return decode_audio(path, sampling_rate=sr)
    except ImportError:
        import librosa
        return librosa.load(path, sr=sr, mono=True)[0]


class FasterWhisperASR:
    """model_size: 'small' | 'medium' | 'large-v3' | 'large-v3-turbo' | thư mục model CT2
    (PhoWhisper có thể chuyển sang CT2 bằng ct2-transformers-converter — xem docs/HUONG_DAN.md)."""

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "default",
                 language: str = "vi", initial_prompt: str | None = None):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language, self.initial_prompt = language, initial_prompt

    def transcribe(self, audio, beam_size: int = 5, vad: bool = True, **kw) -> AsrResult:
        segs, info = self.model.transcribe(audio, language=self.language, task="transcribe",
                                           beam_size=beam_size, vad_filter=vad,
                                           initial_prompt=self.initial_prompt,
                                           condition_on_previous_text=False, **kw)
        out = [AsrSegment(s.start, s.end, s.text.strip(), s.avg_logprob, s.no_speech_prob) for s in segs]
        return AsrResult(" ".join(s.text for s in out).strip(), out, info.language)

    def nbest(self, audio, n: int = 5, beam_size: int = 5, max_seconds: float = 30.0) -> list[str]:
        """N-best thật cho đoạn ≤30 s qua CTranslate2 `num_hypotheses` (thực nghiệm; API nội bộ có thể đổi)."""
        import numpy as np
        from faster_whisper.tokenizer import Tokenizer
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = np.asarray(audio, dtype=np.float32)[: int(max_seconds * SAMPLE_RATE)]
        feats = self.model.feature_extractor(audio)
        T = 3000
        feats = feats[:, :T] if feats.shape[1] >= T else np.pad(feats, ((0, 0), (0, T - feats.shape[1])))
        enc = self.model.encode(feats)
        tok = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual,
                        task="transcribe", language=self.language)
        res = self.model.model.generate(enc, [list(tok.sot_sequence) + [tok.no_timestamps]],
                                        beam_size=max(beam_size, n), num_hypotheses=n,
                                        return_scores=True, max_length=224, suppress_blank=True)
        seen, out = set(), []
        for seq in res[0].sequences_ids:
            t = tok.decode([i for i in seq if i < tok.eot]).strip()
            if t and t not in seen:
                seen.add(t); out.append(t)
        return out


class TransformersWhisperASR:
    """Chạy checkpoint HF, ví dụ vinai/PhoWhisper-small (Whisper fine-tune cho tiếng Việt)."""

    def __init__(self, model_name: str = "vinai/PhoWhisper-small", device: str | None = None):
        import torch
        from transformers import pipeline
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=dev, chunk_length_s=30)

    def transcribe(self, audio, **kw) -> AsrResult:
        return AsrResult(self.pipe(audio, **kw)["text"].strip())

    def nbest(self, audio, n: int = 5, **kw) -> list[str]:
        out = self.pipe(audio, generate_kwargs={"num_beams": max(n, 5), "num_return_sequences": n})
        return [o["text"].strip() for o in out] if isinstance(out, list) else [out["text"].strip()]


def make_asr(kind: str = "faster-whisper", **kw):
    if kind in ("faster-whisper", "fw", "whisper"):
        return FasterWhisperASR(**kw)
    if kind in ("hf", "phowhisper", "transformers"):
        return TransformersWhisperASR(**kw)
    raise ValueError(kind)
