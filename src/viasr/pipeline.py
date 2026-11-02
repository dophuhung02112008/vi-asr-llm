"""audio → ASR (1-best hoặc N-best) → Qwen hậu xử lý bảo thủ."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .corrector import ConservativeCorrector


@dataclass
class PipelineOutput:
    audio: str | None
    asr_text: str
    final_text: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    nbest: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    asr_seconds: float = 0.0
    llm_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


class SpeechPipeline:
    def __init__(self, asr, corrector: ConservativeCorrector | None, use_nbest: bool = False, n: int = 5):
        self.asr, self.corrector, self.use_nbest, self.n = asr, corrector, use_nbest, n

    def run(self, audio, context: str | None = None) -> PipelineOutput:
        name = audio if isinstance(audio, str) else None
        t0 = time.perf_counter()
        nb = self.asr.nbest(audio, n=self.n) if self.use_nbest and hasattr(self.asr, "nbest") else []
        asr_text = nb[0] if nb else self.asr.transcribe(audio).text
        t1 = time.perf_counter()
        if self.corrector is None:
            return PipelineOutput(name, asr_text, asr_text, False, ["llm_disabled"], nb, [], t1 - t0)
        res = self.corrector.correct_nbest(nb, context) if nb else self.corrector.correct(asr_text, context)
        return PipelineOutput(name, asr_text, res.corrected, res.accepted, res.reasons, nb, res.hints,
                              t1 - t0, time.perf_counter() - t1)
