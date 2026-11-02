"""Từ điển địa phương / lóng / viết tắt / chêm tiếng Anh và gợi ý ngữ cảnh cho prompt."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .text_norm import normalize_for_eval, strip_diacritics

DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "vi_lexicon.json"


@dataclass
class Lexicon:
    dialect: dict[str, dict[str, str]] = field(default_factory=dict)
    slang: dict[str, str] = field(default_factory=dict)
    abbrev: dict[str, str] = field(default_factory=dict)
    code_switch: dict[str, list[str]] = field(default_factory=dict)
    confusable_pairs: list[list[str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Lexicon":
        d = json.loads(Path(path or DEFAULT_LEXICON).read_text(encoding="utf-8"))
        return cls(d.get("dialect", {}), d.get("slang", {}), d.get("abbrev", {}),
                   d.get("code_switch", {}), d.get("confusable_pairs", []))

    def protected_terms(self) -> set[str]:
        terms = set(self.slang) | set(self.abbrev) | set(self.code_switch)
        for reg in self.dialect.values():
            terms |= set(reg)
        return {t for t in terms if t}

    def phonetic_index(self) -> dict[str, str]:
        """Cách ASR phiên âm (chuẩn hoá) -> từ tiếng Anh gốc."""
        return {normalize_for_eval(v): en for en, vs in self.code_switch.items() for v in vs}

    def find_hints(self, text: str, max_hints: int = 12) -> list[str]:
        norm = f" {normalize_for_eval(text)} "
        hints: list[str] = []
        for region, mp in self.dialect.items():
            for w, std in mp.items():
                if w != std and f" {normalize_for_eval(w)} " in norm:
                    hints.append(f'"{w}" là từ địa phương ({region}) = "{std}" → GIỮ NGUYÊN "{w}"')
        for w, mean in self.slang.items():
            if f" {normalize_for_eval(w)} " in norm:
                hints.append(f'"{w}" là tiếng lóng/khẩu ngữ (= "{mean}") → GIỮ NGUYÊN')
        for w, full in self.abbrev.items():
            if f" {normalize_for_eval(w)} " in norm:
                hints.append(f'"{w}" là viết tắt của "{full}" → giữ dạng viết tắt')
        for phon, en in self.phonetic_index().items():
            if len(phon) >= 3 and f" {phon} " in norm:
                hints.append(f'"{phon}" có thể là từ tiếng Anh "{en}" bị phiên âm → viết lại "{en}" nếu hợp ngữ cảnh')
        seen, out = set(), []
        for h in hints:
            if h not in seen:
                seen.add(h); out.append(h)
        return out[:max_hints]
