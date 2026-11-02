"""Prompt cho Qwen hậu xử lý ASR tiếng Việt (bảo thủ: chỉ sửa lỗi nghe nhầm)."""
from __future__ import annotations

SYSTEM_PROMPT = """Bạn là bộ hậu xử lý cho hệ thống nhận dạng giọng nói (ASR) tiếng Việt.
Đầu vào là bản nháp do máy nghe, có thể sai do: nhầm dấu thanh (hỏi/ngã, sắc/nặng), nhầm phụ âm/vần
(s/x, tr/ch, d/gi/v, l/n, an/ang...), nuốt âm, dính/tách âm tiết, và phiên âm sai từ tiếng Anh chêm vào
(ví dụ "đét lai" -> "deadline", "mít tinh" -> "meeting").

QUY TẮC BẮT BUỘC:
1. Chỉ sửa chỗ chắc chắn là lỗi nghe nhầm. Không viết lại cho hay hơn, không tóm tắt, không dịch, không thêm/bớt ý.
2. GIỮ NGUYÊN từ địa phương, tiếng lóng, viết tắt người nói thực sự dùng (vd "hông", "mô", "răng", "rứa", "chừ",
   "dìa", "khum", "ko", "HN") — đó KHÔNG phải lỗi, không đổi sang từ phổ thông.
3. Từ tiếng Anh bị phiên âm thì viết lại đúng chính tả tiếng Anh. Từ tiếng Anh đã đúng thì giữ nguyên.
4. Giữ nguyên số, tên riêng, đơn vị nếu không có bằng chứng rõ ràng là sai. Giữ số lượng âm tiết gần như cũ.
5. Không chắc chắn thì giữ nguyên bản nháp.
6. Chỉ trả về đúng một dòng văn bản đã sửa, không giải thích."""

SELECT_PROMPT = """Bạn là bộ chọn kết quả cho hệ thống nhận dạng giọng nói tiếng Việt.
Cho nhiều giả thuyết (N-best) của CÙNG một đoạn âm thanh. Chọn giả thuyết đúng nhất về chính tả, dấu thanh và
ngữ cảnh. Từ địa phương, tiếng lóng, chêm tiếng Anh là hợp lệ. Chỉ trả về SỐ THỨ TỰ (ví dụ: 2)."""


def build_user_prompt(hypothesis: str, context: str | None = None, hints: list[str] | None = None) -> str:
    parts = []
    if context:
        parts.append(f"Ngữ cảnh (chỉ tham khảo, không ép bản nháp khớp theo): {context}")
    if hints:
        parts.append("Gợi ý từ điển:\n" + "\n".join(f"- {h}" for h in hints))
    parts.append(f"Bản nháp ASR: {hypothesis}")
    parts.append("Bản đã sửa:")
    return "\n\n".join(parts)


def build_select_prompt(candidates: list[str], context: str | None = None) -> str:
    head = f"Ngữ cảnh: {context}\n\n" if context else ""
    return head + "Các giả thuyết:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates)) + "\n\nSố thứ tự đúng nhất:"


def chat_messages(hypothesis: str, context: str | None = None, hints: list[str] | None = None,
                  target: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(hypothesis, context, hints)}]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs
