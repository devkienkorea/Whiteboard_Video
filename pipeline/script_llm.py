"""Bước 1: LLM local (Qwen3.6-35B-A3B qua mlx-lm) viết kịch bản video TikTok từ 1 chủ đề.

Đầu ra script.json:
{
  "topic": "...", "title": "...",
  "scenes": [
    {"narration": "lời đọc tiếng Việt của cảnh",
     "image_subject": "mô tả hình doodle bằng tiếng Anh",
     "cue": "cụm từ trong narration — lúc nói tới thì bắt đầu vẽ",
     "on_screen_text": "chữ ngắn viết tay đầu cảnh (từ khoá/số liệu), có thể rỗng"}
  ]
}

Chạy: python -m pipeline.script_llm "Tại sao nên uống nước buổi sáng" --out jobs/nuoc/script.json
"""
import argparse
import gc
import json
import re
import time
import unicodedata
from pathlib import Path

LLM_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"

SYSTEM = """Bạn là biên kịch video TikTok dạng vẽ tay trên bảng trắng (whiteboard animation), giọng đọc tiếng Việt.
Viết kịch bản ngắn gọn, dễ hiểu, lôi cuốn, đúng sự thật, cho người Việt xem trên điện thoại."""

INSTRUCTIONS = """Chủ đề: {topic}

Yêu cầu:
- {n_min} đến {n_max} cảnh, tổng {w_min}–{w_max} từ (khoảng {sec_min}–{sec_max} giây đọc).
- Cảnh 1 là câu "hook" gây tò mò trong 3 giây đầu (câu hỏi hoặc sự thật bất ngờ).
- Cảnh cuối chốt ý chính + kêu gọi hành động ngắn (theo dõi, lưu lại, thử ngay...).
- Mỗi cảnh: "narration" 1–2 câu, 12–35 từ, văn nói tự nhiên, không emoji, không hashtag, không ký hiệu lạ.
- Mỗi cảnh: "image_subject" bằng TIẾNG ANH, mô tả MỘT hình minh hoạ đơn giản vẽ được bằng bút dạ
  (tối đa 3 vật thể cụ thể, có thể là người hoặc đồ vật; KHÔNG có chữ, số, biểu đồ, giao diện, bản đồ).
  Ví dụ tốt: "a smiling woman drinking a glass of water next to an alarm clock".
- Mỗi cảnh: "cue" là cụm 1–4 từ COPY NGUYÊN VĂN từ narration của chính cảnh đó, gọi tên thứ được vẽ
  (hình sẽ bắt đầu được vẽ đúng lúc giọng đọc nói tới cụm này).
- Mỗi cảnh: "on_screen_text" là chữ viết lên bảng, 1–5 từ tiếng Việt tóm ý chính hoặc con số đáng nhớ
  (vd. "Ngủ đủ 8 tiếng", "25°C"), KHÔNG chép nguyên câu narration; để "" nếu cảnh không cần chữ.
- "title": tiêu đề video ngắn, dưới 60 ký tự.

Chỉ trả về JSON hợp lệ, không giải thích, theo đúng dạng:
{{"title": "...", "scenes": [{{"narration": "...", "image_subject": "...", "cue": "...", "on_screen_text": "..."}}]}}"""


def _norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", "", s).split()


def contains_phrase(text, phrase):
    t, p = _norm(text), _norm(phrase)
    return bool(p) and any(t[i:i + len(p)] == p for i in range(len(t) - len(p) + 1))


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("không thấy JSON trong câu trả lời")
    return json.loads(m.group(0))


def validate(script, n_min, n_max, w_min, w_max):
    """Trả về danh sách lỗi (rỗng = hợp lệ). Tự sửa các lỗi nhỏ tại chỗ."""
    errors = []
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not (n_min <= len(scenes) <= n_max):
        return [f"cần {n_min}–{n_max} cảnh trong 'scenes'"]
    words = 0
    for i, sc in enumerate(scenes, 1):
        nar = str(sc.get("narration", "")).strip()
        subj = str(sc.get("image_subject", "")).strip()
        cue = str(sc.get("cue", "")).strip()
        text = str(sc.get("on_screen_text", "")).strip()
        sc["narration"], sc["image_subject"], sc["cue"], sc["on_screen_text"] = nar, subj, cue, text
        n = len(nar.split())
        words += n
        if not 8 <= n <= 45:
            errors.append(f"cảnh {i}: narration có {n} từ, cần 12–35")
        if not subj or re.search(r"[À-ỹđĐ]", subj):
            errors.append(f"cảnh {i}: image_subject phải là tiếng Anh, không rỗng")
        if len(text.split()) > 6 or len(text) > 32:
            errors.append(f"cảnh {i}: on_screen_text '{text}' quá dài, tối đa 5 từ")
        if not contains_phrase(nar, cue):
            errors.append(f"cảnh {i}: cue '{cue}' không có nguyên văn trong narration")
    if not (w_min * 0.8 <= words <= w_max * 1.2):
        errors.append(f"tổng {words} từ, cần {w_min}–{w_max}")
    return errors


class ScriptWriter:
    def __init__(self, model=LLM_MODEL):
        from mlx_lm import load

        self.model, self.tok = load(model)

    def ask(self, messages, max_tokens=2000, temp=0.7, thinking=False):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=thinking)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temp, top_p=0.9))

    def write(self, topic, seconds=(45, 70), retries=3, thinking=False):
        # ~2.6 từ/giây với giọng VieNeu
        w_min, w_max = int(seconds[0] * 2.6), int(seconds[1] * 2.6)
        n_min, n_max = 5, 8
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTIONS.format(topic=topic, n_min=n_min, n_max=n_max, w_min=w_min,
                                                            w_max=w_max, sec_min=seconds[0], sec_max=seconds[1])},
        ]
        last_err = None
        for attempt in range(1, retries + 1):
            t0 = time.time()
            raw = self.ask(messages, thinking=thinking)
            try:
                script = extract_json(raw)
                errors = validate(script, n_min, n_max, w_min, w_max)
            except (ValueError, json.JSONDecodeError) as e:
                errors = [f"JSON lỗi: {e}"]
            print(f"  lần {attempt}: {time.time() - t0:.0f}s, {'OK' if not errors else '; '.join(errors)}")
            if not errors:
                script["topic"] = topic
                return script
            last_err = errors
            # gửi lỗi lại cho model sửa
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "Sửa các lỗi sau rồi trả lại TOÀN BỘ JSON:\n- " + "\n- ".join(errors)}]
        raise RuntimeError(f"LLM không tạo được kịch bản hợp lệ: {last_err}")

    def close(self):
        """Trả bộ nhớ GPU của MLX. Lưu ý: tiến trình vẫn giữ ~10GB tới khi đối tượng bị xoá hẳn —
        muốn chắc chắn thì chạy trong tiến trình riêng (make_video.py làm vậy)."""
        import mlx.core as mx

        del self.model, self.tok
        gc.collect()
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", default="45,70", help="độ dài mong muốn, vd. 30,45")
    ap.add_argument("--thinking", action="store_true", help="cho model suy nghĩ trước (chậm hơn)")
    args = ap.parse_args()
    t0 = time.time()
    writer = ScriptWriter()
    print(f"Nạp LLM: {time.time() - t0:.0f}s")
    script = writer.write(args.topic, tuple(int(x) for x in args.seconds.split(",")), thinking=args.thinking)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(script, ensure_ascii=False, indent=1))
    for i, sc in enumerate(script["scenes"], 1):
        print(f"[{i}] {sc['narration']}\n    🖼 {sc['image_subject']}\n    ▶ cue: {sc['cue']}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
