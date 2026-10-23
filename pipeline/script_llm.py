"""Bước 1: LLM local (Qwen3.6-35B-A3B qua mlx-lm) viết kịch bản video TikTok từ 1 chủ đề.

Đầu ra script.json:
{
  "topic": "...", "title": "...",
  "scenes": [
    {"narration": "lời đọc tiếng Việt của cảnh",
     "image_subject": "mô tả hình doodle bằng tiếng Anh",
     "elements": [{"cue": "cụm từ trong narration — lúc nói tới thì bắt đầu vẽ vật thể",
                   "object": "tên vật thể tiếng Anh, có trong image_subject — để định vị trong ảnh"}],
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
CUE_GAP = 4  # số từ tối thiểu giữa 2 cue liên tiếp (~1.5s đọc) — sát hơn thì vật sau vẽ trễ lời

# 1 âm tiết tiếng Việt: phụ âm đầu + 1–3 nguyên âm + phụ âm cuối
_V = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
SYLLABLE = re.compile(rf"(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnprstvx])?[{_V}]{{1,3}}(ch|ng|nh|[cmnpt])?")

SYSTEM = """Bạn là biên kịch video TikTok dạng vẽ tay trên bảng trắng (whiteboard animation), giọng đọc tiếng Việt.
Viết kịch bản ngắn gọn, dễ hiểu, lôi cuốn, đúng sự thật, cho người Việt xem trên điện thoại."""

INSTRUCTIONS = """Chủ đề: {topic}

Yêu cầu:
- {n_min} đến {n_max} cảnh, tổng {w_min}–{w_max} từ (khoảng {sec_min}–{sec_max} giây đọc).
- Cảnh 1 là câu "hook" gây tò mò trong 3 giây đầu (câu hỏi hoặc sự thật bất ngờ).
- Cảnh cuối chốt ý chính + kêu gọi hành động ngắn (theo dõi, lưu lại, thử ngay...).
- Mỗi cảnh: "narration" 1–2 câu, 12–35 từ, văn nói tự nhiên, không emoji, không hashtag, không ký hiệu lạ,
  không chèn từ tiếng Anh.
- Mỗi cảnh: "image_subject" bằng TIẾNG ANH, mô tả MỘT hình minh hoạ đơn giản vẽ được bằng bút dạ
  (tối đa 3 vật thể cụ thể, có thể là người hoặc đồ vật; KHÔNG có chữ, số, biểu đồ, giao diện, bản đồ).
  Ví dụ tốt: "a smiling woman drinking a glass of water next to an alarm clock".
- Mỗi cảnh: "elements" liệt kê ĐỦ MỌI vật thể được vẽ trong image_subject (1–3, kể cả người), không bỏ sót,
  không thêm thứ không vẽ được (nơi chốn, cảm xúc), theo thứ tự được nhắc trong narration. Mỗi vật thể:
  "cue" là cụm 1–4 từ COPY NGUYÊN VĂN từ narration của chính cảnh đó, nói tới vật thể này
  (vật thể sẽ được vẽ đúng lúc giọng đọc nói tới cụm này); "object" là tên vật thể bằng tiếng Anh,
  COPY NGUYÊN VĂN từ image_subject (vd. "woman", "glass of water", "alarm clock"). Các cue khác nhau,
  cue sau cách cue trước ít nhất {cue_gap} từ (để kịp vẽ); câu ngắn thì ít vật thể thôi.
- Mỗi cảnh: "on_screen_text" là chữ viết lên bảng, 1–5 từ tiếng Việt tóm ý chính hoặc con số đáng nhớ
  (vd. "Ngủ đủ 8 tiếng", "25°C"), KHÔNG chép nguyên câu narration; để "" nếu cảnh không cần chữ.
- "title": tiêu đề video ngắn, dưới 60 ký tự.

Chỉ trả về JSON hợp lệ, không giải thích, theo đúng dạng:
{{"title": "...", "scenes": [{{"narration": "...", "image_subject": "...",
  "elements": [{{"cue": "...", "object": "..."}}], "on_screen_text": "..."}}]}}"""


def _norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", "", s).split()


def phrase_index(text, phrase):
    """Vị trí (theo từ) lần đầu cụm `phrase` xuất hiện trong text, -1 nếu không có."""
    t, p = _norm(text), _norm(phrase)
    return next((i for i in range(len(t) - len(p) + 1) if p and t[i:i + len(p)] == p), -1)


def contains_phrase(text, phrase):
    return phrase_index(text, phrase) >= 0


def glued_words(text):
    """Từ có dấu tiếng Việt nhưng không phải 1 âm tiết — thường là LLM dính tiếng Anh vào (vd. "bộrefresh")."""
    return [w for w in text.split()
            if re.search(r"[À-ỹđĐ]", w) and not SYLLABLE.fullmatch(re.sub(r"[^\w]", "", w.lower()))]


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
        text = str(sc.get("on_screen_text", "")).strip()
        sc["narration"], sc["image_subject"], sc["on_screen_text"] = nar, subj, text
        els = sc.get("elements")
        if not isinstance(els, list) or not 1 <= len(els) <= 3:
            errors.append(f"cảnh {i}: 'elements' cần 1–3 vật thể")
            els = []
        els = [{"cue": str(e.get("cue", "")).strip(), "object": str(e.get("object", "")).strip()}
               for e in els if isinstance(e, dict)]
        sc["elements"] = sorted(els, key=lambda e: phrase_index(nar, e["cue"]))  # theo thứ tự được nói
        n = len(nar.split())
        words += n
        if not 8 <= n <= 45:
            errors.append(f"cảnh {i}: narration có {n} từ, cần 12–35")
        if glued := glued_words(nar):
            errors.append(f"cảnh {i}: narration có từ dính tiếng Anh {glued}, viết lại bằng tiếng Việt")
        if not subj or re.search(r"[À-ỹđĐ]", subj):
            errors.append(f"cảnh {i}: image_subject phải là tiếng Anh, không rỗng")
        if len(text.split()) > 6 or len(text) > 32:
            errors.append(f"cảnh {i}: on_screen_text '{text}' quá dài, tối đa 5 từ")
        for e in els:
            if not contains_phrase(nar, e["cue"]):
                errors.append(f"cảnh {i}: cue '{e['cue']}' không có nguyên văn trong narration")
            if not contains_phrase(subj, e["object"]):
                errors.append(f"cảnh {i}: object '{e['object']}' không có nguyên văn trong image_subject")
        for a, b in zip(sc["elements"], sc["elements"][1:]):
            ia, ib = phrase_index(nar, a["cue"]), phrase_index(nar, b["cue"])
            if ia >= 0 and ib >= 0 and ib - ia < CUE_GAP:
                errors.append(f"cảnh {i}: cue '{a['cue']}' và '{b['cue']}' cách nhau {ib - ia} từ, cần ≥{CUE_GAP}"
                              " — bớt vật thể hoặc chọn cue xa nhau hơn")
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
                                                            w_max=w_max, sec_min=seconds[0], sec_max=seconds[1],
                                                            cue_gap=CUE_GAP)},
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
        cues = ", ".join(f"{e['cue']} → {e['object']}" for e in sc["elements"])
        print(f"[{i}] {sc['narration']}\n    🖼 {sc['image_subject']}\n    ▶ {cues}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
