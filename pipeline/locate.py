"""Định vị từng vật thể trong ảnh bằng phần nhìn ảnh của Qwen3.6 (mlx-vlm) → khung để tách nét theo vật thể.

Cùng model với bước viết kịch bản (đã có sẵn trọng số vision), không phải tải thêm.
Nạp ~13s, ~6s/ảnh, RAM ~19GB → chạy trong tiến trình con riêng (make_video.py làm vậy).

Chạy: python -m pipeline.locate image.png "glass of water" "alarm clock"
"""
import argparse
import gc
import json
import re

from pipeline.script_llm import LLM_MODEL

PROMPT = ("Locate each object in the image and output JSON list "
          '[{{"label": ..., "bbox_2d": [x1, y1, x2, y2]}}] with coordinates in 0-1000. Objects: {objects}')


def parse_boxes(text, objects):
    """Câu trả lời của model → khung [x0, y0, x1, y1] (tỉ lệ 0–1) cho từng vật thể theo thứ tự, None nếu không thấy.
    Khớp theo nhãn; nhãn lệch mà số khung bằng số vật thể thì khớp theo thứ tự."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    items = [it for it in items if isinstance(it, dict) and len(it.get("bbox_2d") or []) == 4]
    by_label = {str(it.get("label", "")).strip().lower(): it for it in items}
    picked = [by_label.get(o.strip().lower()) for o in objects]
    if None in picked and len(items) == len(objects):
        picked = items
    boxes = []
    for it in picked:
        b = [min(1.0, max(0.0, float(v) / 1000)) for v in it["bbox_2d"]] if it else None
        boxes.append(b if b and b[2] > b[0] and b[3] > b[1] else None)
    return boxes


class Locator:
    def __init__(self, model=LLM_MODEL):
        from mlx_vlm import load

        self.model, self.proc = load(model)

    def find(self, image, objects):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.proc, self.model.config, PROMPT.format(objects=", ".join(objects)),
                                     num_images=1, enable_thinking=False)
        out = generate(self.model, self.proc, prompt, [str(image)], max_tokens=60 * len(objects) + 50,
                       temperature=0.0, verbose=False)
        return parse_boxes(getattr(out, "text", out), objects)

    def close(self):
        import mlx.core as mx

        del self.model, self.proc
        gc.collect()
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("objects", nargs="+", help="tên vật thể bằng tiếng Anh")
    args = ap.parse_args()
    for o, b in zip(args.objects, Locator().find(args.image, args.objects)):
        print(f"{o}: {b}")


if __name__ == "__main__":
    main()
