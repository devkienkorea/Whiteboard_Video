"""Bước 7b: ảnh bìa (cover) 1080x1920 cho TikTok: tiêu đề to + hình vẽ đã tô màu của 1 cảnh, trên nền giấy.

Lưới video trên trang cá nhân TikTok cắt bìa về giữa khung 3:4 (12.5–87.5% chiều cao)
→ tiêu đề đặt ở 14–33%, hình ở 36–80%, cả hai đều nằm trong vùng đó.

Chạy: python -m pipeline.cover jobs/<slug> [--scene 1]   → jobs/<slug>/cover.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import render, strokes, subtitles

INK = (30, 30, 30)
MARKER = (255, 214, 0)  # vàng bút dạ, cùng màu từ đang đọc trong phụ đề


def make(job, scene=0, W=1080, H=1920):
    """Ghép bìa từ tiêu đề trong script.json và image.png của cảnh `scene` (chỉ số từ 0)."""
    job = Path(job)
    title = json.loads((job / "script.json").read_text())["title"].upper()
    img = cv2.imread(str(job / f"scene_{scene + 1:02}" / "image.png"), cv2.IMREAD_UNCHANGED)

    paper = render.make_paper(W, H)
    drawing = strokes.fit_to_canvas(img, W, H, box=(0.08, 0.36, 0.84, 0.44))
    canvas = Image.fromarray(cv2.cvtColor((drawing.astype(np.uint16) * paper // 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    # tiêu đề: cỡ chữ lớn nhất mà vẫn vừa 3 dòng trong vùng 14–33%
    maxw, top, bottom = W * 0.84, H * 0.14, H * 0.33
    for size in range(round(W * 0.12), 20, -4):
        font = ImageFont.truetype(subtitles.FONT, size)
        lines = subtitles.wrap(title.split(), font, maxw)
        asc, desc = font.getmetrics()
        lh = (asc + desc) * 1.15
        if len(lines) <= 3 and lh * len(lines) <= bottom - top:
            break
    d = ImageDraw.Draw(canvas)
    space = font.getlength(" ")
    y = top + (bottom - top - lh * len(lines)) / 2
    for line in lines:
        width = sum(tw for *_, tw in line) + space * (len(line) - 1)
        x = (W - width) / 2
        # vệt bút dạ nằm nửa dưới dòng chữ, như được tô highlight bằng tay
        d.rounded_rectangle((x - size * 0.15, y + asc * 0.45, x + width + size * 0.15, y + asc + desc * 0.6),
                            radius=size * 0.2, fill=MARKER)
        d.text((x, y), " ".join(t for _, t, _ in line), font=font, fill=INK)
        y += lh
    out = job / "cover.jpg"
    canvas.save(out, quality=92)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", help="thư mục job (có script.json, scene_XX/image.png)")
    ap.add_argument("--scene", type=int, default=1, help="lấy hình của cảnh số mấy (từ 1)")
    args = ap.parse_args()
    print(make(args.job, args.scene - 1))


if __name__ == "__main__":
    main()
