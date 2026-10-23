"""Bước 7a: phụ đề karaoke vẽ thẳng lên từng frame (ffmpeg Homebrew không có libass/drawtext).

Mỗi lần hiện 1 cụm vài từ, từ đang đọc tô vàng. Khối chữ nằm dưới hình vẽ (hình ở 10–72% chiều cao),
trên vùng chữ của TikTok (đáy ~20%), chừa lề 2 bên cho cột nút like/share.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import align

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WHITE, YELLOW, OUTLINE = (255, 255, 255), (255, 214, 0), (25, 25, 25)


def wrap(texts, font, maxw):
    """Xếp các từ thành dòng không rộng quá maxw px → [[(chỉ số, từ, độ rộng), ...], ...]."""
    space = font.getlength(" ")
    lines, cur, curw = [], [], 0.0
    for i, t in enumerate(texts):
        tw = font.getlength(t)
        if cur and curw + space + tw > maxw:
            lines.append(cur)
            cur, curw = [], 0.0
        curw += (space if cur else 0) + tw
        cur.append((i, t, tw))
    lines.append(cur)
    return lines


class Karaoke:
    def __init__(self, words, W, H, size=76, center_y=0.765, left=0.12, right=0.88, max_words=4):
        self.chunks = align.chunk_words(words, max_words)
        self.font = ImageFont.truetype(FONT, round(size * W / 1080))
        self.stroke = max(2, round(self.font.size * 0.09))
        self.x0, self.maxw, self.cy = round(W * left), round(W * (right - left)), round(H * center_y)
        # cụm i hiện từ lúc bắt đầu nói tới lúc cụm sau bắt đầu (cụm cuối giữ tới hết cảnh)
        self.spans = [(c[0]["start"], self.chunks[i + 1][0]["start"] if i + 1 < len(self.chunks) else float("inf"))
                      for i, c in enumerate(self.chunks)]
        self.cache = {}

    def _patch(self, chunk, active):
        """Ảnh BGR + alpha của 1 cụm, từ thứ `active` tô vàng; tự xuống dòng trong khung an toàn."""
        space = self.font.getlength(" ")
        lines = wrap([w["word"] for w in chunk], self.font, self.maxw)
        asc, desc = self.font.getmetrics()
        lh = asc + desc + self.stroke
        img = Image.new("RGBA", (self.maxw + 2 * self.stroke, lh * len(lines) + 2 * self.stroke), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for li, line in enumerate(lines):
            x = self.stroke + (self.maxw - sum(ww for *_, ww in line) - space * (len(line) - 1)) / 2
            for i, text, ww in line:
                d.text((x, self.stroke + li * lh), text, font=self.font, fill=YELLOW if i == active else WHITE,
                       stroke_width=self.stroke, stroke_fill=OUTLINE)
                x += ww + space
        a = np.asarray(img).astype(np.float32)
        return a[:, :, 2::-1], a[:, :, 3:] / 255

    def draw(self, frame, t):
        ci = next((i for i, (s, e) in enumerate(self.spans) if s <= t < e), None)
        if ci is None:
            return
        chunk = self.chunks[ci]
        wi = max(i for i, w in enumerate(chunk) if w["start"] <= t)
        if (ci, wi) not in self.cache:
            self.cache[(ci, wi)] = self._patch(chunk, wi)
        bgr, a = self.cache[(ci, wi)]
        h, w = a.shape[:2]
        x, y = self.x0 - self.stroke, self.cy - h // 2
        roi = frame[y:y + h, x:x + w]
        roi[:] = (roi * (1 - a) + bgr * a).astype(np.uint8)
