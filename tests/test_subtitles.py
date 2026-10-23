"""Xuống dòng và hiện đúng cụm phụ đề theo thời gian."""
import numpy as np
from PIL import ImageFont

from pipeline import subtitles


def test_wrap_khong_vuot_be_rong():
    font = ImageFont.truetype(subtitles.FONT, 76)
    texts = "Mỗi sáng uống một cốc nước ấm giúp cơ thể tỉnh táo".split()
    lines = subtitles.wrap(texts, font, 500)
    assert len(lines) > 1
    assert [t for line in lines for _, t, _ in line] == texts
    space = font.getlength(" ")
    assert all(sum(w for *_, w in line) + space * (len(line) - 1) <= 500 for line in lines)


def test_karaoke_cum_dung_thoi_gian():
    words = [{"word": w, "start": float(i), "end": i + 0.9} for i, w in enumerate("a b c d e f".split())]
    k = subtitles.Karaoke(words, 1080, 1920, max_words=4)
    assert k.spans == [(0.0, 4.0), (4.0, float("inf"))]
    frame = np.full((1920, 1080, 3), 200, np.uint8)
    k.draw(frame, -0.1)  # trước khi nói: không vẽ gì
    assert (frame == 200).all()
    k.draw(frame, 1.5)
    assert (frame != 200).any() and (0, 1) in k.cache  # cụm đầu, từ "b" đang đọc
