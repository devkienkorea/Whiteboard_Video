"""Tách nét theo khung vật thể (regions) — trường hợp gom theo khoảng cách bị dính thành 1."""
import cv2
import numpy as np

from pipeline import strokes


def two_shapes():
    """Hình vuông trái (tô đỏ) + hình tròn phải, nối bằng 1 đường mực → dính thành 1 cụm."""
    img = np.full((400, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 120), (300, 340), (0, 0, 220), -1)
    cv2.rectangle(img, (60, 120), (300, 340), (0, 0, 0), 8)
    cv2.circle(img, (600, 230), 120, (0, 0, 0), 8)
    cv2.line(img, (300, 230), (480, 230), (0, 0, 0), 8)
    return img


def owner_of(meta, e):
    xs = [x for s in meta["elements"][e]["strokes"] for x, _ in s]
    return np.median(xs)


def test_khong_co_khung_thi_dinh_thanh_1():
    _, _, _, meta = strokes.extract(two_shapes())
    assert sum(1 for el in meta["elements"] if el["strokes"]) == 1


def test_chia_theo_khung_dung_thu_tu():
    # khung tròn trước, vuông sau (thứ tự được nhắc), thêm 1 khung không có gì
    regions = [[0.55, 0.1, 0.95, 0.9], [0.05, 0.25, 0.4, 0.9], [0.0, 0.0, 0.02, 0.02]]
    color, _, labels, meta = strokes.extract(two_shapes(), regions=regions)
    assert len(meta["elements"]) == 3
    assert owner_of(meta, 0) > 540 > owner_of(meta, 1)  # canvas 1080 ngang: tròn bên phải, vuông bên trái
    assert meta["elements"][1]["has_color"] and not meta["elements"][0]["has_color"]
    assert meta["elements"][2]["strokes"] == []  # vẫn giữ chỗ để cue thứ 3 khớp đúng vật thể
    red = (color[:, :, 2] > 150) & (color[:, :, 0] < 80)
    assert (labels[red] == 2).mean() > 0.99  # màu tô theo đúng vật thể
