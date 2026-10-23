"""Chấm điểm 1 ảnh có "dễ vẽ lại bằng bút" không, dựa trên kết quả strokes.extract.

Ảnh tốt cho whiteboard: nét đen dày đều, nền trắng sạch, ít chi tiết vụn, không đổ bóng.

Chạy: python -m pipeline.quality ảnh1.png ảnh2.png ...
"""
import argparse
import json

import cv2
import numpy as np

from pipeline import strokes


def edge_touch(img, band=4):
    """Tỉ lệ viền ảnh gốc có nội dung — hình bị cắt mất mép khi > ~0."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = strokes.whiten_background(img[:, :, :3])
    m = strokes.content_mask(img)
    border = np.concatenate([m[:band].ravel(), m[-band:].ravel(), m[:, :band].ravel(), m[:, -band:].ravel()])
    return float(border.mean())


def assess(img):
    cut = edge_touch(img)
    color, ink, labels, meta = strokes.extract(img)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    nonwhite = strokes.content_mask(color)
    ink_b = ink > 0
    area = max(1, int(nonwhite.sum()))

    all_strokes = [np.array(s) for el in meta["elements"] for s in el["strokes"]]
    lengths = np.array([np.sqrt((np.diff(s, axis=0) ** 2).sum(1)).sum() for s in all_strokes]) if all_strokes else np.zeros(0)

    # bao nhiêu % mực được cọ quét qua (phần còn lại sẽ "hiện bụp" lúc cuối)
    reach = np.zeros_like(ink)
    for s in all_strokes:
        cv2.polylines(reach, [s.astype(np.int32)], False, 255, meta["brush"])
    coverage = float(((reach > 0) & ink_b).sum() / max(1, ink_b.sum()))

    # vùng xám không bão hoà = đổ bóng / gạch chéo → vẽ lại sẽ bẩn
    gray = (val >= 110) & (val < 225) & (sat < 35) & ~ink_b
    # chấm nhỏ nằm cô lập, xa hình chính = nền bẩn (vân giấy, nhiễu)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nonwhite.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 30
    small[0] = False
    big_near = cv2.dilate((~small[lab] & (lab > 0)).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    specks = len(set(np.unique(lab[small[lab]])) - set(np.unique(lab[small[lab] & big_near])))
    m = {
        "edge_touch": cut,                                       # hình chạm/bị cắt ở mép ảnh
        "specks": specks,                                        # chấm lẻ tẻ
        "ink_ratio": float(ink_b.sum() / ink_b.size),           # mực / cả canvas
        "gray_ratio": float(gray.sum() / area),                  # đổ bóng / phần có nội dung
        # tỉ lệ chiều dài nằm trong nét vụn (ngắn hơn 1.5 lần cỡ cọ) — vân, gạch chéo, chi tiết lắt nhắt
        "short_ratio": float(lengths[lengths < max(20, 1.5 * meta["brush"])].sum() / lengths.sum()) if lengths.sum() else 1.0,
        "n_strokes": int(len(all_strokes)),
        "n_elements": len(meta["elements"]),
        "coverage": coverage,
        "brush": meta["brush"],
    }

    # điểm 0–100: trừ dần theo từng lỗi
    score = 100.0
    score -= max(0, m["gray_ratio"] - 0.05) * 300                 # bóng xám
    score -= max(0, m["short_ratio"] - 0.10) * 200                # quá nhiều nét vụn
    score -= max(0, 0.95 - m["coverage"]) * 200                   # cọ không phủ hết
    score -= max(0, m["ink_ratio"] - 0.10) * 300                  # mảng đen lớn
    score -= max(0, 0.008 - m["ink_ratio"]) * 3000                # gần như trống
    score -= max(0, m["n_strokes"] - 400) * 0.05                  # quá chi tiết, vẽ lâu
    score -= max(0, m["brush"] - 45) * 1.5                        # nét quá đậm/mảng tô đen
    score -= max(0, m["specks"] - 20) * 0.5                       # nền lốm đốm
    score -= min(40, m["edge_touch"] * 400)                        # bị cắt mép
    m["score"] = round(max(0.0, score), 1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    for path in args.images:
        m = assess(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        print(f"{m['score']:5.1f}  {path}  {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items() if k != 'score'})}")


if __name__ == "__main__":
    main()
