"""Bước 5: tách ảnh lineart thành các nét bút có thứ tự.

Đầu vào: 1 ảnh (lineart đen, có thể đã tô màu).
Đầu ra (trong thư mục --out):
  color.png    ảnh đã căn vào khung dọc 1080x1920
  ink.png      mask nét mực (0/255)
  labels.png   ảnh 16-bit: mỗi pixel mực/màu mang id vật thể (0 = nền)
  strokes.json danh sách vật thể theo thứ tự vẽ, mỗi vật thể gồm các nét [[x, y], ...]
  preview.png  ảnh kiểm tra: nét tô màu theo thứ tự, đánh số vật thể

Chạy: python -m pipeline.strokes input.png --out jobs/demo
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

# (dy, dx) — 4 hướng trước để khi dò nét ưu tiên đi thẳng, không cắt chéo
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fit_to_canvas(img, W, H, box):
    """Thu nhỏ/phóng ảnh vào ô box=(x, y, w, h) (tỉ lệ theo canvas), nền trắng."""
    return _fit(img, W, H, box)[0]


def _fit(img, W, H, box):
    """Như fit_to_canvas, trả thêm hàm đổi toạ độ (x, y) ảnh gốc → canvas."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    img = whiten_background(img)
    cx0, cy0, cx1, cy1 = content_rect(img)
    img = img[cy0:cy1, cx0:cx1]
    bx, by, bw, bh = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    h, w = img.shape[:2]
    scale = min(bw / w, bh / h)
    nw, nh = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((H, W, 3), 255, np.uint8)
    x0 = int(bx + (bw - nw) / 2)
    y0 = int(by + (bh - nh) / 2)
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, lambda x, y: (x0 + (x - cx0) * scale, y0 + (y - cy0) * scale)


def whiten_background(img):
    """Ảnh AI hay có nền giấy xám/ngả màu: lấy màu nền ở viền ảnh, kéo về trắng tinh."""
    border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3),
                             img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    if paper.min() < 170:  # viền tối → không phải nền giấy, giữ nguyên
        return img
    out = img.astype(np.float32) * (255.0 / np.maximum(paper, 1))
    # vùng gần màu nền (vân giấy) → trắng hẳn
    out[np.abs(out - 255).max(axis=2) < 18] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def content_mask(color):
    """Pixel có nội dung (mực hoặc màu), phân biệt với nền trắng."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] < 235) | (hsv[:, :, 1] > 20)


def content_rect(img, pad=0.04):
    """Khung (x0, y0, x1, y1) bỏ lề trắng thừa để hình lấp đầy ô đặt hình (model hay vẽ vật thể nhỏ giữa ảnh)."""
    h, w = img.shape[:2]
    mask = remove_specks(content_mask(img), 30)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, w, h
    p = int(pad * max(np.ptp(ys), np.ptp(xs)))
    return max(0, xs.min() - p), max(0, ys.min() - p), min(w, xs.max() + p + 1), min(h, ys.max() + p + 1)


def remove_specks(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def trace_skeleton(skel):
    """Dò skeleton 1px thành các polyline liền mạch (danh sách [(y, x), ...])."""
    H, W = skel.shape
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    visited = np.zeros_like(skel, bool)

    def neighbors(y, x, want_visited):
        out = []
        for dy, dx in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and visited[ny, nx] == want_visited:
                out.append((ny, nx))
        return out

    def walk(y, x):
        path = [(y, x)]
        # nối vào điểm giao đã vẽ để không bị hở ở chỗ rẽ nhánh
        joined = neighbors(y, x, True)
        if joined:
            path.insert(0, joined[0])
        visited[y, x] = True
        while True:
            cands = neighbors(*path[-1], False)
            if not cands:
                break
            if len(path) >= 2:
                # chọn hướng đi thẳng nhất so với vài bước trước
                ref = path[max(0, len(path) - 4)]
                dy0, dx0 = path[-1][0] - ref[0], path[-1][1] - ref[1]
                cands.sort(key=lambda p: -((p[0] - path[-1][0]) * dy0 + (p[1] - path[-1][1]) * dx0))
            nxt = cands[0]
            visited[nxt] = True
            path.append(nxt)
        # đường khép kín (vòng tròn...) thì nối lại điểm đầu
        if len(path) > 8 and max(abs(path[0][0] - path[-1][0]), abs(path[0][1] - path[-1][1])) <= 1:
            path.append(path[0])
        return path

    ys, xs = np.nonzero(skel)
    paths = []
    for y, x in zip(ys, xs):
        if deg[y, x] == 1 and not visited[y, x]:
            paths.append(walk(y, x))
    for y, x in zip(ys, xs):
        if not visited[y, x]:
            paths.append(walk(y, x))
    return paths


def order_strokes(strokes, start):
    """Nearest-neighbor: nét tiếp theo là nét có đầu (hoặc cuối, thì đảo chiều) gần ngòi bút nhất."""
    remaining = list(strokes)
    ordered = []
    pos = np.array(start, np.float32)
    while remaining:
        best, best_d, best_rev = 0, 1e18, False
        for i, s in enumerate(remaining):
            d0 = np.sum((s[0] - pos) ** 2)
            d1 = np.sum((s[-1] - pos) ** 2)
            if d0 < best_d:
                best, best_d, best_rev = i, d0, False
            if d1 < best_d:
                best, best_d, best_rev = i, d1, True
        s = remaining.pop(best)
        if best_rev:
            s = s[::-1]
        ordered.append(s)
        pos = s[-1]
    return ordered


def group_by_distance(ink_hard, ink_soft, nonwhite, group_px):
    """Nhãn vật thể khi không có khung: nét cách nhau < group_px là 1 vật thể, màu thuộc nhóm nét bao quanh nó."""
    # gom nét gần nhau thành 1 "vật thể"
    grouped = cv2.dilate(ink_hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_px * 2 + 1,) * 2))
    n_groups, group_lab = cv2.connectedComponents(grouped, connectivity=8)
    labels = np.where(ink_soft, group_lab, 0).astype(np.int32)

    # vùng màu thuộc vật thể nào: tô kín đường viền ngoài của nhóm nét; vật nhỏ được nhận trước
    color_px = nonwhite & ~ink_soft
    fills = []
    for g in range(1, n_groups):
        ys, xs = np.nonzero(group_lab == g)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        comp = ((group_lab[y0:y1, x0:x1] == g) & ink_hard[y0:y1, x0:x1]).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        fills.append((int(filled.sum()), g, (y0, y1, x0, x1), filled.astype(bool)))
    for _, g, (y0, y1, x0, x1), filled in sorted(fills, key=lambda f: f[0]):
        sub = labels[y0:y1, x0:x1]
        sub[filled & color_px[y0:y1, x0:x1] & (sub == 0)] = g

    # mảng màu không có viền → thành vật thể riêng (chỉ tô màu, không có nét)
    leftover = remove_specks(color_px & (labels == 0), 50)
    n_left, left_lab = cv2.connectedComponents(leftover.astype(np.uint8), connectivity=8)
    labels[left_lab > 0] = left_lab[left_lab > 0] + n_groups - 1
    n_total = n_groups - 1 + n_left - 1

    return labels, n_total


def region_owner(rects, W, H):
    """Mỗi pixel thuộc khung nhỏ nhất chứa nó (vật nằm trong vật khác vẫn tách được), ngoài mọi khung thì khung gần nhất.
    Trả về ảnh id 1..len(rects)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cost = []
    for x0, y0, x1, y1 in rects:
        dist = np.hypot(np.maximum(np.maximum(x0 - xx, xx - x1), 0), np.maximum(np.maximum(y0 - yy, yy - y1), 0))
        cost.append(dist * 1e7 + (x1 - x0) * (y1 - y0))  # diện tích < 1e7 px → chỉ quyết khi cùng khoảng cách
    return np.argmin(cost, axis=0) + 1


def snap_to_parts(owner, mask, share=0.6):
    """Mỗi mảng liền trong mask (1 nét, 1 mảng màu) theo khung chứa ≥ share diện tích của nó, khỏi bị cạnh khung
    cắt ngang; mảng trải đều nhiều khung thì chia theo từng pixel. Ngoài mask = 0."""
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    k = int(owner.max()) + 1
    counts = np.bincount((comp * k + owner)[mask], minlength=n * k).reshape(n, k)
    best = counts.argmax(axis=1)
    ok = counts.max(axis=1) >= share * np.maximum(counts.sum(axis=1), 1)
    return np.where(mask, np.where(ok[comp], best[comp], owner), 0).astype(np.int32)


def extract(img, W=1080, H=1920, box=(0.05, 0.10, 0.90, 0.62), ink_v=110, group_px=12, min_area=12, regions=None):
    """regions: khung [x0, y0, x1, y1] (tỉ lệ 0–1 theo ảnh gốc) của từng vật thể, theo thứ tự vẽ →
    mỗi khung thành 1 vật thể. Không có thì gom nét gần nhau, vẽ từ trên xuống, trái sang."""
    color, to_canvas = _fit(img, W, H, box)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    nonwhite = content_mask(color)

    # nét mực = pixel tối (theo V nên màu đậm như xanh dương không bị nhầm là mực)
    ink_hard = remove_specks(val < ink_v, min_area)
    # mask hiển thị gồm cả viền khử răng cưa quanh nét
    ink_soft = cv2.dilate(ink_hard.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & nonwhite

    skel = skeletonize(ink_hard)
    # độ dày cọ đủ phủ hết bề rộng nét gốc
    dt = cv2.distanceTransform(ink_hard.astype(np.uint8), cv2.DIST_L2, 3)
    widths = dt[skel]
    brush = int(np.ceil(2 * np.percentile(widths, 90))) + 4 if widths.size else 8
    # skeleton → polyline; nét ngắn hơn ~bề rộng cọ là "gai" của skeleton ở đầu nét/chỗ giao của nét dày:
    # đã nằm trong nét chính, cọ quét qua rồi; còn sót thì render hiện nốt khi xong vật thể
    paths = [p for p in trace_skeleton(skel) if len(p) >= max(3, brush * 0.75)]

    if regions:
        ih, iw = img.shape[:2]
        rects = [(*to_canvas(x0 * iw, y0 * ih), *to_canvas(x1 * iw, y1 * ih)) for x0, y0, x1, y1 in regions]
        owner = region_owner(rects, W, H)
        # mỗi nét theo khung chứa phần lớn điểm của nó; mực quanh nét mang nhãn của nét → khớp đường bút khi render
        path_elem, by_stroke = [], np.zeros((H, W), np.int32)
        for p in paths:
            ys_, xs_ = np.array(p).T
            e = int(np.bincount(owner[ys_, xs_]).argmax())
            path_elem.append(e)
            cv2.polylines(by_stroke, [np.stack([xs_, ys_], 1).astype(np.int32)], False, e, thickness=brush)
        labels = np.where(ink_soft, np.where(by_stroke > 0, by_stroke, owner), 0) + snap_to_parts(owner, nonwhite & ~ink_soft)
        n_total = len(rects)
    else:
        labels, n_total = group_by_distance(ink_hard, ink_soft, nonwhite, group_px)
    color_px = nonwhite & ~ink_soft

    boxes = {}
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    for g in range(1, n_total + 1):
        sel = lab == g
        if sel.any():
            boxes[g] = (xs[sel].min(), ys[sel].min(), xs[sel].max() + 1, ys[sel].max() + 1)
        elif regions:  # khung không trùng nét nào: giữ vật thể rỗng để cue vẫn khớp đúng thứ tự
            boxes[g] = tuple(int(v) for v in rects[g - 1])
    if regions:
        order = list(range(1, n_total + 1))
    else:
        # thứ tự vẽ mặc định: theo hàng từ trên xuống, trong hàng từ trái sang
        band = H * 0.12
        order = sorted(boxes, key=lambda g: (int(((boxes[g][1] + boxes[g][3]) / 2) // band), boxes[g][0]))
    lut = np.zeros(n_total + 1, np.int32)
    for new_id, g in enumerate(order, start=1):
        lut[g] = new_id
    labels = lut[labels]

    per_elem = {i: [] for i in range(1, len(order) + 1)}
    for p, e in zip(paths, path_elem if regions else (labels[p[len(p) // 2]] for p in paths)):
        if e:
            per_elem[e].append(np.array([(x, y) for y, x in p], np.float32))

    elements = []
    pen = (W * 0.5, H * 0.1)
    for e in range(1, len(order) + 1):
        x0, y0, x1, y1 = boxes[order[e - 1]]
        strokes = order_strokes(per_elem[e], pen) if per_elem[e] else []
        if strokes:
            pen = tuple(strokes[-1][-1])
        has_color = bool(((labels == e) & color_px).any())
        elements.append({
            "id": e,
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "has_color": has_color,
            # giữ 1/2 số điểm cho file gọn; render nội suy lại theo độ dài
            "strokes": [np.vstack([s[::2], s[-1:]]).round().astype(int).tolist() for s in strokes],
        })

    meta = {"canvas": [W, H], "box": list(box), "brush": brush, "elements": elements}
    return color, (ink_soft * 255).astype(np.uint8), labels.astype(np.uint16), meta


def draw_preview(color, meta):
    W, H = meta["canvas"]
    prev = (color * 0.25 + 191).astype(np.uint8)
    all_strokes = [s for el in meta["elements"] for s in el["strokes"]]
    for i, s in enumerate(all_strokes):
        hue = int(150 * i / max(1, len(all_strokes) - 1))  # đỏ → xanh theo thứ tự vẽ
        c = cv2.cvtColor(np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(prev, [np.array(s, np.int32)], False, c, 3, cv2.LINE_AA)
    for el in meta["elements"]:
        x, y, w, h = el["bbox"]
        cv2.rectangle(prev, (x, y), (x + w, y + h), (120, 120, 120), 1)
        cv2.putText(prev, str(el["id"]), (x + 4, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return prev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1080x1920", help="kích thước canvas WxH")
    ap.add_argument("--box", default="0.05,0.10,0.90,0.62", help="ô đặt hình x,y,w,h (tỉ lệ), chừa chỗ phụ đề")
    ap.add_argument("--ink-v", type=int, default=110, help="ngưỡng độ sáng coi là mực")
    ap.add_argument("--group-px", type=int, default=12, help="nét cách nhau < N px thì cùng vật thể")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image}")
    W, H = map(int, args.size.lower().split("x"))
    box = tuple(float(v) for v in args.box.split(","))
    color, ink, labels, meta = extract(img, W, H, box, args.ink_v, args.group_px)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.png"), color)
    cv2.imwrite(str(out / "ink.png"), ink)
    cv2.imwrite(str(out / "labels.png"), labels)
    cv2.imwrite(str(out / "preview.png"), draw_preview(color, meta))
    (out / "strokes.json").write_text(json.dumps(meta))
    n_strokes = sum(len(el["strokes"]) for el in meta["elements"])
    print(f"{len(meta['elements'])} vật thể, {n_strokes} nét, cọ {meta['brush']}px → {out}")


if __name__ == "__main__":
    main()
