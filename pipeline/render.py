"""Bước 6: render video bàn tay vẽ nét rồi tô màu từ kết quả của strokes.py.

Chạy: python -m pipeline.render jobs/demo --out jobs/demo/scene.mp4 [--duration 10]
"""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pipeline import subtitles

PAPER_BGR = (236, 246, 250)  # giấy trắng ngà
TEXT_FONT = ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 2)  # Chalkboard SE Bold: nét bút dạ, đủ dấu tiếng Việt
TEXT_BGR = (40, 40, 40)


def make_paper(W, H, seed=7):
    rng = np.random.default_rng(seed)
    paper = np.empty((H, W, 3), np.float32)
    paper[:] = PAPER_BGR
    grain = cv2.GaussianBlur(rng.normal(0, 4, (H, W)).astype(np.float32), (0, 0), 1.2)
    paper += grain[:, :, None]
    # tối nhẹ ở mép cho có chiều sâu
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    paper *= (1 - 0.06 * np.clip(r - 0.6, 0, 1))[:, :, None]
    return np.clip(paper, 0, 255).astype(np.uint8)


def make_hand(scale=1.0):
    """Vẽ bàn tay cầm bút lông bằng code. Trả về (sprite BGRA, toạ độ ngòi bút trong sprite)."""
    S = 2  # vẽ to rồi thu nhỏ cho mịn
    # sprite cao để cánh tay luôn kéo ra khỏi mép dưới khung dọc
    w, h = int(1400 * scale), int(2400 * scale)
    img = np.zeros((h * S, w * S, 3), np.uint8)
    alpha = np.zeros((h * S, w * S), np.uint8)
    tip = np.array([60, 620]) * scale * S
    d = np.array([0.5, -0.866])   # hướng thân bút (lên, chếch phải)
    n = np.array([0.866, 0.5])    # vuông góc, về phía lòng bàn tay
    k = scale * S

    def P(along, side):
        p = tip + d * along * k + n * side * k
        return (int(p[0]), int(p[1]))

    def poly(pts, fill, line):
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, line, int(3 * k), cv2.LINE_AA)
        cv2.fillPoly(alpha, [pts], 255, cv2.LINE_AA)

    def ellipse(along, side, ax, ay, extra_deg, fill, line):
        c = P(along, side)
        ang = math.degrees(math.atan2(d[1], d[0])) + extra_deg
        axes = (int(ax * k), int(ay * k))
        cv2.ellipse(img, c, axes, ang, 0, 360, fill, -1, cv2.LINE_AA)
        cv2.ellipse(img, c, axes, ang, 0, 360, line, int(3 * k), cv2.LINE_AA)
        cv2.ellipse(alpha, c, axes, ang, 0, 360, 255, -1, cv2.LINE_AA)

    skin, skin_line = (150, 186, 232), (90, 120, 170)
    sleeve, sleeve_line = (170, 120, 70), (110, 70, 40)
    # cánh tay + tay áo, kéo ra khỏi khung về phía dưới phải
    arm_dir = np.array([0.45, 0.893])
    arm_n = np.array([-0.893, 0.45])
    base = np.array(P(210, 50), np.float32)

    def A(along, side):
        p = base + arm_dir * along * k + arm_n * side * k
        return (int(p[0]), int(p[1]))

    poly([A(-20, -62), A(-20, 62), A(160, 70), A(160, -70)], skin, skin_line)
    poly([A(150, -78), A(150, 78), A(3000, 90), A(3000, -90)], sleeve, sleeve_line)
    ellipse(205, 48, 95, 72, 0, skin, skin_line)            # lòng bàn tay
    # bút lông: ngòi, thân, nắp đuôi
    poly([P(0, 0), P(46, -15), P(46, 15)], (45, 45, 45), (20, 20, 20))
    poly([P(44, -17), P(300, -17), P(300, 17), P(44, 17)], (190, 120, 40), (110, 60, 20))
    poly([P(270, -18), P(318, -18), P(318, 18), P(270, 18)], (60, 60, 60), (30, 30, 30))
    ellipse(128, 14, 58, 22, 0, skin, skin_line)            # ngón trỏ ôm bút
    ellipse(158, 36, 50, 22, 8, skin, skin_line)            # ngón giữa
    ellipse(150, -22, 46, 20, -6, skin, skin_line)          # ngón cái

    small = (w, h)
    img = cv2.resize(img, small, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, small, interpolation=cv2.INTER_AREA)
    return np.dstack([img, alpha]), (tip / S).astype(int)


def overlay_hand(frame, sprite, tip_in_sprite, pos, shadow=(14, 18)):
    H, W = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    for dx, dy, is_shadow in ((shadow[0], shadow[1], True), (0, 0, False)):
        x0 = int(pos[0]) - tip_in_sprite[0] + dx
        y0 = int(pos[1]) - tip_in_sprite[1] + dy
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
        if fx0 >= fx1 or fy0 >= fy1:
            continue
        sx0, sy0 = fx0 - x0, fy0 - y0
        a = sprite[sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0, 3:4].astype(np.float32) / 255
        roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        if is_shadow:
            roi *= 1 - 0.22 * a
        else:
            src = sprite[sy0:sy0 + fy1 - fy0, sx0:sx0 + fx1 - fx0, :3].astype(np.float32)
            roi = roi * (1 - a) + src * a
        frame[fy0:fy1, fx0:fx1] = roi.astype(np.uint8)


def text_layer(text, W, H, top=0.09, bottom=0.21, maxw=0.84):
    """Chữ on-screen: tối đa 2 dòng, cỡ lớn nhất vừa vùng top–bottom, căn giữa.
    Trả về (alpha HxW 0–1, các dòng [(x0, x1, y0, y1, số ký tự)] theo thứ tự viết)."""
    from PIL import Image, ImageDraw, ImageFont

    for size in range(round(W * 0.1), 20, -4):
        font = ImageFont.truetype(TEXT_FONT[0], size, index=TEXT_FONT[1])
        lines = [" ".join(t for _, t, _ in ln) for ln in subtitles.wrap(text.split(), font, W * maxw)]
        asc, desc = font.getmetrics()
        lh = asc + desc
        if len(lines) <= 2 and lh * len(lines) <= H * (bottom - top):
            break
    # cân 2 dòng cho đều (tránh 1 từ rơi lẻ xuống dòng dưới): thu hẹp khổ tới khi sắp thêm dòng
    width = W * maxw
    while len(lines) == 2:
        tighter = [" ".join(t for _, t, _ in ln) for ln in subtitles.wrap(text.split(), font, width - 10)]
        if len(tighter) > 2:
            break
        lines, width = tighter, width - 10
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    y = H * top + (H * (bottom - top) - lh * len(lines)) / 2
    out = []
    for line in lines:
        x = (W - font.getlength(line)) / 2
        d.text((x, y), line, font=font, fill=255, stroke_width=max(1, size // 28), stroke_fill=255)  # nét dày như bút dạ
        # mở rộng lên trên/xuống dưới cho dấu thanh chồng (Ấ, Ổ...) cũng được lộ ra khi viết
        out.append((int(x), int(x + font.getlength(line)) + 1, max(0, int(y - lh / 3)), min(H, int(y + lh * 1.15)), len(line)))
        y += lh
    return np.asarray(img, np.float32) / 255, out


def ease(p):
    return p * p * (3 - 2 * p)


def open_writer(path, W, H, fps, audio=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # bản ffmpeg đóng gói sẵn, dùng khi máy chưa cài
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit("Chưa có ffmpeg: brew install ffmpeg (hoặc pip install imageio-ffmpeg)")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in encoders:
        codec = ["-c:v", "h264_videotoolbox", "-b:v", "10M"]  # encode bằng chip Apple
    else:
        codec = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        # kéo dài audio bằng im lặng cho bằng video → ghép nhiều cảnh không bị lệch tiếng
        cmd += ["-i", str(audio), "-af", "apad", "-shortest", "-c:a", "aac", "-b:a", "192k"]
    cmd += [*codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def build_timeline(meta, draw_speed, travel_speed, color_time, W, H, text_lines=()):
    """Các bước (loại, thời lượng, dữ liệu) nhóm theo vật thể, thời lượng ở tốc độ tự nhiên (giây).
    text_lines (từ text_layer) → thêm 1 nhóm viết chữ đứng trước các vật thể.
    Trả về (groups, outro): outro là bước đưa tay ra khỏi khung."""
    groups = []
    offscreen = np.array([W * 1.1, H * 1.05], np.float32)
    pen = offscreen
    if text_lines:
        steps = []
        for x0, x1, y0, y1, n in text_lines:
            mid = y0 + (y1 - y0) * 0.55
            start = np.array([x0, mid], np.float32)
            steps.append(("travel", max(0.08, float(np.linalg.norm(start - pen)) / travel_speed), (pen, start)))
            steps.append(("write", 0.07 * n, (x0, x1, y0, y1)))
            pen = np.array([x1, mid], np.float32)
        groups.append(steps)
    for el in meta["elements"]:
        steps = []
        for s in el["strokes"]:
            pts = np.array(s, np.float32)
            seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
            cum = np.concatenate([[0], np.cumsum(seg)])
            dist = float(np.linalg.norm(pts[0] - pen))
            steps.append(("travel", max(0.08, dist / travel_speed), (pen, pts[0])))
            steps.append(("draw", max(1 / 30, cum[-1] / draw_speed), (el["id"], pts, cum)))
            pen = pts[-1]
        steps.append(("flush", 0.0, el["id"]))
        if el["has_color"]:
            x, y, w, h = el["bbox"]
            start = np.array([x + w / 2, y], np.float32)
            steps.append(("travel", max(0.08, float(np.linalg.norm(start - pen)) / travel_speed), (pen, start)))
            area_factor = min(1.0, (w * h) / (W * H * 0.08))
            steps.append(("color", color_time * (0.5 + area_factor), el))
            pen = np.array([x + w / 2, y + h], np.float32)
        groups.append(steps)
    return groups, ("travel", 0.5, (pen, offscreen))


def schedule(groups, starts, end=None, min_factor=0.15):
    """Đặt mỗi vật thể bắt đầu đúng mốc starts[i] (giây, None = ngay sau vật trước).
    Vẽ không kịp trước mốc kế tiếp thì tăng tốc (tối đa 1/min_factor lần), dư thời gian thì tay chờ."""
    out, t = [], 0.0
    for i, g in enumerate(groups):
        natural = sum(d for _, d, _ in g)
        # mốc là lúc mực chạm giấy → trừ thời gian tay di chuyển tới nét đầu
        pre = g[0][1] if g and g[0][0] == "travel" else 0.0
        s = max(t, starts[i] - pre if starts[i] is not None else t)
        if s > t:
            out.append(("wait", s - t, None))
        nxt = next((starts[j] for j in range(i + 1, len(groups)) if starts[j] is not None), end)
        f = 1.0
        if nxt is not None and natural > nxt - s:
            f = max(min_factor, (nxt - s) / natural)
        out += [(k, d * f, data) for k, d, data in g]
        t = s + natural * f
    return out


def render(job, out, fps=30, duration=None, draw_speed=1800, travel_speed=3000, color_time=0.6, hold=1.0,
           starts=None, audio=None, words=None, text=None):
    """starts: mốc bắt đầu vẽ từng vật thể (theo thứ tự strokes.json); audio: file giọng đọc ghép vào;
    words: mốc thời gian từng từ (align.py) → phụ đề karaoke; text: chữ on-screen viết tay đầu cảnh."""
    job = Path(job)
    meta = json.loads((job / "strokes.json").read_text())
    W, H = meta["canvas"]
    brush = meta["brush"]
    color = cv2.imread(str(job / "color.png"))
    ink = cv2.imread(str(job / "ink.png"), cv2.IMREAD_GRAYSCALE) > 0
    labels = cv2.imread(str(job / "labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)

    paper = make_paper(W, H)
    # "multiply": trắng thành màu giấy, mực vẫn đen
    target = (color.astype(np.uint16) * paper // 255).astype(np.uint8)
    text_lines = []
    if text:
        alpha, text_lines = text_layer(text, W, H)
        target = (target * (1 - alpha[..., None]) + np.array(TEXT_BGR, np.float32) * alpha[..., None]).astype(np.uint8)
        text_px = alpha > 0
        if starts is not None:
            starts = [0.0] + list(starts)  # viết chữ ngay đầu cảnh
    canvas = paper.copy()

    # toạ độ pixel của từng vật thể, chia riêng mực / màu
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    is_ink = ink[ys, xs]
    pix = {}
    for kind, sel in (("ink", is_ink), ("color", ~is_ink)):
        yk, xk, lk = ys[sel], xs[sel], lab[sel]
        order = np.lexsort((xk, yk, lk))  # theo vật thể, rồi từ trên xuống
        yk, xk, lk = yk[order], xk[order], lk[order]
        bounds = np.searchsorted(lk, np.arange(1, lk.max() + 2 if lk.size else 2))
        for e in range(1, len(bounds)):
            pix[(kind, e)] = (yk[bounds[e - 1]:bounds[e]], xk[bounds[e - 1]:bounds[e]])

    groups, outro = build_timeline(meta, draw_speed, travel_speed, color_time, W, H, text_lines)
    if starts is not None:
        steps = schedule(groups, starts, end=duration - hold - outro[1] if duration else None)
    else:
        steps = [s for g in groups for s in g]
        if duration:
            factor = max(0.05, (duration - hold - outro[1]) / sum(d for _, d, _ in steps))
            steps = [(k, d * factor, data) for k, d, data in steps]
    steps.append(outro)
    sprite, tip = make_hand(W / 1080)
    karaoke = subtitles.Karaoke(words, W, H) if words else None

    writer = open_writer(out, W, H, fps, audio)
    t = 0.0
    frames_done = 0
    pen = np.array([W * 1.1, H * 1.05], np.float32)

    def emit(pos):
        frame = canvas.copy()
        overlay_hand(frame, sprite, tip, pos)
        if karaoke:
            karaoke.draw(frame, frames_done / fps)
        writer.stdin.write(frame.tobytes())

    for kind, dur, data in steps:
        t += dur
        n = round(t * fps) - frames_done
        if kind == "flush":
            yk, xk = pix.get(("ink", data), (np.empty(0, int), np.empty(0, int)))
            canvas[yk, xk] = target[yk, xk]
            continue
        last_idx = 0
        color_ptr = 0
        last_x = data[0] if kind == "write" else 0
        for i in range(max(n, 1)):
            p = (i + 1) / n if n > 0 else 1.0
            if kind == "travel":
                a, b = data
                pen = a + (b - a) * ease(p)
            elif kind == "draw":
                e, pts, cum = data
                L = p * cum[-1]
                idx = int(np.searchsorted(cum, L, side="right")) - 1
                idx = min(idx, len(pts) - 1)
                if idx < len(pts) - 1 and cum[idx + 1] > cum[idx]:
                    f = (L - cum[idx]) / (cum[idx + 1] - cum[idx])
                    pen = pts[idx] + (pts[idx + 1] - pts[idx]) * f
                else:
                    pen = pts[idx]
                seg = np.vstack([pts[last_idx:idx + 1], pen[None]])
                last_idx = idx
                # mở mực trong vùng cọ vừa quét, chỉ pixel mực của vật thể này
                x0 = max(0, int(seg[:, 0].min()) - brush)
                y0 = max(0, int(seg[:, 1].min()) - brush)
                x1 = min(W, int(seg[:, 0].max()) + brush + 1)
                y1 = min(H, int(seg[:, 1].max()) + brush + 1)
                scratch = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.polylines(scratch, [(seg - [x0, y0]).round().astype(np.int32)], False, 255, brush)
                sel = (scratch > 0) & ink[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] == e)
                canvas[y0:y1, x0:x1][sel] = target[y0:y1, x0:x1][sel]
            elif kind == "write":
                x0, x1, y0, y1 = data
                cx = int(x0 + (x1 - x0) * p) + 1
                sel = text_px[y0:y1, last_x:cx]
                canvas[y0:y1, last_x:cx][sel] = target[y0:y1, last_x:cx][sel]
                last_x = cx
                # ngòi bút nhấp nhô lên xuống như đang viết từng chữ
                pen = np.array([cx, y0 + (y1 - y0) * (0.55 + 0.12 * math.sin(p * 60))], np.float32)
            elif kind == "color":
                x, y, w, h = data["bbox"]
                yk, xk = pix.get(("color", data["id"]), (np.empty(0, int), np.empty(0, int)))
                row = y + h * p
                new_ptr = int(np.searchsorted(yk, row, side="right"))
                canvas[yk[color_ptr:new_ptr], xk[color_ptr:new_ptr]] = target[yk[color_ptr:new_ptr], xk[color_ptr:new_ptr]]
                color_ptr = new_ptr
                zigzags = max(2, h / 70)
                pen = np.array([x + w * (0.5 + 0.42 * math.sin(p * 2 * math.pi * zigzags)), row], np.float32)
            if n > 0:
                wobble = np.array([math.sin(frames_done * 0.7), math.cos(frames_done * 0.9)]) * 1.5
                emit(pen + wobble)
                frames_done += 1

    # hiện nốt phần sót, giữ khung hình cuối
    canvas[:] = target
    hold_frames = round(hold * fps)
    if duration:
        hold_frames = max(hold_frames, round(duration * fps) - frames_done)
    for _ in range(hold_frames):
        frame = canvas
        if karaoke:
            frame = canvas.copy()
            karaoke.draw(frame, frames_done / fps)
        writer.stdin.write(frame.tobytes())
        frames_done += 1
    writer.stdin.close()
    writer.wait()
    return frames_done / fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", help="thư mục đầu ra của strokes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, help="ép tổng thời lượng (giây), vd. bằng độ dài giọng đọc")
    ap.add_argument("--draw-speed", type=float, default=1800, help="px/giây khi vẽ")
    ap.add_argument("--hold", type=float, default=1.0, help="giữ khung cuối (giây)")
    ap.add_argument("--starts", help="mốc bắt đầu vẽ từng vật thể, vd. 0.8,2.5,- (- = ngay sau vật trước)")
    ap.add_argument("--audio", help="file giọng đọc ghép vào video")
    args = ap.parse_args()
    starts = None
    if args.starts:
        starts = [None if v.strip() == "-" else float(v) for v in args.starts.split(",")]
    secs = render(args.job, args.out, args.fps, args.duration, args.draw_speed, hold=args.hold,
                  starts=starts, audio=args.audio)
    print(f"Đã xuất {args.out} ({secs:.1f}s)")


if __name__ == "__main__":
    main()
