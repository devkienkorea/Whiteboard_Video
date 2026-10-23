"""Chạy trọn 1 cảnh: ảnh + lời thoại → video có giọng đọc, mỗi vật thể được vẽ đúng lúc được nhắc tới.

scene.json:
{
  "image": "sample.png",                      # đường dẫn tương đối theo file scene.json
  "narration": "Ngày xưa có một đám mây ...",
  "cues": ["đám mây", "mặt trời", "ngôi nhà"], # cụm từ kích hoạt từng vật thể, theo thứ tự trong strokes.json
  "regions": [[0.1, 0.0, 0.5, 0.3], null, ...], # (tuỳ chọn) khung từng vật thể [x0, y0, x1, y1] tỉ lệ theo ảnh,
                                              # song song với cues → tách nét theo vật thể; null = không định vị được
  "text": "Ngủ đủ 8 tiếng",                   # (tuỳ chọn) chữ viết tay đầu cảnh, hình thu nhỏ xuống nhường chỗ
  "tts": {"engine": "vieneu", "voice": "Ngọc Huyền", "speed": 1.0}
}

Chạy: python -m pipeline.scene jobs/demo/scene.json
Đầu ra cùng thư mục: voice.wav, words.json, subs.srt, scene.mp4
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from pipeline import align, render, strokes, tts

LEAD = 0.2  # mực bắt đầu chạm giấy sớm hơn lời nói một chút
BOX = (0.05, 0.10, 0.90, 0.62)            # vùng đặt hình (x, y, w, h theo tỉ lệ khung)
BOX_WITH_TEXT = (0.05, 0.22, 0.90, 0.50)  # có chữ on-screen: chữ ở 9–21%, hình thu xuống dưới


def run(scene_path, lead=LEAD):
    scene_path = Path(scene_path)
    job = scene_path.parent
    sc = json.loads(scene_path.read_text())
    clock = time.time()

    def step(msg):
        nonlocal clock
        now = time.time()
        print(f"  [{now - clock:5.1f}s] {msg}")
        clock = now

    # 1. tách nét (bỏ qua nếu ảnh, vùng đặt hình và khung vật thể chưa đổi → giữ được thứ tự vẽ đã sửa tay)
    image = (job / sc["image"]).resolve()
    meta_path = job / "strokes.json"
    text = sc.get("text", "").strip()
    box = list(BOX_WITH_TEXT if text else BOX)
    cues = sc.get("cues", [])
    found = [(c, r) for c, r in zip(cues, sc.get("regions") or []) if r]
    regions = [r for _, r in found] or None
    if regions:  # vật thể không định vị được: bỏ cue, nét của nó theo khung gần nhất
        cues = [c for c, _ in found]
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if (not old or meta_path.stat().st_mtime < image.stat().st_mtime
            or old.get("box", list(BOX)) != box or old.get("regions") != regions):
        color, ink, labels, meta = strokes.extract(cv2.imread(str(image), cv2.IMREAD_UNCHANGED), box=box,
                                                   regions=regions)
        meta["regions"] = regions
        cv2.imwrite(str(job / "color.png"), color)
        cv2.imwrite(str(job / "ink.png"), ink)
        cv2.imwrite(str(job / "labels.png"), labels)
        cv2.imwrite(str(job / "preview.png"), strokes.draw_preview(color, meta))
        meta_path.write_text(json.dumps(meta))
    n_elem = len(json.loads(meta_path.read_text())["elements"])
    step(f"tách nét: {n_elem} vật thể")

    # 2. giọng đọc
    opts = sc.get("tts", {})
    voice_wav = job / "voice.wav"
    secs = tts.synthesize(sc["narration"], voice_wav, opts.get("engine", "vieneu"), opts.get("voice"), opts.get("speed", 1.0))
    step(f"giọng đọc: {secs:.1f}s")

    # 3. mốc thời gian từng từ (khớp với lời đã chuẩn hoá — đúng như TTS đã đọc)
    spoken = tts.normalize(sc["narration"])
    words, ratio = align.align(voice_wav, spoken, job / "words.json", job / "subs.srt")
    step(f"whisper: khớp {ratio:.0%} kịch bản")

    # 4. mốc bắt đầu vẽ từng vật thể
    starts, after = [], 0.0
    for i in range(n_elem):
        cue = cues[i] if i < len(cues) else None
        t = align.find_cue(words, tts.normalize(cue), after) if cue else None
        if cue and t is None:
            print(f"  ! không tìm thấy cụm '{cue}' trong lời đọc → vẽ ngay sau vật trước")
        if t is not None:
            after = t
            t = max(0.0, t - lead)
        starts.append(t)
        print(f"    vật thể {i + 1}: {cue or '-'} → {'%.2fs' % t if t is not None else 'nối tiếp'}")

    # 5. render + ghép giọng
    out = job / "scene.mp4"
    render.render(job, out, duration=secs + 0.8, starts=starts, audio=voice_wav, words=words, text=text)
    step(f"render xong → {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="file scene.json")
    args = ap.parse_args()
    run(args.scene)


if __name__ == "__main__":
    main()
