"""Làm nhiều video liên tiếp (vd. chạy qua đêm): mỗi dòng của file là 1 chủ đề, dòng trống / bắt đầu bằng # bỏ qua.

Chủ đề lỗi → ghi log rồi làm tiếp chủ đề sau. Chủ đề đã có final.mp4 → bỏ qua, nên chạy lại cùng file
sẽ làm tiếp phần còn dở (muốn làm lại 1 video thì xoá final.mp4 của nó).

Chạy: python batch.py topics.txt [--seeds 2] [--voice "Ngọc Huyền"] [--music nhac.mp3]
Log: jobs/batch.log
"""
import argparse
import time
import traceback
from pathlib import Path

import make_video as mv
from pipeline import image_gen

LOG = mv.ROOT / "jobs" / "batch.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(topics, **opts):
    done, skipped, failed = [], [], []
    for k, topic in enumerate(topics, 1):
        job = mv.ROOT / "jobs" / mv.slugify(topic)
        if (job / "final.mp4").exists():
            log(f"[{k}/{len(topics)}] bỏ qua (đã có video): {topic}")
            skipped.append(topic)
            continue
        log(f"[{k}/{len(topics)}] bắt đầu: {topic}")
        t0 = time.time()
        try:
            final = mv.make(topic, **opts)
        except Exception:
            log(f"[{k}/{len(topics)}] LỖI: {topic}\n{traceback.format_exc()}")
            failed.append(topic)
            continue
        log(f"[{k}/{len(topics)}] xong sau {(time.time() - t0) / 60:.1f} phút → {final}")
        done.append(topic)
    log(f"Tổng kết: {len(done)} xong, {len(skipped)} bỏ qua, {len(failed)} lỗi"
        + "".join(f"\n    lỗi: {t}" for t in failed))
    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topics", help="file .txt, mỗi dòng 1 chủ đề")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--voice", default="Ngọc Huyền")
    ap.add_argument("--seconds", default="45,70")
    ap.add_argument("--image-model", choices=image_gen.MODELS, default="flux2-klein-4b")
    ap.add_argument("--music", help="file nhạc nền dùng cho mọi video")
    ap.add_argument("--style", choices=list(image_gen.STYLES), default="color")
    args = ap.parse_args()
    lines = Path(args.topics).read_text().splitlines()
    topics = [t.strip() for t in lines if t.strip() and not t.strip().startswith("#")]
    run(topics, seeds=args.seeds, voice=args.voice, seconds=tuple(int(x) for x in args.seconds.split(",")),
        image_model=args.image_model, music=args.music, style=args.style)


if __name__ == "__main__":
    main()
