"""Chạy trọn: 1 chủ đề → video TikTok whiteboard có giọng đọc.

  1. LLM (Qwen3.6-35B-A3B) viết kịch bản      → jobs/<slug>/script.json
  2. Sinh ảnh từng cảnh (FLUX.2 Klein)         → jobs/<slug>/scene_XX/image.png
  3. Định vị từng vật thể trong ảnh (Qwen3.6 nhìn ảnh)  → scene_XX/regions.json
     Mỗi cảnh: tách nét theo vật thể → giọng đọc → whisper → render (vật nào được nhắc thì vẽ lúc đó) → scene_XX/scene.mp4
  4. Ghép các cảnh (chuyển cảnh trượt giấy 0.4s) + nhạc nền (tự giảm khi có lời) + chuẩn âm lượng -14 LUFS → jobs/<slug>/final.mp4
     + ảnh bìa tiêu đề + hình cảnh 1 (nếu chưa có)                            → jobs/<slug>/cover.jpg

Bước nào đã có file thì bỏ qua (sửa script.json hoặc xoá image.png rồi chạy lại để làm lại phần đó).
Mỗi bước nặng (LLM, sinh ảnh, giọng + whisper) chạy trong 1 tiến trình con riêng: xong là thoát,
macOS thu hồi hết bộ nhớ → các model không bao giờ nằm cùng lúc trong RAM (vừa 32GB).

Chạy: python make_video.py "Tại sao nên uống nước buổi sáng" [--seeds 2] [--voice "Ngọc Huyền"] [--music nhac.mp3]
      [--style ink]   (color = nét đen + tô màu, ink = chỉ nét đen như whiteboard cổ điển)
      --voice assets/voices/giong-cua-toi.wav → clone giọng từ file ghi âm
"""
import argparse
import filecmp
import json
import multiprocessing
import re
import shutil
import subprocess
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pipeline import cover, image_gen

ROOT = Path(__file__).resolve().parent


def slugify(text):
    text = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "video"


TRANSITION = "slideleft"  # hiệu ứng xfade của ffmpeg giữa 2 cảnh (giấy trượt sang trang mới); None = cắt thẳng
TRANSITION_SECONDS = 0.4


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True, check=True).stdout)


def concat(clips, out, transition=TRANSITION, d=TRANSITION_SECONDS):
    """Ghép các cảnh. Có transition: hình 2 cảnh chồng nhau d giây; tiếng không crossfade mà cắt d giây
    cuối mỗi cảnh trước (đoạn giữ khung cuối, vốn im lặng) → lời đọc không bị nhỏ đi hay lệch."""
    if not transition or len(clips) < 2:
        lst = out.parent / "concat.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        lst.unlink()
        return
    durs = [duration(c) for c in clips]
    graph, prev, offset = [], "[0:v]", 0.0
    for i in range(1, len(clips)):
        offset += durs[i - 1] - d
        graph.append(f"{prev}[{i}:v]xfade=transition={transition}:duration={d}:offset={offset:.3f}[v{i}]")
        prev = f"[v{i}]"
    graph += [f"[{i}:a]atrim=0:{durs[i] - d:.3f},asetpts=PTS-STARTPTS[a{i}]" for i in range(len(clips) - 1)]
    last = len(clips) - 1
    graph.append("".join(f"[a{i}]" for i in range(last)) + f"[{last}:a]concat=n={len(clips)}:v=0:a=1[a]")
    inputs = [x for c in clips for x in ("-i", str(c))]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", prev, "-map", "[a]", "-c:v", "h264_videotoolbox", "-b:v", "10M", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)


def finish_audio(video, out, music=None, music_volume=0.2):
    """Trộn nhạc nền (lặp cho đủ dài, nhỏ lại khi có giọng nói, fade-out 2s cuối) rồi chuẩn loudness TikTok."""
    dur = duration(video)
    loud = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
    if music:
        inputs = ["-i", str(video), "-stream_loop", "-1", "-i", str(music)]
        graph = (f"[0:a]asplit[voice][key];[1:a]volume={music_volume}[bg];"
                 "[bg][key]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=500[duck];"
                 f"[voice][duck]amix=inputs=2:duration=first:normalize=0,"
                 f"afade=t=out:st={max(0.0, dur - 2):.2f}:d=2,{loud}[a]")
    else:
        inputs, graph = ["-i", str(video)], f"[0:a]{loud}[a]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out)], check=True)


def isolated(fn, *args):
    """Chạy fn trong tiến trình con mới; trả kết quả khi tiến trình đã thoát (bộ nhớ đã được trả lại).
    Lý do: giải phóng model trong cùng tiến trình không trả hết RAM (đo được: còn ~10GB sau khi xoá LLM)."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


# --- các bước chạy trong tiến trình con (phải là hàm cấp module để truyền sang được) ---

def _write_script(topic, seconds, path):
    from pipeline import script_llm

    script = script_llm.ScriptWriter().write(topic, seconds)
    Path(path).write_text(json.dumps(script, ensure_ascii=False, indent=1))


def _gen_images(items, seeds, model, style):
    gen = image_gen.Generator(model)
    for subject, d in items:
        print(f"  {Path(d).name}: {subject}", flush=True)
        best, _ = image_gen.generate_best(gen, subject, Path(d) / "gen", style=style, seeds=seeds)
        dst = Path(d) / "image.png"
        if not (dst.exists() and filecmp.cmp(best, dst, shallow=False)):  # giữ mtime → video cảnh không bị coi là cũ
            shutil.copy(best, dst)


def _locate(items):
    from pipeline import locate

    loc = locate.Locator()
    for image, objects, out in items:
        boxes = loc.find(image, objects)
        print(f"  {Path(out).parent.name}: " + ", ".join(f"{o} {'✓' if b else '✗'}" for o, b in zip(objects, boxes)),
              flush=True)
        Path(out).write_text(json.dumps({"objects": objects, "boxes": boxes}))


def _build_scenes(scene_files):
    from pipeline import scene

    for i, f in enumerate(scene_files, 1):
        print(f"  {Path(f).parent.name} ({i}/{len(scene_files)})", flush=True)
        scene.run(f)


def scene_dir(job, i):
    return Path(job) / f"scene_{i + 1:02}"


def load_script(job):
    return json.loads((Path(job) / "script.json").read_text())


def elements(sc):
    """Vật thể của cảnh [{cue, object}]; kịch bản cũ chỉ có 1 'cue' → 1 vật thể, không định vị."""
    return sc.get("elements") or [{"cue": sc.get("cue", ""), "object": ""}]


def regions(d, sc):
    """Khung đã định vị (song song với elements), None nếu chưa có hoặc đã cũ (ảnh/vật thể đổi)."""
    f, img = Path(d) / "regions.json", Path(d) / "image.png"
    objects = [e["object"] for e in elements(sc)]
    if not all(objects) or not f.exists() or f.stat().st_mtime < img.stat().st_mtime:
        return None
    r = json.loads(f.read_text())
    return r["boxes"] if r["objects"] == objects else None


# --- 4 bước trên 1 thư mục job (dùng chung cho make() và ui.py); i = chỉ số cảnh từ 0 ---

def write_script(job, topic, seconds=(45, 70)):
    Path(job).mkdir(parents=True, exist_ok=True)
    isolated(_write_script, topic, seconds, str(Path(job) / "script.json"))


def gen_images(job, scenes, seeds=2, model="flux2-klein-4b", style="color"):
    """Sinh thêm `seeds` ảnh cho các cảnh (seed nối tiếp lần trước), ảnh điểm cao nhất → image.png."""
    script = load_script(job)
    items = [(script["scenes"][i]["image_subject"], str(scene_dir(job, i))) for i in scenes]
    isolated(_gen_images, items, seeds, model, style)


def build_scenes(job, scenes, voice="Ngọc Huyền"):
    script = load_script(job)
    todo = []  # cảnh cần định vị vật thể (chưa có / ảnh hoặc vật thể đã đổi)
    for i in scenes:
        sc, d = script["scenes"][i], scene_dir(job, i)
        objects = [e["object"] for e in elements(sc)]
        if all(objects) and regions(d, sc) is None:
            todo.append((str(d / "image.png"), objects, str(d / "regions.json")))
    if todo:
        isolated(_locate, todo)
    files = []
    for i in scenes:
        sc, d = script["scenes"][i], scene_dir(job, i)
        (d / "scene.json").write_text(json.dumps({
            "image": "image.png", "narration": sc["narration"], "cues": [e["cue"] for e in elements(sc)],
            "regions": regions(d, sc), "text": sc.get("on_screen_text", ""),
            "tts": {"engine": "vieneu", "voice": voice},
        }, ensure_ascii=False, indent=1))
        files.append(str(d / "scene.json"))
    isolated(_build_scenes, files)


def finish(job, music=None):
    """Ghép scene.mp4 của mọi cảnh + nhạc nền → final.mp4; tạo cover.jpg nếu chưa có (giữ bìa đã chọn tay)."""
    job = Path(job)
    final, joined = job / "final.mp4", job / "joined.mp4"
    concat([scene_dir(job, i) / "scene.mp4" for i in range(len(load_script(job)["scenes"]))], joined)
    finish_audio(joined, final, music)
    joined.unlink()
    if not (job / "cover.jpg").exists():
        cover.make(job)
    return final


def make(topic, seeds=2, voice="Ngọc Huyền", seconds=(45, 70), image_model="flux2-klein-4b", job=None,
         music=None, style="color"):
    job = Path(job) if job else ROOT / "jobs" / slugify(topic)
    t_all = time.time()

    if not (job / "script.json").exists():
        print("① Viết kịch bản...")
        write_script(job, topic, seconds)
    script = load_script(job)
    n = len(script["scenes"])
    print(f"   {script.get('title', topic)} — {n} cảnh")

    todo = [i for i in range(n) if not (scene_dir(job, i) / "image.png").exists()]
    if todo:
        print(f"② Sinh ảnh {len(todo)} cảnh × {seeds} seed...")
        gen_images(job, todo, seeds, image_model, style)

    print("③ Dựng từng cảnh...")
    build_scenes(job, range(n), voice)

    final = finish(job, music)
    print(f"④ Xong: {final}  (tổng {time.time() - t_all:.0f}s)")
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic")
    ap.add_argument("--seeds", type=int, default=2, help="số ảnh thử mỗi cảnh, giữ ảnh điểm cao nhất")
    ap.add_argument("--voice", default="Ngọc Huyền")
    ap.add_argument("--seconds", default="45,70")
    ap.add_argument("--image-model", choices=image_gen.MODELS, default="flux2-klein-4b")
    ap.add_argument("--job", help="thư mục job (mặc định jobs/<slug chủ đề>)")
    ap.add_argument("--music", help="file nhạc nền không bản quyền (mp3/wav...), bỏ trống = không nhạc")
    ap.add_argument("--style", choices=list(image_gen.STYLES), default="color", help="kiểu hình vẽ")
    args = ap.parse_args()
    make(args.topic, args.seeds, args.voice, tuple(int(x) for x in args.seconds.split(",")), args.image_model, args.job,
         args.music, args.style)


if __name__ == "__main__":
    main()
