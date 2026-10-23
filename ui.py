"""Web UI duyệt từng bước: kịch bản → ảnh → dựng cảnh → xuất video, sửa/làm lại từng phần trước khi đi tiếp.

Chạy: .venv/bin/python ui.py   → mở http://127.0.0.1:7860
Mọi trạng thái nằm trong jobs/<slug>/ (giống make_video.py): tắt UI rồi mở lại vẫn làm tiếp được,
và job làm dở bằng UI vẫn chạy tiếp được bằng make_video.py. Bước nặng vẫn chạy trong tiến trình con riêng.
"""
import json
import shutil
from pathlib import Path

import gradio as gr

import make_video as mv
from pipeline import cover, image_gen

JOBS = mv.ROOT / "jobs"
MUSIC = mv.ROOT / "assets" / "music"
NO_MUSIC = "(không nhạc)"


def _list_voices():
    from pipeline import tts

    return tts.load_vieneu().list_preset_voices()  # [(nhãn hiển thị, tên giọng)]


def voice_choices(presets):
    """Giọng riêng trong assets/voices/ (clone) lên đầu, rồi tới giọng có sẵn của VieNeu."""
    from pipeline import tts

    return [(f"🎙 {Path(p).stem} (giọng riêng)", p) for p in tts.custom_voices()] + presets


def job_names():
    return sorted(p.name for p in JOBS.iterdir() if (p / "script.json").exists())


def music_files():
    return [NO_MUSIC] + sorted(p.name for p in MUSIC.glob("*") if p.is_file()) if MUSIC.exists() else [NO_MUSIC]


def need(job):
    if not job:
        raise gr.Error("Chưa chọn job.")
    return JOBS / job


def scene_state(job, i):
    """'missing' / 'stale' (ảnh hoặc lời đã đổi sau lần dựng) / 'ok' cho video của cảnh i."""
    d = mv.scene_dir(job, i)
    img, vid, sj = d / "image.png", d / "scene.mp4", d / "scene.json"
    if not vid.exists() or not img.exists():
        return "missing"
    sc = mv.load_script(job)["scenes"][i]
    built = json.loads(sj.read_text())
    # scene.json mới hơn video = lần dựng trước bị hỏng giữa chừng
    if (max(img.stat().st_mtime, sj.stat().st_mtime) > vid.stat().st_mtime or built["narration"] != sc["narration"]
            or built["cues"] != [sc["cue"]] or built.get("text", "") != sc.get("on_screen_text", "")):
        return "stale"
    return "ok"


def status(job):
    if not job:
        return "Chọn job có sẵn hoặc viết kịch bản cho chủ đề mới."
    job = JOBS / job
    label = {"missing": "–", "stale": "cũ, cần dựng lại", "ok": "✓"}
    rows = ["| Cảnh | Ảnh | Video cảnh |", "|---|---|---|"]
    for i in range(len(mv.load_script(job)["scenes"])):
        has_img = (mv.scene_dir(job, i) / "image.png").exists()
        rows.append(f"| {i + 1} | {'✓' if has_img else '–'} | {label[scene_state(job, i)]} |")
    if (job / "final.mp4").exists():
        rows.append("\nĐã có `final.mp4` (xem tab 4).")
    return "\n".join(rows)


# --- 1. kịch bản ---

def open_job(job):
    if not job:
        return "", [], gr.update(choices=[], value=None), status(job)
    s = mv.load_script(need(job))
    rows = [[sc["narration"], sc["image_subject"], sc["cue"], sc.get("on_screen_text", "")] for sc in s["scenes"]]
    return s.get("title", ""), rows, gr.update(choices=list(range(1, len(rows) + 1)), value=1), status(job)


def new_job(topic, seconds):
    if not topic.strip():
        raise gr.Error("Nhập chủ đề trước.")
    job = JOBS / mv.slugify(topic)
    if (job / "script.json").exists():
        gr.Info(f"Job {job.name} đã có kịch bản → mở lại (xoá script.json nếu muốn viết lại).")
    else:
        lo, hi = (int(x) for x in seconds.split(","))
        mv.write_script(job, topic, (lo, hi))
    return gr.update(choices=job_names(), value=job.name)


def save_script(job, title, rows):
    path = need(job) / "script.json"
    s = json.loads(path.read_text())
    if len(rows) != len(s["scenes"]):
        raise gr.Error(f"Không đổi được số cảnh ở đây (đang có {len(s['scenes'])}).")
    for k, (narration, _, cue, _) in enumerate(rows, 1):
        if cue.strip().lower() not in narration.lower():
            gr.Warning(f"Cảnh {k}: cue '{cue}' không có trong lời thoại → vật thể sẽ được vẽ ngay từ đầu.")
    s["title"] = title
    s["scenes"] = [{"narration": n, "image_subject": sub, "cue": c, "on_screen_text": t.strip()} for n, sub, c, t in rows]
    path.write_text(json.dumps(s, ensure_ascii=False, indent=1))
    gr.Info("Đã lưu kịch bản.")
    return status(job)


# --- 2. ảnh ---

def scene_view(job, scene):
    """Ảnh các seed (điểm cao trước), ảnh đang dùng, video cảnh."""
    if not job or not scene:
        return [], [], None, None
    d = mv.scene_dir(JOBS / job, int(scene) - 1)
    scores = d / "gen" / "scores.json"
    results = sorted(json.loads(scores.read_text())["results"], key=lambda r: -r["score"]) if scores.exists() else []
    items = [(str(d / "gen" / r["path"]), f"seed {r['seed']} · điểm {r['score']:.0f}") for r in results]
    img, vid = d / "image.png", d / "scene.mp4"
    return items, [p for p, _ in items], str(img) if img.exists() else None, str(vid) if vid.exists() else None


def pick_image(job, scene, paths, evt: gr.SelectData):
    shutil.copy(paths[evt.index], mv.scene_dir(need(job), int(scene) - 1) / "image.png")
    return (*scene_view(job, scene), status(job))


def gen_missing(job, seeds, model, style):
    job_dir = need(job)
    todo = [i for i in range(len(mv.load_script(job_dir)["scenes"]))
            if not (mv.scene_dir(job_dir, i) / "image.png").exists()]
    if not todo:
        gr.Info("Cảnh nào cũng đã có ảnh.")
    else:
        mv.gen_images(job_dir, todo, int(seeds), model, style)
    return status(job)


def gen_more(job, scene, seeds, model, style):
    mv.gen_images(need(job), [int(scene) - 1], int(seeds), model, style)
    return (*scene_view(job, scene), status(job))


# --- 3. dựng cảnh ---

def build(job, scene, voice, which):
    job_dir = need(job)
    n = len(mv.load_script(job_dir)["scenes"])
    if which == "this":
        todo = [int(scene) - 1]
    elif which == "pending":
        todo = [i for i in range(n) if scene_state(job_dir, i) != "ok"]
    else:
        todo = list(range(n))
    no_img = [i + 1 for i in todo if not (mv.scene_dir(job_dir, i) / "image.png").exists()]
    if no_img:
        raise gr.Error(f"Cảnh {no_img} chưa có ảnh (tab 2).")
    if todo:
        mv.build_scenes(job_dir, todo, voice)
    return (*scene_view(job, scene), status(job))


# --- 4. xuất ---

def cover_view(job):
    path = JOBS / job / "cover.jpg" if job else None
    return str(path) if path and path.exists() else None


def make_cover(job, scene):
    if not (mv.scene_dir(need(job), int(scene) - 1) / "image.png").exists():
        raise gr.Error(f"Cảnh {scene} chưa có ảnh.")
    return str(cover.make(JOBS / job, int(scene) - 1))


def export(job, music):
    job_dir = need(job)
    bad = [i + 1 for i in range(len(mv.load_script(job_dir)["scenes"])) if scene_state(job_dir, i) != "ok"]
    if bad:
        raise gr.Error(f"Cảnh {bad} chưa dựng hoặc đã cũ (tab 3).")
    final = mv.finish(job_dir, None if music == NO_MUSIC else MUSIC / music)
    return str(final), cover_view(job), status(job)


def build_ui(voices):
    with gr.Blocks(title="Whiteboard video") as demo:
        gr.Markdown("# Whiteboard video")
        with gr.Row():
            job = gr.Dropdown(job_names(), value=None, label="Job", scale=2)
            topic = gr.Textbox(label="Chủ đề mới", scale=3)
            seconds = gr.Textbox("45,70", label="Độ dài (giây)", scale=1)
            write_btn = gr.Button("① Viết kịch bản", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                scene = gr.Dropdown([], label="Cảnh")
                info = gr.Markdown(status(None))
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("1. Kịch bản"):
                        title = gr.Textbox(label="Tiêu đề")
                        table = gr.Dataframe(headers=["Lời thoại", "Mô tả hình (tiếng Anh)", "Cue (cụm từ bắt đầu vẽ)",
                                                      "Chữ trên màn hình (1–5 từ, để trống = không)"],
                                             type="array", interactive=True, wrap=True,
                                             column_widths=["38%", "28%", "17%", "17%"])
                        save_btn = gr.Button("Lưu kịch bản")
                    with gr.Tab("2. Ảnh"):
                        with gr.Row():
                            seeds = gr.Number(2, label="Số ảnh mỗi lần", precision=0, minimum=1)
                            model = gr.Dropdown(image_gen.MODELS, value="flux2-klein-4b", label="Model ảnh")
                            style = gr.Dropdown([("Nét đen + tô màu", "color"), ("Chỉ nét đen", "ink")],
                                                value="color", label="Kiểu hình (giữ nguyên cho cả video)")
                            missing_btn = gr.Button("Sinh ảnh cho cảnh còn thiếu")
                            more_btn = gr.Button("Sinh thêm ảnh cho cảnh này")
                        with gr.Row():
                            gallery = gr.Gallery(label="Bấm vào 1 ảnh để chọn cho cảnh", columns=4, height=520, fit_columns=False,
                                                 object_fit="contain", allow_preview=False, scale=3)
                            current = gr.Image(label="Ảnh đang dùng", interactive=False, height=520, scale=1)
                        paths = gr.State([])
                    with gr.Tab("3. Dựng cảnh"):
                        with gr.Row():
                            voice = gr.Dropdown(voice_choices(voices), value="Ngọc Huyền",
                                                label="Giọng đọc (file ghi âm trong assets/voices/ = clone)")
                            this_btn = gr.Button("Dựng cảnh này", variant="primary")
                            pending_btn = gr.Button("Dựng các cảnh chưa xong / cũ")
                            all_btn = gr.Button("Dựng lại tất cả (vd. sau khi đổi giọng)")
                        scene_video = gr.Video(label="Video cảnh", height=640)
                    with gr.Tab("4. Xuất video"):
                        with gr.Row():
                            music = gr.Dropdown(music_files(), value=NO_MUSIC,
                                                label="Nhạc nền (bỏ file vào assets/music/)")
                            export_btn = gr.Button("④ Ghép + nhạc nền → final.mp4", variant="primary")
                        with gr.Row():
                            final = gr.Video(label="final.mp4", height=640, scale=1)
                            with gr.Column(scale=1):
                                cover_img = gr.Image(label="Ảnh bìa (cover.jpg)", interactive=False, height=640)
                                cover_btn = gr.Button("Tạo lại ảnh bìa từ cảnh đang chọn")

        view = [gallery, paths, current, scene_video]
        heavy = dict(concurrency_id="heavy", concurrency_limit=1)  # mỗi lúc chỉ 1 bước nặng → không tràn RAM
        job.change(open_job, job, [title, table, scene, info]).then(scene_view, [job, scene], view).then(
            lambda j: str(JOBS / j / "final.mp4") if j and (JOBS / j / "final.mp4").exists() else None, job, final)
        write_btn.click(new_job, [topic, seconds], job, **heavy)
        save_btn.click(save_script, [job, title, table], info)
        scene.change(scene_view, [job, scene], view)
        gallery.select(pick_image, [job, scene, paths], [*view, info])
        missing_btn.click(gen_missing, [job, seeds, model, style], info, **heavy).then(scene_view, [job, scene], view)
        more_btn.click(gen_more, [job, scene, seeds, model, style], [*view, info], **heavy)
        for btn, which in ((this_btn, "this"), (pending_btn, "pending"), (all_btn, "all")):
            btn.click(build, [job, scene, voice, gr.State(which)], [*view, info], **heavy)
        export_btn.click(export, [job, music], [final, cover_img, info], **heavy)
        cover_btn.click(make_cover, [job, scene], cover_img)
        job.change(cover_view, job, cover_img)
        demo.load(lambda: gr.update(choices=music_files()), None, music)
        demo.load(lambda: gr.update(choices=voice_choices(voices)), None, voice)  # thấy giọng mới bỏ vào mà không cần khởi động lại
    return demo


if __name__ == "__main__":
    voices = mv.isolated(_list_voices)
    build_ui(voices).queue().launch(inbrowser=True, allowed_paths=[str(JOBS)])
