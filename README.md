# whiteboard-video

Tự động làm video whiteboard tay vẽ (TikTok 9:16, giọng Việt), chạy local trên Mac.
Kế hoạch đầy đủ: 7 bước (kịch bản LLM → TTS → whisper → sinh ảnh → tách nét → render → ghép).

**Hiện có (Giai đoạn 1–7):**
- `pipeline/strokes.py` — ảnh lineart → nét bút có thứ tự (`strokes.json`, `preview.png` để kiểm tra)
- `pipeline/render.py` — bàn tay vẽ nét, tô màu từng vật thể → MP4 1080x1920
- `make_video.py` — chạy trọn: chủ đề → kịch bản → ảnh → từng cảnh → `final.mp4`
- `pipeline/script_llm.py` — Qwen3.6-35B-A3B (mlx-lm, 4-bit) viết `script.json`: lời thoại, prompt ảnh tiếng Anh, cue từng cảnh
- `pipeline/tts.py` — đọc tiếng Việt offline: VieNeu-TTS v3 Turbo (mặc định, 25 giọng, 48kHz), Piper, hoặc giọng macOS "Linh"; tự đổi số/%/đơn vị (°C, km, kg…) thành chữ
- `pipeline/align.py` — mlx-whisper lấy mốc thời gian từng từ, khớp với kịch bản → `words.json`, `subs.srt`
- `pipeline/scene.py` — chạy trọn 1 cảnh từ `scene.json`: tách nét → giọng đọc → whisper → vẽ từng vật thể đúng lúc được nhắc → MP4 có tiếng
- `pipeline/image_gen.py` — sinh doodle bằng FLUX.2 Klein 4B (mặc định) hoặc Z-Image-Turbo (mflux, 4-bit), mỗi mô tả vài seed, giữ ảnh điểm cao nhất
- `pipeline/quality.py` — chấm điểm 0–100 ảnh có tách nét được không (bóng xám, nét vụn, nền bẩn, độ phủ cọ, chạm mép)
- `pipeline/subtitles.py` — phụ đề karaoke (cụm 4 từ, từ đang đọc tô vàng) vẽ thẳng lên frame, nằm dưới hình, tránh vùng UI TikTok
- Chữ on-screen: `on_screen_text` (1–5 từ, LLM viết, sửa được ở UI) được bàn tay viết tay đầu cảnh bằng font Chalkboard SE (đủ dấu tiếng Việt), ở 9–21% chiều cao; hình thu xuống 22–72%
- `pipeline/cover.py` — ảnh bìa `cover.jpg`: tiêu đề in hoa + vệt bút dạ vàng + hình cảnh 1, nằm gọn trong khung 3:4 của lưới TikTok
- Chuyển cảnh: tờ giấy trượt sang trái 0.4s (`xfade=slideleft`), tiếng giữ nguyên nhờ cắt đoạn lặng cuối cảnh trước; đổi/tắt ở `TRANSITION` trong `make_video.py`
- `make_video.py --music` — nhạc nền tự nhỏ lại khi có lời (sidechain), fade-out, chuẩn âm lượng -14 LUFS
- `tools/make_sample.py` — ảnh doodle mẫu để test

## Làm 1 video từ chủ đề
```bash
.venv/bin/python make_video.py "Tại sao nên uống một cốc nước ngay khi thức dậy" --seeds 2
```
Thêm nhạc nền: `--music assets/music/ten-bai.mp3` (dùng nhạc không bản quyền; không có `--music` thì chỉ chuẩn âm lượng giọng).
Phụ đề vẽ bằng PIL vì ffmpeg Homebrew không có libass/drawtext.
Kết quả ở `jobs/<slug>/final.mp4` + ảnh bìa `cover.jpg` (tạo lại từ cảnh khác: `python -m pipeline.cover jobs/<slug> --scene 3`,
hoặc nút ở tab 4 của UI; bìa đã có thì lần ghép sau không ghi đè). Bước nào đã có file thì bỏ qua: sửa `script.json` hoặc xoá
`scene_XX/image.png` rồi chạy lại để làm lại đúng phần đó. Trên M1 32GB: ~15 phút/video 7 cảnh
(kịch bản ~30s, ảnh ~46s/ảnh, dựng cảnh ~1 phút).

Bộ nhớ: mỗi bước nặng chạy trong tiến trình con riêng, xong là thoát nên RAM được trả lại hết.
Đỉnh đo được: viết kịch bản ~19GB (chỉ ~20–30s), sinh ảnh ~8.4GB (đã giới hạn cache MLX 2GB),
giọng + whisper + render ~4GB. Nên tắt bớt app nặng (máy ảo Docker/Colima, nhiều tab Chrome) khi chạy.

## Làm nhiều video (chạy qua đêm)
```bash
.venv/bin/python batch.py topics.txt --music assets/music/ten-bai.mp3
```
`topics.txt`: mỗi dòng 1 chủ đề (dòng trống / `#` bỏ qua). Chủ đề lỗi được ghi vào `jobs/batch.log` rồi làm tiếp
chủ đề sau; chủ đề đã có `final.mp4` thì bỏ qua → chạy lại cùng file là làm tiếp phần dở. ~15 phút/video.

## Web UI duyệt từng bước
```bash
.venv/bin/python ui.py     # mở http://127.0.0.1:7860
```
Chọn job có sẵn hoặc gõ chủ đề → ① Viết kịch bản. Sau đó lần lượt các tab:
1. **Kịch bản**: sửa lời thoại / mô tả hình / cue rồi Lưu (không đổi số cảnh ở đây).
2. **Ảnh**: sinh ảnh cảnh còn thiếu, hoặc "sinh thêm" cho 1 cảnh (seed nối tiếp, không trùng ảnh cũ); bấm ảnh để chọn.
3. **Dựng cảnh**: chọn giọng, dựng 1 cảnh / các cảnh chưa xong hoặc cũ / tất cả, xem video cảnh.
4. **Xuất video**: chọn nhạc trong `assets/music/` → `final.mp4`.

Bảng bên trái cho biết cảnh nào thiếu ảnh hoặc video đã cũ (ảnh/lời đổi sau lần dựng). Mỗi lúc chỉ chạy 1 bước
nặng. Trạng thái nằm trong `jobs/<slug>/` nên làm dở ở UI vẫn chạy tiếp được bằng `make_video.py` và ngược lại.

## Test
```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q     # ~3 giây, không nạp model
```
Bao phần lõi không cần model: khớp whisper ↔ kịch bản, tìm cue, chia cụm/xuống dòng phụ đề, kiểm tra kịch bản LLM,
đổi số thành chữ cho TTS, trạng thái cảnh thiếu/cũ trong UI.

## Chạy thử từng phần
```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python tools/make_sample.py jobs/sample.png
.venv/bin/python -m pipeline.strokes jobs/sample.png --out jobs/demo
.venv/bin/python -m pipeline.render jobs/demo --out jobs/demo/scene.mp4 --duration 10
```
`--duration` ép tổng thời lượng (sau này = độ dài giọng đọc của cảnh).
Thứ tự vẽ có thể sửa tay trong `strokes.json` (đổi thứ tự mảng `elements`).

## Chạy 1 cảnh có giọng đọc
```bash
.venv/bin/pip install -e . piper-tts mlx-whisper
.venv/bin/python -m pipeline.scene jobs/demo/scene.json
```
Xem mẫu `scene.json` trong docstring của `pipeline/scene.py`. `cues` là cụm từ trong lời thoại
kích hoạt từng vật thể (theo thứ tự trong `strokes.json`, xem số trên `preview.png`).
Lần đầu chạy whisper sẽ tải model `whisper-large-v3-turbo` (~1.6GB).

## Sinh hình bằng AI (local)
```bash
.venv/bin/python -m pipeline.image_gen "a piggy bank with gold coins" --out jobs/gen --seeds 3
.venv/bin/python -m pipeline.quality jobs/gen/*/seed_*.png
```
Benchmark trên M1 32GB (832x1024, 4-bit, 12 ảnh):

| Model | s/ảnh | RAM đỉnh | Ghi chú |
|---|---|---|---|
| FLUX.2 Klein 4B (mặc định) | ~52s | 10.5GB | hiểu prompt tốt hơn (vẽ đúng "người uống nước") |
| Z-Image-Turbo (`--model z-image-turbo`) | ~169s | 8.9GB | hình to, tròn trịa, dễ thương hơn |

Cả hai đều Apache-2.0, không cần đăng nhập HuggingFace (FLUX.1-schnell gốc thì bị khóa).

## Giọng đọc
```bash
.venv/bin/python -m pipeline.tts --list-voices
.venv/bin/python -m pipeline.tts "Xin chào" --out test.wav --voice "Thiện Minh"
.venv/bin/python tools/compare_tts.py   # so sánh nhiều giọng trên cùng lời thoại
```
Lần đầu tự tải VieNeu về `models/vieneu` + `models/moss-codec` (~540MB). Model phải nằm ngoài cache
HuggingFace vì onnxruntime 1.31 chặn file `.data` sau symlink. Chọn giọng trong `scene.json` → `tts.voice`.
Piper vẫn dùng được: `"tts": {"engine": "piper"}` (cần tải `vi_VN-vais1000-medium` vào `models/piper`).
VieNeu mặc định có gắn watermark âm thanh (không nghe thấy).

**Clone giọng riêng:** ghi âm 5–15 giây giọng nói rõ, không nhạc/tiếng ồn (wav/mp3/m4a), bỏ vào `assets/voices/`, rồi
`--voice assets/voices/ten-file.wav` (CLI, batch) hoặc chọn mục 🎙 trong ô Giọng đọc của UI. Lần đầu mất ~20s để
VieNeu học giọng, các lần sau dùng lại cache. Chỉ clone giọng của bạn hoặc người đã đồng ý.

## Kiểu hình
`--style color` (mặc định): nét đen + tô màu phẳng. `--style ink`: chỉ nét đen kiểu whiteboard cổ điển, tách nét
sạch hơn. Dùng 1 kiểu cho cả video (UI: ô "Kiểu hình" ở tab 2).
