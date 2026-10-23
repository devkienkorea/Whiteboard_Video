# Tiến độ

> File này là "bộ nhớ" giữa các phiên Claude. Đầu phiên: đọc. Cuối phiên: cập nhật.

## Trạng thái hiện tại (2026-10-23)
Pipeline 7 bước đã chạy trọn đầu–cuối, gồm cả phụ đề karaoke + nhạc nền: `make_video.py "<chủ đề>"` → `jobs/<slug>/final.mp4` (~15 phút/video 7 cảnh).

## Đã xong
- [x] Tách nét lineart → `strokes.json` (`pipeline/strokes.py`)
- [x] Render bàn tay vẽ + tô màu → MP4 1080x1920 (`pipeline/render.py`)
- [x] TTS tiếng Việt offline: VieNeu v3 Turbo (mặc định), Piper, macOS Linh (`pipeline/tts.py`)
- [x] Căn thời gian từng từ bằng mlx-whisper + phụ đề SRT (`pipeline/align.py`)
- [x] Dựng 1 cảnh: vẽ vật thể đúng lúc được nhắc (`pipeline/scene.py`)
- [x] Sinh ảnh doodle FLUX.2 Klein 4B / Z-Image-Turbo, nhiều seed, chọn ảnh điểm cao nhất (`pipeline/image_gen.py`, `pipeline/quality.py`)
- [x] Kịch bản bằng Qwen3.6-35B-A3B qua mlx-lm (`pipeline/script_llm.py`)
- [x] Chạy trọn chủ đề → video, mỗi bước nặng ở tiến trình con riêng (`make_video.py`)
- [x] **Giai đoạn 7** (2026-10-23): phụ đề karaoke `pipeline/subtitles.py` (gọi trong `render.py` qua `words=`), nhạc nền + ducking + loudnorm `finish_audio()` trong `make_video.py` (`--music`). Đã kiểm: final -14.3 LUFS, peak -1.4 dBFS, nhạc giảm ~14 dB khi có lời.
- [x] **Giai đoạn 6a – Web UI** (2026-10-23): `ui.py` (Gradio 6.30) duyệt 4 bước. `make()` tách thành `write_script / gen_images / build_scenes / finish` trên thư mục job, CLI và UI dùng chung. `generate_best` sinh lại thì seed nối tiếp + giữ điểm cũ. Đã test logic (sửa lời/đổi ảnh → cảnh 'cũ', dựng lại, xuất, seed 1→2→3) và bấm thử UI trên Chrome.
- [x] **Giai đoạn 6b – Batch** (2026-10-23): `batch.py topics.txt` gọi `make()` từng chủ đề, lỗi thì log (`jobs/batch.log`) rồi đi tiếp, đã có `final.mp4` thì bỏ qua. Đã test: bỏ qua / lỗi kịch bản / dựng thành công.
- [x] **Giai đoạn 6c – Clone giọng + style** (2026-10-23): `voice` là file ghi âm (`assets/voices/*.wav`) → VieNeu `ref_audio` clone; `--style ink|color` truyền xuống `generate_best` (CLI, batch, UI). Đã kiểm: clone từ giọng nam 119Hz → ra 122Hz (giọng mặc định 221Hz), whisper khớp 100%; ink 2/2 ảnh điểm 100, render chỉ nét đen.
- [x] **Thumbnail/cover** (2026-10-23): `pipeline/cover.py`, `finish()` tạo `cover.jpg` nếu chưa có; UI tab 4 xem + tạo lại từ cảnh đang chọn. Hàm xuống dòng dùng chung `subtitles.wrap()`. Đã kiểm 2 job thật + tiêu đề dài (tự thu chữ còn 3 dòng).
- [x] **Chữ on-screen** (2026-10-23): `script_llm` thêm `on_screen_text` (≤5 từ, có kiểm tra), `scene.json` → `text`, `render.text_layer()` + bước `write` trong timeline (tay viết trái→phải trước khi vẽ hình). UI thêm cột + đánh dấu cảnh cũ khi đổi chữ. Đã kiểm khung hình: chữ to, cân 2 dòng, không đè hình/phụ đề. Chạy trọn chủ đề mới "Vì sao nên đi bộ 30 phút mỗi ngày" (13 phút, 7 cảnh): LLM viết chữ hợp lý cả 7 cảnh.
- [x] **Chuyển cảnh** (2026-10-23): `concat()` dùng `xfade=slideleft` 0.4s; tiếng cắt 0.4s lặng cuối cảnh trước thay vì crossfade. Đã kiểm: thời lượng = tổng cảnh − 6×0.4s, lệch tiếng 21ms. Đã xuất lại video "đi bộ" + "3 mẹo" (bỏ nhạc test).
- [x] **Git + test lõi** (2026-10-23): `git init` (bỏ `jobs/`, `models/`, `*.mp4`), `tests/` 28 test pytest ~3s không nạp model: align/find_cue/chunk/SRT, validate kịch bản, wrap + Karaoke, số → chữ, `ui.scene_state`. Chạy: `.venv/bin/python -m pytest -q`.
- [x] Video đã làm: `jobs/tai-sao-nen-uong-mot-coc-nuoc-ngay-khi-thuc-day/`, `jobs/3-meo-ngu-ngon-hon-moi-dem/`, `jobs/vi-sao-nen-di-bo-30-phut-moi-ngay/`

## Quyết định quan trọng (và lý do)
- FLUX.2 Klein 4B làm model ảnh mặc định: nhanh hơn ~3x Z-Image (52s vs 169s/ảnh), hiểu prompt tốt hơn.
- Model VieNeu đặt ở `models/` chứ không ở cache HF: onnxruntime 1.31 chặn file `.data` sau symlink.
- Mỗi bước nặng chạy tiến trình con: giải phóng model trong cùng tiến trình vẫn giữ ~10GB RAM.
- Phụ đề vẽ bằng PIL trong render, không dùng ASS: ffmpeg Homebrew không có libass/drawtext. Đổi kiểu phụ đề → phải render lại cảnh.
- Phụ đề đặt giữa ~76.5% chiều cao (hình chiếm 10–72%), lề 12% hai bên, font Arial Bold hệ thống.
- Web UI không có database: trạng thái cảnh suy ra từ file (`image.png`, `scene.json`, `scene.mp4` + mtime) → UI và CLI không bao giờ lệch nhau.
- "Cũ" = video cũ hơn image.png/scene.json, hoặc lời/cue trong scene.json khác script.json. Đổi giọng không tự đánh dấu cũ → dùng nút "Dựng lại tất cả".
- `generate_best` giữ seed cũ khi **prompt** (chủ thể + style) không đổi → đổi style thì ảnh cũ không lẫn vào lựa chọn tốt nhất.
- Style không lưu trong script.json: CLI/batch dùng 1 style cho cả video; trong UI phải tự giữ nguyên.
- Chữ on-screen dùng Chalkboard SE Bold (Marker Felt bị dính dấu sắc trên ê/ô). Có chữ → hình đặt ở box (0.05, 0.22, 0.90, 0.50); `strokes.json` lưu `box`, đổi box thì tự tách nét lại.
- Chọn `slideleft` sau khi so 4 kiểu: `fade`/`smoothleft` làm phụ đề 2 cảnh chồng mờ lên nhau, `wipeleft` cắt ngang dòng chữ.
- Bìa nằm trong khung 3:4 giữa ảnh (y 12.5–87.5%) vì lưới trang cá nhân TikTok cắt bìa như vậy. Đổi tiêu đề → phải bấm tạo lại bìa (không tự cập nhật).
- Bước nào đã có file thì bỏ qua → sửa `script.json` / xoá `image.png` để làm lại đúng phần đó.

## Việc tiếp theo
Danh sách đầy đủ + trạng thái: https://claude.ai/artifact/Y75QUxinA8UCLqAWrZL1ns
<!-- TODO: điền mục tiêu tiếp theo, ví dụ: chất lượng ảnh, nhạc nền, hiển thị phụ đề, UI... -->
- [ ] Bỏ nhạc nền không bản quyền thật vào `assets/music/` (hiện chưa có bài nào)
- [ ] Kiểm tra khung xem video trong UI trên trình duyệt thường (Chrome tự động hoá không tải media nên chưa xem được)
- [ ] Thu âm giọng riêng (5–15s, sạch) bỏ vào `assets/voices/` để dùng clone
- [ ] Config YAML (plan giai đoạn 5) — chưa làm; cân nhắc có cần không vì đã có CLI + UI

## Vấn đề đã biết
- `tts.normalize("25°C")` → "hai mươi lăm°C": chưa đọc ký hiệu °C/°, km, kg… (LLM có thể viết vào narration).
- Plan gốc: `~/.claude/plans/t-i-mu-n-l-n-wondrous-twilight.md`
