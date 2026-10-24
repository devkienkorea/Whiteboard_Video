# Tiến độ

> File này là "bộ nhớ" giữa các phiên Claude. Đầu phiên: đọc. Cuối phiên: cập nhật.

## Trạng thái hiện tại (2026-10-24)
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
- [x] **TTS đọc đơn vị** (2026-10-23): `tts.MEASURES` — sau số: °C/°F/°, km/h, km, cm, mm, kg, mg, ml → chữ ("độ C", "ki-lô-mét"…). Không có số đứng trước thì giữ nguyên. Đã nghe thử VieNeu + whisper: đọc đúng "độ C". Chưa làm m, g, l (dễ nhầm, vd. 5G, 1m70).
- [x] **Nhiều vật thể / cue trong 1 cảnh** (2026-10-23): kịch bản có `elements: [{cue, object}]` (1–3, đủ mọi vật thể vẽ ra). `pipeline/locate.py`: Qwen3.6 nhìn ảnh (mlx-vlm, cùng model, không tải thêm) khoanh khung → `scene_XX/regions.json` (cache theo ảnh + tên vật thể), chạy trong `build_scenes` trước khi dựng. `strokes.extract(regions=)`: mỗi nét skeleton theo khung chứa phần lớn điểm của nó, khung nhỏ thắng khi chồng nhau, màu theo mảng liền. UI cột "Vật thể: cụm từ = tên tiếng Anh; …". Kịch bản cũ (`cue`) chạy như trước. Đã chạy trọn "Vì sao nên ngủ trưa 20 phút": 12/12 vật thể định vị được, thêm ~20s/video.
- [x] **Bắt tiếng Anh dính vào lời + cue quá sát** (2026-10-23): `script_llm.glued_words()` — từ có dấu tiếng Việt mà không phải 1 âm tiết (regex `SYLLABLE`) → lỗi, LLM viết lại. `validate` bắt 2 cue liên tiếp cách < `CUE_GAP`=4 từ (prompt cũng dặn). Quét 540 từ kịch bản cũ: 0 báo nhầm.
- [x] **Cảnh ngắn vẽ trễ lời** (2026-10-23): `render.schedule` cho tăng tốc tới 6.7x (`min_factor` 0.3 → 0.15). Đo trên 4 video cũ: trễ tệ nhất 2.1s → 0.9s (cảnh còn lại có cue cách 3 từ — luật CUE_GAP chặn từ kịch bản mới). Video cũ chưa render lại.
- [x] **Thử sinh nhạc nền bằng AI local** (2026-10-23): ACE-Step 1.5 turbo (MIT, cho phép thương mại) ở `models/ACE-Step-1.5` (venv riêng qua `uv sync`), script `tools/gen_music.py` không dùng LM. Mac MPS: nạp 180s (lần đầu tải model), sinh 60s nhạc trong 44s, RAM ~11GB. Ra `assets/music/ace_lofi_test.wav` -14.9 LUFS; lồng vào video OK. Cuối bài có 1.8s fade + lặng → lặp nhạc sẽ có khoảng hụt.
- [x] **Cue sát nhau thành lỗi mềm** (2026-10-23): hết 3 lần thử mà chỉ còn lỗi CUE_GAP → vẫn nhận kịch bản (`script_llm.only_soft`). Lý do: chủ đề "ổ cứng cũ" fail cả 3 lần chỉ vì cue cách 1–2 từ, mất cả kịch bản; vẽ trễ chút thì `render.schedule` đã bù.
- [x] **Ghi lại bug đã sửa + sửa pytest quét nhầm `models/`** (2026-10-24): bảng "Bug đã sửa" bên dưới; `testpaths` trong `pyproject.toml`. 50 test pass.
- [x] Video đã làm: `jobs/tai-sao-nen-uong-mot-coc-nuoc-ngay-khi-thuc-day/`, `jobs/3-meo-ngu-ngon-hon-moi-dem/`, `jobs/vi-sao-nen-di-bo-30-phut-moi-ngay/`, `jobs/vi-sao-nen-ngu-trua-20-phut/`

## Bug đã sửa — đừng để tái hiện
> Mỗi dòng: **triệu chứng** → nguyên nhân → cách sửa → chốt chặn (test/luật). Sửa code chạm vào chỗ này thì chạy `.venv/bin/python -m pytest -q` và đọc lại dòng tương ứng.

| Triệu chứng | Nguyên nhân → cách sửa | Chốt chặn |
|---|---|---|
| TTS/LLM tiếp theo bị thiếu RAM, máy treo | Giải phóng model trong cùng tiến trình vẫn giữ ~10GB → mỗi bước nặng chạy `isolated()` (tiến trình con) | Luật trong CLAUDE.md |
| VieNeu không nạp được model (onnxruntime báo file `.data`) | onnxruntime 1.31 chặn `.data` sau symlink cache HF → đặt model thật ở `models/` | Không chuyển về cache HF |
| `pip install` kéo Pillow 13, crash trên Python 3.11 | Pillow 13 không hỗ trợ 3.11 → ghim `pillow<13` | `pyproject.toml` |
| Phụ đề không hiện (ffmpeg báo thiếu filter) | ffmpeg Homebrew không có libass/drawtext → vẽ phụ đề bằng PIL trong `render.py` | Không dùng ASS/drawtext |
| Chữ on-screen dấu sắc dính vào ê/ô | Font Marker Felt → đổi Chalkboard SE Bold | — |
| Chuyển cảnh: phụ đề 2 cảnh chồng mờ / dòng chữ bị cắt ngang | `fade`/`smoothleft` chồng hình, `wipeleft` cắt chữ → dùng `xfade=slideleft` 0.4s; tiếng cắt 0.4s lặng cuối cảnh trước (không crossfade) | Đã kiểm lệch tiếng 21ms |
| Đổi style mà ảnh style cũ vẫn được chọn "tốt nhất" | `generate_best` gộp seed cũ bất kể prompt → chỉ giữ seed cũ khi prompt (chủ thể + style) không đổi | — |
| Vật thể vẽ dở dang, nửa nét thuộc vật thể khác | Chia theo pixel, cạnh khung cắt ngang nét → chia theo **cả nét skeleton** (đa số điểm), khung nhỏ thắng khi chồng. Nới khung 2% làm tệ hơn — đừng thử lại | `test_strokes.py` |
| Mọi vật thể vẽ cùng lúc dù kịch bản có nhiều cue | Gom nét theo khoảng cách: ảnh AI gần như luôn dính thành 1 cụm → định vị bằng Qwen3.6 vision (`locate.py`) | `test_locate.py` |
| TTS đọc "bộrefresh", sai âm | LLM dính từ tiếng Anh vào chữ Việt → `glued_words()` (regex `SYLLABLE`), bắt LLM viết lại. **Không** bắt từ tiếng Anh đứng riêng (video, calo…) kẻo LLM viết lại mãi | `test_bat_tieng_anh_dinh_vao_tu_viet` |
| Vật thể sau vẽ trễ lời ~2s ở cảnh ngắn | Cue cách nhau 1–3 từ + tốc độ vẽ tối đa 3.3x → `CUE_GAP`=4 trong validate/prompt; `render.schedule` `min_factor` 0.3→0.15 (6.7x) | `test_cue_qua_sat_nhau` |
| Mất cả kịch bản chủ đề "ổ cứng cũ" | Hết 3 lần thử chỉ còn lỗi CUE_GAP → `raise` → lỗi CUE_GAP thành lỗi mềm (`only_soft`), vẫn nhận kịch bản | `test_chi_con_cue_sat_la_loi_mem` |
| TTS đọc "5 °C" sai | Ký hiệu đơn vị → `tts.MEASURES` chỉ đổi khi có số đứng trước. **Chưa** thêm m, g, l: nhầm với "5G", "1m70" | `test_normalize_don_vi` |
| `pytest -q` báo 109 lỗi collection | pytest quét cả `models/ACE-Step-1.5` → `testpaths = ["tests"]` trong `pyproject.toml` (2026-10-24) | `pyproject.toml` |

## Quyết định quan trọng (và lý do)
- Định vị vật thể bằng Qwen3.6 vision thay vì sinh từng vật thể 1 ảnh: giữ hình liền mạch (người cầm cốc), không thêm model, ~6s/ảnh. Gom nét theo khoảng cách cũ vô dụng cho việc này: ảnh AI gần như luôn dính thành 1 cụm.
- Chia nét theo từng nét skeleton (đa số điểm) chứ không theo pixel: cạnh khung cắt ngang nét gây nét vẽ dở. Đã thử nới khung 2% → tệ hơn, bỏ.
- FLUX.2 Klein 4B làm model ảnh mặc định: nhanh hơn ~3x Z-Image (52s vs 169s/ảnh), hiểu prompt tốt hơn.
- Model VieNeu đặt ở `models/` chứ không ở cache HF: onnxruntime 1.31 chặn file `.data` sau symlink.
- Mỗi bước nặng chạy tiến trình con: giải phóng model trong cùng tiến trình vẫn giữ ~10GB RAM.
- Chỉ bắt tiếng Anh *dính* vào chữ có dấu, không bắt từ tiếng Anh đứng riêng: "video", "calo", "endorphin" có trong 3/4 kịch bản cũ, bắt hết thì LLM phải viết lại liên tục.
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
- [ ] Nhạc nền: đã có `assets/music/ace_lofi_test.wav` (ACE-Step, không commit vì 11MB). Còn: sinh bài dài hơn video hoặc cắt bỏ đoạn fade+lặng cuối để lặp không hụt
- [ ] Kiểm tra khung xem video trong UI trên trình duyệt thường (Chrome tự động hoá không tải media nên chưa xem được)
- [ ] Thu âm giọng riêng (5–15s, sạch) bỏ vào `assets/voices/` để dùng clone
- [ ] Config YAML (plan giai đoạn 5) — chưa làm; cân nhắc có cần không vì đã có CLI + UI

## Vấn đề đã biết
- Vật thể không có trong `elements` → nét của nó bị chia theo khung gần nhất (đã ép prompt liệt kê đủ). Vật thể ôm/chồng nhau (bộ xương ôm đồng hồ) chia chưa sắc.
- Đầu cảnh tay bay từ ngoài khung vào mất ~0.4–0.7s nên nét đầu chạm giấy trễ lời chừng đó.
- Từ tiếng Anh đứng riêng ("stress", "video") vẫn lọt qua validate.
- Plan gốc: `~/.claude/plans/t-i-mu-n-l-n-wondrous-twilight.md`
