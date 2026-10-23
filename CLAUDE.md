# whiteboard-video

Pipeline local (Mac M1 32GB) biến 1 chủ đề → video whiteboard TikTok 9:16 giọng Việt.
Cách chạy và chi tiết từng module: xem `README.md`.

## Đầu mỗi phiên
1. Đọc `PROGRESS.md` để biết đang làm tới đâu, việc tiếp theo là gì.
2. Không đọc lại toàn bộ code nếu `PROGRESS.md` đã đủ thông tin — chỉ mở file liên quan tới việc đang làm.

## Trước khi viết code
- Luôn dùng skill `andrej-karpathy-skills:karpathy-guidelines`: thay đổi tối thiểu, không thêm code/abstraction/tham số khi chưa cần, nêu rõ giả định, có tiêu chí kiểm tra được.
- Khi thiết kế tính năng mới, đổi kiến trúc, hoặc thêm module: dùng `mattpocock-skills:codebase-design` (thiết kế module sâu, giao diện, ranh giới) trước khi code. Nếu ý tưởng còn mơ hồ, dùng `mattpocock-skills:grilling` để chốt yêu cầu.

## Cuối mỗi phiên (hoặc khi xong 1 việc)
Cập nhật `PROGRESS.md`: chuyển việc đã xong sang "Đã xong" (kèm ngày), ghi quyết định quan trọng + lý do,
cập nhật "Việc tiếp theo". Viết ngắn, đủ để phiên mới đọc là làm tiếp được.

Cập nhật luôn trang theo dõi trên web https://claude.ai/artifact/Y75QUxinA8UCLqAWrZL1ns bằng tool `ArtifactData`
(collection `tasks`, mỗi việc 1 document: `status` todo|doing|done, `done_at` YYYY-MM-DD, `owner` claude|ban,
`phase`, `order`, `title`, `detail`). Đọc document trước rồi `update` với `if_version`; việc mới thì `set` document mới.
Người dùng cũng tick trạng thái trên trang → đọc trang trước khi ghi đè. Mã nguồn trang: `docs/tien-do.html`.

## Quy ước
- Python 3.11, chạy bằng `.venv/bin/python`. Comment/docstring/log bằng tiếng Việt, giống code hiện có.
- Bước nặng (LLM, sinh ảnh, TTS+whisper) phải chạy trong tiến trình con riêng (`isolated()` trong `make_video.py`) để trả RAM.
- `jobs/` và `models/` là dữ liệu, không commit.
