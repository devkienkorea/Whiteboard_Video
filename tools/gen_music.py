"""Thử sinh 1 bài nhạc nền không lời bằng ACE-Step 1.5 turbo (MIT), không dùng LM.
Cài 1 lần: git clone https://github.com/ace-step/ACE-Step-1.5 models/ACE-Step-1.5 && (cd models/ACE-Step-1.5 && uv sync)
Chạy (môi trường riêng của ACE-Step):  cd models/ACE-Step-1.5 && uv run python ../../tools/gen_music.py
"""
import shutil
import sys
import time
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

ACE = Path.cwd()
OUT = Path("/Users/hbm/Claude/Projects/whiteboard-video/assets/music")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
dit = AceStepHandler()
status, ok = dit.initialize_service(project_root=str(ACE), config_path="acestep-v15-turbo", device="auto")
print(f"Nạp model: {time.time() - t0:.0f}s — {status}")
if not ok:
    sys.exit(1)

params = GenerationParams(
    task_type="text2music",
    caption="calm lo-fi hip hop instrumental, soft piano, warm bass, gentle drums, relaxing, background music",
    lyrics="[Instrumental]", instrumental=True,
    duration=60, bpm=85, seed=42, inference_steps=8, shift=3.0, infer_method="ode",
    thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")

t0 = time.time()
result = generate_music(dit, None, params, config, save_dir=str(OUT / "_tmp"))
if not result.success:
    sys.exit(f"Lỗi: {result.error}")
dst = OUT / "ace_lofi_test.wav"
shutil.move(result.audios[0]["path"], dst)
shutil.rmtree(OUT / "_tmp", ignore_errors=True)
print(f"Sinh nhạc: {time.time() - t0:.0f}s → {dst}")
