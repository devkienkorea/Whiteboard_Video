"""So sánh các giọng TTS trên cùng lời thoại: tốc độ sinh, độ dài, whisper nghe lại có đúng không."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import align, tts  # noqa: E402

TEXT = ("Bạn có biết, 70% cơ thể chúng ta là nước? Mỗi sáng, chỉ cần uống 1 cốc nước ấm "
        "là đã giúp cơ thể tỉnh táo hơn. Đây là thói quen nhỏ, nhưng sau 30 ngày, "
        "bạn sẽ thấy sự khác biệt rõ rệt!")
CANDIDATES = [
    ("piper", "vi_VN-vais1000-medium"),
    ("vieneu", "Ngọc Huyền"),
    ("vieneu", "Thiện Minh"),
    ("vieneu", "Mai Anh"),
    ("vieneu", "Quang Sơn"),
    ("vieneu", "Thục Đoan"),
]


def main(out_dir="jobs/tts_compare"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spoken = tts.normalize(TEXT)
    rows = []
    if any(e == "vieneu" for e, _ in CANDIDATES):
        t0 = time.time()
        tts.load_vieneu()
        print(f"nạp VieNeu: {time.time() - t0:.1f}s")
    for engine, voice in CANDIDATES:
        name = f"{engine}_{voice}".replace(" ", "_")
        wav = out / f"{name}.wav"
        t0 = time.time()
        secs = tts.synthesize(TEXT, wav, engine, voice)
        gen = time.time() - t0
        _, ratio = align.align(wav, spoken, out / f"{name}.words.json")
        heard = " ".join(w["word"] for w in json.loads((out / f"{name}.words.json").read_text())["heard"])
        rows.append({"engine": engine, "voice": voice, "audio_s": round(secs, 1), "gen_s": round(gen, 1),
                     "realtime_x": round(secs / gen, 1), "whisper_match": round(ratio, 3), "heard": heard})
        print(f"{engine:7} {voice:22} {secs:5.1f}s audio, sinh {gen:5.1f}s ({secs / gen:4.1f}x), whisper khớp {ratio:.0%}")
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
