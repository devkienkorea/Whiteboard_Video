"""Bước 2: đọc lời thoại tiếng Việt thành file WAV (chạy offline).

Engine:
  vieneu VieNeu-TTS v3 Turbo (mặc định): tự nhiên, 48kHz, 25 giọng 3 miền, chạy ONNX trên CPU.
         Clone giọng riêng: bỏ 1 file ghi âm sạch 5–15s vào assets/voices/, dùng --voice assets/voices/ten.wav
  piper  giọng Piper (models/piper/<voice>.onnx), nhanh, nhẹ, hơi đều đều
  say    giọng "Linh" có sẵn trong macOS, không cần tải gì

Chạy: python -m pipeline.tts "Xin chào các bạn" --out jobs/demo/voice.wav [--engine piper] [--voice "Ngọc Huyền"]
      python -m pipeline.tts --list-voices
"""
import argparse
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050  # cho piper/say; vieneu giữ nguyên 48kHz
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "piper"
VIENEU_ONNX = ROOT / "models" / "vieneu" / "onnx_update"
VIENEU_CODEC = ROOT / "models" / "moss-codec"
VOICES = ROOT / "assets" / "voices"  # file ghi âm giọng riêng để clone
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DEFAULT_VOICE = {"vieneu": "Ngọc Huyền", "piper": "vi_VN-vais1000-medium", "say": "Linh"}

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
UNITS = [(10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn")]


def _below_1000(n, full):
    """Đọc số 0–999. full=True thì đọc đủ "không trăm", "lẻ" (khi đứng sau hàng lớn hơn)."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    words = []
    if h or full:
        words += [DIGITS[h], "trăm"]
    if t == 0:
        if u and (h or full):
            words.append("lẻ")
    elif t == 1:
        words.append("mười")
    else:
        words += [DIGITS[t], "mươi"]
    if u:
        if u == 1 and t >= 2:
            words.append("mốt")
        elif u == 5 and t >= 1:
            words.append("lăm")
        elif u == 4 and t >= 2:
            words.append("tư")
        else:
            words.append(DIGITS[u])
    return words


def number_to_words(n):
    if n == 0:
        return "không"
    words, started = [], False
    for value, name in UNITS:
        if n >= value:
            words += number_to_words(n // value).split() + [name]
            n %= value
            started = True
    if n:
        words += _below_1000(n, full=started)
    return " ".join(words)


def normalize(text):
    """Đổi số, %, ký hiệu thành chữ để TTS đọc đúng."""
    text = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + " phần trăm", text)
    # 1.000.000 → 1000000 (dấu chấm phân cách hàng nghìn kiểu Việt)
    text = re.sub(r"\d{1,3}(?:\.\d{3})+(?!\d)", lambda m: m.group(0).replace(".", ""), text)
    text = re.sub(r"(\d+),(\d+)", lambda m: f"{number_to_words(int(m.group(1)))} phẩy {number_to_words(int(m.group(2)))}", text)
    text = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), text)
    for sym, word in (("&", " và "), ("+", " cộng "), ("=", " bằng "), ("/", " trên ")):
        text = text.replace(sym, word)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p.strip()]


_vieneu = None


def load_vieneu():
    """Nạp VieNeu 1 lần. Model đọc từ models/ (file thật) vì onnxruntime mới chặn
    file .data nằm sau symlink trong cache HuggingFace."""
    global _vieneu
    if _vieneu is None:
        if not VIENEU_ONNX.exists() or not VIENEU_CODEC.exists():
            from huggingface_hub import snapshot_download
            snapshot_download("pnnbao-ump/VieNeu-TTS-v3-Turbo", local_dir=ROOT / "models" / "vieneu",
                              allow_patterns=["onnx_update/*", "*.json"])
            snapshot_download("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", local_dir=VIENEU_CODEC,
                              allow_patterns=["moss_audio_tokenizer_*", "codec_browser_onnx_meta.json"])
        import vieneu._v3_turbo_engine.onnx_runtime_lite as orl
        from vieneu import Vieneu

        # Vieneu() không chuyển codec_dir xuống engine → gán mặc định ở đây
        init = orl.OnnxV3LiteEngine.__init__
        if not getattr(init, "_patched", False):
            def patched(self, *a, **k):
                k.setdefault("codec_dir", str(VIENEU_CODEC))
                init(self, *a, **k)
            patched._patched = True
            orl.OnnxV3LiteEngine.__init__ = patched
        _vieneu = Vieneu(onnx_dir=str(VIENEU_ONNX))
    return _vieneu


def custom_voices():
    """Các file giọng riêng trong assets/voices/, dạng đường dẫn tương đối dùng được cho `voice`."""
    if not VOICES.exists():
        return []
    return [str(p.relative_to(ROOT)) for p in sorted(VOICES.iterdir()) if p.suffix.lower() in AUDIO_EXT]


def _synth_vieneu(sentences, voice_name, speed):
    tts = load_vieneu()
    # voice là file ghi âm → clone giọng đó (VieNeu tự cache theo nội dung file); còn lại là tên giọng có sẵn
    if Path(voice_name).suffix.lower() in AUDIO_EXT:
        ref = ROOT / voice_name
        if not ref.exists():
            raise FileNotFoundError(f"Không thấy file giọng mẫu: {ref}")
        kw = {"ref_audio": str(ref)}
    else:
        kw = {"voice": voice_name}
    clips = [np.asarray(tts.infer(s, **kw), np.float32) for s in sentences]
    if speed != 1.0:
        clips = [_change_speed(c, speed, tts.sample_rate) for c in clips]
    return clips, tts.sample_rate


def _change_speed(audio, speed, sr):
    """Đổi tốc độ giữ cao độ bằng ffmpeg atempo."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.wav", Path(d) / "b.wav"
        _write_wav(src, audio, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-filter:a", f"atempo={speed}", dst], check=True)
        return _read_wav(dst, sr)


def _synth_piper(sentences, voice_name, speed):
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(MODELS / f"{voice_name}.onnx")
    cfg = SynthesisConfig(length_scale=1.0 / speed)
    out = []
    for s in sentences:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wf:
                voice.synthesize_wav(s, wf, syn_config=cfg)
            out.append(_read_wav(tmp.name))
    return out, SAMPLE_RATE


def _synth_say(sentences, voice_name, speed):
    out = []
    for s in sentences:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(["say", "-v", voice_name, "-r", str(int(175 * speed)), "-o", tmp.name,
                            "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}", s], check=True)
            out.append(_read_wav(tmp.name))
    return out, SAMPLE_RATE


def _read_wav(path, target_sr=SAMPLE_RATE):
    with wave.open(str(path), "rb") as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        data = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    if ch > 1:
        data = data.reshape(-1, ch).mean(1)
    if sr != target_sr:
        x = np.linspace(0, len(data) - 1, int(len(data) * target_sr / sr))
        data = np.interp(x, np.arange(len(data)), data)
    return data.astype(np.float32)


def _trim(audio, thresh=0.01):
    idx = np.nonzero(np.abs(audio) > thresh)[0]
    return audio[idx[0]:idx[-1] + 1] if idx.size else audio


def _write_wav(path, audio, sr):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def synthesize(text, out, engine="vieneu", voice=None, speed=1.0, pause=0.25):
    """Đọc text → WAV. Trả về thời lượng (giây)."""
    voice = voice or DEFAULT_VOICE[engine]
    sentences = [normalize(s) for s in split_sentences(text)]
    synth = {"vieneu": _synth_vieneu, "piper": _synth_piper, "say": _synth_say}[engine]
    clips, sr = synth(sentences, voice, speed)
    gap = np.zeros(int(pause * sr), np.float32)
    pieces = [np.zeros(int(0.15 * sr), np.float32)]
    for clip in clips:
        pieces += [_trim(clip), gap]
    audio = np.concatenate(pieces)
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.9
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _write_wav(out, audio, sr)
    return len(audio) / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="lời thoại, hoặc @file.txt")
    ap.add_argument("--out")
    ap.add_argument("--engine", choices=list(DEFAULT_VOICE), default="vieneu")
    ap.add_argument("--voice", help='vd. "Ngọc Huyền" (vieneu), vi_VN-vais1000-medium (piper), Linh (say)')
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--list-voices", action="store_true", help="liệt kê giọng VieNeu")
    args = ap.parse_args()
    if args.list_voices:
        for label, _ in load_vieneu().list_preset_voices():
            print(label)
        for path in custom_voices():
            print(f"(giọng riêng) {path}")
        return
    if not args.text or not args.out:
        ap.error("cần text và --out")
    text = Path(args.text[1:]).read_text() if args.text.startswith("@") else args.text
    secs = synthesize(text, args.out, args.engine, args.voice, args.speed)
    print(f"{args.out} ({secs:.1f}s)")


if __name__ == "__main__":
    main()
