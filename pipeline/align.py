"""Bước 3: lấy mốc thời gian từng từ của lời đọc bằng whisper (mlx-whisper, chạy trên GPU Mac).

Vì đã biết trước lời thoại, ta khớp từ whisper nghe được với từ trong kịch bản
(difflib) → mỗi từ của kịch bản có start/end, kể cả khi whisper nghe sai chính tả.

Chạy: python -m pipeline.align voice.wav --text "lời thoại" --out words.json [--srt subs.srt]
"""
import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def norm_word(w):
    w = unicodedata.normalize("NFC", w.lower())
    return re.sub(r"[^\w]", "", w)


def tokenize(text):
    """Tách kịch bản thành từ (giữ nguyên dấu câu để hiển thị phụ đề)."""
    return [w for w in text.split() if norm_word(w)]


def transcribe(audio, prompt=None, model=WHISPER_MODEL):
    import mlx_whisper

    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=model, language="vi",
                                    word_timestamps=True, initial_prompt=prompt)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    return words


def align_to_script(script_text, heard):
    """Gán thời gian cho từng từ kịch bản theo các từ whisper nghe được."""
    tokens = tokenize(script_text)
    a = [norm_word(t) for t in tokens]
    b = [norm_word(h["word"]) for h in heard]
    times = [None] * len(tokens)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k]["start"], heard[j1 + k]["end"])
        elif tag == "replace" and j2 > j1:
            # nghe sai chữ: chia đều khoảng thời gian của đoạn whisper cho các từ kịch bản
            t0, t1 = heard[j1]["start"], heard[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n)
    # từ không khớp được: nội suy giữa hai từ lân cận
    for i, t in enumerate(times):
        if t is None:
            prev = next((times[j][1] for j in range(i - 1, -1, -1) if times[j]), 0.0)
            nxt = next((times[j][0] for j in range(i + 1, len(times)) if times[j]), prev + 0.3)
            times[i] = (prev, max(prev, nxt))
    matched = sum(1 for tag, i1, i2, *_ in sm.get_opcodes() if tag == "equal" for _ in range(i2 - i1))
    words = [{"word": w, "start": round(s, 3), "end": round(e, 3)} for w, (s, e) in zip(tokens, times)]
    return words, matched / max(1, len(tokens))


def find_cue(words, phrase, after=0.0):
    """Thời điểm bắt đầu nói cụm từ `phrase` (lần đầu sau `after` giây). None nếu không thấy."""
    target = [norm_word(w) for w in phrase.split() if norm_word(w)]
    seq = [norm_word(w["word"]) for w in words]
    for i in range(len(seq) - len(target) + 1):
        if seq[i:i + len(target)] == target and words[i]["start"] >= after:
            return words[i]["start"]
    return None


def chunk_words(words, max_words=6):
    """Chia lời thành cụm để hiện phụ đề: tối đa max_words từ, ngắt ở dấu câu."""
    lines, chunk = [], []
    for w in words:
        chunk.append(w)
        if len(chunk) >= max_words or re.search(r"[.!?,…:]$", w["word"]):
            lines.append(chunk)
            chunk = []
    if chunk:
        lines.append(chunk)
    return lines


def to_srt(words, max_words=6):
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"

    lines = chunk_words(words, max_words)
    return "\n".join(f"{i}\n{ts(c[0]['start'])} --> {ts(c[-1]['end'])}\n{' '.join(w['word'] for w in c)}\n"
                     for i, c in enumerate(lines, 1))


def align(audio, script_text, out, srt=None):
    heard = transcribe(audio, prompt=script_text)
    words, ratio = align_to_script(script_text, heard)
    Path(out).write_text(json.dumps({"words": words, "heard": heard, "match_ratio": ratio}, ensure_ascii=False, indent=1))
    if srt:
        Path(srt).write_text(to_srt(words))
    return words, ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--text", required=True, help="lời thoại gốc, hoặc @file.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--srt")
    args = ap.parse_args()
    text = Path(args.text[1:]).read_text() if args.text.startswith("@") else args.text
    words, ratio = align(args.audio, text, args.out, args.srt)
    print(f"{len(words)} từ, khớp {ratio:.0%} với kịch bản → {args.out}")


if __name__ == "__main__":
    main()
