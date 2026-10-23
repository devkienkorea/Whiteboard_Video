"""Khớp từ whisper với kịch bản, tìm cue, chia cụm phụ đề — không cần model."""
from pipeline import align


def heard(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_khop_dung_lay_thoi_gian_whisper():
    words, ratio = align.align_to_script("Uống nước, mỗi sáng.",
                                         heard((" uống", 0.0, 0.3), ("nước", 0.3, 0.6), ("mỗi", 0.7, 0.9), ("sáng", 0.9, 1.2)))
    assert ratio == 1.0
    assert [w["word"] for w in words] == ["Uống", "nước,", "mỗi", "sáng."]  # giữ dấu câu để hiện phụ đề
    assert (words[1]["start"], words[3]["end"]) == (0.3, 1.2)


def test_nghe_sai_chu_chia_deu_thoi_gian():
    words, ratio = align.align_to_script("uống nước lọc", heard(("uống", 0.0, 0.4), ("nuốc lóc", 0.4, 1.0)))
    assert ratio == 1 / 3
    assert (words[1]["start"], words[1]["end"]) == (0.4, 0.7)
    assert (words[2]["start"], words[2]["end"]) == (0.7, 1.0)


def test_tu_khong_nghe_thay_noi_suy_giua_hai_tu():
    words, _ = align.align_to_script("một hai ba", heard(("một", 0.0, 0.2), ("ba", 1.0, 1.2)))
    assert (words[1]["start"], words[1]["end"]) == (0.2, 1.0)


def test_find_cue_bo_dau_cau_va_hoa_thuong():
    words = [{"word": w, "start": float(i)} for i, w in enumerate("Cốc nước, rồi lại cốc nước.".split())]
    assert align.find_cue(words, "cốc nước") == 0.0
    assert align.find_cue(words, "Cốc Nước!", after=0.5) == 4.0  # lần nhắc sau mốc `after`
    assert align.find_cue(words, "bánh mì") is None


def test_chunk_words_ngat_o_dau_cau_hoac_du_so_tu():
    words = [{"word": w} for w in "a b, c d e f g h.".split()]
    assert [[w["word"] for w in c] for c in align.chunk_words(words, max_words=4)] == \
        [["a", "b,"], ["c", "d", "e", "f"], ["g", "h."]]


def test_to_srt():
    words = [{"word": "Xin", "start": 0.0, "end": 0.5}, {"word": "chào.", "start": 0.5, "end": 61.25}]
    assert align.to_srt(words) == "1\n00:00:00,000 --> 00:01:01,250\nXin chào.\n"
