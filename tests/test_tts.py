"""Chuẩn hoá số cho TTS tiếng Việt."""
import pytest

from pipeline import tts


@pytest.mark.parametrize("n, words", [
    (0, "không"), (10, "mười"), (15, "mười lăm"), (21, "hai mươi mốt"), (24, "hai mươi tư"),
    (105, "một trăm lẻ năm"), (1005, "một nghìn không trăm lẻ năm"), (1_000_000, "một triệu"),
])
def test_number_to_words(n, words):
    assert tts.number_to_words(n) == words


def test_normalize():
    assert tts.normalize("70% người, 1.500 đồng, 2,5 lít") == \
        "bảy mươi phần trăm người, một nghìn năm trăm đồng, hai phẩy năm lít"
