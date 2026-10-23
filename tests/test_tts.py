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


@pytest.mark.parametrize("text, spoken", [
    ("25°C", "hai mươi lăm độ C"),
    ("37,5 °C", "ba mươi bảy phẩy năm độ C"),
    ("góc 90°", "góc chín mươi độ"),
    ("100°F", "một trăm độ F"),
    ("chạy 3km mất 20 phút", "chạy ba ki-lô-mét mất hai mươi phút"),
    ("60 km/h", "sáu mươi ki-lô-mét một giờ"),
    ("2 kg, 500 mg, 250ml, 5 cm, 3mm", "hai ki-lô-gam, năm trăm mi-li-gam, hai trăm năm mươi mi-li-lít, năm xen-ti-mét, ba mi-li-mét"),
    ("km nào", "km nào"),       # không có số đứng trước thì giữ nguyên
    ("5 kmh", "năm kmh"),       # không phải đơn vị riêng
])
def test_normalize_don_vi(text, spoken):
    assert tts.normalize(text) == spoken
