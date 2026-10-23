"""Đọc khung từ câu trả lời của model nhìn ảnh."""
from pipeline.locate import parse_boxes

ANSWER = '''```json
[
 {"bbox_2d": [195, 257, 506, 675], "label": "glass of water"},
 {"bbox_2d": [0, 457, 514, 1200], "label": "Hand"}
]
```'''


def test_khop_theo_nhan_va_doi_ti_le():
    assert parse_boxes(ANSWER, ["hand", "glass of water"]) == [[0.0, 0.457, 0.514, 1.0], [0.195, 0.257, 0.506, 0.675]]


def test_nhan_lech_cung_so_luong_thi_theo_thu_tu():
    assert parse_boxes(ANSWER, ["cup", "person"])[0] == [0.195, 0.257, 0.506, 0.675]


def test_khong_thay_hoac_hong():
    assert parse_boxes(ANSWER, ["glass of water", "clock", "dog"]) == [[0.195, 0.257, 0.506, 0.675], None, None]
    assert parse_boxes("không biết", ["dog"]) == [None]
    assert parse_boxes('[{"label": "dog", "bbox_2d": [500, 500, 100, 100]}]', ["dog"]) == [None]
