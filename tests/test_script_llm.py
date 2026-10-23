"""Kiểm tra kịch bản LLM trả về (không nạp model)."""
from pipeline import script_llm as sl


def scene(narration="Mỗi sáng hãy uống một cốc nước ấm để cơ thể tỉnh táo và khoẻ mạnh hơn nhé.",
          subject="a glass of water", cue="cốc nước", text="1 cốc nước"):
    return {"narration": narration, "image_subject": subject, "cue": cue, "on_screen_text": text}


def check(scenes):
    return sl.validate({"scenes": scenes}, 1, 3, 10, 40)


def test_contains_phrase_bo_dau_cau_giu_dau_thanh():
    assert sl.contains_phrase("Uống một CỐC NƯỚC, ngay!", "cốc nước")
    assert not sl.contains_phrase("uống một cốc nước", "cóc nước")
    assert not sl.contains_phrase("uống nước", "")


def test_extract_json_bo_phan_think():
    assert sl.extract_json('<think>{"sai": 1}</think> Đây: {"title": "x"} xong') == {"title": "x"}


def test_kich_ban_hop_le_va_tu_strip():
    sc = scene(cue="  cốc nước ")
    assert check([sc]) == []
    assert sc["cue"] == "cốc nước"


def test_bat_loi_tung_truong():
    errors = check([scene(cue="bánh mì", subject="một cốc nước", text="một hai ba bốn năm sáu bảy")])
    assert len(errors) == 3
    assert any("cue" in e for e in errors)
    assert any("image_subject" in e for e in errors)
    assert any("on_screen_text" in e for e in errors)


def test_sai_so_canh():
    assert check([]) and check([scene()] * 4)
