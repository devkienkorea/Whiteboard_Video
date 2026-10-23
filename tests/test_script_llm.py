"""Kiểm tra kịch bản LLM trả về (không nạp model)."""
from pipeline import script_llm as sl

NAR = "Mỗi sáng hãy uống một cốc nước ấm cạnh chiếc đồng hồ để cơ thể tỉnh táo và khoẻ mạnh hơn nhé."


def scene(narration=NAR, subject="a glass of water next to an alarm clock",
          elements=({"cue": "cốc nước", "object": "glass of water"},), text="1 cốc nước"):
    return {"narration": narration, "image_subject": subject, "elements": [dict(e) for e in elements],
            "on_screen_text": text}


def check(scenes):
    return sl.validate({"scenes": scenes}, 1, 3, 10, 40)


def test_contains_phrase_bo_dau_cau_giu_dau_thanh():
    assert sl.contains_phrase("Uống một CỐC NƯỚC, ngay!", "cốc nước")
    assert not sl.contains_phrase("uống một cốc nước", "cóc nước")
    assert not sl.contains_phrase("uống nước", "")
    assert sl.phrase_index("uống một cốc nước", "cốc nước") == 2


def test_extract_json_bo_phan_think():
    assert sl.extract_json('<think>{"sai": 1}</think> Đây: {"title": "x"} xong') == {"title": "x"}


def test_kich_ban_hop_le_tu_strip_va_sap_theo_thu_tu_noi():
    sc = scene(elements=[{"cue": " đồng hồ ", "object": "alarm clock"}, {"cue": "cốc nước", "object": "glass of water"}])
    assert check([sc]) == []
    assert [e["cue"] for e in sc["elements"]] == ["cốc nước", "đồng hồ"]


def test_bat_loi_tung_truong():
    errors = check([scene(subject="một cốc nước", text="một hai ba bốn năm sáu bảy",
                          elements=[{"cue": "bánh mì", "object": "glass of water"}])])
    assert len(errors) == 4
    assert any("cue 'bánh mì'" in e for e in errors)
    assert any("object 'glass of water'" in e for e in errors)  # không có trong image_subject
    assert any("image_subject phải" in e for e in errors)
    assert any("on_screen_text" in e for e in errors)


def test_so_vat_the():
    assert check([scene(elements=[])])
    assert check([scene(elements=[{"cue": "cốc nước", "object": "glass of water"}] * 4)])
    sc = scene()
    del sc["elements"]
    assert check([sc])


def test_sai_so_canh():
    assert check([]) and check([scene()] * 4)
