"""Trạng thái video cảnh trong UI suy ra từ file: thiếu / cũ / ok."""
import json
import os

import pytest

pytest.importorskip("gradio")
import ui  # noqa: E402


@pytest.fixture
def job(tmp_path):
    sc = {"narration": "Uống một cốc nước.", "image_subject": "a glass", "cue": "cốc nước", "on_screen_text": "Nước"}
    (tmp_path / "script.json").write_text(json.dumps({"title": "t", "scenes": [sc]}, ensure_ascii=False))
    d = tmp_path / "scene_01"
    d.mkdir()
    (d / "image.png").write_bytes(b"x")
    (d / "scene.json").write_text(json.dumps({"narration": sc["narration"], "cues": [sc["cue"]], "text": "Nước"}))
    (d / "scene.mp4").write_bytes(b"x")
    os.utime(d / "scene.mp4", (2e9, 2e9))  # video dựng sau ảnh và scene.json
    return tmp_path


def edit_script(job, **kv):
    s = json.loads((job / "script.json").read_text())
    s["scenes"][0].update(kv)
    (job / "script.json").write_text(json.dumps(s))


def test_ok(job):
    assert ui.scene_state(job, 0) == "ok"


def test_thieu_video(job):
    (job / "scene_01" / "scene.mp4").unlink()
    assert ui.scene_state(job, 0) == "missing"


def test_anh_moi_hon_video(job):
    os.utime(job / "scene_01" / "image.png", (3e9, 3e9))
    assert ui.scene_state(job, 0) == "stale"


@pytest.mark.parametrize("field, value", [("narration", "Lời khác."), ("cue", "nước"), ("on_screen_text", "Khác")])
def test_sua_kich_ban(job, field, value):
    edit_script(job, **{field: value})
    assert ui.scene_state(job, 0) == "stale"


@pytest.fixture
def job2(tmp_path):
    """Kịch bản dạng mới: nhiều vật thể, đã định vị (regions.json) và dựng xong."""
    els = [{"cue": "cốc nước", "object": "glass"}, {"cue": "đồng hồ", "object": "clock"}]
    sc = {"narration": "Uống cốc nước cạnh đồng hồ.", "image_subject": "a glass and a clock", "elements": els}
    (tmp_path / "script.json").write_text(json.dumps({"title": "t", "scenes": [sc]}, ensure_ascii=False))
    d = tmp_path / "scene_01"
    d.mkdir()
    boxes = [[0.1, 0.1, 0.5, 0.9], None]
    (d / "image.png").write_bytes(b"x")
    (d / "regions.json").write_text(json.dumps({"objects": ["glass", "clock"], "boxes": boxes}))
    (d / "scene.json").write_text(json.dumps({"narration": sc["narration"], "cues": ["cốc nước", "đồng hồ"],
                                              "regions": boxes}))
    (d / "scene.mp4").write_bytes(b"x")
    os.utime(d / "scene.mp4", (2e9, 2e9))
    return tmp_path


def test_nhieu_vat_the_ok(job2):
    assert ui.scene_state(job2, 0) == "ok"


def test_doi_ten_vat_the_thi_cu(job2):
    edit_script(job2, elements=[{"cue": "cốc nước", "object": "cup"}, {"cue": "đồng hồ", "object": "clock"}])
    assert ui.scene_state(job2, 0) == "stale"


def test_doi_cue_thi_cu(job2):
    edit_script(job2, elements=[{"cue": "cốc nước", "object": "glass"}])
    assert ui.scene_state(job2, 0) == "stale"


def test_o_bang_vat_the_qua_lai():
    els = [{"cue": "cốc nước", "object": "glass of water"}, {"cue": "đồng hồ", "object": ""}]
    assert ui.format_elements(els) == "cốc nước = glass of water; đồng hồ"
    assert ui.parse_elements(" cốc nước=glass of water ;đồng hồ; ") == els
