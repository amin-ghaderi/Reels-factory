from pathlib import Path

from reels_factory.config import load_config
from reels_factory.faces import BBox
from reels_factory.framing import (
    boxes_equal,
    ffmpeg_crops,
    framing_profile_path,
    layout_from_framing_profile,
    load_framing_profile,
    resolve_framing_profile,
)
from reels_factory.render import choose_reel_layout


PROFILE = Path("data/framing_profiles/17-05.json")
RCLOUD = {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "layout": "stacked_faces",
    "lock_face_crops": True,
}


def test_17_05_profile_locks_approved_crops():
    data = load_framing_profile(PROFILE)
    assert data["person_a"] == {"x": 239.6, "y": 294.0, "w": 483.0, "h": 429.3}
    assert data["person_b"] == {"x": 1133.1, "y": 355.6, "w": 483.0, "h": 429.3}
    layout = layout_from_framing_profile(data, RCLOUD)
    assert layout.method == "framing_profile"
    assert layout.mode == "stacked_faces"
    assert boxes_equal(layout.top, BBox(239.6, 294.0, 483.0, 429.3))
    assert boxes_equal(layout.bottom, BBox(1133.1, 355.6, 483.0, 429.3))
    crops = ffmpeg_crops(layout.filter_complex)
    assert crops == [(482, 428, 240, 294), (482, 428, 1132, 356)]
    assert layout.filter_complex.count("crop=1080:960") == 2


def test_profile_layout_is_identical_for_every_clip():
    data = load_framing_profile(PROFILE)
    a = layout_from_framing_profile(data, RCLOUD)
    b = layout_from_framing_profile(data, RCLOUD)
    assert a.filter_complex == b.filter_complex
    assert a.top == b.top
    assert a.bottom == b.bottom


def test_choose_reel_layout_uses_profile_and_skips_detection(tmp_path, monkeypatch):
    profiles = tmp_path / "framing_profiles"
    profiles.mkdir()
    (profiles / "17-05.json").write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = {
        "paths": {"framing_profiles": profiles, "output": tmp_path / "output"},
        "render": dict(RCLOUD),
    }

    def boom(*_args, **_kwargs):
        raise AssertionError("face detection must not run when a framing profile exists")

    monkeypatch.setattr("reels_factory.render.plan_locked_layout", boom)
    clips = [{"label": "only", "start": 1.0, "end": 2.0}]
    layout = choose_reel_layout(Path("data/inbox/17-05.mp4"), cfg, cfg["render"], clips)
    assert layout is not None
    assert layout.method == "framing_profile"
    assert ffmpeg_crops(layout.filter_complex) == [
        (482, 428, 240, 294),
        (482, 428, 1132, 356),
    ]


def test_resolve_framing_profile_from_loaded_config():
    root = Path(".").resolve()
    cfg = load_config(root)
    source = root / "data" / "inbox" / "17-05.mp4"
    path = framing_profile_path(source, cfg)
    assert path.name == "17-05.json"
    profile = resolve_framing_profile(source, cfg)
    assert profile is not None
    assert profile["person_a"]["x"] == 239.6


def test_missing_profile_is_none(tmp_path):
    cfg = {"paths": {"framing_profiles": tmp_path, "output": tmp_path / "out"}}
    assert resolve_framing_profile(Path("other.mp4"), cfg) is None
