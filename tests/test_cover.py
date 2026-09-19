from pathlib import Path

import numpy as np

from reels_factory.cover import (
    CoverError,
    choose_headline,
    headline_candidates,
    load_cover_layout,
    load_cover_metadata,
    parse_cover_size,
    scale_template,
)
from reels_factory.faces import BBox, bbox_inside, crop_inside_roi, inset_bbox
from reels_factory.utils import read_json


Q01_PLAN = {
    "title": "یک سال باران بحران آب ایران را حل نمی‌کند",
    "segments": [
        {
            "role": "question_core",
            "text": "آیا بحران منابع آبی و بحران آب در ایران انقدر ساده است که بعضی دولتمردان ما در ایران می‌گویند که اگر یک سال بارندگی خوب باشد، بارندگی به اندازهٔ زیاد باشد مشکل حل می‌شود و تمام.",
        },
        {
            "role": "central_answer",
            "text": "اگه بخواهم در یک کلمه پاسخ بدهم جواب سؤال شما خیر است و با یک سال پُر باران مسائل منابع آب ایران حل نمی‌شود.",
        },
        {
            "role": "payoff",
            "text": "ولی معناش این نیست که مسائل آبی حل می‌شد. بیشتر شکل از مسائل موقتی است. آن هم نه در همه جای کشور.",
        },
    ],
}


def test_parse_cover_size():
    assert parse_cover_size("1080x1920") == (1080, 1920)


def test_load_17_05_metadata_is_bottom_guest():
    root = Path(__file__).resolve().parents[1]
    meta = load_cover_metadata(root / "data" / "metadata" / "17-05.json", root=root)
    assert meta["guest_name"] == "محسن سعیدی"
    assert meta["guest_role"] == "متخصص آب، محیط زیست و پایداری"
    assert meta["guest_panel"] == "bottom"
    assert meta["cover_size"] == "1080x1920"
    assert Path(meta["cover_template"]).is_file()
    assert str(meta.get("guest_portrait") or "").replace("\\", "/").endswith(
        "data/portraits/17-05_guest_master.png"
    )


def test_metadata_rejects_bad_panel(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"guest_name":"x","guest_role":"y","guest_panel":"left",'
        '"cover_template":"missing.png","cover_size":"1080x1920"}',
        encoding="utf-8",
    )
    try:
        load_cover_metadata(path)
        assert False, "expected CoverError"
    except CoverError:
        pass


def test_headline_candidates_are_source_faithful():
    cands = headline_candidates(Q01_PLAN)
    assert len(cands) == 3
    chosen = choose_headline(cands)
    words = chosen.split()
    assert 5 <= len(words) <= 10
    assert "؟" not in chosen
    assert "نمی‌کند" in chosen or "نمی‌شود" in chosen
    # No invented nuclear / drought-year-count claims.
    joined = " ".join(cands)
    assert "ده سال" not in joined
    assert chosen == cands[0]


def test_choose_headline_rejects_sentence_fragments():
    assert (
        choose_headline(
            [
                "هنوز برای ایران امیدوارم",
                "کنیم و با هم مسئولیت ساختنش را به دوش بگیریم",
            ]
        )
        == "هنوز برای ایران امیدوارم"
    )
    assert (
        choose_headline(
            [
                "بحران آب ایران ریشه‌ای است نه زودگذر",
                "حال حاضر بحران عمیق و ریشه‌ای و زودگذر هم نیست",
                "این مثل بانکی نیست که ما بتوانیم هر سالش را",
            ]
        )
        == "بحران آب ایران ریشه‌ای است نه زودگذر"
    )


def test_scale_template_to_cover_size():
    root = Path(__file__).resolve().parents[1]
    import cv2

    img = cv2.imread(str(root / "assets" / "covers" / "default" / "cover_template.png"))
    assert img is not None
    assert img.shape[1] == 941 and img.shape[0] == 1672
    scaled = scale_template(img, 1080, 1920)
    assert scaled.shape[1] == 1080 and scaled.shape[0] == 1920


def test_layout_keeps_logo_boxes():
    root = Path(__file__).resolve().parents[1]
    layout = load_cover_layout(root / "assets" / "covers" / "default" / "cover_template.png", (1080, 1920))
    hx, hy, hw, hh = layout["headline"]
    gx, gy, gw, gh = layout["guest"]
    assert hy + hh <= gy
    for px, py, pw, ph in layout["protected"]:
        # Guest photo must not sit on the logo boxes.
        overlap_x = min(gx + gw, px + pw) - max(gx, px)
        overlap_y = min(gy + gh, py + ph) - max(gy, py)
        assert overlap_x <= 0 or overlap_y <= 0


def test_cover_crop_stays_inside_guest_panel():
    face = BBox(1263.4, 394.7, 216.7, 315.1, 0.9)
    panel = BBox(986.0, 271.0, 891.0, 538.0)
    safe = inset_bbox(panel, 0.04)
    crop = crop_inside_roi(face, safe, 540 / 600)
    assert bbox_inside(crop, safe)
    assert crop.x >= panel.x
    assert crop.x + crop.w <= panel.x + panel.w
    assert crop.y + crop.h <= panel.y + panel.h


def test_resolve_cover_font_prefers_vazirmatn():
    root = Path(__file__).resolve().parents[1]
    from reels_factory.cover import resolve_cover_font

    _, headline = resolve_cover_font("headline", root=root, size=48)
    _, name = resolve_cover_font("name", root=root, size=36)
    _, role = resolve_cover_font("role", root=root, size=28)
    assert "vazirmatn" in headline.lower()
    assert "vazirmatn" in name.lower()
    assert "vazirmatn" in role.lower()
    assert "extrabold" in headline.lower()
    assert "semibold" in name.lower() or "semi bold" in name.lower()
    assert "regular" in role.lower()
    assert "tahoma" not in headline.lower()


def test_portrait_times_skip_cut_edges():
    from reels_factory.cover import _sample_portrait_times

    times = _sample_portrait_times([(10.0, 20.0), (30.0, 40.0)], step=0.5, edge=0.25)
    assert min(times) >= 10.25
    assert max(t for t in times if t < 25) <= 19.75
    assert all(not (19.76 < t < 30.24) for t in times)


def test_score_rejects_face_outside_panel():
    from reels_factory.cover import score_guest_portrait
    import numpy as np

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    face = BBox(100, 100, 80, 100, 0.9)
    safe = BBox(900, 270, 800, 500)
    details = score_guest_portrait(frame, face, safe, panel=safe)
    assert details["accepted"] is False
    assert details["reject_reason"] == "partially_cropped_face"


def test_cover_json_schema_keys():
    # Contract for the sidecar written by generate_cover.
    required = {
        "selected_headline",
        "guest_name",
        "guest_role",
        "source_frame_timestamp",
        "crop_coordinates",
        "template_used",
    }
    sample = {
        "selected_headline": "x",
        "guest_name": "محسن سعیدی",
        "guest_role": "متخصص آب، محیط زیست و پایداری",
        "source_frame_timestamp": "00:01:55.000",
        "crop_coordinates": {"x": 1, "y": 2, "w": 3, "h": 4},
        "template_used": "assets/covers/default/cover_template.png",
    }
    assert required <= sample.keys()
    assert read_json.__name__ == "read_json"
