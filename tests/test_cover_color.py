from pathlib import Path
import random

import cv2
import numpy as np

from reels_factory.cover import guest_role_lines, load_cover_layout, load_cover_metadata
from reels_factory.cover_color import (
    apply_background_recolor,
    build_background_recolor_mask,
    colors_too_similar,
    generate_procedural_color,
    locked_color_from_cover_json,
    resolve_cover_background,
)
from reels_factory.word_cuts import snap_cut_to_silence


def test_load_0201_metadata_keeps_two_role_lines():
    root = Path(__file__).resolve().parents[1]
    meta = load_cover_metadata(root / "data" / "metadata" / "0201.json", root=root)
    assert meta["guest_name"] == "بانو میترا سالار"
    assert meta["guest_role_line_1"] == "کنشگر سیاسی"
    assert meta["guest_role_line_2"] == "عضو شورای مدیریت پارمان پادشاهی ایرانیان"
    assert meta["guest_role_lines"] == [
        "کنشگر سیاسی",
        "عضو شورای مدیریت پارمان پادشاهی ایرانیان",
    ]
    assert meta["guest_role"] == ""
    assert meta["procedural_cover_background"] is True
    assert meta["guest_panel"] == "bottom"


def test_legacy_metadata_still_uses_single_guest_role():
    root = Path(__file__).resolve().parents[1]
    meta = load_cover_metadata(root / "data" / "metadata" / "17-05.json", root=root)
    assert meta["guest_role"] == "متخصص آب، محیط زیست و پایداری"
    assert guest_role_lines(meta) == ["متخصص آب، محیط زیست و پایداری"]
    assert meta["procedural_cover_background"] is False


def test_layout_exposes_two_role_boxes_without_moving_legacy_role():
    root = Path(__file__).resolve().parents[1]
    layout = load_cover_layout(root / "assets" / "covers" / "default" / "cover_template.png", (1080, 1920))
    assert layout["role"] == (120, 1394, 840, 70)
    assert layout["name"] == (160, 1310, 760, 78)
    assert "role_line_1" in layout and "role_line_2" in layout
    x1, y1, w1, h1 = layout["role_line_1"]
    x2, y2, w2, h2 = layout["role_line_2"]
    assert y2 >= y1 + h1 - 2
    assert w2 >= w1


def test_procedural_colors_are_distinct_and_locked():
    first = generate_procedural_color(rng=random.Random(7))
    second = generate_procedural_color(rng=random.Random(7))
    assert first == second
    other = generate_procedural_color(rng=random.Random(99), recent=[first])
    assert not colors_too_similar(first, other)
    h, s, l = first["hsl"]
    assert 0.0 <= h < 360.0
    assert 0.35 <= s <= 0.70
    assert 0.18 <= l <= 0.38


def test_resolve_cover_background_reuses_locked_color(tmp_path):
    cover_json = tmp_path / "reel.json"
    cover_json.write_text(
        '{"background_color":{"hex":"#334155","rgb":[51,65,85],"hsl":[210,0.25,0.27]}}',
        encoding="utf-8",
    )
    locked = locked_color_from_cover_json(cover_json)
    reused = resolve_cover_background(
        root=tmp_path,
        source_id="0201",
        reel_id="0201_qa_Q01",
        existing_cover_json=cover_json,
        generate=True,
        rng=random.Random(1),
    )
    assert reused["hex"] == locked["hex"]
    skipped = resolve_cover_background(
        root=tmp_path,
        source_id="0201",
        reel_id="0201_qa_Q02",
        generate=False,
    )
    assert skipped is None


def test_recolor_mask_protects_logos_and_gold():
    root = Path(__file__).resolve().parents[1]
    template = cv2.imread(str(root / "assets" / "covers" / "default" / "cover_template.png"))
    assert template is not None
    mask = build_background_recolor_mask(template)
    assert mask.shape[:2] == template.shape[:2]
    assert int(mask[40:160, 40:180].mean()) < 40
    assert int(mask[40:150, 720:900].mean()) < 40
    assert int(mask[700:900, 400:600].mean()) > 200
    color = {"hex": "#6B2A3A", "rgb": [107, 42, 58], "hsl": [348, 0.44, 0.29]}
    out = apply_background_recolor(template, mask, color)
    assert np.array_equal(out[50:120, 50:160], template[50:120, 50:160])
    delta = cv2.absdiff(out[800:860, 480:540], template[800:860, 480:540]).mean()
    assert delta > 4


def test_word_cut_never_enters_spoken_word():
    snapped = snap_cut_to_silence(
        word_start=10.40,
        word_end=12.10,
        prev_end=10.10,
        next_start=12.40,
    )
    assert snapped["source_start"] <= 10.40
    assert snapped["source_end"] >= 12.10
    assert snapped["source_start"] >= 10.10
    assert snapped["source_end"] <= 12.40
    assert snapped["first_word_start"] == 10.4
    assert snapped["last_word_end"] == 12.1
