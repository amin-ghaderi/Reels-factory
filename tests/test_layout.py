from pathlib import Path

import numpy as np

from reels_factory.faces import (
    BBox,
    FaceTrack,
    bbox_inside,
    crop_inside_roi,
    decide_layout,
    detect_participant_panels,
    inset_bbox,
    plan_locked_layout,
    smooth_bboxes,
    track_faces,
)


def _track(*boxes: BBox) -> FaceTrack:
    return FaceTrack(boxes=list(boxes))


def test_two_face_layout_selection():
    left = _track(*[BBox(360, 400, 150, 200, 0.9) for _ in range(8)])
    right = _track(*[BBox(1340, 350, 145, 200, 0.92) for _ in range(8)])
    plan = decide_layout(
        [left, right],
        1920,
        1080,
        face_counts=[2] * 8,
        method="opencv_yunet",
    )
    assert plan.mode == "stacked_faces"
    assert plan.top is not None and plan.bottom is not None
    assert plan.top.cx < plan.bottom.cx
    assert "vstack" in plan.filter_complex
    assert "1080:960" in plan.filter_complex
    assert plan.filter_complex.endswith("[v]")
    # Face-safe: each crop must fully contain its source face.
    for face, crop in ((left.boxes[0], plan.top), (right.boxes[0], plan.bottom)):
        assert crop.x <= face.x + 1
        assert crop.y <= face.y + 1
        assert crop.x + crop.w >= face.x + face.w - 1
        assert crop.y + crop.h >= face.y + face.h - 1
        assert abs(crop.cx - face.cx) < crop.w * 0.2
        # Tighter than the old wide pane crop (~836x744).
        assert crop.w < 650
        assert crop.h < 600


def test_one_face_fallback():
    only = _track(*[BBox(900, 320, 180, 220, 0.9) for _ in range(6)])
    plan = decide_layout([only], 1920, 1080, face_counts=[1] * 6)
    assert plan.mode == "single_face"
    assert plan.single is not None
    assert "vstack" not in plan.filter_complex
    assert "1080:1920" in plan.filter_complex
    face = only.boxes[0]
    crop = plan.single
    assert crop.x <= face.x + 1
    assert crop.y <= face.y + 1
    assert crop.x + crop.w >= face.x + face.w - 1
    assert crop.y + crop.h >= face.y + face.h - 1


def test_detection_failure_fallback():
    empty = decide_layout([], 1920, 1080, face_counts=[0, 0, 0], method="none")
    assert empty.mode == "fit_blur"
    assert "gblur" in empty.filter_complex
    assert "overlay" in empty.filter_complex

    tiny = _track(*[BBox(10, 10, 12, 12, 0.4) for _ in range(5)])
    weak = _track(*([None] * 8 + [BBox(400, 400, 160, 180, 0.9)]))
    failed = decide_layout([tiny, weak], 1920, 1080, min_hit_ratio=0.4, face_counts=[0] * 5)
    assert failed.mode == "fit_blur"


def test_bounding_box_smoothing():
    stable = [BBox(100, 120, 80, 90) for _ in range(5)]
    spike = BBox(180, 120, 80, 90)
    rest = [BBox(100, 120, 80, 90) for _ in range(5)]
    raw = stable + [spike] + rest
    smoothed = smooth_bboxes(raw, alpha=0.3)
    assert len(smoothed) == len(raw)
    assert abs(smoothed[5].x - 100) < abs(raw[5].x - 100)
    assert smoothed[5].x < spike.x
    raw_spread = max(b.x for b in raw) - min(b.x for b in raw)
    smooth_spread = max(b.x for b in smoothed) - min(b.x for b in smoothed)
    assert smooth_spread < raw_spread


def test_stacked_zoom_tightens_crop():
    left = _track(*[BBox(360, 400, 150, 200, 0.9) for _ in range(6)])
    right = _track(*[BBox(1340, 350, 145, 200, 0.92) for _ in range(6)])
    loose = decide_layout([left, right], 1920, 1080, crop={"zoom": 1.0})
    tight = decide_layout([left, right], 1920, 1080, crop={"zoom": 1.32})
    assert tight.top.w < loose.top.w
    assert tight.top.h < loose.top.h
    face = left.boxes[0]
    assert tight.top.x <= face.x + 1
    assert tight.top.y <= face.y + 1
    assert tight.top.x + tight.top.w >= face.x + face.w - 1
    assert tight.top.y + tight.top.h >= face.y + face.h - 1


def test_locked_layout_samples_every_window_once(monkeypatch):
    calls = []

    def fake_sample(video, start, end, **kwargs):
        calls.append((start, end))
        faces = [
            BBox(360, 400, 150, 200, 0.9),
            BBox(1340, 350, 145, 200, 0.92),
        ]
        return [faces for _ in range(4)], "opencv_yunet", 1920, 1080

    monkeypatch.setattr("reels_factory.faces.sample_clip_faces", fake_sample)
    plan = plan_locked_layout(
        Path("clip.mp4"),
        [(10.0, 20.0), (40.0, 50.0)],
        {"layout": "stacked_faces", "lock_face_crops": True},
    )
    assert calls == [(10.0, 20.0), (40.0, 50.0)]
    assert plan.mode == "stacked_faces"
    assert plan.top is not None and plan.bottom is not None
    assert plan.filter_complex.count("crop=") >= 2


def test_combined_samples_lock_one_crop_pair():
    left_a = BBox(300, 400, 150, 200, 0.9)
    left_b = BBox(420, 400, 150, 200, 0.9)
    right_a = BBox(1300, 350, 145, 200, 0.92)
    right_b = BBox(1420, 350, 145, 200, 0.92)
    samples_a = [[left_a, right_a] for _ in range(6)]
    samples_b = [[left_b, right_b] for _ in range(6)]
    per_a = decide_layout(track_faces(samples_a), 1920, 1080)
    per_b = decide_layout(track_faces(samples_b), 1920, 1080)
    locked = decide_layout(track_faces(samples_a + samples_b), 1920, 1080)
    assert per_a.mode == per_b.mode == locked.mode == "stacked_faces"
    assert abs(per_a.top.x - per_b.top.x) > 10
    assert locked.top is not None and locked.bottom is not None
    # One frozen pair, not a per-segment recrop.
    assert locked.filter_complex == decide_layout(
        track_faces(samples_a + samples_b), 1920, 1080
    ).filter_complex


def test_inset_bbox_pulls_edges_inward():
    panel = BBox(100, 200, 800, 500)
    safe = inset_bbox(panel, 0.03)
    assert safe.x > panel.x
    assert safe.y > panel.y
    assert safe.x + safe.w < panel.x + panel.w
    assert safe.y + safe.h < panel.y + panel.h
    assert abs((safe.x - panel.x) / panel.w - 0.03) < 0.002
    assert abs((safe.y - panel.y) / panel.h - 0.03) < 0.002


def test_crop_inside_roi_never_crosses_panel():
    face = BBox(1340, 350, 220, 280, 0.92)
    panel = BBox(980, 270, 900, 400)
    safe = inset_bbox(panel, 0.03)
    crop = crop_inside_roi(face, safe, 1080 / 960)
    assert bbox_inside(crop, safe)
    assert abs(crop.w / crop.h - 1.125) < 0.02
    unconstrained = decide_layout(
        [_track(*[BBox(360, 400, 150, 200, 0.9) for _ in range(6)]),
         _track(*[face for _ in range(6)])],
        1920,
        1080,
    )
    assert unconstrained.bottom is not None
    assert unconstrained.bottom.y + unconstrained.bottom.h > safe.y + safe.h


def test_stacked_crop_stays_inside_safe_panel():
    left = _track(*[BBox(360, 400, 150, 200, 0.9) for _ in range(6)])
    right = _track(*[BBox(1340, 350, 220, 280, 0.92) for _ in range(6)])
    pa = BBox(40, 270, 800, 400)
    pb = BBox(980, 270, 900, 400)
    plan = decide_layout(
        [left, right],
        1920,
        1080,
        panels=(pa, pb),
        crop={"panel_safety_margin": 0.03},
    )
    assert plan.mode == "stacked_faces"
    assert plan.panel_a is not None and plan.panel_b is not None
    assert abs(plan.safety_margin - 0.03) < 1e-6
    assert bbox_inside(plan.top, plan.panel_a)
    assert bbox_inside(plan.bottom, plan.panel_b)
    assert plan.bottom.y + plan.bottom.h <= pb.y + pb.h - 4
    assert plan.top.y + plan.top.h <= pa.y + pa.h - 4


def test_detect_participant_panels_on_split_graphic():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (40, 190, 230)
    frame[270:810, 48:828] = (90, 90, 90)
    frame[270:810, 990:1876] = (100, 100, 100)
    frame[268:276, 48:828] = 15
    frame[804:812, 48:828] = 15
    frame[268:276, 990:1876] = 15
    frame[804:812, 990:1876] = 15
    frame[270:810, 44:52] = 15
    frame[270:810, 824:832] = 15
    frame[270:810, 986:994] = 15
    frame[270:810, 1872:1880] = 15
    face_a = BBox(300, 420, 160, 200, 0.9)
    face_b = BBox(1300, 400, 160, 200, 0.92)
    panels = detect_participant_panels([frame, frame], face_a, face_b)
    assert panels is not None
    pa, pb = panels
    assert pa.x < 80
    assert pa.x + pa.w < 980
    assert pb.x > 900
    assert abs(pa.y - 270) <= 12
    assert abs((pa.y + pa.h) - 810) <= 12
    assert face_a.cx > pa.x and face_a.cx < pa.x + pa.w
    assert face_b.cx > pb.x and face_b.cx < pb.x + pb.w
