from reels_factory.faces import (
    BBox,
    FaceTrack,
    decide_layout,
    smooth_bboxes,
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
