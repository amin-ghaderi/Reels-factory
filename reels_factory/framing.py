from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .faces import BBox, LayoutPlan, bbox_inside, stacked_faces_filter
from .utils import read_json

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def framing_profile_path(source: Path, cfg: dict) -> Path:
    folder = cfg.get("paths", {}).get("framing_profiles")
    if folder:
        base = Path(folder)
    else:
        base = Path(cfg["paths"]["output"]).parent / "framing_profiles"
    return base / f"{Path(source).stem}.json"


def load_framing_profile(path: Path) -> dict:
    data = read_json(path)
    for key in ("person_a", "person_b"):
        box = data.get(key) or {}
        if not all(k in box for k in ("x", "y", "w", "h")):
            raise ValueError(f"Framing profile {path.name} missing {key} x/y/w/h")
    return data


def resolve_framing_profile(source: Path, cfg: dict) -> dict | None:
    path = framing_profile_path(source, cfg)
    if not path.is_file():
        return None
    return load_framing_profile(path)


def _box(raw: dict) -> BBox:
    return BBox(float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"]))


def layout_from_framing_profile(profile: dict, rcfg: dict) -> LayoutPlan:
    """Frozen stacked_faces layout. No detection, tracking, or smoothing."""
    frame_w, frame_h = [int(v) for v in (profile.get("frame_size") or (1920, 1080))]
    width = int(rcfg.get("width", 1080))
    height = int(rcfg.get("height", 1920))
    fps = int(rcfg.get("fps", 30))
    top = _box(profile["person_a"])
    bottom = _box(profile["person_b"])
    panel_a = _box(profile["panel_a"]) if profile.get("panel_a") else None
    panel_b = _box(profile["panel_b"]) if profile.get("panel_b") else None
    if panel_a is not None and not bbox_inside(top, panel_a, eps=1.0):
        raise ValueError("Framing profile person_a is outside panel_a")
    if panel_b is not None and not bbox_inside(bottom, panel_b, eps=1.0):
        raise ValueError("Framing profile person_b is outside panel_b")
    return LayoutPlan(
        mode="stacked_faces",
        filter_complex=stacked_faces_filter(top, bottom, frame_w, frame_h, width, height, fps),
        method="framing_profile",
        face_counts=[2],
        avg_faces=2.0,
        top=top,
        bottom=bottom,
        panel_a=panel_a,
        panel_b=panel_b,
        safety_margin=float(profile.get("safety_margin") or 0.04),
    )


def ffmpeg_crops(filter_complex: str) -> list[tuple[int, int, int, int]]:
    return [tuple(int(v) for v in m.groups()) for m in _CROP_RE.finditer(filter_complex or "")]


def boxes_equal(a: BBox, b: BBox, eps: float = 0.05) -> bool:
    return (
        abs(a.x - b.x) <= eps
        and abs(a.y - b.y) <= eps
        and abs(a.w - b.w) <= eps
        and abs(a.h - b.h) <= eps
    )


def _frame_at(video: Path, seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        return None
    finally:
        cap.release()


def _graphic_strip_score(frame: np.ndarray) -> float:
    """Share of highly saturated graphic pixels in the bottom 6 rows."""
    strip = frame[-6:, :, :]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 80), (45, 255, 255))
    cyan = cv2.inRange(hsv, (85, 70, 50), (130, 255, 255))
    return max(float(yellow.mean()), float(cyan.mean())) / 255.0


def output_cut_times(clips: list[dict]) -> list[float]:
    cuts = []
    acc = 0.0
    for clip in clips[:-1]:
        acc += float(clip["end"]) - float(clip["start"])
        cuts.append(acc)
    return cuts


def validate_fixed_framing(
    *,
    layout: LayoutPlan,
    profile: dict,
    output_mp4: Path,
    clips: list[dict],
) -> dict:
    expected_top = _box(profile["person_a"])
    expected_bot = _box(profile["person_b"])
    framing_diffs = 0
    if layout.top is None or layout.bottom is None:
        framing_diffs += 1
    else:
        if not boxes_equal(layout.top, expected_top):
            framing_diffs += 1
        if not boxes_equal(layout.bottom, expected_bot):
            framing_diffs += 1
    crops = ffmpeg_crops(layout.filter_complex)
    expected_crops = ffmpeg_crops(
        stacked_faces_filter(
            expected_top,
            expected_bot,
            int((profile.get("frame_size") or (1920, 1080))[0]),
            int((profile.get("frame_size") or (1920, 1080))[1]),
            1080,
            1920,
            30,
        )
    )
    if crops != expected_crops:
        framing_diffs += 1

    duration = sum(float(c["end"]) - float(c["start"]) for c in clips)
    times = [0.04, max(0.04, duration / 2.0), max(0.08, duration - 0.08)]
    for cut in output_cut_times(clips):
        times.append(max(0.04, cut - 0.05))
        times.append(cut + 0.05)
    artifacts = 0
    sampled = 0
    for t in times:
        if t < 0 or t > duration:
            continue
        frame = _frame_at(output_mp4, t)
        if frame is None:
            continue
        sampled += 1
        if _graphic_strip_score(frame) > 0.12:
            artifacts += 1
    return {
        "framing_differences": framing_diffs,
        "outside_panel_artifacts": artifacts,
        "frames_sampled": sampled,
        "cut_boundaries": len(clips) - 1,
        "ffmpeg_crops": crops,
        "filter": layout.filter_complex,
    }
