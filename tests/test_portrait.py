from pathlib import Path

import cv2
import numpy as np

from reels_factory.cover import _paste_guest
from reels_factory.portrait import (
    _grabcut_cutout,
    _is_camera_facing,
    crop_interview_still_inside_roi,
    enhance_guest_cutout,
    enhance_interview_still,
    master_portrait_path,
    resolve_master_portrait,
)
from reels_factory.faces import BBox, bbox_inside


def test_master_portrait_path_from_metadata(tmp_path):
    root = tmp_path
    meta = {"guest_portrait": "data/portraits/17-05_guest_master.png"}
    path = master_portrait_path("17-05", root=root, meta=meta, cfg=None)
    assert path == root / "data" / "portraits" / "17-05_guest_master.png"
    assert resolve_master_portrait("17-05", root=root, meta=meta) is None
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert resolve_master_portrait("17-05", root=root, meta=meta) == path


def test_master_portrait_default_stem_path(tmp_path):
    cfg = {"paths": {"portraits": tmp_path / "portraits"}}
    path = master_portrait_path("17-05", root=tmp_path, cfg=cfg)
    assert path == tmp_path / "portraits" / "17-05_guest_master.png"


def test_camera_facing_requires_open_eyes_and_gaze():
    ok = {
        "accepted": True,
        "gaze_frontal": 0.88,
        "iris_gaze": 0.62,
        "yaw": 0.02,
        "eyes_open": 0.80,
    }
    assert _is_camera_facing(ok) is True
    bad = dict(ok, iris_gaze=0.20)
    assert _is_camera_facing(bad) is False


def test_grabcut_cutout_has_alpha():
    img = np.full((120, 100, 3), 30, dtype=np.uint8)
    cv2.circle(img, (50, 50), 28, (180, 160, 140), -1)
    out = _grabcut_cutout(img)
    assert out.shape[2] == 4
    assert int(out[:, :, 3].max()) > 0


def test_enhance_guest_cutout_upscales_without_rgb_only():
    img = np.full((80, 72, 3), 40, dtype=np.uint8)
    cv2.ellipse(img, (36, 34), (22, 28), 0, 0, 360, (190, 170, 150), -1)
    out, steps = enhance_guest_cutout(img)
    assert out.shape[0] == 80 * 3
    assert out.shape[1] == 72 * 3
    assert out.shape[2] == 4
    assert any("upscale" in s.lower() for s in steps)
    assert any("cutout" in s.lower() for s in steps)
    assert not any("generative" in s.lower() and "used" in s.lower() for s in steps)


def test_paste_guest_uses_alpha_over_template():
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas[:] = (10, 20, 80)
    guest = np.zeros((40, 40, 4), dtype=np.uint8)
    guest[8:32, 8:32, :3] = (40, 180, 40)
    guest[8:32, 8:32, 3] = 255
    out = _paste_guest(canvas, guest, (20, 20, 40, 40), radius=4, gold=(200, 180, 80))
    assert tuple(int(v) for v in out[5, 5]) == (10, 20, 80)
    # Transparent interior of the PNG should keep the template, not the guest fill.
    assert int(out[26, 26, 1]) < 80
    # Opaque guest pixels should show the green cutout.
    assert out[40, 40, 1] > 100


def test_interview_still_crop_uses_full_safe_roi():
    face = BBox(1263.4, 394.7, 216.7, 315.1, 0.9)
    safe = BBox(1021.6, 292.5, 818.8, 495.0)
    crop = crop_interview_still_inside_roi(face, safe)
    assert bbox_inside(crop, safe)
    assert abs(crop.x - safe.x) < 0.01
    assert abs(crop.y - safe.y) < 0.01
    assert abs(crop.w - safe.w) < 0.01
    assert abs(crop.h - safe.h) < 0.01


def test_enhance_interview_still_keeps_background_and_rgb():
    img = np.full((80, 120, 3), 40, dtype=np.uint8)
    img[:, :40] = (90, 40, 20)
    out, steps = enhance_interview_still(img)
    assert out.shape[0] == 80 * 3
    assert out.shape[1] == 120 * 3
    assert out.ndim == 3 and out.shape[2] == 3
    joined = " ".join(steps).lower()
    assert "rembg" not in joined or "no rembg" in joined
    assert "cutout" not in joined or "no cutout" in joined
    assert "upscale" in joined
    assert "original background preserved" in joined
