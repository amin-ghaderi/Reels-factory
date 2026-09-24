"""Procedural cover background color. Logos, gold, and branding stay protected.

Randomization happens only when a production cover is first created.
Later regenerations reuse the locked color in that Reel's cover JSON.
"""
from __future__ import annotations

import colorsys
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .utils import read_json, write_json

DEFAULT_SATURATION = (0.35, 0.70)
DEFAULT_LIGHTNESS = (0.18, 0.38)
HUE_MIN_DISTANCE_DEG = 28.0
RGB_MIN_DISTANCE = 48.0
HISTORY_LOOKBACK = 5
MAX_ATTEMPTS = 80

# OpenCV HLS: H in [0, 180], L/S in [0, 255]
_GOLD_HSV_LO = (8, 40, 70)
_GOLD_HSV_HI = (42, 255, 255)


class CoverColorError(ValueError):
    """Procedural background color could not be generated or applied."""


def color_history_path(root: Path) -> Path:
    return Path(root) / "data" / "cover_color_history.json"


def default_mask_path(root: Path) -> Path:
    return Path(root) / "assets" / "covers" / "default" / "background_recolor_mask.png"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def hue_distance_deg(h1: float, h2: float) -> float:
    d = abs(float(h1) - float(h2)) % 360.0
    return min(d, 360.0 - d)


def rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def colors_too_similar(a: dict, b: dict) -> bool:
    ha, sa, la = [float(v) for v in a["hsl"]]
    hb, sb, lb = [float(v) for v in b["hsl"]]
    if hue_distance_deg(ha, hb) < HUE_MIN_DISTANCE_DEG and abs(la - lb) < 0.10 and abs(sa - sb) < 0.22:
        return True
    ra = tuple(int(v) for v in a["rgb"])
    rb = tuple(int(v) for v in b["rgb"])
    return rgb_distance(ra, rb) < RGB_MIN_DISTANCE


def color_payload(h: float, s: float, l: float) -> dict:
    rgb = hsl_to_rgb(h, s, l)
    return {
        "hex": rgb_to_hex(rgb),
        "rgb": list(rgb),
        "hsl": [round(h, 2), round(s, 4), round(l, 4)],
        "procedurally_generated": True,
    }


def generate_procedural_color(
    *,
    rng: random.Random | None = None,
    recent: list[dict] | None = None,
    saturation: tuple[float, float] = DEFAULT_SATURATION,
    lightness: tuple[float, float] = DEFAULT_LIGHTNESS,
) -> dict:
    rng = rng or random.Random()
    lookback = list(recent or [])[-HISTORY_LOOKBACK:]
    last_error = "could not generate a distinct cover color"
    for _ in range(MAX_ATTEMPTS):
        h = rng.uniform(0.0, 360.0)
        s = rng.uniform(*saturation)
        l = rng.uniform(*lightness)
        if s < 0.32 or l < 0.16 or l > 0.42:
            continue
        payload = color_payload(h, s, l)
        if any(colors_too_similar(payload, prev) for prev in lookback):
            last_error = "candidate too similar to recent covers"
            continue
        return payload
    raise CoverColorError(last_error)


def load_color_history(path: Path) -> dict:
    if not path.is_file():
        return {"covers": []}
    data = read_json(path)
    if not isinstance(data, dict):
        return {"covers": []}
    covers = data.get("covers")
    if not isinstance(covers, list):
        covers = []
    return {"covers": covers}


def recent_cover_colors(history: dict, n: int = HISTORY_LOOKBACK) -> list[dict]:
    covers = [c for c in (history.get("covers") or []) if isinstance(c, dict) and c.get("hsl") and c.get("rgb")]
    return covers[-n:]


def append_color_history(path: Path, record: dict) -> dict:
    history = load_color_history(path)
    history["covers"].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, history)
    return history


def locked_color_from_cover_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    data = read_json(path)
    color = data.get("background_color") if isinstance(data, dict) else None
    if not isinstance(color, dict):
        return None
    if not color.get("hex") or not color.get("rgb") or not color.get("hsl"):
        return None
    return {
        "hex": str(color["hex"]),
        "rgb": [int(v) for v in color["rgb"][:3]],
        "hsl": [float(v) for v in color["hsl"][:3]],
        "procedurally_generated": True,
    }


def resolve_cover_background(
    *,
    root: Path,
    source_id: str,
    reel_id: str,
    existing_cover_json: Path | None = None,
    generate: bool = False,
    rng: random.Random | None = None,
) -> dict | None:
    """Reuse a locked Reel color. Generate only on first production cover create."""
    locked = locked_color_from_cover_json(existing_cover_json)
    if locked is not None:
        return locked
    if not generate:
        return None
    history_path = color_history_path(root)
    history = load_color_history(history_path)
    payload = generate_procedural_color(rng=rng, recent=recent_cover_colors(history))
    append_color_history(history_path, {
        "source_id": source_id,
        "reel_id": reel_id,
        "hex": payload["hex"],
        "rgb": payload["rgb"],
        "hsl": payload["hsl"],
        "created_at": _now_iso(),
    })
    return payload


def _logo_boxes_template_space(template_w: int, template_h: int) -> list[tuple[int, int, int, int]]:
    sx = template_w / 1080.0
    sy = template_h / 1920.0
    boxes = [(24, 24, 240, 200), (800, 24, 256, 190)]
    out = []
    for x, y, w, h in boxes:
        out.append((
            max(0, int(round(x * sx))),
            max(0, int(round(y * sy))),
            max(1, int(round(w * sx))),
            max(1, int(round(h * sy))),
        ))
    return out


def build_background_recolor_mask(template_bgr: np.ndarray) -> np.ndarray:
    """White = recolor background/artwork. Black = protect logos, gold, branding."""
    h, w = template_bgr.shape[:2]
    recolor = np.full((h, w), 255, dtype=np.uint8)
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, _GOLD_HSV_LO, _GOLD_HSV_HI)
    # Thin bright gold strokes / sparkles
    bright_gold = cv2.inRange(hsv, (8, 20, 160), (50, 255, 255))
    protect = cv2.bitwise_or(gold, bright_gold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    protect = cv2.dilate(protect, kernel, iterations=1)
    recolor[protect > 0] = 0
    for x, y, bw, bh in _logo_boxes_template_space(w, h):
        recolor[y:y + bh, x:x + bw] = 0
    return recolor


def write_background_recolor_mask(template_path: Path, output_path: Path) -> Path:
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise CoverColorError(f"Could not read cover template: {template_path}")
    mask = build_background_recolor_mask(template)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), mask)
    if not ok:
        raise CoverColorError(f"Failed to write {output_path}")
    return output_path


def load_recolor_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise CoverColorError(f"Could not read recolor mask: {mask_path}")
    width, height = size
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_background_recolor(
    canvas_bgr: np.ndarray,
    mask: np.ndarray,
    color: dict,
) -> np.ndarray:
    """Recolor background via HLS hue/saturation swap. Preserve lightness/texture."""
    if canvas_bgr.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (canvas_bgr.shape[1], canvas_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    rgb = [int(v) for v in color["rgb"][:3]]
    target_bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    target_hls = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)[0, 0]
    hls = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2HLS)
    select = mask > 127
    hls[:, :, 0][select] = int(target_hls[0])
    hls[:, :, 2][select] = int(target_hls[2])
    recolored = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    out = canvas_bgr.copy()
    out[select] = recolored[select]
    return out
