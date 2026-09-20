from __future__ import annotations

import math
import re
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

from .faces import (
    BBox,
    CachedYunet,
    crop_inside_roi,
    detect_participant_panels,
    ensure_yunet_model,
    face_crop_params,
    filter_faces,
    inset_bbox,
    read_frames_at,
    bbox_inside,
)
from .render import _clips_from_plan, _resolve_source_video
from .utils import read_json, ts, write_json

GUEST_PANELS = {"top", "bottom"}
_WORD_RE = re.compile(r"\s+")
_FONT_EXTS = {".ttf", ".otf", ".ttc"}
_CUT_EDGE_S = 0.25
_SAMPLE_STEP_S = 0.12

# Production lock (17-05 covers): Vazirmatn ExtraBold / SemiBold / Regular.
# Fallbacks are only used if Vazirmatn is missing. Never jump to Tahoma if a
# modern Persian/Arabic sans is present.
_FONT_PREF = {
    "headline": [
        ("vazirmatn", ("extrabold", "extra bold")),
        ("shabnam", ("bold", "extrabold")),
        ("sahel", ("bold", "black")),
        ("noto sans arabic", ("extrabold", "black", "bold")),
    ],
    "name": [
        ("vazirmatn", ("semibold", "semi bold")),
        ("shabnam", ("semibold", "bold")),
        ("sahel", ("semibold", "bold")),
        ("noto sans arabic", ("semibold", "medium", "bold")),
    ],
    "role": [
        ("vazirmatn", ("regular",)),
        ("shabnam", ("regular", "medium")),
        ("sahel", ("regular", "medium")),
        ("noto sans arabic", ("regular", "medium")),
    ],
}
_VAR_WEIGHT = {
    "thin": 100,
    "extralight": 200,
    "light": 300,
    "regular": 400,
    "normal": 400,
    "medium": 500,
    "semibold": 600,
    "demibold": 600,
    "bold": 700,
    "extrabold": 800,
    "black": 900,
}


class CoverError(ValueError):
    """Cover metadata or inputs are not usable."""


def parse_cover_size(value: str) -> tuple[int, int]:
    text = str(value or "").strip().lower().replace(" ", "")
    parts = text.split("x")
    if len(parts) != 2:
        raise CoverError(f"Invalid cover_size: {value!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise CoverError(f"Invalid cover_size: {value!r}") from exc
    if width < 2 or height < 2:
        raise CoverError(f"cover_size must be at least 2x2, got {value!r}")
    return width, height


def load_cover_metadata(path: Path, *, root: Path | None = None) -> dict:
    data = read_json(path)
    required = ("guest_name", "guest_role", "guest_panel", "cover_template", "cover_size")
    missing = [key for key in required if not str(data.get(key) or "").strip()]
    if missing:
        raise CoverError(f"{path.name} missing fields: {', '.join(missing)}")
    panel = str(data["guest_panel"]).strip().lower()
    if panel not in GUEST_PANELS:
        raise CoverError(f"guest_panel must be top or bottom, got {data['guest_panel']!r}")
    data["guest_panel"] = panel
    data["guest_name"] = str(data["guest_name"]).strip()
    data["guest_role"] = str(data["guest_role"]).strip()
    template_rel = str(data["cover_template"]).replace("\\", "/")
    template = Path(template_rel)
    if not template.is_absolute():
        base = root if root is not None else Path.cwd()
        template = (base / template).resolve()
    if not template.is_file():
        raise CoverError(f"Cover template not found: {template}")
    data["cover_template"] = template
    data["cover_template_rel"] = template_rel
    data["cover_size"] = f"{parse_cover_size(data['cover_size'])[0]}x{parse_cover_size(data['cover_size'])[1]}"
    return data


def load_cover_layout(template: Path, canvas: tuple[int, int]) -> dict:
    layout_path = template.with_name("layout.json")
    width, height = canvas
    if layout_path.is_file():
        raw = read_json(layout_path)
        src = tuple(int(v) for v in (raw.get("canvas") or canvas))
        sx = width / max(1, src[0])
        sy = height / max(1, src[1])

        def _box(key: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            vals = raw.get(key) or fallback
            x, y, w, h = (int(v) for v in vals[:4])
            return (
                int(round(x * sx)),
                int(round(y * sy)),
                int(round(w * sx)),
                int(round(h * sy)),
            )

        protected = []
        for item in raw.get("protected") or []:
            x, y, w, h = (int(v) for v in item[:4])
            protected.append((
                int(round(x * sx)),
                int(round(y * sy)),
                int(round(w * sx)),
                int(round(h * sy)),
            ))
        return {
            "headline": _box("headline", (70, 250, 940, 390)),
            "guest": _box("guest", (270, 680, 540, 600)),
            "name": _box("name", (160, 1310, 760, 78)),
            "role": _box("role", (120, 1394, 840, 70)),
            "guest_radius": int(round(float(raw.get("guest_radius", 42)) * min(sx, sy))),
            "protected": protected,
        }
    return {
        "headline": (int(0.065 * width), int(0.13 * height), int(0.87 * width), int(0.20 * height)),
        "guest": (int(0.25 * width), int(0.35 * height), int(0.50 * width), int(0.31 * height)),
        "name": (int(0.15 * width), int(0.68 * height), int(0.70 * width), int(0.04 * height)),
        "role": (int(0.11 * width), int(0.73 * height), int(0.78 * width), int(0.036 * height)),
        "guest_radius": int(0.04 * width),
        "protected": [
            (int(0.02 * width), int(0.012 * height), int(0.22 * width), int(0.105 * height)),
            (int(0.74 * width), int(0.012 * height), int(0.24 * width), int(0.10 * height)),
        ],
    }


def scale_template(template: np.ndarray, width: int, height: int) -> np.ndarray:
    if template.shape[1] == width and template.shape[0] == height:
        return template.copy()
    return cv2.resize(template, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _persian_words(text: str) -> list[str]:
    return [part for part in _WORD_RE.split(str(text or "").strip()) if part]


def _trim_to_words(text: str, lo: int = 5, hi: int = 10) -> str:
    words = _persian_words(text.replace(".", " ").replace("،", " "))
    if not words:
        return ""
    if len(words) <= hi:
        return " ".join(words)
    markers = ("نمی‌کند", "نمی‌شود", "نمي‌کند", "نمي‌شود", "نیست")
    for start in range(0, len(words) - hi + 1):
        window = words[start : start + hi]
        joined = " ".join(window)
        if any(token in joined for token in markers):
            return joined
    return " ".join(words[-hi:])


def _drop_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    out = str(text or "").strip()
    for prefix in prefixes:
        if out.startswith(prefix):
            out = out[len(prefix):].strip(" ،,.و")
    return out


def headline_candidates(plan: dict) -> list[str]:
    """Three source-faithful headline options. No invented claims."""
    title = str(plan.get("title") or "").strip()
    by_role = {
        str(seg.get("role") or ""): str(seg.get("text") or "").strip()
        for seg in plan.get("segments") or []
    }
    answer = by_role.get("central_answer", "")
    payoff = by_role.get("payoff", "")

    trimmed_answer = _drop_prefix(
        answer,
        (
            "اگه بخواهم در یک کلمه پاسخ بدهم جواب سؤال شما خیر است و",
            "اگر بخواهم در یک کلمه پاسخ بدهم جواب سؤال شما خیر است و",
        ),
    )
    payoff_core = _drop_prefix(payoff, ("ولی معناش این نیست که", "معناش این نیست که"))
    payoff_core = payoff_core.replace("حل می‌شد", "حل نمی‌شود")

    raw = [
        title,
        trimmed_answer or answer,
        payoff_core or payoff,
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _trim_to_words(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    if len(out) < 3:
        for extra in (title, answer, payoff):
            text = _trim_to_words(extra)
            if text and text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= 3:
                break
    return out[:3]


_INCOMPLETE_FIRST = frozenset({
    "کنیم",
    "کنید",
    "را",
    "و",
    "پس",
    "که",
    "ولی",
    "این",
})


def _looks_incomplete(text: str) -> bool:
    words = _persian_words(text)
    if not words:
        return True
    if words[0] in _INCOMPLETE_FIRST:
        return True
    if text.startswith("حال حاضر"):
        return True
    last = words[-1]
    return last in {"را", "که", "و", "اگر"}


def choose_headline(candidates: list[str]) -> str:
    if not candidates:
        raise CoverError("No headline candidates")

    def score(text: str) -> tuple[int, int]:
        words = _persian_words(text)
        n = len(words)
        points = 0
        if _looks_incomplete(text):
            points -= 6
        if 5 <= n <= 10:
            points += 3
        elif 4 <= n <= 12:
            points += 1
        if "؟" not in text and not text.startswith("آیا"):
            points += 2
        if any(token in text for token in ("نمی‌کند", "نمی‌شود", "نیست")):
            points += 1
        return points, -abs(8 - n)

    return max(candidates, key=score)


def _guest_windows(plan: dict) -> list[tuple[float, float]]:
    clips = _clips_from_plan(plan)
    preferred = {"central_answer", "reasoning", "payoff", "answer"}
    chosen = [
        (c["start"], c["end"])
        for c in clips
        if str(c.get("label") or "") in preferred
    ]
    return chosen or [(c["start"], c["end"]) for c in clips]


def _all_plan_windows(plan: dict) -> list[tuple[float, float]]:
    clips = _clips_from_plan(plan)
    return [(c["start"], c["end"]) for c in clips] or _guest_windows(plan)


def _sample_times(windows: list[tuple[float, float]], step: float = 0.33) -> list[float]:
    times: list[float] = []
    for start, end in windows:
        t = start + 0.12
        while t < end - 0.08:
            times.append(t)
            t += step
        if not times or times[-1] < end - 0.2:
            times.append(max(start, (start + end) / 2.0))
    return times


def _sample_portrait_times(windows: list[tuple[float, float]], *, step: float = _SAMPLE_STEP_S, edge: float = _CUT_EDGE_S) -> list[float]:
    """Dense samples, skipping frames next to edit/cut boundaries."""
    times: list[float] = []
    for start, end in windows:
        lo = start + edge
        hi = end - edge
        if hi <= lo:
            times.append((start + end) / 2.0)
            continue
        t = lo
        while t <= hi + 1e-6:
            times.append(round(t, 3))
            t += step
    # unique, sorted
    out: list[float] = []
    for t in sorted(times):
        if not out or abs(t - out[-1]) > 1e-3:
            out.append(t)
    return out


def _pick_guest_face(faces: list[BBox], panel: str) -> BBox | None:
    if not faces:
        return None
    ordered = sorted(faces, key=lambda b: b.cx)
    if panel == "bottom":
        return ordered[-1]
    return ordered[0]


def _face_roi(gray: np.ndarray, box: BBox, pad: float = 0.0) -> np.ndarray:
    x0 = max(0, int(box.x - box.w * pad))
    y0 = max(0, int(box.y - box.h * pad))
    x1 = min(gray.shape[1], int(box.x + box.w * (1.0 + pad)))
    y1 = min(gray.shape[0], int(box.y + box.h * (1.0 + pad)))
    return gray[y0:y1, x0:x1]


def _eye_patches(gray: np.ndarray, face: BBox, scale: float = 0.16) -> list[np.ndarray]:
    if not face.landmarks or len(face.landmarks) < 2:
        return []
    patches = []
    r = max(6, int(round(min(face.w, face.h) * scale)))
    for x, y in face.landmarks[:2]:
        x0, y0 = max(0, int(x - r)), max(0, int(y - r))
        x1, y1 = min(gray.shape[1], int(x + r)), min(gray.shape[0], int(y + r))
        patch = gray[y0:y1, x0:x1]
        if patch.size >= 16:
            patches.append(patch)
    return patches


def _eye_open_score(gray: np.ndarray, face: BBox) -> float:
    """0..1, higher = both eyes look open. Uses lid contrast + iris structure."""
    patches = _eye_patches(gray, face, 0.15)
    if len(patches) < 2:
        return 0.2
    scores = []
    for patch in patches:
        mid = patch.shape[0] // 2
        lid = float(patch[mid:].mean() - patch[:mid].mean())
        structure = float(cv2.Laplacian(patch, cv2.CV_64F).var())
        # Open eyes: lower half brighter (iris/glints) and some internal structure.
        scores.append(0.6 * min(max(lid, 0.0) / 12.0, 1.0) + 0.4 * min(structure / 140.0, 1.0))
    return float(min(scores))  # both eyes must be open


_HAAR_EYES = None


def _haar_eye_count(gray: np.ndarray, face: BBox) -> int:
    global _HAAR_EYES
    if _HAAR_EYES is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
        if not cascade_path.is_file():
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
        _HAAR_EYES = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.is_file() else False
    if _HAAR_EYES is False:
        return 0
    roi = _face_roi(gray, face, pad=0.05)
    if roi.size < 64:
        return 0
    eyes = _HAAR_EYES.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=3, minSize=(14, 14))
    return 0 if eyes is None else len(eyes)


def _head_pose(face: BBox) -> dict[str, float]:
    if not face.landmarks or len(face.landmarks) < 3:
        return {"yaw": 1.0, "pitch": 0.0, "roll": 0.0, "frontal": 0.4}
    (rx, ry), (lx, ly), (nx, ny) = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_dx = lx - rx
    eye_dy = ly - ry
    eye_dist = max(math.hypot(eye_dx, eye_dy), 1.0)
    mid_x = (rx + lx) / 2.0
    mid_y = (ry + ly) / 2.0
    yaw = (nx - mid_x) / eye_dist
    pitch = (ny - mid_y) / eye_dist
    roll = math.degrees(math.atan2(eye_dy, eye_dx))
    frontal = 1.0
    frontal *= max(0.0, 1.0 - min(1.0, abs(yaw) / 0.22))
    frontal *= max(0.0, 1.0 - min(1.0, abs(roll) / 10.0))
    # Cover stills want the nose just below the eyes — not chin-down, not looking up.
    if pitch < 0.48 or pitch > 0.78:
        frontal *= 0.2
    elif pitch < 0.52 or pitch > 0.70:
        frontal *= 0.55
    return {"yaw": yaw, "pitch": pitch, "roll": roll, "frontal": max(0.0, min(1.0, frontal))}


def _iris_gaze_score(gray: np.ndarray, face: BBox) -> float:
    """Estimate camera-facing gaze from iris/pupil position inside each eye."""
    patches = _eye_patches(gray, face, 0.14)
    if len(patches) < 2:
        return 0.35
    scores = []
    for patch in patches:
        h, w = patch.shape[:2]
        inner = patch[int(h * 0.18) : int(h * 0.82), int(w * 0.16) : int(w * 0.84)]
        if inner.size < 16:
            continue
        blur = cv2.GaussianBlur(inner, (5, 5), 0)
        _min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(blur)
        cx, cy = inner.shape[1] / 2.0, inner.shape[0] / 2.0
        dx = (min_loc[0] - cx) / max(cx, 1.0)
        dy = (min_loc[1] - cy) / max(cy, 1.0)
        scores.append(max(0.0, 1.0 - math.hypot(dx, dy) / 0.85))
    return float(min(scores)) if scores else 0.35


def _mouth_relaxed_score(gray: np.ndarray, face: BBox) -> float:
    """High when mouth is closed or only slightly open — not mid-word."""
    if not face.landmarks or len(face.landmarks) < 5:
        return 0.5
    (nx, ny) = face.landmarks[2]
    (x1, y1), (x2, y2) = face.landmarks[3], face.landmarks[4]
    width_px = math.hypot(x2 - x1, y2 - y1)
    width = width_px / max(face.w, 1.0)
    drop = ((y1 + y2) / 2.0 - ny) / max(face.h, 1.0)
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    pad = max(2.0, 0.08 * width_px)
    x0 = max(0, int(min(x1, x2) + pad))
    x1b = min(gray.shape[1], int(max(x1, x2) - pad))
    y0 = max(0, int(my - 0.09 * face.h))
    y1b = min(gray.shape[0], int(my + 0.12 * face.h))
    roi = gray[y0:y1b, x0:x1b]
    dark_frac = 0.0
    center_mean = 128.0
    if roi.size >= 32:
        dark_frac = float((roi < 50).mean())
        w = roi.shape[1]
        center = roi[:, w // 3 : 2 * w // 3] if w >= 9 else roi
        center_mean = float(center.mean())
    talking = width > 0.50 or drop > 0.42 or dark_frac > 0.055
    if talking:
        return 0.05
    closed = dark_frac <= 0.032 and center_mean >= 104.0 and width <= 0.44
    if closed:
        return 1.0
    if dark_frac <= 0.040 and width <= 0.46:
        return 0.55
    return 0.25


def _sharpness(gray: np.ndarray, box: BBox) -> float:
    roi = _face_roi(gray, box)
    if roi.size < 64:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _exposure_score(gray: np.ndarray, face: BBox) -> float:
    roi = _face_roi(gray, face)
    if roi.size < 64:
        return 0.0
    mean = float(roi.mean())
    if mean < 28 or mean > 230:
        return 0.0
    if 70 <= mean <= 180:
        return 1.0
    if 45 <= mean <= 210:
        return 0.7
    return 0.35


def score_guest_portrait(
    frame: np.ndarray,
    face: BBox,
    safe: BBox,
    *,
    panel: BBox,
) -> dict:
    """Score one guest still. Rejects blinks, profile, blur, mid-word, and cropped faces."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    details = {
        "accepted": False,
        "reject_reason": None,
        "eyes_open": 0.0,
        "haar_eyes": 0,
        "gaze_frontal": 0.0,
        "iris_gaze": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "sharpness": 0.0,
        "mouth_relaxed": 0.0,
        "exposure": 0.0,
        "face_in_panel": False,
        "score": 0.0,
    }
    in_panel = bbox_inside(face, safe, eps=2.0)
    details["face_in_panel"] = in_panel
    if not in_panel:
        details["reject_reason"] = "partially_cropped_face"
        return details
    if face.x < 1 or face.y < 1 or face.x + face.w > w - 1 or face.y + face.h > h - 1:
        details["reject_reason"] = "partially_cropped_face"
        return details

    pose = _head_pose(face)
    iris = _iris_gaze_score(gray, face)
    details["yaw"] = round(pose["yaw"], 3)
    details["pitch"] = round(pose["pitch"], 3)
    details["roll"] = round(pose["roll"], 2)
    details["gaze_frontal"] = round(pose["frontal"], 3)
    details["iris_gaze"] = round(iris, 3)
    looking_away = (
        abs(pose["yaw"]) > 0.20
        or abs(pose["roll"]) > 12
        or pose["pitch"] < 0.48
        or pose["pitch"] > 0.78
        or pose["frontal"] < 0.50
        or iris < 0.38
    )
    if looking_away:
        details["reject_reason"] = "looking_away_or_rotated"
        return details

    eyes = _eye_open_score(gray, face)
    haar = _haar_eye_count(gray, face)
    details["eyes_open"] = round(eyes, 3)
    details["haar_eyes"] = int(haar)
    if eyes < 0.42 or (haar < 1 and eyes < 0.55):
        details["reject_reason"] = "blink_or_closed_eyes"
        return details

    sharp = _sharpness(gray, face)
    details["sharpness"] = round(sharp, 1)
    if sharp < 90:
        details["reject_reason"] = "motion_blur"
        return details

    mouth = _mouth_relaxed_score(gray, face)
    details["mouth_relaxed"] = round(mouth, 3)
    if mouth < 0.40:
        details["reject_reason"] = "awkward_mouth"
        return details

    exposure = _exposure_score(gray, face)
    details["exposure"] = round(exposure, 3)
    if exposure <= 0.0:
        details["reject_reason"] = "bad_exposure"
        return details

    score = (
        0.22 * eyes
        + 0.26 * pose["frontal"]
        + 0.12 * iris
        + 0.20 * mouth
        + 0.10 * min(sharp / 280.0, 1.0)
        + 0.07 * exposure
        + 0.05 * min(float(face.score), 1.0)
    )
    details["score"] = round(float(score), 4)
    details["accepted"] = True
    return details


def detect_guest_panel(
    video: Path,
    windows: list[tuple[float, float]],
    guest_panel: str,
) -> BBox:
    probe_times = _sample_times(windows, step=max(1.2, (windows[-1][1] - windows[0][0]) / 6.0))[:8]
    frames = read_frames_at(video, probe_times)
    if len(frames) < 2:
        raise CoverError("Could not sample frames for guest panel detection")
    model = ensure_yunet_model()
    yunet = CachedYunet(model, score_threshold=0.55) if model else None
    faces_a = None
    faces_b = None
    for frame in frames:
        if yunet is None:
            break
        found = filter_faces(yunet.detect(frame), frame.shape[1], frame.shape[0], 40)
        if len(found) >= 2:
            found.sort(key=lambda b: b.cx)
            faces_a, faces_b = found[0], found[-1]
            break
    if faces_a is None or faces_b is None:
        raise CoverError("Need two faces to locate the guest panel")
    panels = detect_participant_panels(frames, faces_a, faces_b)
    if panels is None:
        raise CoverError("Could not detect split-screen guest panel")
    return panels[1] if guest_panel == "bottom" else panels[0]


def select_guest_still(
    video: Path,
    plan: dict,
    *,
    guest_panel: str,
    panel: BBox,
    crop_cfg: dict,
    aspect: float,
) -> dict:
    windows = _all_plan_windows(plan)
    times = _sample_portrait_times(windows)
    frames = read_frames_at(video, times)
    if not frames:
        raise CoverError("No frames available for guest still")
    model = ensure_yunet_model()
    yunet = CachedYunet(model, score_threshold=0.5) if model else None
    safe = inset_bbox(panel, float(crop_cfg.get("panel_safety_margin", 0.04)))
    evaluated = 0
    rejected = 0
    reject_counts: dict[str, int] = {}
    best = None
    ranked: list[dict] = []
    for t, frame in zip(times, frames):
        if yunet is None:
            break
        faces = filter_faces(yunet.detect(frame), frame.shape[1], frame.shape[0], 40)
        face = _pick_guest_face(faces, guest_panel)
        if face is None:
            continue
        evaluated += 1
        details = score_guest_portrait(frame, face, safe, panel=panel)
        details["timestamp"] = round(float(t), 3)
        if not details["accepted"]:
            rejected += 1
            reason = str(details.get("reject_reason") or "rejected")
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        ranked.append({
            "timestamp": round(float(t), 3),
            "score": details["score"],
            "eyes_open": details["eyes_open"],
            "gaze_frontal": details["gaze_frontal"],
            "iris_gaze": details.get("iris_gaze", 0.0),
            "mouth_relaxed": details["mouth_relaxed"],
            "sharpness": details["sharpness"],
        })
        if best is None or details["score"] > best["details"]["score"]:
            crop = crop_inside_roi(face, safe, aspect, **crop_cfg)
            best = {"timestamp": t, "crop": crop, "frame": frame, "face": face, "details": details}
    if best is None:
        raise CoverError("Could not find a clean guest portrait inside the panel")
    crop = best["crop"]
    frame = best["frame"]
    x, y, w, h = int(round(crop.x)), int(round(crop.y)), int(round(crop.w)), int(round(crop.h))
    x = max(0, x)
    y = max(0, y)
    patch = frame[y:y + h, x:x + w]
    if patch.size == 0:
        raise CoverError("Guest crop is empty")
    reasons = []
    d = best["details"]
    if d["eyes_open"] >= 0.55:
        reasons.append("both eyes open")
    if d["gaze_frontal"] >= 0.55 and d.get("iris_gaze", 0) >= 0.45:
        reasons.append("gaze toward camera / near-frontal pose")
    if d["mouth_relaxed"] >= 0.7:
        reasons.append("mouth relaxed / not mid-word")
    if d["sharpness"] >= 150:
        reasons.append("sharp face")
    if d["exposure"] >= 0.7:
        reasons.append("good exposure")
    if d["face_in_panel"]:
        reasons.append("fully inside guest panel")
    return {
        "timestamp": best["timestamp"],
        "crop": crop,
        "patch": patch,
        "evaluated": evaluated,
        "accepted": evaluated - rejected,
        "rejected": rejected,
        "reject_counts": reject_counts,
        "score": d["score"],
        "details": d,
        "reasons": reasons,
        "top_candidates": sorted(ranked, key=lambda r: r["score"], reverse=True)[:8],
    }


def _font_search_dirs(root: Path | None) -> list[Path]:
    dirs = []
    if root is not None:
        dirs.append(root / "assets" / "fonts")
    dirs.append(Path("C:/Windows/Fonts"))
    local = Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts"
    dirs.append(local)
    return [d for d in dirs if d.is_dir()]


def _iter_font_files(root: Path | None) -> list[Path]:
    found: list[Path] = []
    for folder in _font_search_dirs(root):
        try:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in _FONT_EXTS:
                    found.append(path)
        except OSError:
            continue
    return found


def _norm_style(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _file_family_style(path: Path) -> tuple[str, str]:
    stem = _norm_style(path.stem)
    family = stem
    style = "regular"
    for token in ("extrabold", "semibold", "demibold", "medium", "black", "bold", "regular", "light"):
        if token in stem:
            style = token
            family = stem.replace(token, "").replace("wght", "")
            break
    if "wght" in stem:
        family = family.replace("wght", "")
        if style == "regular":
            style = "variable"
    return family, style


def _open_variable(path: Path, size: int, style: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), size)
    wanted = style.replace(" ", "")
    names = []
    try:
        names = [n.decode("utf-8", "ignore") if isinstance(n, bytes) else str(n) for n in font.get_variation_names()]
    except Exception:
        names = []
    for name in names:
        if _norm_style(name) == _norm_style(wanted):
            font.set_variation_by_name(name)
            return font
    weight = _VAR_WEIGHT.get(_norm_style(wanted))
    if weight is not None:
        try:
            font.set_variation_by_axes([weight])
        except Exception:
            pass
    return font


def resolve_cover_font(role: str, *, root: Path | None, size: int) -> tuple[ImageFont.FreeTypeFont, str]:
    files = _iter_font_files(root)
    prefs = _FONT_PREF[role]
    for family, styles in prefs:
        fam_key = _norm_style(family)
        matches = [p for p in files if fam_key in _norm_style(p.stem) or fam_key in _norm_style(p.name)]
        if not matches:
            continue
        # Prefer a file whose name contains the requested style; else variable.
        for style in styles:
            sty_key = _norm_style(style)
            named = [p for p in matches if sty_key in _norm_style(p.stem)]
            if named:
                path = named[0]
                font = ImageFont.truetype(str(path), size)
                label = f"{path.stem} ({path.name})"
                return font, label
        variable = [p for p in matches if "wght" in p.stem.lower() or "variable" in p.stem.lower()]
        pool = variable or matches
        path = pool[0]
        font = _open_variable(path, size, styles[0])
        label = f"{family.title()} {styles[0]} ({path.name})"
        return font, label
    raise CoverError(
        "No professional Persian/Arabic sans-serif font found. "
        "Install Vazirmatn, Shabnam, Sahel, or Noto Sans Arabic."
    )


def _fit_font_spec(text: str, box: tuple[int, int, int, int], role: str, root: Path | None, max_size: int, min_size: int = 22) -> tuple[ImageFont.FreeTypeFont, str]:
    _, _, width, height = box
    chosen_label = ""
    size = max_size
    while size >= min_size:
        font, label = resolve_cover_font(role, root=root, size=size)
        chosen_label = label
        lines = _wrap_text(text, font, width - 8)
        line_h = font.getbbox("آ")[3] - font.getbbox("آ")[1]
        total_h = len(lines) * int(line_h * 1.18)
        if total_h <= height and all(
            font.getbbox(_rtl(line))[2] - font.getbbox(_rtl(line))[0] <= width - 8
            for line in lines
        ):
            return font, label
        size -= 2
    font, label = resolve_cover_font(role, root=root, size=min_size)
    return font, label or chosen_label


def _rtl(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def _fit_font(text: str, box: tuple[int, int, int, int], font_path: str, max_size: int, min_size: int = 22) -> ImageFont.FreeTypeFont:
    _, _, width, height = box
    size = max_size
    while size >= min_size:
        font = ImageFont.truetype(font_path, size)
        lines = _wrap_text(text, font, width - 8)
        line_h = font.getbbox("آ")[3] - font.getbbox("آ")[1]
        total_h = len(lines) * int(line_h * 1.22)
        if total_h <= height and all(
            font.getbbox(_rtl(line))[2] - font.getbbox(_rtl(line))[0] <= width - 8
            for line in lines
        ):
            return font
        size -= 2
    return ImageFont.truetype(font_path, min_size)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = _persian_words(text)
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        visual = _rtl(trial)
        w = font.getbbox(visual)[2] - font.getbbox(visual)[0]
        if current and w > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_rtl_block(
    canvas: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    *,
    stroke_fill: tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = box
    lines = _wrap_text(text, font, w - 8)
    line_h = font.getbbox("آ")[3] - font.getbbox("آ")[1]
    gap = int(line_h * 0.22)
    total = len(lines) * line_h + max(0, len(lines) - 1) * gap
    cy = y + max(0, (h - total) // 2)
    for line in lines:
        visual = _rtl(line)
        lw = font.getbbox(visual)[2] - font.getbbox(visual)[0]
        lx = x + max(0, (w - lw) // 2)
        kwargs = {"fill": fill}
        if stroke_width and stroke_fill is not None:
            kwargs["stroke_width"] = stroke_width
            kwargs["stroke_fill"] = stroke_fill
        draw.text((lx, cy), visual, font=font, **kwargs)
        cy += line_h + gap


def _sample_gold(template: np.ndarray) -> tuple[int, int, int]:
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 60, 80), (40, 255, 255))
    if int(mask.sum()) < 100:
        return (212, 175, 90)
    mean = cv2.mean(template, mask=mask)
    return (int(mean[2]), int(mean[1]), int(mean[0]))


def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=max(0, radius), fill=255)
    return np.array(mask)


def _paste_guest(
    canvas: np.ndarray,
    guest: np.ndarray,
    box: tuple[int, int, int, int],
    radius: int,
    gold: tuple[int, int, int],
) -> np.ndarray:
    x, y, w, h = box
    if guest.ndim == 2:
        guest = cv2.cvtColor(guest, cv2.COLOR_GRAY2BGR)
    if guest.shape[2] == 4:
        rgb = guest[:, :, :3]
        person_a = guest[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = guest[:, :, :3]
        person_a = np.ones(guest.shape[:2], dtype=np.float32)
    gh, gw = rgb.shape[:2]
    scale = min(w / max(1, gw), h / max(1, gh))
    new_w = max(1, int(round(gw * scale)))
    new_h = max(1, int(round(gh * scale)))
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    fitted = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    fitted_a = cv2.resize(person_a, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ox = (w - new_w) // 2
    oy = (h - new_h) // 2
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    alpha_layer = np.zeros((h, w), dtype=np.float32)
    layer[oy : oy + new_h, ox : ox + new_w] = fitted
    alpha_layer[oy : oy + new_h, ox : ox + new_w] = fitted_a
    round_mask = _rounded_mask(w, h, radius)
    roi = canvas[y : y + h, x : x + w]
    alpha = (alpha_layer * (round_mask.astype(np.float32) / 255.0))[..., None]
    blended = (layer.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    canvas[y : y + h, x : x + w] = blended
    overlay = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (x, y, x + w - 1, y + h - 1),
        radius=max(0, radius),
        outline=gold,
        width=4,
    )
    return cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)


def _largest_face(image: np.ndarray) -> BBox | None:
    bgr = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
    model = ensure_yunet_model()
    if model is None:
        return None
    yunet = CachedYunet(model, score_threshold=0.5)
    faces = filter_faces(yunet.detect(bgr), bgr.shape[1], bgr.shape[0], 40)
    if not faces:
        return None
    return max(faces, key=lambda f: f.area)


def face_aware_cover_crop(image: np.ndarray, width: int, height: int) -> dict:
    """Crop-to-fill a box. Uniform scale, no letterbox, face-centered with headroom."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgr = image[:, :, :3]
    src_h, src_w = bgr.shape[:2]
    if src_w < 2 or src_h < 2:
        raise CoverError("Portrait image is empty")
    scale = max(width / src_w, height / src_h)
    crop_w = min(float(src_w), width / scale)
    crop_h = min(float(src_h), height / scale)
    face = _largest_face(bgr)
    if face is None:
        cx = src_w / 2.0
        head_top = 0.0
    else:
        cx = float(face.cx)
        head_top = float(face.y) - 0.16 * float(face.h)
    x0 = cx - crop_w / 2.0
    y0 = head_top - 0.06 * crop_h
    x0 = min(max(0.0, x0), max(0.0, src_w - crop_w))
    y0 = min(max(0.0, y0), max(0.0, src_h - crop_h))
    x = int(round(x0))
    y = int(round(y0))
    w = max(1, int(round(crop_w)))
    h = max(1, int(round(crop_h)))
    x = min(max(0, x), max(0, src_w - w))
    y = min(max(0, y), max(0, src_h - h))
    patch = bgr[y : y + h, x : x + w]
    if patch.size == 0:
        raise CoverError("Cover crop is empty")
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    filled = cv2.resize(patch, (width, height), interpolation=interp)
    return {
        "patch": filled,
        "crop": {"x": float(x), "y": float(y), "w": float(w), "h": float(h)},
        "scale": round(float(scale), 4),
        "face": None
        if face is None
        else {
            "x": round(float(face.x), 1),
            "y": round(float(face.y), 1),
            "w": round(float(face.w), 1),
            "h": round(float(face.h), 1),
        },
    }


def resolve_concept_label(value: str | None) -> str | None:
    """Kickers are off unless a non-empty phrase is explicitly supplied."""
    text = str(value or "").strip()
    return text or None


def _draw_editorial_headline(
    canvas: Image.Image,
    lines: list[str],
    concept: str | None,
    box: tuple[int, int, int, int],
    root: Path | None,
    gold: tuple[int, int, int],
) -> tuple[str, str | None]:
    x, y, w, h = box
    draw = ImageDraw.Draw(canvas)
    concept_font, concept_label = resolve_cover_font("role", root=root, size=30)
    headline_font = None
    headline_label = ""
    for size in range(74, 47, -2):
        font, label = resolve_cover_font("headline", root=root, size=size)
        line_h = font.getbbox("آ")[3] - font.getbbox("آ")[1]
        gap = int(line_h * 0.30)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * gap
        concept_h = 0
        if concept:
            ch = concept_font.getbbox("آ")[3] - concept_font.getbbox("آ")[1]
            concept_h = ch + 26
        widths_ok = all(
            (font.getbbox(_rtl(line))[2] - font.getbbox(_rtl(line))[0]) <= (w - 24)
            for line in lines
        )
        if widths_ok and concept_h + total_h <= h - 12:
            headline_font = font
            headline_label = label
            break
    if headline_font is None:
        headline_font, headline_label = resolve_cover_font("headline", root=root, size=48)
    line_h = headline_font.getbbox("آ")[3] - headline_font.getbbox("آ")[1]
    gap = int(line_h * 0.30)
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * gap
    concept_h = 0
    if concept:
        concept_line_h = concept_font.getbbox("آ")[3] - concept_font.getbbox("آ")[1]
        concept_h = concept_line_h + 26
    cy = y + max(0, (h - (concept_h + total_h)) // 2)
    if concept:
        visual = _rtl(concept)
        lw = concept_font.getbbox(visual)[2] - concept_font.getbbox(visual)[0]
        lx = x + max(0, (w - lw) // 2)
        draw.text((lx, cy), visual, font=concept_font, fill=gold)
        cy += concept_h
    for line in lines:
        visual = _rtl(line)
        lw = headline_font.getbbox(visual)[2] - headline_font.getbbox(visual)[0]
        lx = x + max(0, (w - lw) // 2)
        draw.text(
            (lx, cy),
            visual,
            font=headline_font,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(8, 18, 40),
        )
        cy += line_h + gap
    return headline_label, (concept_label if concept else None)


def generate_cover_finaltest(
    plan_path: Path,
    metadata_path: Path,
    cfg: dict,
    *,
    root: Path,
    master_portrait: Path,
    output_jpg: Path,
    output_json: Path,
    headline: str,
    headline_lines: list[str],
    headline_candidates: list[str],
    concept_label: str | None = None,
) -> dict:
    from .portrait import load_master_guest_portrait

    concept_label = resolve_concept_label(concept_label)
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    meta = load_cover_metadata(metadata_path, root=root)
    width, height = parse_cover_size(meta["cover_size"])
    template_img = cv2.imread(str(meta["cover_template"]), cv2.IMREAD_COLOR)
    if template_img is None:
        raise CoverError(f"Could not read template: {meta['cover_template']}")
    canvas = scale_template(template_img, width, height)
    layout = load_cover_layout(Path(meta["cover_template"]), (width, height))
    master_file = Path(master_portrait)
    master = load_master_guest_portrait(master_file)
    gx, gy, gw, gh = layout["guest"]
    filled = face_aware_cover_crop(master["image"], gw, gh)
    gold = _sample_gold(canvas)
    canvas = _paste_guest(canvas, filled["patch"], layout["guest"], layout["guest_radius"], gold)
    canvas = _restore_protected(canvas, scale_template(template_img, width, height), layout["protected"])

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    headline_font_label, concept_font_label = _draw_editorial_headline(
        pil, headline_lines, concept_label, layout["headline"], root, gold
    )
    name_font, name_font_label = _fit_font_spec(meta["guest_name"], layout["name"], "name", root, 42, 24)
    role_font, role_font_label = _fit_font_spec(meta["guest_role"], layout["role"], "role", root, 28, 18)
    _draw_rtl_block(pil, meta["guest_name"], layout["name"], name_font, gold)
    _draw_rtl_block(pil, meta["guest_role"], layout["role"], role_font, (232, 220, 190))
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    canvas = _restore_protected(canvas, scale_template(template_img, width, height), layout["protected"])
    problems = layout_problems(canvas, layout, scale_template(template_img, width, height))

    jpg_path = Path(output_jpg)
    json_path = Path(output_json)
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(jpg_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    if not ok:
        raise CoverError(f"Failed to write {jpg_path}")

    png_crop = filled["crop"]
    origin = master["crop"]
    src_h = int(master["image"].shape[0])
    scale_from_master = (origin.h / src_h) if src_h else 1.0
    source_crop = {
        "x": round(origin.x + png_crop["x"] * scale_from_master, 1),
        "y": round(origin.y + png_crop["y"] * scale_from_master, 1),
        "w": round(png_crop["w"] * scale_from_master, 1),
        "h": round(png_crop["h"] * scale_from_master, 1),
    }
    payload = {
        "selected_headline": headline,
        "short_concept_phrase": concept_label,
        "headline_candidates": headline_candidates,
        "headline_lines": headline_lines,
        "number_of_headline_lines": len(headline_lines),
        "guest_name": meta["guest_name"],
        "guest_role": meta["guest_role"],
        "guest_panel": meta["guest_panel"],
        "guest_portrait": str(master_file.as_posix()),
        "portrait_mode": "master_cover_fill",
        "source_frame_timestamp": ts(master["timestamp"]),
        "source_frame_seconds": round(float(master["timestamp"]), 3),
        "master_portrait_crop_coordinates": {
            "x": round(png_crop["x"], 1),
            "y": round(png_crop["y"], 1),
            "w": round(png_crop["w"], 1),
            "h": round(png_crop["h"], 1),
        },
        "source_crop_coordinates": source_crop,
        "final_portrait_scale": filled["scale"],
        "face": filled.get("face"),
        "crop_coordinates": source_crop,
        "panel": {
            "x": round(float(master["panel"].x), 1),
            "y": round(float(master["panel"].y), 1),
            "w": round(float(master["panel"].w), 1),
            "h": round(float(master["panel"].h), 1),
        },
        "template_used": meta.get("cover_template_rel") or str(Path(meta["cover_template"]).as_posix()),
        "cover_size": f"{width}x{height}",
        "fonts": {
            "headline": headline_font_label,
            "concept": concept_font_label,
            "guest_name": name_font_label,
            "guest_role": role_font_label,
        },
        "layout_problems": problems,
        "output": str(jpg_path.as_posix()),
        "reel_id": plan.get("reel_id") or plan_path.stem,
    }
    write_json(json_path, payload)
    return payload


def _restore_protected(canvas: np.ndarray, template: np.ndarray, protected: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = canvas
    for x, y, w, h in protected:
        x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        x0, y0 = max(0, x), max(0, y)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = template[y0:y1, x0:x1]
    return out


def layout_problems(
    canvas: np.ndarray,
    layout: dict,
    template: np.ndarray,
) -> list[str]:
    problems: list[str] = []
    h, w = canvas.shape[:2]
    if (w, h) != (template.shape[1], template.shape[0]):
        problems.append("canvas size does not match scaled template")
    for x, y, bw, bh in layout.get("protected") or []:
        region_c = canvas[y:y + bh, x:x + bw]
        region_t = template[y:y + bh, x:x + bw]
        if region_c.size == 0:
            problems.append("protected logo region is empty")
            continue
        delta = cv2.absdiff(region_c, region_t).mean()
        if delta > 8:
            problems.append(f"protected template region changed (mean delta {delta:.1f})")
    gx, gy, gw, gh = layout["guest"]
    if gx < 0 or gy < 0 or gx + gw > w or gy + gh > h:
        problems.append("guest image box is outside the canvas")
    return problems


def generate_cover(
    plan_path: Path,
    metadata_path: Path,
    cfg: dict,
    *,
    root: Path,
    output_jpg: Path | None = None,
    output_json: Path | None = None,
    master_portrait: Path | None = None,
) -> dict:
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    meta = load_cover_metadata(metadata_path, root=root)
    width, height = parse_cover_size(meta["cover_size"])
    source = Path(plan.get("source_video") or "")
    source = _resolve_source_video(source, plan_path)
    template_img = cv2.imread(str(meta["cover_template"]), cv2.IMREAD_COLOR)
    if template_img is None:
        raise CoverError(f"Could not read template: {meta['cover_template']}")
    canvas = scale_template(template_img, width, height)
    layout = load_cover_layout(Path(meta["cover_template"]), (width, height))
    crop_cfg = face_crop_params(cfg.get("render") or {})
    gx, gy, gw, gh = layout["guest"]
    aspect = gw / max(1, gh)
    from .portrait import load_master_guest_portrait, master_portrait_path, resolve_master_portrait

    source_stem = Path(plan.get("source_video") or source).stem or source.stem
    declared = str(meta.get("guest_portrait") or "").strip()
    if master_portrait is not None:
        master_file = Path(master_portrait)
        if not master_file.is_file():
            raise CoverError(f"Master guest portrait not found: {master_file}")
        master = load_master_guest_portrait(master_file)
    else:
        master_path = master_portrait_path(source_stem, root=root, meta=meta, cfg=cfg)
        if declared and not master_path.is_file():
            raise CoverError(f"Master guest portrait not found: {master_path}")
        master_file = resolve_master_portrait(source_stem, root=root, meta=meta, cfg=cfg)
        master = load_master_guest_portrait(master_file) if master_file else None
    if master is not None:
        timestamp = master["timestamp"]
        crop = master["crop"]
        guest_patch = master["image"]
        panel = master["panel"]
        still = {
            "evaluated": (master["sidecar"].get("portrait_selection") or {}).get("frames_evaluated", 0),
            "accepted": (master["sidecar"].get("portrait_selection") or {}).get("frames_accepted", 0),
            "rejected": (master["sidecar"].get("portrait_selection") or {}).get("frames_rejected", 0),
            "reject_counts": (master["sidecar"].get("portrait_selection") or {}).get("reject_counts", {}),
            "score": master["score"],
            "details": (master["sidecar"].get("portrait_selection") or {}).get("details", {}),
            "reasons": (master["sidecar"].get("portrait_selection") or {}).get("reasons", ["master guest portrait"]),
            "top_candidates": (master["sidecar"].get("portrait_selection") or {}).get("top_candidates", []),
        }
    else:
        panel = detect_guest_panel(source, _all_plan_windows(plan), meta["guest_panel"])
        still = select_guest_still(
            source,
            plan,
            guest_panel=meta["guest_panel"],
            panel=panel,
            crop_cfg=crop_cfg,
            aspect=aspect,
        )
        timestamp, crop, guest_patch = still["timestamp"], still["crop"], still["patch"]
    gold = _sample_gold(canvas)
    canvas = _paste_guest(canvas, guest_patch, layout["guest"], layout["guest_radius"], gold)
    canvas = _restore_protected(canvas, scale_template(template_img, width, height), layout["protected"])

    candidates = headline_candidates(plan)
    headline = choose_headline(candidates)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    headline_font, headline_font_label = _fit_font_spec(headline, layout["headline"], "headline", root, 68, 28)
    name_font, name_font_label = _fit_font_spec(meta["guest_name"], layout["name"], "name", root, 42, 24)
    role_font, role_font_label = _fit_font_spec(meta["guest_role"], layout["role"], "role", root, 28, 18)
    _draw_rtl_block(
        pil, headline, layout["headline"], headline_font, (255, 255, 255),
        stroke_fill=(8, 18, 40), stroke_width=2,
    )
    _draw_rtl_block(pil, meta["guest_name"], layout["name"], name_font, gold)
    _draw_rtl_block(pil, meta["guest_role"], layout["role"], role_font, (232, 220, 190))
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    canvas = _restore_protected(canvas, scale_template(template_img, width, height), layout["protected"])
    problems = layout_problems(canvas, layout, scale_template(template_img, width, height))

    reel_id = plan.get("reel_id") or plan_path.stem
    out_dir = Path(cfg["paths"]["output"])
    jpg_path = Path(output_jpg) if output_jpg else out_dir / f"{reel_id}_cover.jpg"
    json_path = Path(output_json) if output_json else out_dir / f"{reel_id}_cover.json"
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(jpg_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    if not ok:
        raise CoverError(f"Failed to write {jpg_path}")
    payload = {
        "selected_headline": headline,
        "headline_candidates": candidates,
        "guest_name": meta["guest_name"],
        "guest_role": meta["guest_role"],
        "guest_panel": meta["guest_panel"],
        "guest_portrait": str(master_file.as_posix()) if master is not None else None,
        "portrait_mode": "master" if master is not None else "per_reel",
        "source_frame_timestamp": ts(timestamp),
        "source_frame_seconds": round(float(timestamp), 3),
        "crop_coordinates": {
            "x": round(float(crop.x), 1),
            "y": round(float(crop.y), 1),
            "w": round(float(crop.w), 1),
            "h": round(float(crop.h), 1),
        },
        "panel": {
            "x": round(float(panel.x), 1),
            "y": round(float(panel.y), 1),
            "w": round(float(panel.w), 1),
            "h": round(float(panel.h), 1),
        },
        "template_used": meta.get("cover_template_rel") or str(Path(meta["cover_template"]).as_posix()),
        "cover_size": f"{width}x{height}",
        "fonts": {
            "headline": headline_font_label,
            "guest_name": name_font_label,
            "guest_role": role_font_label,
        },
        "portrait_selection": {
            "frames_evaluated": still["evaluated"],
            "frames_accepted": still["accepted"],
            "frames_rejected": still["rejected"],
            "reject_counts": still["reject_counts"],
            "score": still["score"],
            "reasons": still["reasons"],
            "details": still["details"],
            "top_candidates": still.get("top_candidates", []),
        },
        "layout_problems": problems,
        "output": str(jpg_path.as_posix()),
    }
    write_json(json_path, payload)
    return payload
