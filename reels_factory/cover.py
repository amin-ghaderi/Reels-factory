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
)
from .render import _clips_from_plan, _resolve_source_video
from .utils import parse_timestamp, read_json, ts, write_json

GUEST_PANELS = {"top", "bottom"}
_WORD_RE = re.compile(r"\s+")


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


def choose_headline(candidates: list[str]) -> str:
    if not candidates:
        raise CoverError("No headline candidates")

    def score(text: str) -> tuple[int, int]:
        words = _persian_words(text)
        n = len(words)
        points = 0
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


def _pick_guest_face(faces: list[BBox], panel: str) -> BBox | None:
    if not faces:
        return None
    ordered = sorted(faces, key=lambda b: b.cx)
    if panel == "bottom":
        return ordered[-1]
    return ordered[0]


def _eye_score(gray: np.ndarray, face: BBox) -> float:
    if not face.landmarks or len(face.landmarks) < 2:
        return 0.45
    scores = []
    for x, y in face.landmarks[:2]:
        r = max(4, int(round(min(face.w, face.h) * 0.12)))
        x0, y0 = max(0, int(x - r)), max(0, int(y - r))
        x1, y1 = min(gray.shape[1], int(x + r)), min(gray.shape[0], int(y + r))
        patch = gray[y0:y1, x0:x1]
        if patch.size < 16:
            continue
        scores.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    if not scores:
        return 0.45
    return min(1.0, float(np.mean(scores)) / 180.0)


def _eye_lid_contrast(gray: np.ndarray, face: BBox) -> float:
    """Open eyes usually make the lower half of the eye box brighter than the lid."""
    if not face.landmarks or len(face.landmarks) < 2:
        return 0.0
    vals = []
    for x, y in face.landmarks[:2]:
        r = max(6, int(round(min(face.w, face.h) * 0.14)))
        x0, y0 = max(0, int(x - r)), max(0, int(y - r))
        x1, y1 = min(gray.shape[1], int(x + r)), min(gray.shape[0], int(y + r))
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] < 8:
            continue
        mid = patch.shape[0] // 2
        vals.append(float(patch[mid:].mean() - patch[:mid].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _frontal_score(face: BBox) -> float:
    if not face.landmarks or len(face.landmarks) < 3:
        return 0.5
    (rx, ry), (lx, ly), (nx, ny) = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    mid_x = (rx + lx) / 2.0
    eye_y = (ry + ly) / 2.0
    offset = abs(nx - mid_x) / max(face.w, 1.0)
    looking_down = (ny - eye_y) / max(face.h, 1.0)
    score = 1.0 - min(1.0, offset / 0.18)
    if looking_down < 0.12:
        score *= 0.45
    return max(0.1, score)


def _mouth_score(face: BBox) -> float:
    if not face.landmarks or len(face.landmarks) < 5:
        return 0.7
    (x1, y1), (x2, y2) = face.landmarks[3], face.landmarks[4]
    width = math.hypot(x2 - x1, y2 - y1)
    ratio = width / max(face.w, 1.0)
    if 0.26 <= ratio <= 0.42:
        return 1.0
    if ratio < 0.18 or ratio > 0.55:
        return 0.2
    return 0.6


def _sharpness(gray: np.ndarray, box: BBox) -> float:
    x0 = max(0, int(box.x))
    y0 = max(0, int(box.y))
    x1 = min(gray.shape[1], int(box.x + box.w))
    y1 = min(gray.shape[0], int(box.y + box.h))
    roi = gray[y0:y1, x0:x1]
    if roi.size < 64:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


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
) -> tuple[float, BBox, np.ndarray]:
    windows = _guest_windows(plan)
    times = _sample_times(windows, step=0.33)
    frames = read_frames_at(video, times)
    if not frames:
        raise CoverError("No frames available for guest still")
    model = ensure_yunet_model()
    yunet = CachedYunet(model, score_threshold=0.5) if model else None
    safe = inset_bbox(panel, float(crop_cfg.get("panel_safety_margin", 0.04)))
    best = None
    for t, frame in zip(times, frames):
        if yunet is None:
            break
        faces = filter_faces(yunet.detect(frame), frame.shape[1], frame.shape[0], 40)
        face = _pick_guest_face(faces, guest_panel)
        if face is None:
            continue
        if not (
            safe.x - 8 <= face.cx <= safe.x + safe.w + 8
            and safe.y - 8 <= face.cy <= safe.y + safe.h + 8
        ):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lid = _eye_lid_contrast(gray, face)
        if lid < 5.0:
            continue
        if _frontal_score(face) < 0.7:
            continue
        sharp = _sharpness(gray, face)
        score = (
            0.28 * min(sharp / 220.0, 1.0)
            + 0.14 * float(face.score)
            + 0.18 * _eye_score(gray, face)
            + 0.14 * _mouth_score(face)
            + 0.14 * _frontal_score(face)
            + 0.12 * min(max(lid, 0.0) / 14.0, 1.0)
        )
        if best is None or score > best[0]:
            crop = crop_inside_roi(face, safe, aspect, **crop_cfg)
            best = (score, t, crop, frame)
    if best is None:
        raise CoverError("Could not find a clean guest frame inside the panel")
    _, timestamp, crop, frame = best
    x, y, w, h = int(round(crop.x)), int(round(crop.y)), int(round(crop.w)), int(round(crop.h))
    x = max(0, x)
    y = max(0, y)
    patch = frame[y:y + h, x:x + w]
    if patch.size == 0:
        raise CoverError("Guest crop is empty")
    return timestamp, crop, patch


def _windows_font(*names: str) -> str:
    windir = Path("C:/Windows/Fonts")
    for name in names:
        path = windir / name
        if path.is_file():
            return str(path)
    raise CoverError("No Persian-capable font found (expected Tahoma/Segoe UI/Arial)")


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
    fitted = cv2.resize(guest, (w, h), interpolation=cv2.INTER_AREA)
    mask = _rounded_mask(w, h, radius)
    roi = canvas[y:y + h, x:x + w]
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = (fitted.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    canvas[y:y + h, x:x + w] = blended
    overlay = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (x, y, x + w - 1, y + h - 1),
        radius=max(0, radius),
        outline=gold,
        width=4,
    )
    return cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)


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
    panel = detect_guest_panel(source, _guest_windows(plan), meta["guest_panel"])
    timestamp, crop, guest_patch = select_guest_still(
        source,
        plan,
        guest_panel=meta["guest_panel"],
        panel=panel,
        crop_cfg=crop_cfg,
        aspect=aspect,
    )
    gold = _sample_gold(canvas)
    canvas = _paste_guest(canvas, guest_patch, layout["guest"], layout["guest_radius"], gold)
    canvas = _restore_protected(canvas, scale_template(template_img, width, height), layout["protected"])

    candidates = headline_candidates(plan)
    headline = choose_headline(candidates)
    bold = _windows_font("tahomabd.ttf", "Tahoma Bold.ttf", "segoeuib.ttf", "arialbd.ttf")
    regular = _windows_font("tahoma.ttf", "segoeui.ttf", "arial.ttf")
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    headline_font = _fit_font(headline, layout["headline"], bold, 72, 28)
    name_font = _fit_font(meta["guest_name"], layout["name"], bold, 44, 26)
    role_font = _fit_font(meta["guest_role"], layout["role"], regular, 30, 20)
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
        "layout_problems": problems,
        "output": str(jpg_path.as_posix()),
    }
    write_json(json_path, payload)
    return payload
