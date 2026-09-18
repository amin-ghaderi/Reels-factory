from __future__ import annotations

import hashlib
import json
import statistics
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    landmarks: tuple[tuple[float, float], ...] | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class FaceTrack:
    boxes: list[BBox | None]

    @property
    def hit_ratio(self) -> float:
        if not self.boxes:
            return 0.0
        hits = sum(1 for b in self.boxes if b is not None)
        return hits / len(self.boxes)

    def known(self) -> list[BBox]:
        return [b for b in self.boxes if b is not None]


@dataclass
class LayoutPlan:
    mode: str
    filter_complex: str
    method: str
    face_counts: list[int] = field(default_factory=list)
    avg_faces: float = 0.0
    top: BBox | None = None
    bottom: BBox | None = None
    single: BBox | None = None
    panel_a: BBox | None = None
    panel_b: BBox | None = None
    safety_margin: float = 0.0


def _even(value: float, minimum: int = 2) -> int:
    n = int(round(value))
    if n % 2:
        n -= 1
    return max(minimum, n)


def _lerp_box(a: BBox, b: BBox, t: float) -> BBox:
    t = min(1.0, max(0.0, t))
    return BBox(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        w=a.w + (b.w - a.w) * t,
        h=a.h + (b.h - a.h) * t,
        score=a.score + (b.score - a.score) * t,
    )


def interpolate_missing(boxes: list[BBox | None]) -> list[BBox | None]:
    """Fill detection gaps by carrying edges and linearly interpolating interiors."""
    n = len(boxes)
    known = [i for i, box in enumerate(boxes) if box is not None]
    if not known:
        return list(boxes)
    out: list[BBox | None] = list(boxes)
    first, last = known[0], known[-1]
    for i in range(first):
        out[i] = boxes[first]
    for i in range(last + 1, n):
        out[i] = boxes[last]
    for a, b in zip(known, known[1:]):
        if b <= a + 1:
            continue
        left, right = boxes[a], boxes[b]
        span = b - a
        for i in range(a + 1, b):
            out[i] = _lerp_box(left, right, (i - a) / span)
    return out


def smooth_bboxes(boxes: list[BBox], alpha: float = 0.35) -> list[BBox]:
    """Exponential moving average that damps jitter without dropping samples."""
    if not boxes:
        return []
    alpha = min(1.0, max(0.0, float(alpha)))
    acc = boxes[0]
    out = [acc]
    keep = 1.0 - alpha
    for box in boxes[1:]:
        acc = BBox(
            x=alpha * box.x + keep * acc.x,
            y=alpha * box.y + keep * acc.y,
            w=alpha * box.w + keep * acc.w,
            h=alpha * box.h + keep * acc.h,
            score=alpha * box.score + keep * acc.score,
        )
        out.append(acc)
    return out


def median_bbox(boxes: list[BBox]) -> BBox:
    return BBox(
        x=statistics.median(b.x for b in boxes),
        y=statistics.median(b.y for b in boxes),
        w=statistics.median(b.w for b in boxes),
        h=statistics.median(b.h for b in boxes),
        score=statistics.median(b.score for b in boxes),
    )


def stable_track_bbox(track: FaceTrack, alpha: float = 0.35) -> BBox | None:
    filled = [b for b in interpolate_missing(track.boxes) if b is not None]
    if not filled:
        return None
    return median_bbox(smooth_bboxes(filled, alpha=alpha))


def face_crop_params(rcfg: dict | None = None) -> dict:
    raw = (rcfg or {}).get("face_crop") or {}
    return {
        "zoom": float(raw.get("zoom", 1.32)),
        "headroom": float(raw.get("headroom", 0.38)),
        "horizontal_margin": float(raw.get("horizontal_margin", 0.15)),
        "body": float(raw.get("body", 1.05)),
        "panel_safety_margin": float(raw.get("panel_safety_margin", 0.04)),
    }


def _face_crop_kwargs(crop_cfg: dict) -> dict:
    return {
        key: crop_cfg[key]
        for key in ("zoom", "headroom", "horizontal_margin", "body")
        if key in crop_cfg
    }


def crop_around_face(
    face: BBox,
    frame_w: int,
    frame_h: int,
    aspect: float,
    *,
    zoom: float = 1.32,
    headroom: float = 0.38,
    horizontal_margin: float = 0.15,
    body: float = 1.05,
    side: float | None = None,
) -> BBox:
    """Tight face-safe crop: head + shoulders + upper torso, no stretching."""
    if side is not None:
        horizontal_margin = side
    zoom = max(1.0, float(zoom))
    pad_top = max(0.0, float(headroom)) * face.h
    pad_x = max(0.0, float(horizontal_margin)) * face.w
    pad_bot = max(0.2, float(body)) * face.h

    desired_w = max(face.w + 2 * pad_x, face.w * 1.02, 2.0)
    desired_h = max(pad_top + face.h + pad_bot, face.h * 1.02, 2.0)
    if desired_w / desired_h < aspect:
        desired_w = desired_h * aspect
    else:
        desired_h = desired_w / aspect

    crop_w = desired_w / zoom
    crop_h = desired_h / zoom

    # Zoom may not eat the face or the requested headroom.
    min_h = face.h + pad_top
    min_w = face.w + 2 * pad_x
    if crop_h < min_h:
        crop_h = min_h
        crop_w = crop_h * aspect
    if crop_w < min_w:
        crop_w = min_w
        crop_h = crop_w / aspect
        if crop_h < min_h:
            crop_h = min_h
            crop_w = crop_h * aspect

    max_h = float(frame_h)
    max_w = max_h * aspect
    if max_w > frame_w:
        max_w = float(frame_w)
        max_h = max_w / aspect
    if crop_w > max_w:
        crop_w = max_w
        crop_h = crop_w / aspect
    if crop_h > max_h:
        crop_h = max_h
        crop_w = crop_h * aspect

    crop_w = min(crop_w, float(frame_w))
    crop_h = min(crop_h, float(frame_h))

    x0 = face.cx - crop_w / 2.0
    y0 = face.y - pad_top
    if face.x < x0:
        x0 = face.x
    if face.x + face.w > x0 + crop_w:
        x0 = face.x + face.w - crop_w
    if face.y < y0:
        y0 = face.y
    if face.y + face.h > y0 + crop_h:
        y0 = face.y + face.h - crop_h

    x0 = min(max(0.0, x0), max(0.0, frame_w - crop_w))
    y0 = min(max(0.0, y0), max(0.0, frame_h - crop_h))
    return BBox(x=x0, y=y0, w=crop_w, h=crop_h, score=face.score)


def inset_bbox(box: BBox, margin: float) -> BBox:
    """Pull every edge inward by `margin` of the panel size (and at least 4px)."""
    margin = min(0.45, max(0.0, float(margin)))
    mx = max(4.0, box.w * margin)
    my = max(4.0, box.h * margin)
    mx = min(mx, box.w * 0.45)
    my = min(my, box.h * 0.45)
    w = max(2.0, box.w - 2.0 * mx)
    h = max(2.0, box.h - 2.0 * my)
    return BBox(x=box.x + mx, y=box.y + my, w=w, h=h, score=box.score)


def bbox_inside(inner: BBox, outer: BBox, eps: float = 1.0) -> bool:
    return (
        inner.x >= outer.x - eps
        and inner.y >= outer.y - eps
        and inner.x + inner.w <= outer.x + outer.w + eps
        and inner.y + inner.h <= outer.y + outer.h + eps
    )


def clamp_bbox_to_roi(box: BBox, roi: BBox, aspect: float) -> BBox:
    """Shrink a 9:8 crop until it sits entirely inside roi. Never expand."""
    x0 = max(box.x, roi.x)
    y0 = max(box.y, roi.y)
    x1 = min(box.x + box.w, roi.x + roi.w)
    y1 = min(box.y + box.h, roi.y + roi.h)
    w = max(2.0, x1 - x0)
    h = max(2.0, y1 - y0)
    if w / h > aspect:
        w = h * aspect
        cx = min(max(box.cx, roi.x + w / 2.0), roi.x + roi.w - w / 2.0)
        x0 = cx - w / 2.0
    else:
        h = w / aspect
        cy = min(max(box.cy, roi.y + h / 2.0), roi.y + roi.h - h / 2.0)
        y0 = cy - h / 2.0
    x0 = min(max(roi.x, x0), roi.x + roi.w - w)
    y0 = min(max(roi.y, y0), roi.y + roi.h - h)
    return BBox(x=x0, y=y0, w=w, h=h, score=box.score)


def crop_inside_roi(face: BBox, roi: BBox, aspect: float, **crop_cfg) -> BBox:
    """Face-centered 9:8 crop that cannot cross the safe ROI boundary."""
    roi_w = max(2.0, roi.w)
    roi_h = max(2.0, roi.h)
    fx = min(max(face.x, roi.x), roi.x + roi_w - 2.0)
    fy = min(max(face.y, roi.y), roi.y + roi_h - 2.0)
    fw = min(max(2.0, face.w), roi.x + roi_w - fx)
    fh = min(max(2.0, face.h), roi.y + roi_h - fy)
    local_face = BBox(fx - roi.x, fy - roi.y, fw, fh, score=face.score)
    kwargs = _face_crop_kwargs(crop_cfg)
    local = crop_around_face(
        local_face,
        max(2, int(round(roi_w))),
        max(2, int(round(roi_h))),
        aspect,
        **kwargs,
    )
    crop = BBox(local.x + roi.x, local.y + roi.y, local.w, local.h, score=face.score)
    return clamp_bbox_to_roi(crop, roi, aspect)


def _shrink_crop_to_height(crop: BBox, face: BBox, height: float, aspect: float, roi: BBox) -> BBox:
    height = min(crop.h, max(2.0, height))
    width = height * aspect
    if width > roi.w:
        width = roi.w
        height = width / aspect
    x0 = face.cx - width / 2.0
    y0 = face.y - max(0.0, (height - face.h) * 0.28)
    if face.x < x0:
        x0 = face.x
    if face.x + face.w > x0 + width:
        x0 = face.x + face.w - width
    if face.y < y0:
        y0 = face.y
    if face.y + face.h > y0 + height:
        y0 = face.y + face.h - height
    return clamp_bbox_to_roi(BBox(x0, y0, width, height, crop.score), roi, aspect)


def _argmax_range(energy: np.ndarray, lo: float, hi: float) -> tuple[int | None, float]:
    lo_i = max(0, int(lo))
    hi_i = min(int(energy.shape[0]), int(hi))
    if hi_i <= lo_i:
        return None, 0.0
    sl = energy[lo_i:hi_i]
    idx = int(np.argmax(sl))
    return lo_i + idx, float(sl[idx])


def _smooth_1d(energy: np.ndarray, k: int = 11) -> np.ndarray:
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(energy.astype(np.float64), kernel, mode="same")


def _local_maxima(energy: np.ndarray, *, min_frac: float = 0.28, radius: int = 8) -> list[int]:
    sm = _smooth_1d(energy)
    peak = float(sm.max()) if sm.size else 0.0
    if peak <= 1e-6:
        return []
    thresh = peak * min_frac
    peaks: list[int] = []
    n = int(sm.shape[0])
    r = max(3, radius)
    for x in range(r, n - r):
        window = sm[x - r : x + r + 1]
        if sm[x] >= float(window.max()) and sm[x] >= thresh:
            peaks.append(x)
    return peaks


def _sobel_profiles(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    row_acc: np.ndarray | None = None
    col_acc: np.ndarray | None = None
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        row = np.mean(np.abs(sy), axis=1)
        col = np.mean(np.abs(sx), axis=0)
        row_acc = row if row_acc is None else row_acc + row
        col_acc = col if col_acc is None else col_acc + col
    n = float(max(1, len(frames)))
    assert row_acc is not None and col_acc is not None
    return row_acc / n, col_acc / n


def _border_col_energy(frames: list[np.ndarray], top: int, bot: int) -> np.ndarray:
    acc: np.ndarray | None = None
    h = int(frames[0].shape[0])
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        parts = []
        for y in (top, bot):
            y0 = max(0, y - 4)
            y1 = min(h, y + 5)
            parts.append(np.mean(sx[y0:y1], axis=0))
        col = np.mean(np.stack(parts, axis=0), axis=0)
        acc = col if acc is None else acc + col
    n = float(max(1, len(frames)))
    assert acc is not None
    return acc / n


def detect_participant_panels(
    frames: list[np.ndarray],
    face_a: BBox,
    face_b: BBox,
) -> tuple[BBox, BBox] | None:
    """Find the two clean source tiles of a graphic split-screen.

    Returns None when the frame is not a framed split (full-bleed two-shot),
    so callers can keep the existing full-frame crop.
    """
    if not frames:
        return None
    frame_h, frame_w = frames[0].shape[:2]
    row_e, col_e = _sobel_profiles(frames)
    top, top_e = _argmax_range(row_e, 0.08 * frame_h, 0.45 * frame_h)
    bot, bot_e = _argmax_range(row_e, 0.55 * frame_h, 0.92 * frame_h)
    if top is None or bot is None or bot - top < 0.22 * frame_h:
        return None
    median_row = float(np.median(row_e))
    if top_e < median_row * 1.8 or bot_e < median_row * 1.8:
        return None

    split, split_e = _argmax_range(col_e, 0.38 * frame_w, 0.62 * frame_w)
    median_col = float(np.median(col_e))
    if split is None or split_e < median_col * 2.0:
        return None
    left_face, right_face = (face_a, face_b) if face_a.cx <= face_b.cx else (face_b, face_a)
    if not (left_face.cx < split < right_face.cx):
        return None

    border_col = _border_col_energy(frames, top, bot)
    peaks = _local_maxima(border_col, min_frac=0.28, radius=8)
    left_outer = max((x for x in peaks if x < left_face.x), default=0)
    right_outer = min((x for x in peaks if x > right_face.x + right_face.w), default=frame_w - 1)
    # Inner walls are the peaks nearest the center split, not features inside the face.
    left_inner = min((x for x in peaks if left_face.cx < x <= split), default=split)
    right_inner = min((x for x in peaks if split <= x < right_face.cx), default=split)

    if left_inner - left_outer < 0.12 * frame_w or right_outer - right_inner < 0.12 * frame_w:
        return None
    panel_h = float(bot - top)
    panel_a = BBox(float(left_outer), float(top), float(left_inner - left_outer), panel_h)
    panel_b = BBox(float(right_inner), float(top), float(right_outer - right_inner), panel_h)
    if left_face.cx < right_face.cx:
        return panel_a, panel_b
    return panel_b, panel_a


_YELLOW_LO = (12, 60, 60)
_YELLOW_HI = (45, 255, 255)


def crop_edge_yellow_fraction(frame: np.ndarray, crop: BBox, edge: str, strip_frac: float = 0.05) -> float:
    h_img, w_img = frame.shape[:2]
    x0 = max(0, int(round(crop.x)))
    y0 = max(0, int(round(crop.y)))
    x1 = min(w_img, int(round(crop.x + crop.w)))
    y1 = min(h_img, int(round(crop.y + crop.h)))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    strip_h = max(2, int(round(rh * strip_frac)))
    strip_w = max(2, int(round(rw * strip_frac)))
    if edge == "bottom":
        strip = roi[-strip_h:]
    elif edge == "top":
        strip = roi[:strip_h]
    elif edge == "left":
        strip = roi[:, :strip_w]
    else:
        strip = roi[:, -strip_w:]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)
    return float(mask.mean()) / 255.0


def crop_has_outside_panel_pixels(
    crop: BBox,
    panel: BBox,
    frames: list[np.ndarray],
    *,
    max_yellow: float = 0.40,
) -> bool:
    if not bbox_inside(crop, panel, eps=1.0):
        return True
    for frame in frames:
        # Film-strip / lower-third chrome below a tile is overwhelmingly yellow.
        # Warm skin or studio light is not; keep this threshold high.
        if crop_edge_yellow_fraction(frame, crop, "bottom") > max_yellow:
            return True
    return False


def _sample_times(windows: list[tuple[float, float]], count: int = 8) -> list[float]:
    spans = [(s, e, max(0.0, e - s)) for s, e in windows]
    total = sum(d for _, _, d in spans) or 1.0
    times: list[float] = []
    for i in range(max(1, count)):
        t_rel = (i + 0.5) / max(1, count) * total
        acc = 0.0
        placed = False
        for start, _end, dur in spans:
            if acc + dur >= t_rel:
                times.append(start + (t_rel - acc))
                placed = True
                break
            acc += dur
        if not placed:
            times.append(windows[-1][1])
    return times


def read_frames_at(video: Path, times: list[float]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for panel sampling: {video}")
    frames: list[np.ndarray] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def _ffmpeg_crop(box: BBox, frame_w: int, frame_h: int) -> str:
    w = _even(min(box.w, frame_w))
    h = _even(min(box.h, frame_h))
    x = _even(max(0, box.x), minimum=0)
    y = _even(max(0, box.y), minimum=0)
    if x + w > frame_w:
        x = max(0, _even(frame_w - w, minimum=0))
    if y + h > frame_h:
        y = max(0, _even(frame_h - h, minimum=0))
    w = min(w, _even(frame_w - x) if frame_w - x >= 2 else 2)
    h = min(h, _even(frame_h - y) if frame_h - y >= 2 else 2)
    return f"crop={w}:{h}:{x}:{y}"


def fit_blur_filter(width: int, height: int, fps: int) -> str:
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=16[bgbl];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgfit];"
        f"[bgbl][fgfit]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[v]"
    )


def stacked_faces_filter(top: BBox, bottom: BBox, frame_w: int, frame_h: int, width: int, height: int, fps: int) -> str:
    half_h = height // 2
    top_crop = _ffmpeg_crop(top, frame_w, frame_h)
    bot_crop = _ffmpeg_crop(bottom, frame_w, frame_h)
    return (
        f"[0:v]split=2[a][b];"
        f"[a]{top_crop},scale={width}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={width}:{half_h}[top];"
        f"[b]{bot_crop},scale={width}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={width}:{half_h}[bot];"
        f"[top][bot]vstack=inputs=2,setsar=1,fps={fps}[v]"
    )


def single_face_filter(crop: BBox, frame_w: int, frame_h: int, width: int, height: int, fps: int) -> str:
    return (
        f"[0:v]{_ffmpeg_crop(crop, frame_w, frame_h)},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps}[v]"
    )


def yunet_model_path() -> Path:
    return Path.home() / ".cache" / "reels-factory" / YUNET_NAME


def ensure_yunet_model() -> Path | None:
    path = yunet_model_path()
    if path.exists() and path.stat().st_size > 100_000:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".onnx.part")
        print(f"[faces] downloading YuNet model -> {path}")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != YUNET_SHA256:
            tmp.unlink(missing_ok=True)
            print("[faces] YuNet checksum mismatch; using Haar fallback")
            return None
        tmp.replace(path)
        return path
    except Exception as exc:
        print(f"[faces] YuNet download failed ({exc}); using Haar fallback")
        return None


class CachedYunet:
    def __init__(self, model_path: Path, score_threshold: float = 0.55):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self._detector = None
        self._size: tuple[int, int] | None = None

    def detect(self, frame) -> list[BBox]:
        h, w = frame.shape[:2]
        size = (w, h)
        if self._detector is None or self._size != size:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                size,
                float(self.score_threshold),
                0.3,
                5000,
            )
            self._size = size
        self._detector.setInputSize(size)
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [
            BBox(
                x=float(row[0]),
                y=float(row[1]),
                w=float(row[2]),
                h=float(row[3]),
                score=float(row[-1]),
                landmarks=tuple(
                    (float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)
                )
                if len(row) >= 15
                else None,
            )
            for row in faces
        ]


def detect_faces_yunet(frame, model_path: Path, score_threshold: float = 0.55) -> list[BBox]:
    return CachedYunet(model_path, score_threshold=score_threshold).detect(frame)


def detect_faces_haar(frame, score_threshold: float = 0.55) -> list[BBox]:
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [
        BBox(x=float(x), y=float(y), w=float(w), h=float(h), score=max(score_threshold, 0.5))
        for (x, y, w, h) in detected
    ]


def filter_faces(faces: list[BBox], frame_w: int, frame_h: int, min_size: float) -> list[BBox]:
    min_area = max(min_size * min_size, 0.002 * frame_w * frame_h)
    return [f for f in faces if min(f.w, f.h) >= min_size and f.area >= min_area]


def detect_faces(frame, *, method: str | None = None, score_threshold: float = 0.55) -> tuple[list[BBox], str]:
    """Return (faces, method_used). Prefers OpenCV YuNet, then Haar."""
    requested = (method or "auto").lower()
    if requested in {"auto", "yunet", "opencv_yunet"}:
        model = ensure_yunet_model()
        if model is not None:
            try:
                return detect_faces_yunet(frame, model, score_threshold=score_threshold), "opencv_yunet"
            except Exception as exc:
                print(f"[faces] YuNet failed ({exc}); trying Haar")
    return detect_faces_haar(frame, score_threshold=score_threshold), "opencv_haar"


def track_faces(samples: list[list[BBox]], max_match_mult: float = 2.5) -> list[FaceTrack]:
    tracks: list[list[BBox | None]] = []
    for si, faces in enumerate(samples):
        used: set[int] = set()
        for boxes in tracks:
            last = next((b for b in reversed(boxes) if b is not None), None)
            match = None
            if last is not None:
                best_i = None
                best_d = None
                for i, face in enumerate(faces):
                    if i in used:
                        continue
                    dx = face.cx - last.cx
                    dy = face.cy - last.cy
                    dist = (dx * dx + dy * dy) ** 0.5
                    if best_d is None or dist < best_d:
                        best_d = dist
                        best_i = i
                limit = max(last.w, last.h) * max_match_mult
                if best_i is not None and best_d is not None and best_d <= limit:
                    match = faces[best_i]
                    used.add(best_i)
            boxes.append(match)
        for i, face in enumerate(faces):
            if i in used:
                continue
            tracks.append([None] * si + [face])
    n = len(samples)
    return [FaceTrack(boxes=(boxes + [None] * n)[:n]) for boxes in tracks]


def confident_tracks(
    tracks: list[FaceTrack],
    *,
    min_hit_ratio: float,
    min_size: float,
    frame_w: int,
    frame_h: int,
) -> list[FaceTrack]:
    min_area = max(min_size * min_size, 0.002 * frame_w * frame_h)
    kept = []
    for track in tracks:
        known = track.known()
        if not known or track.hit_ratio < min_hit_ratio:
            continue
        med = median_bbox(known)
        if min(med.w, med.h) < min_size or med.area < min_area:
            continue
        kept.append(track)
    kept.sort(key=lambda t: median_bbox(t.known()).area, reverse=True)
    return kept[:2]


def _stacked_crops(
    face_a: BBox,
    face_b: BBox,
    frame_w: int,
    frame_h: int,
    aspect: float,
    crop_cfg: dict,
    panels: tuple[BBox, BBox] | None,
) -> tuple[BBox, BBox, BBox | None, BBox | None, float]:
    face_kwargs = _face_crop_kwargs(crop_cfg)
    margin = float(crop_cfg.get("panel_safety_margin", 0.03))
    if panels is None:
        top = crop_around_face(face_a, frame_w, frame_h, aspect, **face_kwargs)
        bottom = crop_around_face(face_b, frame_w, frame_h, aspect, **face_kwargs)
        return top, bottom, None, None, 0.0
    panel_a, panel_b = panels
    safe_a = inset_bbox(panel_a, margin)
    safe_b = inset_bbox(panel_b, margin)
    top = crop_inside_roi(face_a, safe_a, aspect, **face_kwargs)
    bottom = crop_inside_roi(face_b, safe_b, aspect, **face_kwargs)
    target_h = min(top.h, bottom.h)
    if top.h > target_h * 1.08:
        top = _shrink_crop_to_height(top, face_a, target_h, aspect, safe_a)
    if bottom.h > target_h * 1.08:
        bottom = _shrink_crop_to_height(bottom, face_b, target_h, aspect, safe_b)
    return top, bottom, safe_a, safe_b, margin


def decide_layout(
    tracks: list[FaceTrack],
    frame_w: int,
    frame_h: int,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    min_hit_ratio: float = 0.4,
    min_size: float = 40.0,
    smooth_alpha: float = 0.35,
    method: str = "opencv_yunet",
    face_counts: list[int] | None = None,
    crop: dict | None = None,
    panels: tuple[BBox, BBox] | None = None,
) -> LayoutPlan:
    counts = face_counts or []
    avg = (sum(counts) / len(counts)) if counts else 0.0
    chosen = confident_tracks(
        tracks,
        min_hit_ratio=min_hit_ratio,
        min_size=min_size,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    half_aspect = width / max(1, (height // 2))
    full_aspect = width / max(1, height)
    crop_cfg = {**face_crop_params(), **(crop or {})}

    if len(chosen) >= 2:
        boxes = [stable_track_bbox(t, alpha=smooth_alpha) for t in chosen[:2]]
        boxes = [b for b in boxes if b is not None]
        if len(boxes) >= 2:
            boxes.sort(key=lambda b: b.cx)
            top, bottom, panel_a, panel_b, margin = _stacked_crops(
                boxes[0],
                boxes[1],
                frame_w,
                frame_h,
                half_aspect,
                crop_cfg,
                panels,
            )
            return LayoutPlan(
                mode="stacked_faces",
                filter_complex=stacked_faces_filter(top, bottom, frame_w, frame_h, width, height, fps),
                method=method,
                face_counts=counts,
                avg_faces=avg,
                top=top,
                bottom=bottom,
                panel_a=panel_a,
                panel_b=panel_b,
                safety_margin=margin,
            )
        if len(boxes) == 1:
            chosen = [FaceTrack(boxes=[boxes[0]])]

    if len(chosen) == 1:
        face = stable_track_bbox(chosen[0], alpha=smooth_alpha)
        if face is not None:
            single = crop_around_face(face, frame_w, frame_h, full_aspect, headroom=0.85, body=3.2, side=0.95)
            return LayoutPlan(
                mode="single_face",
                filter_complex=single_face_filter(single, frame_w, frame_h, width, height, fps),
                method=method,
                face_counts=counts,
                avg_faces=avg,
                single=single,
            )

    return LayoutPlan(
        mode="fit_blur",
        filter_complex=fit_blur_filter(width, height, fps),
        method=method,
        face_counts=counts,
        avg_faces=avg,
    )


def sample_clip_faces(
    video: Path,
    start: float,
    end: float,
    *,
    interval: float = 0.5,
    score_threshold: float = 0.55,
    min_size: float = 40.0,
) -> tuple[list[list[BBox]], str, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for face sampling: {video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0
    start_frame = max(0, int(round(start * fps)))
    end_frame = max(start_frame + 1, int(round(end * fps)))
    step = max(1, int(round(max(interval, 1.0 / fps) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples: list[list[BBox]] = []
    method_used = "opencv_haar"
    yunet = None
    model = ensure_yunet_model()
    if model is not None:
        try:
            yunet = CachedYunet(model, score_threshold=score_threshold)
            method_used = "opencv_yunet"
        except Exception as exc:
            print(f"[faces] YuNet init failed ({exc}); using Haar")
            yunet = None
    fno = start_frame
    first = True
    while fno < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if first:
            frame_h, frame_w = frame.shape[:2]
            first = False
        if yunet is not None:
            try:
                faces = yunet.detect(frame)
                method_used = "opencv_yunet"
            except Exception:
                faces, method_used = detect_faces_haar(frame, score_threshold=score_threshold), "opencv_haar"
                yunet = None
        else:
            faces, method_used = detect_faces_haar(frame, score_threshold=score_threshold), "opencv_haar"
        samples.append(filter_faces(faces, frame_w, frame_h, min_size))
        skip = step - 1
        skipped = 0
        while skipped < skip:
            if not cap.grab():
                fno = end_frame
                break
            skipped += 1
        fno += step
    cap.release()
    if not samples:
        raise RuntimeError("No frames could be sampled for face detection")
    return samples, method_used, frame_w, frame_h


def plan_locked_layout(
    video: Path,
    windows: list[tuple[float, float]],
    rcfg: dict,
) -> LayoutPlan:
    """Sample every window, then freeze one pair of crops for the whole Reel."""
    if not windows:
        raise ValueError("plan_locked_layout needs at least one time window")
    width = int(rcfg.get("width", 1080))
    height = int(rcfg.get("height", 1920))
    fps = int(rcfg.get("fps", 30))
    interval = float(rcfg.get("face_sample_interval", 0.5))
    score_threshold = float(rcfg.get("face_score_threshold", 0.55))
    min_size = float(rcfg.get("face_min_size", 40))
    min_hit_ratio = float(rcfg.get("face_track_min_hit_ratio", 0.4))
    alpha = float(rcfg.get("face_smooth_alpha", 0.35))
    crop_cfg = face_crop_params(rcfg)
    print(
        "[layout] face_crop "
        f"zoom={crop_cfg['zoom']:.2f} headroom={crop_cfg['headroom']:.2f} "
        f"horizontal_margin={crop_cfg['horizontal_margin']:.2f} body={crop_cfg['body']:.2f} "
        f"panel_safety_margin={crop_cfg['panel_safety_margin']:.2f}"
    )

    all_samples: list[list[BBox]] = []
    method = "none"
    frame_w = frame_h = 0
    try:
        for start, end in windows:
            samples, method, frame_w, frame_h = sample_clip_faces(
                video,
                start,
                end,
                interval=interval,
                score_threshold=score_threshold,
                min_size=min_size,
            )
            all_samples.extend(samples)
    except Exception as exc:
        print(f"[faces] locked detection failed ({exc}); falling back to fit_blur")
        return LayoutPlan(
            mode="fit_blur",
            filter_complex=fit_blur_filter(width, height, fps),
            method="none",
        )
    if not all_samples:
        return LayoutPlan(
            mode="fit_blur",
            filter_complex=fit_blur_filter(width, height, fps),
            method="none",
        )

    tracks = track_faces(all_samples)
    counts = [len(s) for s in all_samples]
    panels, panel_frames = _detect_locked_panels(
        video, windows, tracks, frame_w, frame_h, min_hit_ratio, min_size, alpha
    )
    return _decide_layout_with_panel_validation(
        tracks,
        frame_w,
        frame_h,
        width=width,
        height=height,
        fps=fps,
        min_hit_ratio=min_hit_ratio,
        min_size=min_size,
        smooth_alpha=alpha,
        method=method,
        face_counts=counts,
        crop_cfg=crop_cfg,
        panels=panels,
        panel_frames=panel_frames,
    )


def _detect_locked_panels(
    video: Path,
    windows: list[tuple[float, float]],
    tracks: list[FaceTrack],
    frame_w: int,
    frame_h: int,
    min_hit_ratio: float,
    min_size: float,
    alpha: float,
) -> tuple[tuple[BBox, BBox] | None, list]:
    chosen = confident_tracks(
        tracks,
        min_hit_ratio=min_hit_ratio,
        min_size=min_size,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    if len(chosen) < 2:
        return None, []
    boxes = [stable_track_bbox(t, alpha=alpha) for t in chosen[:2]]
    boxes = [b for b in boxes if b is not None]
    if len(boxes) < 2:
        return None, []
    boxes.sort(key=lambda b: b.cx)
    try:
        panel_frames = read_frames_at(video, _sample_times(windows, 8))
    except Exception as exc:
        print(f"[layout] panel detection skipped ({exc})")
        return None, []
    if len(panel_frames) < 2:
        return None, []
    panels = detect_participant_panels(panel_frames, boxes[0], boxes[1])
    if panels is None:
        print("[layout] no split-screen panels; using full-frame crop bounds")
        return None, panel_frames
    pa, pb = panels
    print(
        f"[layout] Person A panel x={pa.x:.1f} y={pa.y:.1f} w={pa.w:.1f} h={pa.h:.1f}"
    )
    print(
        f"[layout] Person B panel x={pb.x:.1f} y={pb.y:.1f} w={pb.w:.1f} h={pb.h:.1f}"
    )
    return panels, panel_frames


def _decide_layout_with_panel_validation(
    tracks: list[FaceTrack],
    frame_w: int,
    frame_h: int,
    *,
    width: int,
    height: int,
    fps: int,
    min_hit_ratio: float,
    min_size: float,
    smooth_alpha: float,
    method: str,
    face_counts: list[int],
    crop_cfg: dict,
    panels: tuple[BBox, BBox] | None,
    panel_frames: list,
) -> LayoutPlan:
    base_margin = float(crop_cfg.get("panel_safety_margin", 0.03))
    margins = [base_margin]
    if panels is not None:
        extra = [0.04, 0.05]
        for extra_margin in extra:
            if extra_margin > base_margin + 1e-6:
                margins.append(extra_margin)
        plan = None
        for margin in margins:
            attempt = {**crop_cfg, "panel_safety_margin": margin}
            plan = decide_layout(
                tracks,
                frame_w,
                frame_h,
                width=width,
                height=height,
                fps=fps,
                min_hit_ratio=min_hit_ratio,
                min_size=min_size,
                smooth_alpha=smooth_alpha,
                method=method,
                face_counts=face_counts,
                crop=attempt,
                panels=panels,
            )
            if plan.mode != "stacked_faces" or plan.top is None or plan.bottom is None:
                return plan
            dirty_a = crop_has_outside_panel_pixels(plan.top, panels[0], panel_frames)
            dirty_b = crop_has_outside_panel_pixels(plan.bottom, panels[1], panel_frames)
            if plan.panel_a is not None:
                dirty_a = dirty_a or not bbox_inside(plan.top, plan.panel_a)
            if plan.panel_b is not None:
                dirty_b = dirty_b or not bbox_inside(plan.bottom, plan.panel_b)
            if not dirty_a and not dirty_b:
                print(f"[layout] panel validation: clean (margin={margin:.0%})")
                return plan
            print(
                f"[layout] panel validation found outside-panel pixels "
                f"(A={'dirty' if dirty_a else 'ok'} B={'dirty' if dirty_b else 'ok'}) "
                f"at margin={margin:.0%}"
            )
        return plan
    return decide_layout(
        tracks,
        frame_w,
        frame_h,
        width=width,
        height=height,
        fps=fps,
        min_hit_ratio=min_hit_ratio,
        min_size=min_size,
        smooth_alpha=smooth_alpha,
        method=method,
        face_counts=face_counts,
        crop=crop_cfg,
        panels=None,
    )


def plan_clip_layout(video: Path, start: float, end: float, rcfg: dict) -> LayoutPlan:
    width = int(rcfg.get("width", 1080))
    height = int(rcfg.get("height", 1920))
    fps = int(rcfg.get("fps", 30))
    requested = str(rcfg.get("layout", "stacked_faces")).strip().lower()
    if requested == "fit_blur":
        return LayoutPlan(mode="fit_blur", filter_complex=fit_blur_filter(width, height, fps), method="none")

    interval = float(rcfg.get("face_sample_interval", 0.5))
    score_threshold = float(rcfg.get("face_score_threshold", 0.55))
    min_size = float(rcfg.get("face_min_size", 40))
    min_hit_ratio = float(rcfg.get("face_track_min_hit_ratio", 0.4))
    alpha = float(rcfg.get("face_smooth_alpha", 0.35))
    crop_cfg = face_crop_params(rcfg)
    print(
        "[layout] face_crop "
        f"zoom={crop_cfg['zoom']:.2f} headroom={crop_cfg['headroom']:.2f} "
        f"horizontal_margin={crop_cfg['horizontal_margin']:.2f} body={crop_cfg['body']:.2f} "
        f"panel_safety_margin={crop_cfg['panel_safety_margin']:.2f}"
    )

    try:
        samples, method, frame_w, frame_h = sample_clip_faces(
            video,
            start,
            end,
            interval=interval,
            score_threshold=score_threshold,
            min_size=min_size,
        )
    except Exception as exc:
        print(f"[faces] detection failed ({exc}); falling back to fit_blur")
        return LayoutPlan(
            mode="fit_blur",
            filter_complex=fit_blur_filter(width, height, fps),
            method="none",
        )

    tracks = track_faces(samples)
    counts = [len(s) for s in samples]
    panels, panel_frames = _detect_locked_panels(
        video, [(start, end)], tracks, frame_w, frame_h, min_hit_ratio, min_size, alpha
    )
    plan = _decide_layout_with_panel_validation(
        tracks,
        frame_w,
        frame_h,
        width=width,
        height=height,
        fps=fps,
        min_hit_ratio=min_hit_ratio,
        min_size=min_size,
        smooth_alpha=alpha,
        method=method,
        face_counts=counts,
        crop_cfg=crop_cfg,
        panels=panels,
        panel_frames=panel_frames,
    )
    if requested == "single_face" and plan.mode == "stacked_faces" and plan.top is not None:
        # Honor an explicit single-person request without changing detection.
        crop = plan.top
        plan = LayoutPlan(
            mode="single_face",
            filter_complex=single_face_filter(crop, frame_w, frame_h, width, height, fps),
            method=method,
            face_counts=counts,
            avg_faces=plan.avg_faces,
            single=crop,
        )
    return plan


def _box_dict(box: BBox, label: str | None = None) -> dict:
    payload = {
        "label": label,
        "x": round(box.x, 1),
        "y": round(box.y, 1),
        "w": round(box.w, 1),
        "h": round(box.h, 1),
        "cx": round(box.cx, 1),
        "cy": round(box.cy, 1),
        "score": round(box.score, 3),
    }
    if label is None:
        payload.pop("label")
    return payload


def assign_person_labels(faces: list[BBox]) -> list[tuple[str, BBox]]:
    """Label the two largest faces left=A, right=B; extras stay unlabeled."""
    ordered = sorted(faces, key=lambda b: b.area, reverse=True)
    primary = ordered[:2]
    primary.sort(key=lambda b: b.cx)
    labels = ["A", "B"][: len(primary)]
    labeled = list(zip(labels, primary))
    used = {id(b) for _, b in labeled}
    for face in faces:
        if id(face) not in used:
            labeled.append(("extra", face))
    return labeled


def _annotate_frame(frame, labeled: list[tuple[str, BBox]]):
    colors = {
        "A": (40, 220, 40),
        "B": (40, 200, 255),
        "extra": (0, 220, 255),
    }
    out = frame.copy()
    for label, box in labeled:
        x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
        color = colors.get(label, (180, 180, 180))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        caption = f"{label} {box.score:.2f}"
        cv2.putText(
            out,
            caption,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def diagnose_clip_faces(
    video: Path,
    start: float,
    end: float,
    out_dir: Path,
    *,
    frame_count: int = 8,
    score_threshold: float = 0.55,
    min_size: float = 40.0,
) -> dict:
    """Sample representative frames, save annotated boxes, and summarize detections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError("Diagnostic window must have end > start")
    n = max(2, int(frame_count))
    timestamps = [start + (i + 0.5) * (end - start) / n for i in range(n)]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for diagnostics: {video}")
    method = "opencv_haar"
    yunet = None
    model = ensure_yunet_model()
    if model is not None:
        yunet = CachedYunet(model, score_threshold=score_threshold)
        method = "opencv_yunet"

    frames_report = []
    two_person_hits = 0
    for idx, t in enumerate(timestamps, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            frames_report.append({"index": idx, "time": t, "ok": False, "faces": []})
            continue
        frame_h, frame_w = frame.shape[:2]
        if yunet is not None:
            faces = yunet.detect(frame)
            method = "opencv_yunet"
        else:
            faces = detect_faces_haar(frame, score_threshold=score_threshold)
            method = "opencv_haar"
        faces = filter_faces(faces, frame_w, frame_h, min_size)
        labeled = assign_person_labels(faces)
        if sum(1 for label, _ in labeled if label in {"A", "B"}) >= 2:
            two_person_hits += 1
        annotated = _annotate_frame(frame, labeled)
        image_name = f"frame_{idx:02d}.jpg"
        cv2.imwrite(str(out_dir / image_name), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frames_report.append({
            "index": idx,
            "time": round(t, 3),
            "time_clock": f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{t % 60:06.3f}",
            "ok": True,
            "image": image_name,
            "face_count": len(faces),
            "faces": [_box_dict(box, label) for label, box in labeled],
        })
    cap.release()

    counts = [row["face_count"] for row in frames_report if row.get("ok")]
    avg_faces = (sum(counts) / len(counts)) if counts else 0.0
    ok_frames = sum(1 for row in frames_report if row.get("ok"))
    two_person_ok = ok_frames > 0 and two_person_hits / ok_frames >= 0.75
    summary = {
        "video": str(video),
        "start": start,
        "end": end,
        "method": method,
        "frame_count": len(frames_report),
        "ok_frames": ok_frames,
        "two_person_frames": two_person_hits,
        "avg_faces": round(avg_faces, 3),
        "two_person_ok": two_person_ok,
        "frames": frames_report,
    }
    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[faces] diagnostics method={method} avg_faces={avg_faces:.2f} "
        f"two_person_frames={two_person_hits}/{ok_frames} ok={two_person_ok}"
    )
    for row in frames_report:
        print(
            f"  t={row.get('time_clock', row.get('time'))} count={row.get('face_count', 0)} "
            f"boxes={row.get('faces', [])}"
        )
    return summary
