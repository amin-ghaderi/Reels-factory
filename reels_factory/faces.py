from __future__ import annotations

import hashlib
import json
import statistics
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2

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
            top = crop_around_face(boxes[0], frame_w, frame_h, half_aspect, **crop_cfg)
            bottom = crop_around_face(boxes[1], frame_w, frame_h, half_aspect, **crop_cfg)
            return LayoutPlan(
                mode="stacked_faces",
                filter_complex=stacked_faces_filter(top, bottom, frame_w, frame_h, width, height, fps),
                method=method,
                face_counts=counts,
                avg_faces=avg,
                top=top,
                bottom=bottom,
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
        f"horizontal_margin={crop_cfg['horizontal_margin']:.2f} body={crop_cfg['body']:.2f}"
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
    plan = decide_layout(
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
        crop=crop_cfg,
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
