from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .cover import (
    CoverError,
    _pick_guest_face,
    detect_guest_panel,
    load_cover_metadata,
    score_guest_portrait,
)
from .faces import (
    BBox,
    CachedYunet,
    bbox_inside,
    crop_inside_roi,
    ensure_yunet_model,
    face_crop_params,
    filter_faces,
    inset_bbox,
    read_frames_at,
)
from .utils import read_json, ts, write_json

_UPSCALE = 3
_COARSE_STEP_S = 0.5
_FINE_STEP_S = 0.10
_FINE_RADIUS_S = 1.20
_FINE_PEAKS = 12
_COVER_ASPECT = 540 / 600
_CAMERA_FRONTAL = 0.70
_CAMERA_IRIS = 0.50
_CAMERA_YAW = 0.16
_CAMERA_EYES = 0.55


def _video_probe(video: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise CoverError(f"Could not open video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = n / fps if n > 0 else 0.0
    finally:
        cap.release()
    if duration < 1.0:
        raise CoverError(f"Video is too short for a master portrait: {video}")
    return duration, fps


def _offset_face(face: BBox, ox: float, oy: float) -> BBox:
    landmarks = None
    if face.landmarks:
        landmarks = tuple((x + ox, y + oy) for x, y in face.landmarks)
    return BBox(face.x + ox, face.y + oy, face.w, face.h, face.score, landmarks)


def _bbox_payload(box: BBox) -> dict:
    return {
        "x": round(float(box.x), 1),
        "y": round(float(box.y), 1),
        "w": round(float(box.w), 1),
        "h": round(float(box.h), 1),
    }


def _is_camera_facing(details: dict) -> bool:
    return (
        bool(details.get("accepted"))
        and float(details.get("gaze_frontal") or 0.0) >= _CAMERA_FRONTAL
        and float(details.get("iris_gaze") or 0.0) >= _CAMERA_IRIS
        and abs(float(details.get("yaw") or 1.0)) <= _CAMERA_YAW
        and float(details.get("eyes_open") or 0.0) >= _CAMERA_EYES
    )


def _master_rank(details: dict) -> float:
    return (
        float(details.get("score") or 0.0)
        + 0.10 * float(details.get("mouth_relaxed") or 0.0)
        + 0.08 * float(details.get("iris_gaze") or 0.0)
        + 0.05 * float(details.get("eyes_open") or 0.0)
    )


def master_portrait_path(
    source_stem: str,
    *,
    root: Path,
    meta: dict | None = None,
    cfg: dict | None = None,
) -> Path:
    if meta and str(meta.get("guest_portrait") or "").strip():
        path = Path(str(meta["guest_portrait"]).replace("\\", "/"))
        return path if path.is_absolute() else (root / path)
    portraits = root / "data" / "portraits"
    if cfg and (cfg.get("paths") or {}).get("portraits"):
        portraits = Path(cfg["paths"]["portraits"])
    return portraits / f"{source_stem}_guest_master.png"


def resolve_master_portrait(
    source_stem: str,
    *,
    root: Path,
    meta: dict | None = None,
    cfg: dict | None = None,
) -> Path | None:
    path = master_portrait_path(source_stem, root=root, meta=meta, cfg=cfg)
    return path if path.is_file() else None


def load_master_guest_portrait(path: Path) -> dict:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise CoverError(f"Could not read master portrait: {path}")
    sidecar = path.with_suffix(".json")
    meta = read_json(sidecar) if sidecar.is_file() else {}
    crop_raw = meta.get("original_crop_coordinates") or meta.get("crop_coordinates") or {}
    panel_raw = meta.get("panel") or {}
    crop = BBox(
        float(crop_raw.get("x") or 0.0),
        float(crop_raw.get("y") or 0.0),
        float(crop_raw.get("w") or image.shape[1]),
        float(crop_raw.get("h") or image.shape[0]),
    )
    panel = BBox(
        float(panel_raw.get("x") or 0.0),
        float(panel_raw.get("y") or 0.0),
        float(panel_raw.get("w") or 0.0),
        float(panel_raw.get("h") or 0.0),
    )
    return {
        "path": path,
        "image": image,
        "sidecar": meta,
        "timestamp": float(meta.get("source_frame_seconds") or 0.0),
        "crop": crop,
        "panel": panel,
        "score": float(meta.get("portrait_score") or 0.0),
        "naturally_camera_facing": bool(meta.get("naturally_camera_facing")),
    }


def _iter_coarse_frames(video: Path, step_s: float, duration: float, fps: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise CoverError(f"Could not open video: {video}")
    interval = max(1, int(round(fps * step_s)))
    index = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % interval == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield index / fps, frame
            index += 1
            if index / fps > duration + 0.5:
                break
    finally:
        cap.release()


def _panel_xywh(panel: BBox, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x = max(0, int(round(panel.x)))
    y = max(0, int(round(panel.y)))
    pw = max(2, int(round(panel.w)))
    ph = max(2, int(round(panel.h)))
    pw = min(pw, w - x)
    ph = min(ph, h - y)
    return x, y, pw, ph


def _detect_guest_in_panel(
    yunet: CachedYunet,
    frame: np.ndarray,
    panel: BBox,
    guest_panel: str,
) -> BBox | None:
    x, y, pw, ph = _panel_xywh(panel, frame.shape)
    roi = frame[y : y + ph, x : x + pw]
    if roi.size == 0:
        return None
    faces = filter_faces(yunet.detect(roi), pw, ph, 40)
    local = _pick_guest_face(faces, guest_panel)
    if local is None:
        return None
    return _offset_face(local, float(x), float(y))


def _evaluate_frame(
    yunet: CachedYunet,
    frame: np.ndarray,
    t: float,
    panel: BBox,
    safe: BBox,
    guest_panel: str,
) -> dict | None:
    face = _detect_guest_in_panel(yunet, frame, panel, guest_panel)
    if face is None:
        return None
    details = score_guest_portrait(frame, face, safe, panel=panel)
    details["timestamp"] = round(float(t), 3)
    return {"timestamp": round(float(t), 3), "face": face, "details": details}


def _rank_key(item: dict) -> tuple[int, float]:
    details = item["details"]
    return (1 if _is_camera_facing(details) else 0, _master_rank(details))


def select_master_guest_still(
    video: Path,
    *,
    guest_panel: str,
    crop_cfg: dict,
    aspect: float = _COVER_ASPECT,
    coarse_step: float = _COARSE_STEP_S,
) -> dict:
    duration, fps = _video_probe(video)
    windows = [(2.0, max(3.0, duration - 2.0))]
    panel = detect_guest_panel(video, windows, guest_panel)
    safe = inset_bbox(panel, float(crop_cfg.get("panel_safety_margin", 0.04)))
    model = ensure_yunet_model()
    if model is None:
        raise CoverError("YuNet model is required for master portrait selection")
    yunet = CachedYunet(model, score_threshold=0.5)

    evaluated = 0
    rejected = 0
    reject_counts: dict[str, int] = {}
    accepted: list[dict] = []

    print(f"[portrait] coarse scan {video.name} duration={duration:.1f}s step={coarse_step:.2f}s", flush=True)
    last_log = -30.0
    for t, frame in _iter_coarse_frames(video, coarse_step, duration, fps):
        item = _evaluate_frame(yunet, frame, t, panel, safe, guest_panel)
        if item is None:
            continue
        evaluated += 1
        if t - last_log >= 30.0:
            print(f"[portrait]   t={ts(t)} evaluated={evaluated} accepted={len(accepted)}", flush=True)
            last_log = t
        if not item["details"]["accepted"]:
            rejected += 1
            reason = str(item["details"].get("reject_reason") or "rejected")
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        accepted.append(item)

    if not accepted:
        raise CoverError("Could not find a usable guest portrait in the full interview")

    peaks = sorted(accepted, key=_rank_key, reverse=True)[:_FINE_PEAKS]
    fine_times: list[float] = []
    for peak in peaks:
        center = float(peak["timestamp"])
        t = max(0.0, center - _FINE_RADIUS_S)
        end = min(duration, center + _FINE_RADIUS_S)
        while t <= end + 1e-6:
            fine_times.append(round(t, 3))
            t += _FINE_STEP_S
    unique_fine: list[float] = []
    for t in sorted(fine_times):
        if not unique_fine or abs(t - unique_fine[-1]) > 1e-3:
            unique_fine.append(t)

    print(f"[portrait] fine scan {len(unique_fine)} frames around {len(peaks)} peaks", flush=True)
    seen = {round(float(item["timestamp"]), 3) for item in accepted}
    for t, frame in zip(unique_fine, read_frames_at(video, unique_fine)):
        if round(float(t), 3) in seen:
            continue
        item = _evaluate_frame(yunet, frame, t, panel, safe, guest_panel)
        if item is None:
            continue
        evaluated += 1
        seen.add(round(float(t), 3))
        if not item["details"]["accepted"]:
            rejected += 1
            reason = str(item["details"].get("reject_reason") or "rejected")
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        accepted.append(item)

    facing = [item for item in accepted if _is_camera_facing(item["details"])]
    pool = facing if facing else accepted
    best = max(pool, key=_rank_key)
    frames = read_frames_at(video, [float(best["timestamp"])])
    if not frames:
        raise CoverError("Could not re-read the selected master portrait frame")
    frame = frames[0]
    face = _detect_guest_in_panel(yunet, frame, panel, guest_panel) or best["face"]
    crop = crop_inside_roi(face, safe, aspect, **crop_cfg)
    x, y, w, h = int(round(crop.x)), int(round(crop.y)), int(round(crop.w)), int(round(crop.h))
    x = max(0, x)
    y = max(0, y)
    patch = frame[y : y + h, x : x + w]
    if patch.size == 0:
        raise CoverError("Master guest crop is empty")

    ranked = sorted(
        (
            {
                "timestamp": item["timestamp"],
                "score": item["details"]["score"],
                "master_rank": round(_master_rank(item["details"]), 4),
                "camera_facing": _is_camera_facing(item["details"]),
                "eyes_open": item["details"]["eyes_open"],
                "gaze_frontal": item["details"]["gaze_frontal"],
                "iris_gaze": item["details"].get("iris_gaze", 0.0),
                "mouth_relaxed": item["details"]["mouth_relaxed"],
                "sharpness": item["details"]["sharpness"],
            }
            for item in accepted
        ),
        key=lambda r: (int(r["camera_facing"]), r["master_rank"]),
        reverse=True,
    )
    d = best["details"]
    reasons = []
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
        "frame": frame,
        "face": face,
        "panel": panel,
        "evaluated": evaluated,
        "accepted": len(accepted),
        "rejected": rejected,
        "reject_counts": reject_counts,
        "score": d["score"],
        "details": d,
        "reasons": reasons,
        "top_candidates": ranked[:12],
        "naturally_camera_facing": bool(facing),
        "duration": duration,
    }


def _grabcut_cutout(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    border = max(6, min(h, w) // 14)
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.46)),
        (max(8, int(w * 0.36)), max(8, int(h * 0.42))),
        0,
        0,
        360,
        int(cv2.GC_FGD),
        -1,
    )
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.medianBlur(fg, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    alpha = np.clip(dist / 2.5, 0.0, 1.0)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (alpha * 255.0).astype(np.uint8)
    return bgra


def _rembg_cutout(bgr: np.ndarray) -> np.ndarray | None:
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    from PIL import Image

    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    session = None
    for model in ("u2net_human_seg", "u2net"):
        try:
            session = new_session(model)
            break
        except Exception:
            continue
    if session is None:
        return None
    cut = remove(rgb, session=session)
    arr = np.array(cut)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def enhance_guest_cutout(bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Local, non-generative cleanup. Does not restore or reshape the face."""
    steps: list[str] = []
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise CoverError("Portrait crop must be a BGR video frame")
    den = cv2.bilateralFilter(bgr, d=5, sigmaColor=18, sigmaSpace=18)
    steps.append("bilateral denoise (conservative)")

    cut = _rembg_cutout(den)
    if cut is not None and int((cut[:, :, 3] > 12).sum()) > 64:
        steps.append("background cutout (rembg, local ONNX segmentation)")
    else:
        cut = _grabcut_cutout(den)
        steps.append("background cutout (GrabCut, local)")

    h, w = cut.shape[:2]
    up = cv2.resize(cut, (w * _UPSCALE, h * _UPSCALE), interpolation=cv2.INTER_LANCZOS4)
    steps.append(f"Lanczos upscale {_UPSCALE}x")
    color = up[:, :, :3]
    blur = cv2.GaussianBlur(color, (0, 0), 0.7)
    sharp = cv2.addWeighted(color, 1.18, blur, -0.18, 0)
    up[:, :, :3] = sharp
    steps.append("unsharp mask (conservative)")
    return up, steps


def crop_interview_still_inside_roi(face: BBox, roi: BBox) -> BBox:
    """Full Safe-ROI rectangular still. No face-tight zoom, no background cutout."""
    if roi.w < 2 or roi.h < 2:
        raise CoverError("Guest Safe ROI is empty")
    if not bbox_inside(face, roi, eps=2.0):
        raise CoverError("Guest face is not fully inside the Safe ROI")
    return BBox(roi.x, roi.y, roi.w, roi.h, score=face.score)


def enhance_interview_still(bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Keep the real frame. Denoise / Lanczos / light sharpen only."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise CoverError("Portrait crop must be a BGR video frame")
    steps = ["original background preserved (no cutout, no rembg, no segmentation)"]
    den = cv2.bilateralFilter(bgr, d=5, sigmaColor=18, sigmaSpace=18)
    steps.append("bilateral denoise (conservative)")
    h, w = den.shape[:2]
    up = cv2.resize(den, (w * _UPSCALE, h * _UPSCALE), interpolation=cv2.INTER_LANCZOS4)
    steps.append(f"Lanczos upscale {_UPSCALE}x")
    blur = cv2.GaussianBlur(up, (0, 0), 0.7)
    sharp = cv2.addWeighted(up, 1.18, blur, -0.18, 0)
    steps.append("unsharp mask (conservative)")
    steps.append("identity preserved: real extracted frame, no generative face restoration")
    return sharp, steps


def extract_guest_still_at(
    video: Path,
    timestamp: float,
    *,
    guest_panel: str,
    crop_cfg: dict,
) -> dict:
    duration, _fps = _video_probe(video)
    if timestamp < 0 or timestamp > duration:
        raise CoverError(f"Timestamp {timestamp} is outside the source video")
    windows = [(2.0, max(3.0, duration - 2.0))]
    panel = detect_guest_panel(video, windows, guest_panel)
    safe = inset_bbox(panel, float(crop_cfg.get("panel_safety_margin", 0.04)))
    frames = read_frames_at(video, [float(timestamp)])
    if not frames:
        raise CoverError(f"Could not read source frame at {ts(timestamp)}")
    frame = frames[0]
    model = ensure_yunet_model()
    if model is None:
        raise CoverError("YuNet model is required for master portrait extraction")
    yunet = CachedYunet(model, score_threshold=0.5)
    face = _detect_guest_in_panel(yunet, frame, panel, guest_panel)
    if face is None:
        raise CoverError("Could not detect the guest in the locked source frame")
    details = score_guest_portrait(frame, face, safe, panel=panel)
    details["timestamp"] = round(float(timestamp), 3)
    crop = crop_interview_still_inside_roi(face, safe)
    x, y, w, h = int(round(crop.x)), int(round(crop.y)), int(round(crop.w)), int(round(crop.h))
    x = max(0, x)
    y = max(0, y)
    patch = frame[y : y + h, x : x + w]
    if patch.size == 0:
        raise CoverError("Interview still crop is empty")
    if not bbox_inside(crop, safe, eps=1.0):
        raise CoverError("Interview still crop escaped the Safe ROI")
    return {
        "timestamp": round(float(timestamp), 3),
        "crop": crop,
        "patch": patch,
        "frame": frame,
        "face": face,
        "panel": panel,
        "safe": safe,
        "score": float(details.get("score") or 0.0),
        "details": details,
        "naturally_camera_facing": _is_camera_facing(details),
    }


def build_rectangular_master_portrait(
    video: Path,
    metadata_path: Path,
    cfg: dict,
    *,
    root: Path,
    timestamp: float,
    output_png: Path,
    output_json: Path | None = None,
) -> dict:
    """Locked-timestamp rectangular interview still. Does not overwrite other portraits."""
    video = Path(video)
    meta = load_cover_metadata(metadata_path, root=root)
    crop_cfg = face_crop_params(cfg.get("render") or {})
    still = extract_guest_still_at(
        video,
        timestamp,
        guest_panel=meta["guest_panel"],
        crop_cfg=crop_cfg,
    )
    image, steps = enhance_interview_still(still["patch"])
    steps = [
        "panel-aware Safe ROI rectangular crop (bottom guest)",
        "complete head, hair, shoulders and upper torso inside guest panel",
        *steps,
    ]
    png_path = Path(output_png)
    json_path = Path(output_json) if output_json else png_path.with_suffix(".json")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(png_path), image)
    if not ok:
        raise CoverError(f"Failed to write {png_path}")
    payload = {
        "source_video": str(video.as_posix()),
        "source_timestamp": ts(still["timestamp"]),
        "source_frame_seconds": round(float(still["timestamp"]), 3),
        "original_crop_coordinates": _bbox_payload(still["crop"]),
        "panel": _bbox_payload(still["panel"]),
        "safe_roi": _bbox_payload(still["safe"]),
        "portrait_score": still["score"],
        "guest_name": meta["guest_name"],
        "guest_role": meta["guest_role"],
        "guest_panel": meta["guest_panel"],
        "naturally_camera_facing": still["naturally_camera_facing"],
        "enhancement_steps": steps,
        "output": str(png_path.as_posix()),
        "portrait_size": {"w": int(image.shape[1]), "h": int(image.shape[0])},
        "portrait_selection": {
            "mode": "locked_timestamp_rectangular_still",
            "score": still["score"],
            "details": still["details"],
        },
    }
    write_json(json_path, payload)
    print(
        f"[portrait] wrote {png_path} t={payload['source_timestamp']} "
        f"crop={payload['original_crop_coordinates']} size={payload['portrait_size']}",
        flush=True,
    )
    return payload


def build_master_guest_portrait(
    video: Path,
    metadata_path: Path,
    cfg: dict,
    *,
    root: Path,
    output_png: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    video = Path(video)
    meta = load_cover_metadata(metadata_path, root=root)
    crop_cfg = face_crop_params(cfg.get("render") or {})
    still = select_master_guest_still(
        video,
        guest_panel=meta["guest_panel"],
        crop_cfg=crop_cfg,
        aspect=_COVER_ASPECT,
    )
    rgba, steps = enhance_guest_cutout(still["patch"])
    steps = ["panel-aware Safe ROI crop (bottom guest)", *steps]
    steps.append("identity preserved: real extracted frame, no generative face restoration")

    stem = video.stem
    png_path = output_png or master_portrait_path(stem, root=root, meta=meta, cfg=cfg)
    png_path = Path(png_path)
    json_path = Path(output_json) if output_json else png_path.with_suffix(".json")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(png_path), rgba)
    if not ok:
        raise CoverError(f"Failed to write {png_path}")

    payload = {
        "source_video": str(video.as_posix()),
        "source_timestamp": ts(still["timestamp"]),
        "source_frame_seconds": round(float(still["timestamp"]), 3),
        "original_crop_coordinates": _bbox_payload(still["crop"]),
        "panel": _bbox_payload(still["panel"]),
        "portrait_score": still["score"],
        "guest_name": meta["guest_name"],
        "guest_role": meta["guest_role"],
        "guest_panel": meta["guest_panel"],
        "naturally_camera_facing": still["naturally_camera_facing"],
        "enhancement_steps": steps,
        "output": str(png_path.as_posix()),
        "portrait_selection": {
            "frames_evaluated": still["evaluated"],
            "frames_accepted": still["accepted"],
            "frames_rejected": still["rejected"],
            "reject_counts": still["reject_counts"],
            "score": still["score"],
            "reasons": still["reasons"],
            "details": still["details"],
            "top_candidates": still["top_candidates"],
        },
    }
    write_json(json_path, payload)
    print(
        f"[portrait] wrote {png_path} t={payload['source_timestamp']} "
        f"score={payload['portrait_score']} camera_facing={payload['naturally_camera_facing']}",
        flush=True,
    )
    return payload
