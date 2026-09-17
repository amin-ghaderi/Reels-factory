from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from .utils import (
    parse_timestamp,
    read_json,
    require_binary,
    run,
    srt_ts,
    to_source_time,
    ts,
    write_json,
)

DURATION_SLACK = 0.25


def display_clock(seconds: float) -> str:
    stamp = ts(seconds)
    if seconds < 3600:
        return stamp[3:8]
    return stamp[:8]


def refined_output_paths(out_dir: Path, video_stem: str, candidate_id: str) -> dict[str, Path]:
    safe_id = str(candidate_id).strip()
    if not safe_id:
        raise ValueError("candidate id is empty")
    base = f"{video_stem}.{safe_id}.refined"
    return {
        "json": out_dir / f"{base}.json",
        "md": out_dir / f"{base}.md",
        "srt": out_dir / f"{base}.srt",
    }


def refined_work_packet_path(out_dir: Path, video_stem: str) -> Path:
    return out_dir / f"{video_stem}.refined_work_packet.md"


def validate_window(start: float, end: float, duration: float | None = None) -> None:
    if start < 0:
        raise ValueError(f"start must be >= 0 (got {ts(start)})")
    if end <= start:
        raise ValueError(f"start must be before end (got {ts(start)} → {ts(end)})")
    if duration is not None and end > duration + DURATION_SLACK:
        raise ValueError(
            f"Requested end {ts(end)} is past source duration {ts(duration)}"
        )


def resolve_video_path(video: Path, root: Path) -> Path:
    path = Path(video)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source video not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Source video is not a file: {path}")
    return path


def probe_duration(video: Path) -> float:
    ffprobe = require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed for {video}: {err or proc.returncode}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not read duration for {video}: {proc.stdout!r}") from exc


def extract_window(source: Path, start: float, end: float, dest: Path) -> None:
    ffmpeg = require_binary("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Output-side -ss/-to decode accurately so Whisper local times match the request.
    run([
        ffmpeg, "-y",
        "-i", str(source),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dest),
    ])


def rebase_segment(seg, source_start: float) -> dict:
    local_start = float(getattr(seg, "start", 0.0) or 0.0)
    local_end = float(getattr(seg, "end", 0.0) or 0.0)
    words = []
    for w in getattr(seg, "words", None) or []:
        ls = float(getattr(w, "start", 0.0) or 0.0)
        le = float(getattr(w, "end", 0.0) or 0.0)
        words.append({
            "word": getattr(w, "word", ""),
            "local_start": ls,
            "local_end": le,
            "source_start": to_source_time(ls, source_start),
            "source_end": to_source_time(le, source_start),
            "probability": float(getattr(w, "probability", 0.0) or 0.0),
        })
    return {
        "local_start": local_start,
        "local_end": local_end,
        "source_start": to_source_time(local_start, source_start),
        "source_end": to_source_time(local_end, source_start),
        "text": str(getattr(seg, "text", "")).strip(),
        "words": words,
    }


def parse_refine_request(payload: dict, root: Path) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Refine request must be a JSON object")
    if "video" not in payload:
        raise ValueError("Refine request is missing 'video'")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Refine request needs a non-empty 'windows' list")

    parsed = []
    seen = set()
    for i, raw in enumerate(windows):
        if not isinstance(raw, dict):
            raise ValueError(f"windows[{i}] must be an object")
        cid = str(raw.get("id") or "").strip()
        if not cid:
            raise ValueError(f"windows[{i}] is missing 'id'")
        if cid in seen:
            raise ValueError(f"Duplicate window id: {cid}")
        if "start" not in raw or "end" not in raw:
            raise ValueError(f"windows[{i}] ({cid}) needs 'start' and 'end'")
        start = parse_timestamp(raw["start"])
        end = parse_timestamp(raw["end"])
        validate_window(start, end)
        seen.add(cid)
        parsed.append({
            "id": cid,
            "start": start,
            "end": end,
            "start_label": str(raw["start"]),
            "end_label": str(raw["end"]),
        })
    return {
        "video": resolve_video_path(Path(payload["video"]), root),
        "windows": parsed,
    }


def local_whisper_dir(model_name: str) -> Path | None:
    path = Path.home() / ".cache" / "reels-factory" / "whisper" / model_name
    model_bin = path / "model.bin"
    if not (path / "config.json").exists() or not model_bin.exists():
        return None
    # Ignore in-progress curl downloads (model.bin is ~1.5GB).
    if model_bin.stat().st_size < 500_000_000:
        return None
    return path


def load_refine_model(cfg: dict, *, model_name: str | None = None):
    # Hugging Face xet downloads often stall on Windows; plain HTTP is reliable here.
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    rcfg = cfg.get("refine") or {}
    preferred = model_name or rcfg.get("model") or "large-v3-turbo"
    fallback = rcfg.get("fallback_model") or "medium"
    device = rcfg.get("device") or "cpu"
    compute_type = rcfg.get("compute_type") or "int8"
    try:
        from faster_whisper import WhisperModel, available_models
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    supported = set(available_models())
    candidates = []
    for name in (preferred, fallback):
        if name and name not in candidates:
            candidates.append(name)

    last_err = None
    for name in candidates:
        local = local_whisper_dir(name)
        if local is None and name not in supported:
            print(f"[refine] model '{name}' is not in this faster-whisper build, skipping")
            continue
        source = str(local) if local else name
        print(f"Loading {name} on {device}/{compute_type}...")
        if local:
            print(f"Using local model files: {local}")
        try:
            model = WhisperModel(source, device=device, compute_type=compute_type)
            print(f"Using model {name}")
            return model, name
        except Exception as exc:
            print(f"[refine] failed to load {name}: {exc}")
            last_err = exc
    raise RuntimeError(
        "Could not load a second-pass Whisper model. "
        f"Tried: {', '.join(candidates)}"
    ) from last_err


def _write_refined_markdown(payload: dict, out_path: Path) -> None:
    lines = [
        f"# Refined transcript — {Path(payload['source_video']).name} / {payload['candidate_id']}",
        "",
        f"- Source window: {ts(payload['source_start'])} → {ts(payload['source_end'])}",
        f"- Language: {payload.get('language')}",
        f"- Model: {payload.get('model')}",
        "",
        "## Full transcript",
        "",
        payload.get("text") or "(empty)",
        "",
        "## Segments (source timestamps)",
        "",
    ]
    for seg in payload.get("segments") or []:
        lines.append(f"**{ts(seg['source_start'])} → {ts(seg['source_end'])}**")
        lines.append("")
        lines.append(seg.get("text") or "")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_refined_srt(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(payload.get("segments") or [], 1):
            f.write(
                f"{idx}\n{srt_ts(seg['source_start'])} --> {srt_ts(seg['source_end'])}\n"
                f"{seg.get('text') or ''}\n\n"
            )


def write_refined_work_packet(source_name: str, payloads: list[dict], out_path: Path) -> Path:
    lines = [
        f"# Reels Factory — Refined Work Packet: {source_name}",
        "",
        "Second-pass local transcripts for shortlisted windows only.",
        "Use these source timestamps for final edit plans. Do not re-read the full first-pass transcript.",
        "",
    ]
    for payload in payloads:
        lines += [
            f"## {payload['candidate_id']} — {ts(payload['source_start'])} → {ts(payload['source_end'])}",
            "",
            "### Transcript",
            "",
            payload.get("text") or "(empty)",
            "",
            "### Segments",
            "",
        ]
        for seg in payload.get("segments") or []:
            lines.append(f"- **{ts(seg['source_start'])} → {ts(seg['source_end'])}**")
            lines.append(f"  {seg.get('text') or ''}")
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def transcribe_window(clip_path: Path, cfg: dict, model, model_name: str) -> tuple[list[dict], dict]:
    rcfg = cfg.get("refine") or {}
    language = rcfg.get("language") or "fa"
    segments_iter, info = model.transcribe(
        str(clip_path),
        language=language,
        vad_filter=bool(rcfg.get("vad_filter", True)),
        beam_size=int(rcfg.get("beam_size", 5)),
        word_timestamps=True,
    )
    raw_segments = list(segments_iter)
    meta = {
        "language": getattr(info, "language", language) or language,
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "model": model_name,
    }
    return raw_segments, meta


def refine_window(
    video: Path,
    cfg: dict,
    *,
    candidate_id: str,
    start: float,
    end: float,
    model,
    model_name: str,
    force: bool = False,
    duration: float | None = None,
    index: int | None = None,
    total: int | None = None,
) -> dict:
    validate_window(start, end, duration)
    out_dir = Path(cfg["paths"]["refined_transcripts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = refined_output_paths(out_dir, video.stem, candidate_id)
    prefix = f"[{index}/{total}] " if index and total else ""
    print(f"{prefix}{candidate_id}")
    print(f"Extracting {display_clock(start)} → {display_clock(end)}")

    if paths["json"].exists() and not force:
        print(f"Skipping existing refined transcript: {paths['json']}")
        return read_json(paths["json"])

    tmp_path = None
    started = time.perf_counter()
    try:
        tmp = tempfile.NamedTemporaryFile(prefix=f"refine_{video.stem}_{candidate_id}_", suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        extract_window(video, start, end, tmp_path)
        print(f"Transcribing with {model_name}...")
        raw_segments, meta = transcribe_window(tmp_path, cfg, model, model_name)
        segments = [rebase_segment(seg, start) for seg in raw_segments]
        elapsed = time.perf_counter() - started
        payload = {
            "source_video": str(video),
            "candidate_id": candidate_id,
            "source_start": start,
            "source_end": end,
            "source_start_ts": ts(start),
            "source_end_ts": ts(end),
            "language": meta["language"],
            "language_probability": meta["language_probability"],
            "model": model_name,
            "elapsed_seconds": round(elapsed, 2),
            "text": " ".join(seg["text"] for seg in segments if seg["text"]).strip(),
            "segments": segments,
        }
        write_json(paths["json"], payload)
        _write_refined_markdown(payload, paths["md"])
        _write_refined_srt(payload, paths["srt"])
        print(f"Saved refined transcript ({elapsed:.1f}s).")
        print(f"  {paths['json']}")
        print(f"  {paths['md']}")
        print(f"  {paths['srt']}")
        return payload
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                print(f"[warn] could not delete temp clip: {tmp_path}")


def refine_windows(
    video: Path,
    windows: list[dict],
    cfg: dict,
    *,
    force: bool = False,
    model_name: str | None = None,
    write_packet: bool = True,
) -> list[dict]:
    duration = probe_duration(video)
    model, used_model = load_refine_model(cfg, model_name=model_name)
    payloads = []
    errors = []
    total = len(windows)
    for i, window in enumerate(windows, 1):
        try:
            payload = refine_window(
                video,
                cfg,
                candidate_id=window["id"],
                start=window["start"],
                end=window["end"],
                model=model,
                model_name=used_model,
                force=force,
                duration=duration,
                index=i,
                total=total,
            )
            payloads.append(payload)
        except Exception as exc:
            print(f"[error] {window['id']}: {exc}")
            errors.append((window["id"], exc))
    if write_packet and payloads:
        packet = refined_work_packet_path(Path(cfg["paths"]["refined_transcripts"]), video.stem)
        write_refined_work_packet(video.name, payloads, packet)
        print(f"Work packet: {packet}")
    if errors:
        detail = "; ".join(f"{cid}: {err}" for cid, err in errors)
        raise RuntimeError(f"{len(errors)} window(s) failed: {detail}")
    return payloads