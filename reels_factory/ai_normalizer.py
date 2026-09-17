from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .cursor_ai import CursorAIError, extract_json_payload, invoke_cursor_agent, resolve_model_id
from .refine import resolve_video_path
from .utils import read_json, ts, write_json


def compact_segments_for_normalizer(transcript: dict) -> list[dict]:
    rows = []
    for idx, seg in enumerate(transcript.get("segments") or []):
        rows.append({
            "segment_id": int(seg.get("id", idx)),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "raw_text": str(seg.get("text") or "").strip(),
        })
    return rows


def merge_corrections(segments: list[dict], corrections) -> list[dict]:
    """Apply a compact patch list. Timestamps always come from the original segments."""
    patches: dict[int, str] = {}
    for item in corrections or []:
        if not isinstance(item, dict):
            continue
        if "segment_id" not in item or "clean_text" not in item:
            continue
        patches[int(item["segment_id"])] = str(item["clean_text"])
    merged = []
    for seg in segments:
        sid = int(seg["segment_id"])
        raw = str(seg.get("raw_text") or "")
        merged.append({
            "segment_id": sid,
            "start": seg["start"],
            "end": seg["end"],
            "raw_text": raw,
            "clean_text": patches.get(sid, raw),
        })
    return merged


def timestamps_unchanged(original: list[dict], normalized: list[dict]) -> bool:
    if len(original) != len(normalized):
        return False
    for a, b in zip(original, normalized):
        if float(a["start"]) != float(b["start"]) or float(a["end"]) != float(b["end"]):
            return False
        if int(a["segment_id"]) != int(b["segment_id"]):
            return False
    return True


def normalized_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "json": out_dir / f"{stem}.normalized.json",
        "md": out_dir / f"{stem}.normalized.md",
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        f"# Normalized transcript — {payload.get('source_video', '')}",
        "",
        f"- language: {payload.get('language')}",
        f"- duration: {ts(payload.get('duration') or 0)}",
        f"- corrections: {payload.get('correction_count', 0)}",
        f"- model: {payload.get('normalization_model')}",
        "",
    ]
    for seg in payload.get("segments") or []:
        changed = " ✱" if seg.get("clean_text") != seg.get("raw_text") else ""
        lines.append(f"## {seg['segment_id']}  {ts(seg['start'])} → {ts(seg['end'])}{changed}")
        if changed:
            lines.append(f"- raw: {seg.get('raw_text')}")
        lines.append(seg.get("clean_text") or "")
        lines.append("")
    return "\n".join(lines)


def load_normalizer_prompt(root: Path) -> str:
    path = root / "prompts" / "transcript_normalizer.md"
    return path.read_text(encoding="utf-8")


def normalize_transcript(
    video: Path,
    cfg: dict,
    *,
    root: Path,
    force: bool = False,
    invoke=None,
) -> dict:
    video = resolve_video_path(video, root)
    stem = video.stem
    transcripts_dir = Path(cfg["paths"]["transcripts"])
    source_json = transcripts_dir / f"{stem}.transcript.json"
    if not source_json.exists():
        raise FileNotFoundError(f"Raw transcript not found: {source_json}")
    out_dir = Path(cfg["paths"]["normalized_transcripts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = normalized_paths(out_dir, stem)
    if paths["json"].exists() and not force:
        print(f"[normalize-ai] cache hit {paths['json']}")
        return read_json(paths["json"])

    raw = read_json(source_json)
    compact = compact_segments_for_normalizer(raw)
    aicfg = (cfg.get("ai_editor") or {}).get("normalization") or {}
    model = resolve_model_id(str(aicfg.get("model") or "composer-2.5"), kind="normalization")
    mode = str(aicfg.get("mode") or "ask")
    if mode.lower() == "standard":
        mode = "ask"

    user_payload = {
        "source_video": str(raw.get("source_video") or video.name),
        "language": raw.get("language"),
        "duration": raw.get("duration"),
        "segments": compact,
    }
    prompt = (
        load_normalizer_prompt(root)
        + "\n\n## Transcript\n\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )
    caller = invoke or invoke_cursor_agent
    print(f"[normalize-ai] calling {model} for {len(compact)} segments")
    try:
        with tempfile.TemporaryDirectory(prefix="reels_ai_norm_") as td:
            output = caller(prompt, model=model, mode=mode, workspace=Path(td), timeout=600)
        payload = extract_json_payload(output)
    except CursorAIError:
        raise
    except Exception as exc:
        raise CursorAIError(f"Normalization model call failed: {exc}") from exc

    corrections = payload.get("corrections") if isinstance(payload, dict) else None
    if corrections is None and isinstance(payload, list):
        corrections = payload
    merged = merge_corrections(compact, corrections or [])
    if not timestamps_unchanged(compact, merged):
        raise CursorAIError("Normalization attempted to change timestamps; refusing to save")

    result = {
        "source_video": str(raw.get("source_video") or video),
        "source_transcript": str(source_json),
        "language": raw.get("language"),
        "duration": raw.get("duration"),
        "normalization_model": model,
        "correction_count": sum(1 for s in merged if s["clean_text"] != s["raw_text"]),
        "segments": merged,
    }
    write_json(paths["json"], result)
    paths["md"].write_text(_render_markdown(result), encoding="utf-8")
    print(f"[normalize-ai] wrote {paths['json']} ({result['correction_count']} corrections)")
    return result
