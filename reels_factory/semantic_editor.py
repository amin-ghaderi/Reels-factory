from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .cursor_ai import CursorAIError, extract_json_payload, invoke_cursor_agent, resolve_model_id
from .plan_validate import (
    PlanValidationError,
    estimated_duration,
    source_span_duration,
    validate_semantic_plan,
)
from .refine import resolve_video_path
from .utils import read_json, ts, write_json

FULL_CONTEXT_CHAR_LIMIT = 400_000


def compact_normalized_for_editor(normalized: dict) -> list[dict]:
    rows = []
    for seg in normalized.get("segments") or []:
        rows.append({
            "id": int(seg["segment_id"]),
            "start": ts(seg["start"]),
            "end": ts(seg["end"]),
            "text": seg.get("clean_text") or seg.get("raw_text") or "",
        })
    return rows


def load_editor_prompt(root: Path) -> str:
    return (root / "prompts" / "semantic_reel_editor.md").read_text(encoding="utf-8")


def _rel_source(video: Path, root: Path) -> str:
    try:
        return video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(video)


def enrich_reel(reel: dict, *, source_video: str, source_duration: float | None) -> dict:
    plan = dict(reel)
    plan["source_video"] = source_video
    plan.setdefault("avoid_hook_repeat", True)
    validate_semantic_plan(plan, source_duration=source_duration)
    plan["estimated_duration_s"] = round(estimated_duration(plan), 3)
    plan["original_source_span_s"] = round(source_span_duration(plan), 3)
    return plan


def write_semantic_plans(reels: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for reel in reels:
        reel_id = reel.get("reel_id") or f"semantic_{len(paths)+1:02d}"
        path = out_dir / f"{reel_id}.json"
        write_json(path, reel)
        paths.append(path)
    return paths


def semantic_edit(
    video: Path,
    cfg: dict,
    *,
    root: Path,
    force: bool = False,
    invoke=None,
) -> dict:
    video = resolve_video_path(video, root)
    stem = video.stem
    normalized_path = Path(cfg["paths"]["normalized_transcripts"]) / f"{stem}.normalized.json"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized transcript not found: {normalized_path}. Run normalize-ai first."
        )
    out_dir = Path(cfg["paths"]["semantic_plans"])
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_stem = stem.replace("-", "")
    index_path = out_dir / f"{stem}.semantic_index.json"
    existing = sorted({
        *out_dir.glob(f"{compact_stem}_semantic_*.json"),
        *out_dir.glob(f"{stem}_semantic_*.json"),
    })
    if existing and not force:
        print(f"[semantic-edit] cache hit ({len(existing)} plans)")
        if index_path.exists():
            cached = read_json(index_path)
            cached["cached"] = True
            return cached
        return {
            "cached": True,
            "reel_count": len(existing),
            "plans": [str(p) for p in existing],
            "clearest_example": str(existing[0]),
        }

    normalized = read_json(normalized_path)
    aicfg = (cfg.get("ai_editor") or {}).get("semantic_editor") or {}
    model = resolve_model_id(str(aicfg.get("model") or "grok-4.6"), kind="semantic")
    rows = compact_normalized_for_editor(normalized)
    source_rel = _rel_source(video, root)
    payload = {
        "source_video": source_rel,
        "language": normalized.get("language"),
        "duration": normalized.get("duration"),
        "duration_clock": ts(normalized.get("duration") or 0),
        "reel_id_prefix": f"{compact_stem}_semantic",
        "segments": rows,
    }
    max_reels = (cfg.get("ai_editor") or {}).get("max_reels")
    extra = f"\nReturn at most {int(max_reels)} reels.\n" if max_reels else "\n"
    prompt = (
        load_editor_prompt(root)
        + extra
        + "\n## Normalized transcript\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    caller = invoke or invoke_cursor_agent
    mode_label = "full transcript"
    if len(prompt) > FULL_CONTEXT_CHAR_LIMIT:
        print(
            f"[semantic-edit] prompt is {len(prompt)} chars; dropping empty/tiny "
            "segments as a context-window fallback"
        )
        payload["segments"] = [
            row for row in rows
            if str(row.get("text") or "").strip()
        ]
        prompt = (
            load_editor_prompt(root)
            + extra
            + "\n## Normalized transcript\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        mode_label = "context-fallback"
    print(f"[semantic-edit] calling {model} on {len(payload['segments'])} segments ({mode_label})")
    try:
        with tempfile.TemporaryDirectory(prefix="reels_ai_edit_") as td:
            output = caller(prompt, model=model, mode="ask", workspace=Path(td), timeout=900)
        parsed = extract_json_payload(output)
    except CursorAIError:
        raise
    except Exception as exc:
        raise CursorAIError(f"Semantic editor call failed: {exc}") from exc

    if isinstance(parsed, dict) and "reels" in parsed:
        reels = parsed.get("reels") or []
        example_id = parsed.get("clearest_example_reel_id")
    elif isinstance(parsed, list):
        reels = parsed
        example_id = None
    else:
        raise CursorAIError("Semantic editor JSON did not contain reels[]")

    source_duration = float(normalized.get("duration") or 0) or None
    enriched = []
    for reel in reels:
        if not isinstance(reel, dict):
            continue
        if max_reels and len(enriched) >= int(max_reels):
            break
        if not reel.get("reel_id"):
            reel["reel_id"] = f"{compact_stem}_semantic_{len(enriched)+1:02d}"
        try:
            enriched.append(
                enrich_reel(reel, source_video=source_rel, source_duration=source_duration)
            )
        except PlanValidationError as exc:
            print(f"[semantic-edit] rejecting {reel.get('reel_id')}: {exc}")
            continue

    if not enriched:
        raise CursorAIError("Semantic editor returned no valid reels")

    paths = write_semantic_plans(enriched, out_dir)
    example = None
    if example_id:
        example = next((p for p in paths if p.stem == example_id), None)
    if example is None:
        high = [p for p, r in zip(paths, enriched) if r.get("context_integrity") == "high"]
        example = (high[0] if high else paths[0])

    index = {
        "source_video": source_rel,
        "normalized_transcript": str(normalized_path),
        "semantic_model": model,
        "reel_count": len(paths),
        "plans": [str(p) for p in paths],
        "clearest_example": str(example),
        "cached": False,
    }
    write_json(out_dir / f"{stem}.semantic_index.json", index)
    print(f"[semantic-edit] wrote {len(paths)} plans; example={example.name}")
    return index
