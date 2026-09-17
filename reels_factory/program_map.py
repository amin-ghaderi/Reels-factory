from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .cursor_ai import CursorAIError, extract_json_payload, invoke_cursor_agent, resolve_model_id
from .program_map_validate import ProgramMapError, validate_program_map
from .refine import resolve_video_path
from .utils import read_json, ts, write_json

FULL_CONTEXT_CHAR_LIMIT = 400_000


def compact_normalized_for_mapper(normalized: dict) -> list[dict]:
    rows = []
    for seg in normalized.get("segments") or []:
        rows.append({
            "id": int(seg["segment_id"]),
            "start": ts(seg["start"]),
            "end": ts(seg["end"]),
            "text": seg.get("clean_text") or seg.get("raw_text") or "",
        })
    return rows


def load_mapper_prompt(root: Path) -> str:
    return (root / "prompts" / "program_mapper.md").read_text(encoding="utf-8")


def _rel_source(video: Path, root: Path) -> str:
    try:
        return video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(video)


def render_program_map_markdown(payload: dict) -> str:
    lines = [
        f"# Program map — {payload.get('source_video', '')}",
        "",
        f"- duration: {payload.get('program_duration')}",
        f"- units: {len(payload.get('units') or [])}",
        f"- model: {payload.get('mapper_model')}",
        "",
    ]
    for unit in payload.get("units") or []:
        uid = unit.get("unit_id")
        lines.append(f"## {uid} — {unit.get('topic', '')}")
        lines.append(f"- type: {unit.get('question_type')}")
        q = unit.get("question") or {}
        a = unit.get("answer") or {}
        if unit.get("start") and not q and not a:
            lines.append(f"- span: {unit.get('start')} → {unit.get('end')}")
        if q:
            lines.append(f"- question: {q.get('start')} → {q.get('end')}")
            lines.append(f"- question core: {q.get('core_start')} → {q.get('core_end')}")
            if q.get("text"):
                lines.append(f"- question text: {q.get('text')}")
        else:
            lines.append("- question: none")
        if a:
            lines.append(f"- answer: {a.get('start')} → {a.get('end')}")
            lines.append(f"- answer core start: {a.get('core_start')}")
            lines.append(f"- answer status: {a.get('status')}")
            if a.get("summary"):
                lines.append(f"- guest answers: {a.get('summary')}")
        else:
            lines.append("- answer: none")
        refs = unit.get("references_to_previous_unit") or []
        if refs:
            lines.append(f"- references: {', '.join(str(r) for r in refs)}")
        if unit.get("notes"):
            lines.append(f"- notes: {unit.get('notes')}")
        lines.append("")
    return "\n".join(lines)


def program_map_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "json": out_dir / f"{stem}.program_map.json",
        "md": out_dir / f"{stem}.program_map.md",
    }


def map_program(
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
    out_dir = Path(cfg["paths"]["program_maps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = program_map_paths(out_dir, stem)
    if paths["json"].exists() and not force:
        print(f"[map-program] cache hit {paths['json']}")
        return read_json(paths["json"])

    normalized = read_json(normalized_path)
    aicfg = (cfg.get("ai_editor") or {}).get("program_mapper") or {}
    semantic_cfg = (cfg.get("ai_editor") or {}).get("semantic_editor") or {}
    preferred = str(aicfg.get("model") or semantic_cfg.get("model") or "grok-4.6")
    model = resolve_model_id(preferred, kind="semantic")
    rows = compact_normalized_for_mapper(normalized)
    source_rel = _rel_source(video, root)
    duration = normalized.get("duration")
    payload = {
        "source_video": source_rel,
        "language": normalized.get("language"),
        "duration": duration,
        "duration_clock": ts(duration or 0),
        "segments": rows,
    }
    prompt = (
        load_mapper_prompt(root)
        + "\n\n## Normalized transcript\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    if len(prompt) > FULL_CONTEXT_CHAR_LIMIT:
        print(f"[map-program] prompt is {len(prompt)} chars; sending full transcript anyway")
    caller = invoke or invoke_cursor_agent
    print(f"[map-program] calling {model} on {len(rows)} segments (full transcript)")
    try:
        with tempfile.TemporaryDirectory(prefix="reels_ai_map_") as td:
            output = caller(prompt, model=model, mode="ask", workspace=Path(td), timeout=900)
        parsed = extract_json_payload(output)
    except CursorAIError:
        raise
    except Exception as exc:
        raise CursorAIError(f"Program mapper call failed: {exc}") from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("units"), list):
        raise CursorAIError("Program mapper JSON did not contain units[]")

    source_duration = float(duration or 0) or None
    try:
        validate_program_map(parsed, source_duration=source_duration)
    except ProgramMapError as exc:
        raise CursorAIError(str(exc)) from exc

    result = {
        "source_video": source_rel,
        "source_transcript": str(normalized_path),
        "program_duration": parsed.get("program_duration") or ts(duration or 0),
        "mapper_model": model,
        "unit_count": len(parsed["units"]),
        "units": parsed["units"],
    }
    write_json(paths["json"], result)
    paths["md"].write_text(render_program_map_markdown(result), encoding="utf-8")
    print(f"[map-program] wrote {paths['json']} ({result['unit_count']} units)")
    return result
