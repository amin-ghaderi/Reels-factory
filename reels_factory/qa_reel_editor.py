from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .cursor_ai import CursorAIError, extract_json_payload, invoke_cursor_agent, resolve_model_id
from .plan_validate import PlanValidationError, estimated_duration, validate_semantic_plan
from .refine import resolve_video_path
from .semantic_editor import compact_normalized_for_editor, enrich_reel
from .utils import parse_timestamp, read_json, ts, write_json

WORD_PAD_S = 2.0
SEGMENT_PAD_S = 0.75
TRANSITION_TYPES = {"transition"}
QA_TYPES = {"main_question", "follow_up", "clarification"}


def load_qa_editor_prompt(root: Path) -> str:
    return (root / "prompts" / "qa_reel_editor.md").read_text(encoding="utf-8")


def compact_stem(stem: str) -> str:
    return stem.replace("-", "")


def qa_reel_id(stem: str, unit_id: str) -> str:
    return f"{compact_stem(stem)}_qa_{unit_id}"


def qa_plan_paths(out_dir: Path, stem: str, unit_id: str) -> dict[str, Path]:
    base = f"{stem}.{unit_id}.qa"
    return {
        "json": out_dir / f"{base}.json",
        "skip": out_dir / f"{base}.skip.json",
    }


def is_transition_unit(unit: dict) -> bool:
    qtype = str(unit.get("question_type") or "").strip().lower()
    if qtype in TRANSITION_TYPES:
        return True
    if unit.get("question") or unit.get("answer"):
        return False
    return True


def is_qa_unit(unit: dict) -> bool:
    if not isinstance(unit, dict):
        return False
    if is_transition_unit(unit):
        return False
    qtype = str(unit.get("question_type") or "").strip().lower()
    if qtype and qtype not in QA_TYPES:
        return False
    return bool(unit.get("question") or unit.get("answer"))


def unit_bounds(unit: dict) -> tuple[float, float]:
    times: list[float] = []
    for block in (unit.get("question"), unit.get("answer"), unit):
        if not isinstance(block, dict):
            continue
        for key in ("start", "end", "core_start", "core_end"):
            raw = block.get(key)
            if raw in (None, ""):
                continue
            times.append(parse_timestamp(raw))
    if len(times) < 2:
        raise ValueError(f"{unit.get('unit_id') or 'unit'} has no usable time range")
    return min(times), max(times)


def slice_normalized_for_unit(normalized: dict, start: float, end: float) -> list[dict]:
    rows = []
    lo, hi = start - SEGMENT_PAD_S, end + SEGMENT_PAD_S
    for row in compact_normalized_for_editor(normalized):
        seg_start = parse_timestamp(row["start"])
        seg_end = parse_timestamp(row["end"])
        if seg_end < lo or seg_start > hi:
            continue
        rows.append(row)
    return rows


def slice_words_for_unit(transcript: dict, start: float, end: float) -> list[dict]:
    lo, hi = start - WORD_PAD_S, end + WORD_PAD_S
    words = []
    for seg in transcript.get("segments") or []:
        for word in seg.get("words") or []:
            try:
                ws = float(word["start"])
                we = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if we < lo or ws > hi:
                continue
            token = str(word.get("word") or "").strip()
            if not token:
                continue
            words.append({"start": ts(ws), "end": ts(we), "word": token})
    return words


def referenced_unit_summaries(program_map: dict, unit: dict) -> list[dict]:
    wanted = {str(x) for x in (unit.get("references_to_previous_unit") or [])}
    if not wanted:
        return []
    summaries = []
    for other in program_map.get("units") or []:
        uid = str(other.get("unit_id") or "")
        if uid not in wanted:
            continue
        answer = other.get("answer") or {}
        summaries.append({
            "unit_id": uid,
            "topic": other.get("topic"),
            "answer_summary": answer.get("summary"),
        })
    return summaries


def _rel_source(video: Path, root: Path) -> str:
    try:
        return video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(video)


def parse_qa_editor_payload(parsed, *, reel_id: str, unit_id: str) -> dict:
    if not isinstance(parsed, dict):
        raise CursorAIError("Q&A editor JSON must be an object")
    if parsed.get("skip") is True:
        reason = str(parsed.get("reason") or "editor skipped this unit").strip()
        return {"skip": True, "reason": reason, "source_unit": unit_id, "reel_id": reel_id}
    plan = parsed.get("plan") if isinstance(parsed.get("plan"), dict) else parsed
    if not isinstance(plan.get("segments"), list) or not plan.get("segments"):
        raise CursorAIError("Q&A editor JSON did not contain segments[]")
    out = dict(plan)
    out["skip"] = False
    out["reel_id"] = reel_id
    out["source_unit"] = unit_id
    out.setdefault("hook_used", any(
        str(seg.get("role") or "").lower() == "hook" for seg in out.get("segments") or []
    ))
    out.setdefault("avoid_hook_repeat", True)
    return out


def _write_skip(path: Path, payload: dict) -> None:
    write_json(path, payload)


def edit_qa_unit(
    unit: dict,
    *,
    video: Path,
    cfg: dict,
    root: Path,
    normalized: dict,
    transcript: dict,
    program_map: dict,
    force: bool = False,
    invoke=None,
) -> dict:
    unit_id = str(unit.get("unit_id") or "").strip()
    if not unit_id:
        raise CursorAIError("Q&A unit is missing unit_id")
    stem = video.stem
    out_dir = Path(cfg["paths"]["qa_plans"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = qa_plan_paths(out_dir, stem, unit_id)
    if not force and paths["json"].exists():
        print(f"[qa-editor] cache hit {paths['json'].name}")
        cached = read_json(paths["json"])
        cached["cached"] = True
        cached["skip"] = False
        return cached
    if not force and paths["skip"].exists():
        print(f"[qa-editor] cache hit skip {paths['skip'].name}")
        cached = read_json(paths["skip"])
        cached["cached"] = True
        cached["skip"] = True
        return cached

    start, end = unit_bounds(unit)
    source_rel = _rel_source(video, root)
    reel_id = qa_reel_id(stem, unit_id)
    aicfg = (cfg.get("ai_editor") or {}).get("qa_reel_editor") or {}
    semantic_cfg = (cfg.get("ai_editor") or {}).get("semantic_editor") or {}
    preferred = str(aicfg.get("model") or semantic_cfg.get("model") or "grok-4.6")
    model = resolve_model_id(preferred, kind="semantic")
    payload = {
        "source_video": source_rel,
        "reel_id": reel_id,
        "language": normalized.get("language"),
        "program_duration": ts(normalized.get("duration") or 0),
        "unit": unit,
        "referenced_units": referenced_unit_summaries(program_map, unit),
        "segments": slice_normalized_for_unit(normalized, start, end),
        "words": slice_words_for_unit(transcript, start, end),
    }
    prompt = (
        load_qa_editor_prompt(root)
        + "\n\n## Q&A unit request\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    caller = invoke or invoke_cursor_agent
    print(f"[qa-editor] calling {model} for {unit_id} ({len(payload['segments'])} segments)")
    try:
        with tempfile.TemporaryDirectory(prefix="reels_ai_qa_") as td:
            output = caller(prompt, model=model, mode="ask", workspace=Path(td), timeout=900)
        parsed = extract_json_payload(output)
        plan = parse_qa_editor_payload(parsed, reel_id=reel_id, unit_id=unit_id)
    except CursorAIError:
        raise
    except Exception as exc:
        raise CursorAIError(f"Q&A editor call failed for {unit_id}: {exc}") from exc

    if plan.get("skip"):
        record = {
            "skip": True,
            "reason": plan.get("reason"),
            "source_unit": unit_id,
            "reel_id": reel_id,
            "cached": False,
        }
        _write_skip(paths["skip"], record)
        if paths["json"].exists():
            paths["json"].unlink()
        print(f"[qa-editor] skipped {unit_id}: {record['reason']}")
        return record

    source_duration = float(normalized.get("duration") or 0) or None
    try:
        enriched = enrich_reel(plan, source_video=source_rel, source_duration=source_duration)
    except PlanValidationError as exc:
        raise CursorAIError(f"Q&A editor plan for {unit_id} failed validation: {exc}") from exc
    enriched["source_unit"] = unit_id
    enriched["reel_id"] = reel_id
    enriched["estimated_duration_s"] = round(estimated_duration(enriched), 3)
    enriched["cached"] = False
    enriched["skip"] = False
    write_json(paths["json"], enriched)
    if paths["skip"].exists():
        paths["skip"].unlink()
    print(f"[qa-editor] wrote {paths['json'].name} ({enriched['estimated_duration_s']}s)")
    return enriched


def edit_program_qa_units(
    video: Path,
    cfg: dict,
    *,
    root: Path,
    force: bool = False,
    invoke=None,
    program_map: dict | None = None,
    normalized: dict | None = None,
    transcript: dict | None = None,
) -> dict:
    video = resolve_video_path(video, root)
    stem = video.stem
    if program_map is None:
        map_path = Path(cfg["paths"]["program_maps"]) / f"{stem}.program_map.json"
        if not map_path.exists():
            raise FileNotFoundError(f"Program map not found: {map_path}. Run map-program first.")
        program_map = read_json(map_path)
    if normalized is None:
        norm_path = Path(cfg["paths"]["normalized_transcripts"]) / f"{stem}.normalized.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalized transcript not found: {norm_path}")
        normalized = read_json(norm_path)
    if transcript is None:
        tr_path = Path(cfg["paths"]["transcripts"]) / f"{stem}.transcript.json"
        if not tr_path.exists():
            raise FileNotFoundError(f"Raw transcript not found: {tr_path}")
        transcript = read_json(tr_path)

    out_dir = Path(cfg["paths"]["qa_plans"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plans: list[dict] = []
    skipped: list[dict] = []
    for unit in program_map.get("units") or []:
        uid = str(unit.get("unit_id") or "")
        if not is_qa_unit(unit):
            skipped.append({
                "source_unit": uid,
                "reason": "transition/intro/outro",
                "skip": True,
                "cached": False,
            })
            print(f"[qa-editor] skipping non-Q&A unit {uid or '?'}")
            continue
        try:
            result = edit_qa_unit(
                unit,
                video=video,
                cfg=cfg,
                root=root,
                normalized=normalized,
                transcript=transcript,
                program_map=program_map,
                force=force,
                invoke=invoke,
            )
        except (CursorAIError, ValueError, PlanValidationError) as exc:
            skipped.append({
                "source_unit": uid,
                "reason": str(exc),
                "skip": True,
                "cached": False,
                "error": True,
            })
            print(f"[qa-editor] {uid} failed — skipping unit: {exc}")
            continue
        if result.get("skip"):
            skipped.append(result)
        else:
            plans.append(result)

    qa_results = [p for p in plans] + [s for s in skipped if s.get("source_unit") and "transition" not in str(s.get("reason") or "")]
    index = {
        "source_video": _rel_source(video, root),
        "program_map": str(Path(cfg["paths"]["program_maps"]) / f"{stem}.program_map.json"),
        "plan_count": len(plans),
        "skipped_count": len(skipped),
        "plans": [str(qa_plan_paths(out_dir, stem, p["source_unit"])["json"]) for p in plans],
        "skipped": [
            {"unit_id": s.get("source_unit"), "reason": s.get("reason")}
            for s in skipped
        ],
        "cached": bool(qa_results) and all(item.get("cached") for item in qa_results),
    }
    write_json(out_dir / f"{stem}.qa_index.json", index)
    return {"index": index, "plans": plans, "skipped": skipped}
