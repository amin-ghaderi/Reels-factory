from __future__ import annotations

from .utils import parse_timestamp

OVERLAP_EPS = 0.15


class PlanValidationError(ValueError):
    """Edit plan failed structural / overlap validation."""


def plan_has_segments(plan: dict) -> bool:
    return bool(plan.get("segments"))


def clips_from_segments(plan: dict) -> list[dict]:
    clips = []
    for idx, seg in enumerate(plan.get("segments") or []):
        start = parse_timestamp(seg["start"])
        end = parse_timestamp(seg["end"])
        clips.append({
            "label": str(seg.get("role") or f"seg_{idx:02d}"),
            "start": start,
            "end": end,
            "role": seg.get("role"),
        })
    return clips


def _ranges_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def validate_semantic_plan(plan: dict, *, source_duration: float | None = None) -> None:
    """Fail clearly on invalid multi-segment plans. Never rewrite timestamps."""
    errors: list[str] = []
    segments = plan.get("segments")
    if not segments:
        raise PlanValidationError("Semantic plan is missing segments[]")

    clips = []
    for idx, seg in enumerate(segments):
        try:
            start = parse_timestamp(seg["start"])
            end = parse_timestamp(seg["end"])
        except (KeyError, ValueError) as exc:
            errors.append(f"segment[{idx}] has an invalid timestamp: {exc}")
            continue
        if end <= start:
            errors.append(f"segment[{idx}] start >= end ({start} → {end})")
        if start < 0:
            errors.append(f"segment[{idx}] start is negative ({start})")
        if source_duration is not None and end > source_duration + 0.25:
            errors.append(
                f"segment[{idx}] end {end:.3f} is past source duration {source_duration:.3f}"
            )
        clips.append({"index": idx, "start": start, "end": end, "role": seg.get("role")})

    seen: dict[tuple[float, float], int] = {}
    for clip in clips:
        key = (round(clip["start"], 3), round(clip["end"], 3))
        if key in seen:
            errors.append(
                f"exact duplicate segment: index {seen[key]} and {clip['index']} "
                f"({clip['start']:.3f}–{clip['end']:.3f})"
            )
        else:
            seen[key] = clip["index"]

    for i, a in enumerate(clips):
        for b in clips[i + 1 :]:
            overlap = _ranges_overlap(a["start"], a["end"], b["start"], b["end"])
            if overlap > OVERLAP_EPS:
                errors.append(
                    f"unnecessary overlap of {overlap:.3f}s between segment[{a['index']}] "
                    f"({a['start']:.3f}–{a['end']:.3f}) and segment[{b['index']}] "
                    f"({b['start']:.3f}–{b['end']:.3f})"
                )

    avoid_repeat = bool(plan.get("avoid_hook_repeat", True))
    hook = next((c for c in clips if str(c.get("role") or "").lower() == "hook"), None)
    if avoid_repeat and hook is not None:
        for clip in clips:
            if clip["index"] == hook["index"]:
                continue
            overlap = _ranges_overlap(hook["start"], hook["end"], clip["start"], clip["end"])
            if overlap > OVERLAP_EPS:
                errors.append(
                    f"hook repeated in later segment[{clip['index']}] "
                    f"(overlap {overlap:.3f}s) while avoid_hook_repeat is true"
                )

    if errors:
        raise PlanValidationError("Invalid semantic edit plan:\n- " + "\n- ".join(errors))


def estimated_duration(plan: dict) -> float:
    if plan_has_segments(plan):
        return sum(max(0.0, parse_timestamp(s["end"]) - parse_timestamp(s["start"])) for s in plan["segments"])
    body = plan.get("body") or {}
    hook = plan.get("hook") or {}
    total = 0.0
    if hook and hook.get("enabled", True) and hook.get("start") is not None:
        total += parse_timestamp(hook["end"]) - parse_timestamp(hook["start"])
    if body.get("start") is not None:
        total += parse_timestamp(body["end"]) - parse_timestamp(body["start"])
    return total


def source_span_duration(plan: dict) -> float:
    """Wall-clock source coverage from earliest start to latest end."""
    times: list[float] = []
    if plan_has_segments(plan):
        for seg in plan["segments"]:
            times.extend([parse_timestamp(seg["start"]), parse_timestamp(seg["end"])])
    else:
        for block in (plan.get("hook"), plan.get("body")):
            if not block:
                continue
            if block.get("start") is None:
                continue
            times.extend([parse_timestamp(block["start"]), parse_timestamp(block["end"])])
    if not times:
        return 0.0
    return max(times) - min(times)
