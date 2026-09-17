from __future__ import annotations

from .utils import parse_timestamp


class ProgramMapError(ValueError):
    """Program map failed structural validation. Timestamps are never rewritten."""


ALLOWED_QUESTION_TYPES = {
    "main_question",
    "follow_up",
    "clarification",
    "transition",
}
ALLOWED_ANSWER_STATUS = {"complete", "partial", "interrupted"}


def _clock(block: dict | None, key: str) -> float | None:
    if not block or block.get(key) in (None, ""):
        return None
    return parse_timestamp(block[key])


def _check_range(label: str, start, end, *, duration: float | None, errors: list[str]) -> None:
    if start is None or end is None:
        return
    if end <= start:
        errors.append(f"{label}: start >= end ({start} → {end})")
    if start < 0:
        errors.append(f"{label}: start is negative ({start})")
    if duration is not None and end > duration + 0.25:
        errors.append(f"{label}: end {end:.3f} is past source duration {duration:.3f}")


def _check_core(label: str, parent_start, parent_end, core_start, core_end, errors: list[str]) -> None:
    if core_start is None:
        return
    if parent_start is not None and core_start + 0.05 < parent_start:
        errors.append(f"{label}.core_start {core_start:.3f} is before parent start {parent_start:.3f}")
    if parent_end is not None and core_start > parent_end + 0.05:
        errors.append(f"{label}.core_start {core_start:.3f} is after parent end {parent_end:.3f}")
    if core_end is None:
        return
    if core_end <= core_start:
        errors.append(f"{label}.core: start >= end ({core_start} → {core_end})")
    if parent_end is not None and core_end > parent_end + 0.05:
        errors.append(f"{label}.core_end {core_end:.3f} is after parent end {parent_end:.3f}")


def validate_program_map(payload: dict, *, source_duration: float | None = None) -> None:
    """Fail clearly. Never rewrite timestamps."""
    errors: list[str] = []
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ProgramMapError("Program map is missing units[]")

    seen_ids: set[str] = set()
    seen_spans: dict[tuple, str] = {}
    prev_start: float | None = None

    for idx, unit in enumerate(units):
        if not isinstance(unit, dict):
            errors.append(f"units[{idx}] is not an object")
            continue
        uid = str(unit.get("unit_id") or "").strip() or f"units[{idx}]"
        if uid in seen_ids:
            errors.append(f"duplicate unit_id {uid}")
        seen_ids.add(uid)

        qtype = unit.get("question_type")
        if qtype and str(qtype) not in ALLOWED_QUESTION_TYPES:
            errors.append(f"{uid}: unknown question_type {qtype!r}")

        question = unit.get("question")
        answer = unit.get("answer")
        q0 = _clock(question, "start")
        q1 = _clock(question, "end")
        qc0 = _clock(question, "core_start")
        qc1 = _clock(question, "core_end")
        a0 = _clock(answer, "start")
        a1 = _clock(answer, "end")
        ac0 = _clock(answer, "core_start")
        u0 = _clock(unit, "start")
        u1 = _clock(unit, "end")
        if not question and not answer:
            if u0 is None or u1 is None:
                errors.append(f"{uid}: unit has neither question, answer, nor start/end")
            else:
                _check_range(f"{uid}.span", u0, u1, duration=source_duration, errors=errors)

        _check_range(f"{uid}.question", q0, q1, duration=source_duration, errors=errors)
        _check_range(f"{uid}.answer", a0, a1, duration=source_duration, errors=errors)
        _check_core(f"{uid}.question", q0, q1, qc0, qc1, errors)
        _check_core(f"{uid}.answer", a0, a1, ac0, None, errors)

        if answer and answer.get("status") and str(answer.get("status")) not in ALLOWED_ANSWER_STATUS:
            errors.append(f"{uid}: unknown answer.status {answer.get('status')!r}")

        span_key = (
            None if q0 is None else round(q0, 3),
            None if q1 is None else round(q1, 3),
            None if a0 is None else round(a0, 3),
            None if a1 is None else round(a1, 3),
            None if u0 is None else round(u0, 3),
            None if u1 is None else round(u1, 3),
        )
        if span_key in seen_spans:
            errors.append(
                f"{uid} duplicates the same Q&A span as {seen_spans[span_key]}"
            )
        else:
            seen_spans[span_key] = uid

        unit_start = q0 if q0 is not None else a0 if a0 is not None else u0
        if unit_start is not None:
            if prev_start is not None and unit_start + 0.05 < prev_start:
                errors.append(
                    f"{uid} is out of chronological order "
                    f"(starts {unit_start:.3f} before previous {prev_start:.3f})"
                )
            prev_start = unit_start

    if errors:
        raise ProgramMapError("Invalid program map:\n- " + "\n- ".join(errors))
