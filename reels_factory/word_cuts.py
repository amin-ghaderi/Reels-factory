"""Word-level cut helpers. Timing is a precision reference, not a hard knife."""
from __future__ import annotations


def snap_cut_to_silence(
    *,
    word_start: float,
    word_end: float,
    prev_end: float | None,
    next_start: float | None,
    pad_before_ms: int = 0,
    pad_after_ms: int = 0,
    min_pad_before_ms: int = 40,
    max_pad_before_ms: int = 160,
    min_pad_after_ms: int = 60,
    max_pad_after_ms: int = 180,
) -> dict:
    """Place a cut in nearby silence/breath. Never inside the spoken word."""
    start = float(word_start)
    end = float(word_end)
    if prev_end is not None:
        gap_before = max(0.0, start - float(prev_end))
        natural = min(max(gap_before * 0.45, min_pad_before_ms / 1000.0), max_pad_before_ms / 1000.0)
        start = start - min(natural, gap_before * 0.85)
    else:
        start = start - (min_pad_before_ms / 1000.0)
    if next_start is not None:
        gap_after = max(0.0, float(next_start) - end)
        natural = min(max(gap_after * 0.45, min_pad_after_ms / 1000.0), max_pad_after_ms / 1000.0)
        end = end + min(natural, gap_after * 0.85)
    else:
        end = end + (min_pad_after_ms / 1000.0)
    if pad_before_ms:
        start = min(float(word_start), start) - (pad_before_ms / 1000.0)
        start = min(start, float(word_start) - 0.01)
    if pad_after_ms:
        end = max(float(word_end), end) + (pad_after_ms / 1000.0)
        end = max(end, float(word_end) + 0.01)
    # Never enter the word from the wrong side.
    start = min(start, float(word_start))
    end = max(end, float(word_end))
    return {
        "source_start": round(start, 3),
        "source_end": round(end, 3),
        "first_word_start": round(float(word_start), 3),
        "last_word_end": round(float(word_end), 3),
        "boundary_padding_before_ms": int(round((float(word_start) - start) * 1000)),
        "boundary_padding_after_ms": int(round((end - float(word_end)) * 1000)),
    }


def word_fields(
    *,
    first_word: str,
    last_word: str,
    first_word_start: float,
    last_word_end: float,
    prev_end: float | None = None,
    next_start: float | None = None,
) -> dict:
    snapped = snap_cut_to_silence(
        word_start=first_word_start,
        word_end=last_word_end,
        prev_end=prev_end,
        next_start=next_start,
    )
    return {
        "source_start": snapped["source_start"],
        "source_end": snapped["source_end"],
        "first_word": first_word,
        "first_word_start": snapped["first_word_start"],
        "last_word": last_word,
        "last_word_end": snapped["last_word_end"],
        "boundary_padding_before_ms": snapped["boundary_padding_before_ms"],
        "boundary_padding_after_ms": snapped["boundary_padding_after_ms"],
        "padding_before_ms": snapped["boundary_padding_before_ms"],
        "padding_after_ms": snapped["boundary_padding_after_ms"],
    }
