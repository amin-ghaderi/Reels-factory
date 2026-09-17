from __future__ import annotations

from pathlib import Path
from .utils import srt_ts


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_rebased_srt(transcript: dict, clips: list[dict], out_path: Path) -> Path:
    """Rebase source transcript segments onto the concatenated reel timeline."""
    rows = []
    timeline_cursor = 0.0
    for clip in clips:
        c0, c1 = float(clip["start"]), float(clip["end"])
        for seg in transcript.get("segments", []):
            s0, s1 = float(seg["start"]), float(seg["end"])
            if _overlap(c0, c1, s0, s1) <= 0:
                continue
            local_start = timeline_cursor + max(s0, c0) - c0
            local_end = timeline_cursor + min(s1, c1) - c0
            if local_end - local_start < 0.08:
                continue
            rows.append((local_start, local_end, seg["text"].strip()))
        timeline_cursor += c1 - c0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(rows, 1):
            f.write(f"{idx}\n{srt_ts(start)} --> {srt_ts(end)}\n{text}\n\n")
    return out_path
