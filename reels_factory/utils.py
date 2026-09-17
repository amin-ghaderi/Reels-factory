from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary '{name}' was not found in PATH.")
    return path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"[run] {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def srt_ts(seconds: float) -> str:
    return ts(seconds).replace(".", ",")


def parse_timestamp(value) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS[.mmm] into seconds."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Timestamp cannot be negative: {value}")
        return seconds
    text = str(value).strip()
    if not text:
        raise ValueError("Timestamp is empty")
    text = text.replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, sec = parts
            seconds = int(minutes) * 60 + float(sec)
        elif len(parts) == 3:
            hours, minutes, sec = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(sec)
        else:
            raise ValueError(f"Invalid timestamp format: {value!r}")
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp format: {value!r}") from exc
    if seconds < 0:
        raise ValueError(f"Timestamp cannot be negative: {value}")
    return seconds


def to_source_time(local_seconds: float, source_start: float) -> float:
    """Convert a clip-local timestamp into an absolute source timestamp."""
    return float(local_seconds) + float(source_start)
