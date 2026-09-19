from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FOLDERS = (
    "inbox",
    "transcripts",
    "candidates",
    "edit_plans",
    "output",
    "approved",
    "rejected",
    "refined_transcripts",
    "refine_requests",
    "normalized_transcripts",
    "semantic_plans",
    "program_maps",
    "qa_plans",
    "metadata",
    "portraits",
    "framing_profiles",
    "final",
)
WRITABLE_FOLDERS = (
    "transcripts",
    "candidates",
    "edit_plans",
    "output",
    "approved",
    "rejected",
    "refined_transcripts",
    "refine_requests",
    "normalized_transcripts",
    "semantic_plans",
    "program_maps",
    "qa_plans",
    "metadata",
    "portraits",
    "framing_profiles",
    "final",
)


def _ok(label: str, detail: str = "OK") -> None:
    print(f"{label:<14} {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"{label:<14} FAIL — {detail}")


def _binary_version(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        first = (proc.stdout or proc.stderr or "").splitlines()
        return first[0].strip() if first else path
    except OSError:
        return path


def main() -> int:
    print("Reels Factory Health Check")
    failed = False

    py = sys.version_info
    py_label = f"{py.major}.{py.minor}.{py.micro}"
    if py.major == 3 and py.minor in (11, 12):
        _ok("Python", f"OK ({py_label})")
    elif py.major == 3 and py.minor >= 13:
        _fail("Python", f"{py_label} is not recommended; use Python 3.12")
        failed = True
    else:
        _fail("Python", f"{py_label} is unsupported; install Python 3.12")
        failed = True

    ffmpeg_ver = _binary_version("ffmpeg")
    if ffmpeg_ver:
        _ok("FFmpeg", "OK")
    else:
        _fail("FFmpeg", "ffmpeg was not found in PATH")
        failed = True

    ffprobe_ver = _binary_version("ffprobe")
    if ffprobe_ver:
        _ok("FFprobe", "OK")
    else:
        _fail("FFprobe", "ffprobe was not found in PATH")
        failed = True

    try:
        import faster_whisper  # noqa: F401
        _ok("Whisper", "OK")
    except Exception as exc:
        _fail("Whisper", f"faster-whisper import failed: {exc}")
        failed = True

    try:
        import yaml  # noqa: F401
        from reels_factory.config import load_config

        cfg_path = ROOT / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} is missing — run: python run_pipeline.py setup")
        cfg = load_config(ROOT, cfg_path)
        tcfg = cfg.get("transcription") or {}
        if tcfg.get("model") != "small" or tcfg.get("device") != "cpu":
            _ok("Config", f"OK (model={tcfg.get('model')}, device={tcfg.get('device')})")
        else:
            _ok("Config", "OK")
    except Exception as exc:
        _fail("Config", str(exc))
        failed = True
        cfg = None

    folder_ok = True
    if cfg is None:
        folder_ok = False
    else:
        for key in REQUIRED_FOLDERS:
            path = Path(cfg["paths"].get(key, ROOT / "data" / key))
            if not path.exists():
                print(f"               missing folder: {path}")
                folder_ok = False
                continue
            if key in WRITABLE_FOLDERS:
                probe = path / ".healthcheck_write_probe"
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    probe.write_text("ok", encoding="utf-8")
                except OSError as exc:
                    print(f"               not writable: {path} ({exc})")
                    folder_ok = False
                finally:
                    if probe.exists():
                        try:
                            probe.unlink()
                        except OSError:
                            pass
        if folder_ok:
            _ok("Folders", "OK")
        else:
            _fail("Folders", "one or more required folders are missing or not writable")
            failed = True

    print()
    if failed:
        print("NOT READY")
        return 1
    print("READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
