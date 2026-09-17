from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


class CursorAIError(RuntimeError):
    """Cursor Agent CLI could not be invoked or returned unusable output."""


def find_agent_binary() -> str | None:
    env = os.environ.get("CURSOR_AGENT_BIN")
    if env and Path(env).exists():
        return env
    for name in ("agent", "agent.exe", "agent.cmd"):
        found = shutil.which(name)
        if found:
            return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "agent.exe",
        home / ".local" / "bin" / "agent.cmd",
        home / ".local" / "bin" / "agent",
        Path(os.environ.get("LOCALAPPDATA", "")) / "cursor-agent" / "agent.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "cursor-agent" / "agent.exe",
    ]
    for path in candidates:
        if path and path.exists():
            return str(path)
    return None


def _run_agent(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    binary = find_agent_binary()
    if not binary:
        raise CursorAIError(
            "Cursor Agent CLI was not found. Install it with PowerShell: "
            "irm 'https://cursor.com/install?win32=true' | iex   then reopen the terminal."
        )
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def list_cursor_models() -> list[str]:
    """Return model identifiers from the local Cursor CLI, or an empty list."""
    try:
        proc = _run_agent(["--list-models"], timeout=60)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            proc = _run_agent(["models"], timeout=60)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except CursorAIError:
        return []
    except (OSError, subprocess.TimeoutExpired):
        return []
    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("model", "available", "name", "-")):
            continue
        token = line.split()[0].strip("`,* ")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{1,80}", token):
            ids.append(token)
    return ids


def resolve_model_id(preferred: str, *, kind: str) -> str:
    """Pick a real CLI model id. Prefer non-Fast variants when listed."""
    wanted = (preferred or "").strip()
    models = list_cursor_models()
    if not models:
        return wanted or ("composer-2.5" if kind == "normalization" else "grok-4.6")

    lowered = {m.lower(): m for m in models}

    def pick(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        for cand in candidates:
            for key, orig in lowered.items():
                if cand.lower() in key and "fast" not in key:
                    return orig
        return None

    if kind == "normalization":
        chosen = pick([
            wanted,
            "composer-2.5",
            "composer-2.5-standard",
            "composer-2.5-low",
            "Cursor Composer 2.5",
        ])
        if chosen and "fast" in chosen.lower() and wanted and "fast" not in wanted.lower():
            nonfast = pick(["composer-2.5", "composer-2.5-standard"])
            if nonfast and "fast" not in nonfast.lower():
                return nonfast
        return chosen or wanted or "composer-2.5"

    chosen = pick([
        wanted,
        "grok-4.6",
        "cursor-grok-4.6-high",
        "grok-4.6-high",
        "Cursor Grok 4.6",
    ])
    return chosen or wanted or "grok-4.6"


def extract_json_payload(text: str):
    """Parse JSON from an Agent CLI print response, including fenced dumps."""
    if not text or not str(text).strip():
        raise CursorAIError("Cursor Agent returned empty output")
    raw = str(text).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("result"), str):
            return extract_json_payload(payload["result"])
        return payload
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise CursorAIError("Cursor Agent output did not contain JSON")


def invoke_cursor_agent(
    prompt: str,
    *,
    model: str,
    mode: str = "ask",
    workspace: Path | None = None,
    timeout: int = 600,
) -> str:
    """One-shot Cursor Agent call. Ask mode, no repo tools required."""
    workspace = workspace or Path.cwd()
    args = [
        "-p",
        "--output-format",
        "text",
        "--mode",
        mode,
        "--model",
        model,
        "--trust",
        "--workspace",
        str(workspace),
        prompt,
    ]
    print(f"[cursor-ai] model={model} mode={mode} workspace={workspace}")
    try:
        proc = _run_agent(args, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CursorAIError(f"Cursor Agent timed out after {timeout}s") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise CursorAIError(f"Cursor Agent failed (exit {proc.returncode}): {err[:800]}")
    return proc.stdout or ""
