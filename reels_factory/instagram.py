from __future__ import annotations

import re
from pathlib import Path

_SENTENCE_SPLIT = re.compile(r"(?<=[。.؟!])\s+")


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" \t")


def _payoff_text(plan: dict) -> str:
    for seg in plan.get("segments") or []:
        if str(seg.get("role") or "") == "payoff":
            text = _clean_sentence(str(seg.get("text") or ""))
            if text:
                return text
    return ""


def _hashtags(plan: dict, meta: dict) -> list[str]:
    blob = " ".join(
        [
            str(plan.get("title") or ""),
            str(meta.get("guest_role") or ""),
            _payoff_text(plan),
        ]
    )
    tags = ["#ایران"]
    if "آب" in blob:
        tags.append("#بحران_آب")
        tags.append("#منابع_آب")
    if "محیط" in blob or "زیست" in blob:
        tags.append("#محیط_زیست")
    if len(tags) < 2:
        tags.append("#گفتگو")
    # unique, max 4
    out = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
        if len(out) == 4:
            break
    return out


def build_instagram_caption(plan: dict, meta: dict) -> str:
    """Short source-faithful Persian caption. Does not copy the cover headline."""
    guest = _clean_sentence(str(meta.get("guest_name") or ""))
    title = _clean_sentence(str(plan.get("title") or "")).rstrip("؟?.")
    payoff = _payoff_text(plan)
    lines: list[str] = []
    if guest and title:
        lines.append(f"از نظر {guest}، {title}.")
    elif title:
        lines.append(f"{title}.")
    elif guest:
        lines.append(guest)

    extra = payoff
    if extra and title and extra.replace(".", "") == title.replace(".", ""):
        extra = ""
    if extra:
        parts = [_clean_sentence(p) for p in _SENTENCE_SPLIT.split(extra) if _clean_sentence(p)]
        for part in parts:
            if len(lines) >= 4:
                break
            if part and part not in lines:
                lines.append(part)

    # Keep 2–4 body lines; drop a trailing duplicate if needed.
    lines = [ln for ln in lines if ln][:4]
    if not lines:
        lines = ["گفتگوی برنامه."]
    caption = "\n".join(lines)
    tags = _hashtags(plan, meta)
    return f"{caption}\n\n{' '.join(tags)}\n"


def write_instagram_caption(path: Path, plan: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_instagram_caption(plan, meta).rstrip() + "\n", encoding="utf-8")
    return path
