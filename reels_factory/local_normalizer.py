"""Local Persian ASR cleanup. Timestamps stay untouched. No paraphrasing."""
from __future__ import annotations

import re
from pathlib import Path

from .utils import read_json, write_json, ts

_ARABIC_YE = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})
_DIGIT = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Longest-first obvious ASR / spacing repairs. Meaning is not rewritten.
_TERM_REPAIRS: list[tuple[str, str]] = [
    ("تلویزیون ایرانگرا", "تلویزیون ایران‌گرا"),
    ("ایرانگرایی", "ایران‌گرایی"),
    ("ایرانگرا", "ایران‌گرا"),
    ("پادشاهی خواه", "پادشاهی‌خواه"),
    ("پادشاهیخواهان", "پادشاهی‌خواهان"),
    ("پادشاهیخواهان", "پادشاهی‌خواهان"),
    ("پادشاهی خواهان", "پادشاهی‌خواهان"),
    ("خوشال", "خوشحال"),
    ("می شود", "می‌شود"),
    ("می شود", "می‌شود"),
    ("نمی شود", "نمی‌شود"),
    ("نمی کند", "نمی‌کند"),
    ("می کند", "می‌کند"),
    ("می کنم", "می‌کنم"),
    ("می کنیم", "می‌کنیم"),
    ("می کنید", "می‌کنید"),
    ("می کنند", "می‌کنند"),
    ("می خواهم", "می‌خواهم"),
    ("می خواهد", "می‌خواهد"),
    ("می توان", "می‌توان"),
    ("نمی توان", "نمی‌توان"),
    ("می دهم", "می‌دهم"),
    ("می دهد", "می‌دهد"),
    ("تلویزیون نها دازاری", "تلویزیون ایران‌گرا"),
    ("تلیوزیون نها دازاری", "تلویزیون ایران‌گرا"),
    ("بانو میترا سال داری", "بانو میترا سالار"),
    ("پرمان پادشاهی", "پارمان پادشاهی"),
    ("پادشایی ایرانیان", "پادشاهی ایرانیان"),
    ("پادشای ایران", "پادشاهی ایران"),
    ("اوپوزو سیان", "اپوزیسیون"),
    ("ایرانگرها", "ایران‌گرا"),
    ("شایست سالاری", "شایسته‌سالاری"),
    ("سازمان دهی", "سازماندهی"),
    ("تلیوزیون", "تلویزیون"),
    ("برود بر شما", "درود بر شما"),
    ("با فرمت.", "درود."),
    ("خیل خوشه روام", "خیلی خوشحالم"),
    ("سپوسه از", "سپاس از"),
    ("پیروز خاهیت شود", "پیروز خواهید شد"),
    ("بر میگردیم", "برمی‌گردیم"),
    ("پرمان", "پارمان"),
]


def _normalize_chars(text: str) -> str:
    out = (text or "").translate(_ARABIC_YE).translate(_DIGIT)
    out = out.replace("\u200c\u200c", "\u200c")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\s+([،.,؛:!?؟])", r"\1", out)
    out = re.sub(r"([،.,؛:!?؟])(\S)", r"\1 \2", out)
    return out.strip()


def apply_term_repairs(text: str, extra: list[tuple[str, str]] | None = None) -> str:
    out = _normalize_chars(text)
    for src, dst in list(_TERM_REPAIRS) + list(extra or []):
        out = out.replace(src, dst)
    out = re.sub(r"(?<![می‌ن])\bمی ([آاأإءبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+)", r"می‌\1", out)
    out = re.sub(r"\bنمی ([آاأإءبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+)", r"نمی‌\1", out)
    return re.sub(r"[ \t]+", " ", out).strip()


def link_words(segments: list[dict], words: list[dict]) -> list[dict]:
    by_seg: dict[int, list[dict]] = {}
    for idx, word in enumerate(words):
        sid = int(word.get("segment_id", 0))
        row = dict(word)
        row["word_index"] = int(word.get("word_index", idx))
        by_seg.setdefault(sid, []).append(row)
    out = []
    for seg in segments:
        sid = int(seg["segment_id"])
        linked = by_seg.get(sid, [])
        start_i = linked[0]["word_index"] if linked else None
        end_i = linked[-1]["word_index"] if linked else None
        out.append({
            **seg,
            "word_index_start": start_i,
            "word_index_end": end_i,
            "words": [
                {
                    "text": w.get("text") or w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "probability": w.get("probability"),
                    "word_index": w.get("word_index"),
                }
                for w in linked
            ],
        })
    return out


def normalize_local(
    *,
    transcript: dict,
    words: list[dict],
    extra_repairs: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], int]:
    merged = []
    corrections = 0
    for idx, seg in enumerate(transcript.get("segments") or []):
        raw = str(seg.get("text") or "").strip()
        clean = apply_term_repairs(raw, extra_repairs)
        if clean != raw:
            corrections += 1
        merged.append({
            "segment_id": int(seg.get("id", idx)),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "raw_text": raw,
            "clean_text": clean,
        })
    return link_words(merged, words), corrections


def write_normalized(
    *,
    root: Path,
    stem: str,
    source_video: str,
    transcript_path: Path,
    words_path: Path,
    extra_repairs: list[tuple[str, str]] | None = None,
) -> dict:
    raw = read_json(transcript_path)
    words_payload = read_json(words_path)
    words = words_payload.get("words") or []
    segments, nfix = normalize_local(transcript=raw, words=words, extra_repairs=extra_repairs)
    payload = {
        "source_video": source_video,
        "source_transcript": str(transcript_path.as_posix()),
        "source_words": str(words_path.as_posix()),
        "language": raw.get("language"),
        "duration": raw.get("duration"),
        "normalization_model": "local-rule-corrections+word-linkage",
        "correction_count": nfix,
        "word_count": len(words),
        "segments": segments,
    }
    out_dir = root / "data" / "normalized_transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.normalized.json"
    md_path = out_dir / f"{stem}.normalized.md"
    write_json(json_path, payload)
    lines = [
        f"# Normalized transcript — {source_video}",
        "",
        f"- language: {payload.get('language')}",
        f"- duration: {ts(payload.get('duration') or 0)}",
        f"- corrections: {nfix}",
        f"- model: {payload.get('normalization_model')}",
        "",
    ]
    for seg in segments:
        changed = " ✱" if seg.get("clean_text") != seg.get("raw_text") else ""
        lines.append(f"## {seg['segment_id']}  {ts(seg['start'])} → {ts(seg['end'])}{changed}")
        if changed:
            lines.append(f"- raw: {seg.get('raw_text')}")
        lines.append(seg.get("clean_text") or "")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload
