from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .cover import _open_variable, _rtl, resolve_cover_font
from .package_final import INTRO_S, _duration
from .render import _clips_from_plan
from .utils import parse_timestamp, read_json, require_binary, run, srt_ts, write_json

VAZIRMATN_VAR = Path("assets/fonts/vazirmatn/Vazirmatn-wght.ttf")
FONT_WEIGHT = 600
FONT_SIZE = 46
CANVAS = (1080, 1920)
# Stable split-band: between Person A (top) and Person B (bottom).
SUBTITLE_Y = 900
MAX_TEXT_WIDTH = 720
SIDE_MARGIN = 90
MAX_LINES = 2
MIN_CUE_S = 0.85
SHORT_CUE_CHARS = 12

_PHRASE_SPLIT = re.compile(r"(?<=[،؛.!?؟])\s+")


def polish_subtitle_text(text: str) -> str:
    """Minor subtitle cleanup. Does not paraphrase."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.replace("محل می‌شد", "حل می‌شد")
    text = text.replace("موقعاتی", "موقتی")
    if text.startswith("اروپا "):
        text = text[len("اروپا "):].lstrip()
    return text.strip(" ،")


def _seg_text(seg: dict) -> str:
    return str(seg.get("clean_text") or seg.get("text") or seg.get("raw_text") or "").strip()


def _seg_range(seg: dict) -> tuple[float, float]:
    start = seg.get("start", seg.get("source_start"))
    end = seg.get("end", seg.get("source_end"))
    return parse_timestamp(start), parse_timestamp(end)


def collect_source_words(transcript: dict) -> list[tuple[float, float, str]]:
    rows = []
    for seg in transcript.get("segments") or []:
        for word in seg.get("words") or []:
            try:
                rows.append((float(word["start"]), float(word["end"]), str(word.get("word") or "").strip()))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def trim_text_to_window(
    text: str,
    seg_start: float,
    seg_end: float,
    win_start: float,
    win_end: float,
    source_words: list[tuple[float, float, str]] | None = None,
) -> str:
    """Keep only the words whose source audio sits inside the edit window."""
    words = text.split()
    if not words:
        return ""
    dur = max(0.001, seg_end - seg_start)
    in_seg = [
        w for w in (source_words or [])
        if w[1] > seg_start + 0.02 and w[0] < seg_end - 0.02
    ]
    if in_seg:
        before = sum(1 for w in in_seg if w[1] <= win_start + 0.04)
        after = sum(1 for w in in_seg if w[0] >= win_end - 0.04)
        scale = len(words) / max(1, len(in_seg))
        i0 = min(len(words), max(0, math.floor(before * scale + 1e-6)))
        i1 = max(i0, len(words) - math.floor(after * scale + 1e-6))
        return " ".join(words[i0:i1]).strip()
    start_frac = max(0.0, (win_start - seg_start) / dur)
    end_frac = min(1.0, (win_end - seg_start) / dur)
    i0 = min(len(words), max(0, math.floor(start_frac * len(words) + 1e-6)))
    i1 = min(len(words), max(i0, math.ceil(end_frac * len(words) - 1e-6)))
    return " ".join(words[i0:i1]).strip()


def extract_plan_cues(
    plan: dict,
    normalized: dict,
    *,
    source_words: list[tuple[float, float, str]] | None = None,
    intro_s: float = INTRO_S,
    root: Path | None = None,
) -> list[dict]:
    clips = _clips_from_plan(plan)
    raw: list[dict] = []
    cursor = 0.0
    for clip in clips:
        c0, c1 = float(clip["start"]), float(clip["end"])
        for seg in normalized.get("segments") or []:
            s0, s1 = _seg_range(seg)
            overlap = min(s1, c1) - max(s0, c0)
            if overlap < 0.08:
                continue
            w0, w1 = max(s0, c0), min(s1, c1)
            text = polish_subtitle_text(trim_text_to_window(_seg_text(seg), s0, s1, w0, w1, source_words))
            if not text:
                continue
            raw.append({
                "start": intro_s + cursor + (w0 - c0),
                "end": intro_s + cursor + (w1 - c0),
                "source_start": w0,
                "source_end": w1,
                "role": clip["label"],
                "text": text,
            })
        cursor += c1 - c0
    return chunk_cues(raw, root=root)


def _line_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    visual = _rtl(text) if text else ""
    box = font.getbbox(visual)
    return int(box[2] - box[0])


def wrap_subtitle_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int = MAX_TEXT_WIDTH) -> list[str]:
    words = [w for w in text.split() if w]
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and _line_width(font, trial) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return _balance_orphan_line(lines, font, max_width)


def _balance_orphan_line(lines: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if len(lines) < 2:
        return lines
    last_words = lines[-1].split()
    prev_words = lines[-2].split()
    if len(last_words) == 1 and len(prev_words) > 3:
        moved = prev_words[-1]
        trial_prev = " ".join(prev_words[:-1])
        trial_last = moved + " " + lines[-1]
        if _line_width(font, trial_last) <= max_width and trial_prev:
            lines[-2] = trial_prev
            lines[-1] = trial_last
    return lines


def _measure_font(root: Path | None) -> ImageFont.FreeTypeFont:
    base = Path(root) if root is not None else Path(".")
    path = base / VAZIRMATN_VAR
    if path.is_file():
        return _open_variable(path, FONT_SIZE, "semibold")
    try:
        font, _ = resolve_cover_font("name", root=base, size=FONT_SIZE)
        return font
    except Exception:
        return ImageFont.load_default()


def chunk_cues(cues: list[dict], *, root: Path | None = None) -> list[dict]:
    font = _measure_font(root)
    out: list[dict] = []
    for cue in cues:
        pieces = [p.strip() for p in _PHRASE_SPLIT.split(cue["text"]) if p.strip()]
        if not pieces:
            pieces = [cue["text"]]
        chunks: list[list[str]] = []
        for piece in pieces:
            lines = wrap_subtitle_lines(piece, font)
            if not lines:
                continue
            for i in range(0, len(lines), MAX_LINES):
                block = lines[i:i + MAX_LINES]
                if len(block) > MAX_LINES:
                    raise ValueError("subtitle wrap exceeded two lines")
                chunks.append(block)
        if not chunks:
            continue
        weights = [sum(len(line) for line in block) + 1 for block in chunks]
        total = sum(weights)
        t0, t1 = float(cue["start"]), float(cue["end"])
        span = t1 - t0
        if span < 0.12:
            continue
        cursor = t0
        src0, src1 = float(cue["source_start"]), float(cue["source_end"])
        src_cursor = src0
        for idx, block in enumerate(chunks):
            share = weights[idx] / total
            dur = span * share
            end = t1 if idx == len(chunks) - 1 else min(t1, cursor + dur)
            src_end = src1 if idx == len(chunks) - 1 else src_cursor + (src1 - src0) * share
            if end - cursor < MIN_CUE_S and idx < len(chunks) - 1:
                # fold a tiny leftover into the next piece by extending
                end = min(t1, cursor + MIN_CUE_S)
            out.append({
                "start": round(cursor, 3),
                "end": round(end, 3),
                "source_start": round(src_cursor, 3),
                "source_end": round(src_end, 3),
                "role": cue["role"],
                "lines": block,
                "text": " ".join(block),
            })
            cursor = end
            src_cursor = src_end
    return merge_short_cues(out, font)


def merge_short_cues(cues: list[dict], font: ImageFont.FreeTypeFont) -> list[dict]:
    """Join isolated leftover words with the previous cue when it still fits on 2 lines."""
    if not cues:
        return []
    merged: list[dict] = []
    for cue in cues:
        text = " ".join(cue["lines"]).strip()
        short = len(text) <= SHORT_CUE_CHARS or (len(cue["lines"]) == 1 and len(text.split()) <= 1)
        if merged and cue["role"] == merged[-1]["role"] and short:
            prev = merged[-1]
            combined = (prev["text"] + " " + text).strip()
            wrapped = wrap_subtitle_lines(combined, font)
            if len(wrapped) <= MAX_LINES:
                prev["lines"] = wrapped
                prev["text"] = combined
                prev["end"] = cue["end"]
                prev["source_end"] = cue["source_end"]
                continue
            if len(prev["lines"]) == MAX_LINES:
                trial = (prev["lines"][-1] + " " + text).strip()
                if _line_width(font, trial) <= MAX_TEXT_WIDTH:
                    prev["lines"][-1] = trial
                    prev["text"] = combined
                    prev["end"] = cue["end"]
                    prev["source_end"] = cue["source_end"]
                    continue
        merged.append(dict(cue))
    i = 0
    while i < len(merged) - 1:
        text = merged[i]["text"].strip()
        short = len(text) <= SHORT_CUE_CHARS or (len(merged[i]["lines"]) == 1 and len(text.split()) <= 2)
        nxt = merged[i + 1]
        if short and merged[i]["role"] == nxt["role"]:
            combined = (text + " " + nxt["text"]).strip()
            wrapped = wrap_subtitle_lines(combined, font)
            if len(wrapped) <= MAX_LINES:
                nxt["lines"] = wrapped
                nxt["text"] = combined
                nxt["start"] = merged[i]["start"]
                nxt["source_start"] = merged[i]["source_start"]
                del merged[i]
                continue
        i += 1
    cleaned = [c for c in merged if c["end"] - c["start"] >= 0.18 and c["lines"]]
    for row in cleaned:
        if len(row["lines"]) > MAX_LINES:
            raise ValueError(f"cue exceeds {MAX_LINES} lines: {row['lines']}")
    return cleaned


def write_srt(cues: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for idx, cue in enumerate(cues, 1):
        body = "\n".join(cue["lines"])
        parts.append(f"{idx}\n{srt_ts(cue['start'])} --> {srt_ts(cue['end'])}\n{body}\n")
    path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    return path


def _draw_cue_png(lines: list[str], dest: Path, font: ImageFont.FreeTypeFont) -> None:
    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    line_h = font.getbbox("آ")[3] - font.getbbox("آ")[1]
    gap = int(line_h * 0.18)
    widths = [_line_width(font, line) for line in lines]
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * gap
    max_w = max(widths) if widths else 0
    box_w = max_w + 56
    box_h = total_h + 28
    box_x = (CANVAS[0] - box_w) // 2
    box_y = int(SUBTITLE_Y - box_h / 2)
    draw.rounded_rectangle(
        (box_x, box_y, box_x + box_w, box_y + box_h),
        radius=18,
        fill=(0, 0, 0, 118),
    )
    cy = box_y + (box_h - total_h) // 2
    for line, lw in zip(lines, widths):
        visual = _rtl(line)
        lx = (CANVAS[0] - lw) // 2
        draw.text(
            (lx, cy),
            visual,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 210),
        )
        cy += line_h + gap
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def burn_cues_onto_mp4(
    source_mp4: Path,
    cues: list[dict],
    output_mp4: Path,
    *,
    root: Path,
) -> Path:
    ffmpeg = require_binary("ffmpeg")
    font = _measure_font(root)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pub_subs_") as td:
        temp = Path(td)
        pngs = []
        for idx, cue in enumerate(cues):
            png = temp / f"cue_{idx:03d}.png"
            _draw_cue_png(cue["lines"], png, font)
            pngs.append(png)
        script = temp / "filter.txt"
        lines = []
        prev = "0:v"
        for idx, cue in enumerate(cues):
            nxt = f"v{idx}"
            start, end = cue["start"], cue["end"]
            src = "[0:v]" if idx == 0 else f"[{prev}]"
            lines.append(
                f"{src}[{idx + 1}:v]overlay=0:0:eof_action=repeat:enable='between(t,{start:.3f},{end:.3f})'[{nxt}]"
            )
            prev = nxt
        script.write_text(";\n".join(lines) + "\n", encoding="utf-8")
        duration = max(0.1, _duration(source_mp4))
        cmd = [ffmpeg, "-y", "-i", str(source_mp4)]
        for png in pngs:
            cmd.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", str(png)])
        cmd.extend([
            "-filter_complex_script", str(script),
            "-map", f"[{prev}]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-t", f"{duration:.3f}",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_mp4),
        ])
        run(cmd)
    return output_mp4


def validate_publication_cues(cues: list[dict], *, intro_s: float = INTRO_S) -> list[str]:
    errors = []
    if not cues:
        errors.append("no subtitle cues")
        return errors
    if cues[0]["start"] + 1e-6 < intro_s:
        errors.append(f"first cue starts during cover: {cues[0]['start']}")
    for cue in cues:
        if cue["start"] < intro_s - 1e-6:
            errors.append(f"cue during cover at {cue['start']}")
        if len(cue.get("lines") or []) > MAX_LINES:
            errors.append(f"more than {MAX_LINES} lines: {cue['lines']}")
        if cue["end"] <= cue["start"]:
            errors.append(f"inverted cue {cue}")
    for a, b in zip(cues, cues[1:]):
        if b["start"] + 1e-6 < a["start"]:
            errors.append("cues not in time order")
    return errors


Q01_INSTAGRAM = """\
از نظر محسن سعیدی، بحران آب ایران با یک سال بارندگی خوب تمام نمی‌شود.
حتی رسیدن بارش به میانگین درازمدت به‌معنای حل مسئله نیست؛
هر گشایشی هم بیشتر موقتی است و در همه جای کشور دیده نمی‌شود.

#بحران_آب #منابع_آب #ایران
"""


def build_q01_publication_test(root: Path) -> dict:
    root = Path(root)
    plan = read_json(root / "data" / "qa_plans" / "17-05.Q01.qa.json")
    normalized = read_json(root / "data" / "normalized_transcripts" / "17-05.normalized.json")
    transcript = read_json(root / "data" / "transcripts" / "17-05.transcript.json")
    source_mp4 = root / "data" / "final" / "1705_qa_Q01_final.mp4"
    if not source_mp4.is_file():
        raise FileNotFoundError(f"Approved Q01 package missing: {source_mp4}")

    cues = extract_plan_cues(
        plan,
        normalized,
        source_words=collect_source_words(transcript),
        intro_s=INTRO_S,
        root=root,
    )
    errors = validate_publication_cues(cues, intro_s=INTRO_S)
    if errors:
        raise ValueError("subtitle validation failed: " + "; ".join(errors))

    srt_path = root / "data" / "final" / "1705_qa_Q01_subtitles.srt"
    json_path = root / "data" / "final" / "1705_qa_Q01_subtitles.json"
    caption_path = root / "data" / "final" / "1705_qa_Q01_instagram.txt"
    out_mp4 = root / "data" / "final" / "1705_qa_Q01_final_subtitled.mp4"
    write_srt(cues, srt_path)
    font_path = root / VAZIRMATN_VAR
    write_json(json_path, {
        "reel_id": "1705_qa_Q01",
        "source_mp4": str(source_mp4.as_posix()),
        "intro_s": INTRO_S,
        "font_family": "Vazirmatn",
        "font_weight": "SemiBold",
        "font_file": str(font_path.as_posix()),
        "font_size": FONT_SIZE,
        "vertical_center_px": SUBTITLE_Y,
        "max_lines": MAX_LINES,
        "cue_count": len(cues),
        "cues": cues,
    })
    caption_path.write_text(Q01_INSTAGRAM.strip() + "\n", encoding="utf-8")
    burn_cues_onto_mp4(source_mp4, cues, out_mp4, root=root)
    return {
        "cues": cues,
        "srt": srt_path,
        "json": json_path,
        "caption": caption_path,
        "mp4": out_mp4,
        "font": "Vazirmatn SemiBold",
        "vertical_center_px": SUBTITLE_Y,
        "errors": [],
    }
