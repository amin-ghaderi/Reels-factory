from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .faces import LayoutPlan, plan_clip_layout, plan_locked_layout
from .framing import layout_from_framing_profile, resolve_framing_profile
from .plan_validate import plan_has_segments, validate_semantic_plan
from .subtitles import build_rebased_srt
from .utils import parse_timestamp, read_json, require_binary, run


def _clips_from_plan(plan: dict) -> list[dict]:
    if plan_has_segments(plan):
        clips = []
        for idx, seg in enumerate(plan.get("segments") or []):
            clips.append({
                "label": str(seg.get("role") or f"seg_{idx:02d}"),
                "start": parse_timestamp(seg["start"]),
                "end": parse_timestamp(seg["end"]),
            })
        if not clips:
            raise ValueError("Edit plan segments[] is empty.")
        for clip in clips:
            if clip["end"] <= clip["start"]:
                raise ValueError(f"Invalid clip range: {clip}")
        return clips

    clips = []
    hook = plan.get("hook")
    body = plan.get("body")
    hook_enabled = bool(hook) and hook.get("enabled", True)
    if hook_enabled:
        clips.append({
            "label": "hook",
            "start": parse_timestamp(hook["start"]),
            "end": parse_timestamp(hook["end"]),
        })
    if body:
        b0, b1 = parse_timestamp(body["start"]), parse_timestamp(body["end"])
        avoid_repeat = bool(plan.get("avoid_hook_repeat", False))
        if avoid_repeat and hook_enabled:
            h0, h1 = parse_timestamp(hook["start"]), parse_timestamp(hook["end"])
            # If the hook is contained in the body, omit its duplicate occurrence.
            # Cold open first, then story/context before the hook, then payoff after it.
            if b0 <= h0 and h1 <= b1 and h1 > h0:
                if h0 - b0 > 0.15:
                    clips.append({"label": "body_before_hook", "start": b0, "end": h0})
                if b1 - h1 > 0.15:
                    clips.append({"label": "body_after_hook", "start": h1, "end": b1})
            else:
                clips.append({"label": "body", "start": b0, "end": b1})
        else:
            clips.append({"label": "body", "start": b0, "end": b1})
    if not clips:
        raise ValueError("Edit plan needs at least a body or enabled hook.")
    for clip in clips:
        if clip["end"] <= clip["start"]:
            raise ValueError(f"Invalid clip range: {clip}")
    return clips


def _should_burn_captions(rcfg: dict) -> bool:
    """Reel renders default to no burned captions; plan caption blocks are ignored."""
    return bool(rcfg.get("burn_captions", False))


def _resolve_source_video(source: Path, plan_path: Path) -> Path:
    if source.exists():
        return source.resolve()
    if not source.is_absolute():
        cwd_try = (Path.cwd() / source).resolve()
        if cwd_try.exists():
            return cwd_try
        for parent in plan_path.resolve().parents:
            candidate = (parent / source).resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Source video not found: {source}")


def resolve_transcript_for_plan(plan_path: Path, plan: dict, cfg: dict) -> Path:
    """Prefer a refined window transcript for the plan's candidate, else the full transcript."""
    source = Path(plan.get("source_video") or "")
    stem = source.stem or plan_path.stem
    candidate_id = plan.get("candidate_id")
    refined_dir = Path(cfg["paths"]["refined_transcripts"])
    if candidate_id:
        refined = refined_dir / f"{stem}.{candidate_id}.refined.json"
        if refined.exists():
            return refined
    transcripts_dir = Path(cfg["paths"]["transcripts"])
    full = transcripts_dir / f"{stem}.transcript.json"
    normalized_dir = Path(cfg["paths"].get("normalized_transcripts") or "")
    if normalized_dir:
        normalized = normalized_dir / f"{stem}.normalized.json"
        if normalized.exists():
            return normalized
    if full.exists():
        return full
    raise FileNotFoundError(
        f"No transcript found for {plan_path.name} (candidate_id={candidate_id!r})"
    )


def choose_reel_layout(
    source: Path,
    cfg: dict,
    rcfg: dict,
    clips: list[dict],
) -> LayoutPlan | None:
    """Prefer a source-level framing profile over per-reel face detection."""
    profile = resolve_framing_profile(source, cfg)
    if profile is not None:
        return layout_from_framing_profile(profile, rcfg)
    if _lock_crops_for_reel(rcfg, clips):
        return plan_locked_layout(
            source,
            [(clip["start"], clip["end"]) for clip in clips],
            rcfg,
        )
    return None


def _lock_crops_for_reel(rcfg: dict, clips: list[dict]) -> bool:
    requested = str(rcfg.get("layout", "stacked_faces")).strip().lower()
    if requested not in {"stacked_faces", "single_face"}:
        return False
    if not bool(rcfg.get("lock_face_crops", True)):
        return False
    return len(clips) > 1


def _print_locked_crops(layout: LayoutPlan) -> None:
    if layout.panel_a is not None:
        p = layout.panel_a
        print(
            f"[layout] Person A safe ROI "
            f"x={p.x:.1f} y={p.y:.1f} w={p.w:.1f} h={p.h:.1f}"
        )
    if layout.panel_b is not None:
        p = layout.panel_b
        print(
            f"[layout] Person B safe ROI "
            f"x={p.x:.1f} y={p.y:.1f} w={p.w:.1f} h={p.h:.1f}"
        )
    if layout.safety_margin:
        print(f"[layout] panel safety margin={layout.safety_margin:.0%}")
    if layout.top is not None:
        t = layout.top
        print(
            f"[layout] locked Person A/top crop "
            f"x={t.x:.1f} y={t.y:.1f} w={t.w:.1f} h={t.h:.1f}"
        )
    if layout.bottom is not None:
        b = layout.bottom
        print(
            f"[layout] locked Person B/bottom crop "
            f"x={b.x:.1f} y={b.y:.1f} w={b.w:.1f} h={b.h:.1f}"
        )
    if layout.single is not None and layout.top is None:
        s = layout.single
        print(
            f"[layout] locked single crop "
            f"x={s.x:.1f} y={s.y:.1f} w={s.w:.1f} h={s.h:.1f}"
        )


def render_reel(
    plan_path: Path,
    transcript_path: Path,
    cfg: dict,
    *,
    output_path: Path | None = None,
) -> Path:
    ffmpeg = require_binary("ffmpeg")
    plan = read_json(plan_path)
    transcript = read_json(transcript_path)
    source = Path(plan.get("source_video") or transcript["source_video"])
    source = _resolve_source_video(source, plan_path)

    if plan_has_segments(plan):
        duration = None
        try:
            duration = float((transcript or {}).get("duration") or 0) or None
        except (TypeError, ValueError):
            duration = None
        validate_semantic_plan(plan, source_duration=duration)

    rcfg = cfg["render"]
    crf = str(rcfg["video_crf"])
    audio_bitrate = str(rcfg["audio_bitrate"])
    clips = _clips_from_plan(plan)

    output_dir = Path(cfg["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reel_id = plan.get("reel_id") or plan_path.stem
    final_path = Path(output_path) if output_path else (output_dir / f"{reel_id}.mp4")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reels_factory_") as td:
        temp = Path(td)
        parts = []
        layouts: list[LayoutPlan] = []
        locked_layout = choose_reel_layout(source, cfg, rcfg, clips)
        if locked_layout is not None:
            print(
                f"[layout] locked crops for all {len(clips)} clips "
                f"mode={locked_layout.mode} method={locked_layout.method} "
                f"avg_faces={locked_layout.avg_faces:.2f} "
                f"samples={len(locked_layout.face_counts)}"
            )
            _print_locked_crops(locked_layout)
        for idx, clip in enumerate(clips):
            if locked_layout is not None:
                layout = locked_layout
            else:
                layout = plan_clip_layout(source, clip["start"], clip["end"], rcfg)
            layouts.append(layout)
            print(
                f"[layout] clip={clip['label']} mode={layout.mode} "
                f"method={layout.method} avg_faces={layout.avg_faces:.2f} "
                f"samples={len(layout.face_counts)} "
                f"locked={'yes' if locked_layout is not None else 'no'}"
            )
            part = temp / f"part_{idx:02d}.mp4"
            run([
                ffmpeg, "-y",
                "-ss", str(clip["start"]), "-to", str(clip["end"]),
                "-i", str(source),
                "-filter_complex", layout.filter_complex,
                "-map", "[v]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                "-c:a", "aac", "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(part),
            ])
            parts.append(part)

        all_counts = [n for layout in layouts for n in layout.face_counts]
        avg_faces = (sum(all_counts) / len(all_counts)) if all_counts else 0.0
        method = next((lay.method for lay in layouts if lay.method != "none"), "none")
        modes = ",".join(lay.mode for lay in layouts)
        print(f"[layout] detection={method} avg_faces={avg_faces:.2f} modes={modes}")

        concat_file = temp / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{str(p).replace(os.sep, '/')}'" for p in parts),
            encoding="utf-8",
        )
        joined = temp / "joined.mp4"
        run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(joined)])

        if _should_burn_captions(rcfg):
            srt = temp / f"{reel_id}.srt"
            build_rebased_srt(transcript, clips, srt)
            font = rcfg.get("subtitle_font", "Tahoma")
            font_size = int(rcfg.get("subtitle_font_size", 18))
            margin_v = int(rcfg.get("subtitle_margin_v", 110))
            outline = int(rcfg.get("subtitle_outline", 2))
            sub_path = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
            style = f"FontName={font},FontSize={font_size},Alignment=2,MarginV={margin_v},Outline={outline},Shadow=0"
            sub_filter = f"subtitles='{sub_path}':charenc='UTF-8':force_style='{style}'"
            run([
                ffmpeg, "-y", "-i", str(joined),
                "-vf", sub_filter,
                "-af", f"loudnorm=I={rcfg.get('loudness_target_lufs', -16)}:TP=-1.5:LRA=11",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                "-c:a", "aac", "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(final_path),
            ])
        else:
            print("[captions] disabled")
            run([
                ffmpeg, "-y", "-i", str(joined),
                "-af", f"loudnorm=I={rcfg.get('loudness_target_lufs', -16)}:TP=-1.5:LRA=11",
                "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(final_path),
            ])
    return final_path
