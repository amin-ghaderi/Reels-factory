from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .faces import LayoutPlan, plan_clip_layout
from .subtitles import build_rebased_srt
from .utils import parse_timestamp, read_json, require_binary, run


def _clips_from_plan(plan: dict) -> list[dict]:
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


def render_reel(plan_path: Path, transcript_path: Path, cfg: dict) -> Path:
    ffmpeg = require_binary("ffmpeg")
    plan = read_json(plan_path)
    transcript = read_json(transcript_path)
    source = Path(plan.get("source_video") or transcript["source_video"])
    source = _resolve_source_video(source, plan_path)

    rcfg = cfg["render"]
    crf = str(rcfg["video_crf"])
    audio_bitrate = str(rcfg["audio_bitrate"])
    clips = _clips_from_plan(plan)

    output_dir = Path(cfg["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reel_id = plan.get("reel_id") or plan_path.stem
    final_path = output_dir / f"{reel_id}.mp4"

    with tempfile.TemporaryDirectory(prefix="reels_factory_") as td:
        temp = Path(td)
        parts = []
        layouts: list[LayoutPlan] = []
        for idx, clip in enumerate(clips):
            layout = plan_clip_layout(source, clip["start"], clip["end"], rcfg)
            layouts.append(layout)
            print(
                f"[layout] clip={clip['label']} mode={layout.mode} "
                f"method={layout.method} avg_faces={layout.avg_faces:.2f} "
                f"samples={len(layout.face_counts)}"
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
