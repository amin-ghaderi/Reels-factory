from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .subtitles import build_rebased_srt
from .utils import read_json, require_binary, run


def _clips_from_plan(plan: dict) -> list[dict]:
    clips = []
    hook = plan.get("hook")
    body = plan.get("body")
    hook_enabled = bool(hook) and hook.get("enabled", True)
    if hook_enabled:
        clips.append({"label": "hook", "start": float(hook["start"]), "end": float(hook["end"])})
    if body:
        b0, b1 = float(body["start"]), float(body["end"])
        avoid_repeat = bool(plan.get("avoid_hook_repeat", False))
        if avoid_repeat and hook_enabled:
            h0, h1 = float(hook["start"]), float(hook["end"])
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


def render_reel(plan_path: Path, transcript_path: Path, cfg: dict) -> Path:
    ffmpeg = require_binary("ffmpeg")
    plan = read_json(plan_path)
    transcript = read_json(transcript_path)
    source = Path(plan.get("source_video") or transcript["source_video"])
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    rcfg = cfg["render"]
    width, height = int(rcfg["width"]), int(rcfg["height"])
    fps = int(rcfg["fps"])
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
        vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps}"
        for idx, clip in enumerate(clips):
            part = temp / f"part_{idx:02d}.mp4"
            run([
                ffmpeg, "-y",
                "-ss", str(clip["start"]), "-to", str(clip["end"]),
                "-i", str(source),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                "-c:a", "aac", "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(part),
            ])
            parts.append(part)

        concat_file = temp / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{str(p).replace(os.sep, '/')}'" for p in parts),
            encoding="utf-8",
        )
        joined = temp / "joined.mp4"
        run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(joined)])

        burn = bool(plan.get("captions", {}).get("enabled", rcfg.get("burn_captions", True)))
        if burn:
            srt = temp / f"{reel_id}.srt"
            build_rebased_srt(transcript, clips, srt)
            font = plan.get("captions", {}).get("font", rcfg.get("subtitle_font", "Tahoma"))
            font_size = int(plan.get("captions", {}).get("font_size", rcfg.get("subtitle_font_size", 18)))
            margin_v = int(plan.get("captions", {}).get("margin_v", rcfg.get("subtitle_margin_v", 110)))
            outline = int(plan.get("captions", {}).get("outline", rcfg.get("subtitle_outline", 2)))
            # Escape backslashes/colons for ffmpeg's subtitles filter.
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
            run([
                ffmpeg, "-y", "-i", str(joined),
                "-af", f"loudnorm=I={rcfg.get('loudness_target_lufs', -16)}:TP=-1.5:LRA=11",
                "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(final_path),
            ])
    return final_path
