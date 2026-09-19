from __future__ import annotations

from pathlib import Path

from .ai_normalizer import normalize_transcript
from .program_map import map_program
from .package_final import packaging_enabled, package_production_reel
from .qa_reel_editor import edit_program_qa_units, qa_plan_paths
from .refine import resolve_video_path
from .render import render_reel
from .transcribe import transcribe
from .utils import read_json, write_json


def _transcript_paths(cfg: dict, stem: str) -> dict[str, Path]:
    transcripts = Path(cfg["paths"]["transcripts"])
    return {
        "json": transcripts / f"{stem}.transcript.json",
        "srt": transcripts / f"{stem}.srt",
    }


def ensure_transcript(
    video: Path,
    cfg: dict,
    *,
    root: Path,
    force: bool = False,
    transcribe_fn=None,
) -> dict:
    """Reuse Whisper output when present. --force does not redo transcription."""
    video = resolve_video_path(video, root)
    paths = _transcript_paths(cfg, video.stem)
    if paths["json"].exists():
        # Whisper is a local stage, not an AI editor call. --force does not redo it.
        print(f"[make-reels] transcript cache hit {paths['json']}")
        return read_json(paths["json"])
    print(f"[make-reels] transcribing {video.name}")
    caller = transcribe_fn or transcribe
    return caller(video, paths["json"], paths["srt"], cfg)


def render_config_for_qa(cfg: dict) -> dict:
    """stacked_faces + captions off; keep current crop and the rest of render cfg."""
    out = dict(cfg)
    render = dict(cfg.get("render") or {})
    render["burn_captions"] = False
    render["layout"] = "stacked_faces"
    out["render"] = render
    pack = dict(cfg.get("packaging") or {})
    pack["burn_subtitles"] = False
    pack["write_srt"] = False
    out["packaging"] = pack
    return out


def _metadata_path(cfg: dict, stem: str, root: Path) -> Path:
    meta_dir = cfg.get("paths", {}).get("metadata")
    if meta_dir:
        return Path(meta_dir) / f"{stem}.json"
    return Path(root) / "data" / "metadata" / f"{stem}.json"


def _output_mp4(cfg: dict, reel_id: str) -> Path:
    return Path(cfg["paths"]["output"]) / f"{reel_id}.mp4"


def make_reels(
    video: Path,
    cfg: dict,
    *,
    root: Path,
    force: bool = False,
    invoke_normalize=None,
    invoke_map=None,
    invoke_editor=None,
    transcribe_fn=None,
    render_fn=None,
    package_fn=None,
) -> dict:
    """
    Production path:
    video → Whisper → normalize → program map → Q&A editor → stacked_faces MP4
    → cover + whoosh package → Instagram caption.
    Subtitles are not generated.
    """
    video = resolve_video_path(video, root)
    stem = video.stem
    transcript = ensure_transcript(video, cfg, root=root, force=force, transcribe_fn=transcribe_fn)
    normalized = normalize_transcript(
        video, cfg, root=root, force=force, invoke=invoke_normalize,
    )
    program_map = map_program(video, cfg, root=root, force=force, invoke=invoke_map)
    edited = edit_program_qa_units(
        video,
        cfg,
        root=root,
        force=force,
        invoke=invoke_editor,
        program_map=program_map,
        normalized=normalized,
        transcript=transcript,
    )

    render_cfg = render_config_for_qa(cfg)
    renderer = render_fn or render_reel
    packer = package_fn or package_production_reel
    rendered = []
    skipped_render = []
    out_dir = Path(cfg["paths"]["qa_plans"])
    meta_path = _metadata_path(render_cfg, stem, root)
    for plan in edited["plans"]:
        unit_id = plan["source_unit"]
        plan_path = qa_plan_paths(out_dir, stem, unit_id)["json"]
        mp4 = _output_mp4(cfg, plan["reel_id"])
        if mp4.exists() and plan.get("cached") and not force:
            print(f"[make-reels] render cache hit {mp4.name}")
            skipped_render.append(str(mp4))
            out = mp4
            cached = True
        else:
            transcript_path = _transcript_paths(cfg, stem)["json"]
            print(f"[make-reels] rendering {plan_path.name}")
            out = renderer(plan_path, transcript_path, render_cfg)
            cached = False
        packaged = None
        if packaging_enabled(render_cfg):
            if not meta_path.is_file():
                print(f"[make-reels] skip package {plan['reel_id']}: no guest metadata")
            else:
                final_dir = Path(render_cfg["paths"].get("final") or (root / "data" / "final"))
                final_mp4 = final_dir / f"{plan['reel_id']}_final.mp4"
                caption = final_dir / f"{plan['reel_id']}_instagram.txt"
                if final_mp4.exists() and not force:
                    print(f"[make-reels] package cache hit {final_mp4.name}")
                    packaged = {
                        "final_mp4": str(final_mp4),
                        "instagram_txt": str(caption) if caption.exists() else None,
                        "cached": True,
                    }
                else:
                    print(f"[make-reels] packaging {plan['reel_id']}")
                    packaged = packer(
                        plan_path=plan_path,
                        reel_mp4=out,
                        cfg=render_cfg,
                        root=root,
                        metadata_path=meta_path,
                    )
                    packaged["cached"] = False
        rendered.append({
            "unit_id": unit_id,
            "plan": str(plan_path),
            "mp4": str(out),
            "cached": cached,
            "package": packaged,
        })

    summary = {
        "video": str(video),
        "transcript": str(_transcript_paths(cfg, stem)["json"]),
        "normalized": str(Path(cfg["paths"]["normalized_transcripts"]) / f"{stem}.normalized.json"),
        "program_map": str(Path(cfg["paths"]["program_maps"]) / f"{stem}.program_map.json"),
        "plan_count": len(edited["plans"]),
        "skipped_units": edited["index"]["skipped"],
        "rendered": rendered,
        "render_cache_hits": skipped_render,
    }
    write_json(Path(cfg["paths"]["qa_plans"]) / f"{stem}.make_reels.json", summary)
    return summary
