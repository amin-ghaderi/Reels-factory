from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reels_factory.ai_normalizer import normalize_transcript
from reels_factory.config import load_config
from reels_factory.cursor_ai import CursorAIError
from reels_factory.make_reels import make_reels
from reels_factory.pipeline import preprocess_video
from reels_factory.program_map import map_program
from reels_factory.refine import parse_refine_request, refine_window, refine_windows
from reels_factory.render import render_reel, resolve_transcript_for_plan
from reels_factory.semantic_editor import semantic_edit
from reels_factory.utils import parse_timestamp, read_json

ROOT = Path(__file__).resolve().parent


def cmd_setup(args):
    cfg_path = ROOT / "config.yaml"
    example = ROOT / "config.example.yaml"
    if not cfg_path.exists():
        shutil.copy2(example, cfg_path)
        print(f"Created {cfg_path}")
    else:
        print(f"Already exists: {cfg_path}")
    cfg = load_config(ROOT, cfg_path)
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    print("Folder structure ready.")


def cmd_preprocess(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video).resolve()
    result = preprocess_video(video, cfg, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_inbox(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    inbox = Path(cfg["paths"]["inbox"])
    exts = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
    videos = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not videos:
        print(f"No videos found in {inbox}")
        return
    for video in videos:
        try:
            result = preprocess_video(video, cfg, force=args.force)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[error] {video.name}: {exc}")


def cmd_render(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    plan = Path(args.plan)
    transcript = Path(args.transcript)
    if not plan.is_absolute():
        plan = (ROOT / plan).resolve()
    if not transcript.is_absolute():
        transcript = (ROOT / transcript).resolve()
    out = render_reel(plan, transcript, cfg)
    print(f"Rendered: {out}")


def cmd_render_all(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    if args.plans:
        plans = []
        for item in args.plans:
            path = Path(item)
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            plans.append(path)
    else:
        plans_dir = Path(args.dir) if args.dir else Path(cfg["paths"]["edit_plans"])
        if not plans_dir.is_absolute():
            plans_dir = (ROOT / plans_dir).resolve()
        plans = sorted(plans_dir.glob(args.pattern))
    if not plans:
        raise FileNotFoundError("No edit plans found to render")

    succeeded = []
    failed = []
    for plan_path in plans:
        if not plan_path.exists():
            failed.append((plan_path.name, f"missing file: {plan_path}"))
            print(f"[error] {plan_path.name}: missing file")
            continue
        try:
            plan = read_json(plan_path)
            transcript = resolve_transcript_for_plan(plan_path, plan, cfg)
            print(f"[render-all] {plan_path.name} transcript={transcript.name}")
            out = render_reel(plan_path, transcript, cfg)
            succeeded.append((plan_path.stem, out))
            print(f"[ok] {plan_path.name} -> {out}")
        except Exception as exc:
            failed.append((plan_path.stem, str(exc)))
            print(f"[error] {plan_path.name}: {exc}")

    print()
    print(f"Successful: {len(succeeded)}")
    for name, out in succeeded:
        print(f"  {name}: {out}")
    print(f"Failed: {len(failed)}")
    for name, err in failed:
        print(f"  {name}: {err}")
    if failed:
        raise SystemExit(1)


def cmd_refine(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    from reels_factory.refine import load_refine_model, probe_duration, resolve_video_path

    video = resolve_video_path(Path(args.video), ROOT)
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    duration = probe_duration(video)
    model, model_name = load_refine_model(cfg, model_name=args.model)
    refine_window(
        video,
        cfg,
        candidate_id=args.id,
        start=start,
        end=end,
        model=model,
        model_name=model_name,
        force=args.force,
        duration=duration,
        index=1,
        total=1,
    )


def cmd_refine_batch(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    request = Path(args.request)
    if not request.is_absolute():
        request = (ROOT / request).resolve()
    if not request.exists():
        raise FileNotFoundError(f"Refine request not found: {request}")
    parsed = parse_refine_request(read_json(request), ROOT)
    refine_windows(
        parsed["video"],
        parsed["windows"],
        cfg,
        force=args.force,
        model_name=args.model,
        write_packet=True,
    )


def cmd_normalize_ai(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    video = Path(args.video)
    try:
        result = normalize_transcript(video, cfg, root=ROOT, force=args.force)
    except CursorAIError as exc:
        print(f"[normalize-ai] FAILED — existing local workflow is unchanged. {exc}")
        raise SystemExit(1) from exc
    print(json.dumps({
        "source_transcript": result.get("source_transcript"),
        "normalized": str(Path(cfg["paths"]["normalized_transcripts"]) / f"{Path(args.video).stem}.normalized.json"),
        "correction_count": result.get("correction_count"),
        "model": result.get("normalization_model"),
    }, ensure_ascii=False, indent=2))


def cmd_semantic_edit(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    video = Path(args.video)
    try:
        result = semantic_edit(video, cfg, root=ROOT, force=args.force)
    except CursorAIError as exc:
        print(f"[semantic-edit] FAILED — heuristic candidate miner is still available. {exc}")
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_map_program(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    video = Path(args.video)
    try:
        result = map_program(video, cfg, root=ROOT, force=args.force)
    except CursorAIError as exc:
        print(f"[map-program] FAILED — existing local workflow is unchanged. {exc}")
        raise SystemExit(1) from exc
    print(json.dumps({
        "program_map": str(Path(cfg["paths"]["program_maps"]) / f"{Path(args.video).stem}.program_map.json"),
        "unit_count": result.get("unit_count"),
        "model": result.get("mapper_model"),
    }, ensure_ascii=False, indent=2))


def cmd_make_reels(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    video = Path(args.video)
    try:
        result = make_reels(video, cfg, root=ROOT, force=args.force)
    except CursorAIError as exc:
        print(f"[make-reels] FAILED — existing local workflow is unchanged. {exc}")
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_render_semantic(args):
    cfg = load_config(ROOT, Path(args.config) if args.config else None)
    plan = Path(args.plan)
    if not plan.is_absolute():
        plan = (ROOT / plan).resolve()
    payload = read_json(plan)
    transcript = Path(args.transcript) if args.transcript else resolve_transcript_for_plan(plan, payload, cfg)
    if not transcript.is_absolute():
        transcript = (ROOT / transcript).resolve()
    out = render_reel(plan, transcript, cfg)
    print(f"Rendered: {out}")


def build_parser():
    p = argparse.ArgumentParser(description="Low-cost local Reels Factory V1")
    p.add_argument("--config", help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Create config.yaml and working folders")
    s.set_defaults(func=cmd_setup)

    pp = sub.add_parser("preprocess", help="Transcribe one video and mine candidate reels")
    pp.add_argument("video")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_preprocess)

    ib = sub.add_parser("inbox", help="Process every video in data/inbox")
    ib.add_argument("--force", action="store_true")
    ib.set_defaults(func=cmd_inbox)

    r = sub.add_parser("render", help="Render a reel from an edit-plan JSON")
    r.add_argument("--plan", required=True)
    r.add_argument("--transcript", required=True)
    r.set_defaults(func=cmd_render)

    ra = sub.add_parser("render-all", help="Render every edit-plan JSON (or listed plans)")
    ra.add_argument("plans", nargs="*", help="Optional explicit plan JSON paths")
    ra.add_argument("--dir", help="Edit-plan directory (default: paths.edit_plans)")
    ra.add_argument("--pattern", default="*.json", help="Glob inside --dir when no plans are listed")
    ra.set_defaults(func=cmd_render_all)

    rf = sub.add_parser("refine", help="Second-pass transcribe one shortlisted time window")
    rf.add_argument("--video", required=True)
    rf.add_argument("--start", required=True)
    rf.add_argument("--end", required=True)
    rf.add_argument("--id", required=True)
    rf.add_argument("--force", action="store_true")
    rf.add_argument("--model", help="Override refine model (default: large-v3-turbo, else medium)")
    rf.set_defaults(func=cmd_refine)

    rb = sub.add_parser("refine-batch", help="Second-pass transcribe windows from a JSON request")
    rb.add_argument("--request", required=True)
    rb.add_argument("--force", action="store_true")
    rb.add_argument("--model", help="Override refine model (default: large-v3-turbo, else medium)")
    rb.set_defaults(func=cmd_refine_batch)

    na = sub.add_parser("normalize-ai", help="Context-aware ASR cleanup via Cursor (does not overwrite Whisper)")
    na.add_argument("--video", required=True)
    na.add_argument("--force", action="store_true")
    na.set_defaults(func=cmd_normalize_ai)

    se = sub.add_parser("semantic-edit", help="Full-transcript semantic Reel editor via Cursor Grok")
    se.add_argument("--video", required=True)
    se.add_argument("--force", action="store_true")
    se.set_defaults(func=cmd_semantic_edit)

    mp = sub.add_parser("map-program", help="Map full-program Q&A / conversation structure via Cursor Grok")
    mp.add_argument("--video", required=True)
    mp.add_argument("--force", action="store_true")
    mp.set_defaults(func=cmd_map_program)

    mr = sub.add_parser("make-reels", help="Whisper → normalize → program map → Q&A Reel Editor → stacked_faces MP4")
    mr.add_argument("--video", required=True)
    mr.add_argument("--force", action="store_true", help="Redo AI stages (normalize, map, Q&A editor) and re-render")
    mr.set_defaults(func=cmd_make_reels)

    rs = sub.add_parser("render-semantic", help="Render a multi-segment semantic plan (no captions)")
    rs.add_argument("--plan", required=True)
    rs.add_argument("--transcript", help="Optional transcript JSON; defaults to normalized transcript")
    rs.set_defaults(func=cmd_render_semantic)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
