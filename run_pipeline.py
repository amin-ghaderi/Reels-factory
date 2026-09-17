from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reels_factory.config import load_config
from reels_factory.pipeline import preprocess_video
from reels_factory.refine import parse_refine_request, refine_window, refine_windows
from reels_factory.render import render_reel
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
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
