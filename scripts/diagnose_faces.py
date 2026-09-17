from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reels_factory.faces import diagnose_clip_faces
from reels_factory.render import _clips_from_plan, _resolve_source_video
from reels_factory.utils import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Save face-detection diagnostics for an edit plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", help="Output directory for annotated frames")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = (ROOT / plan_path).resolve()
    plan = read_json(plan_path)
    source = Path(plan.get("source_video") or "")
    source = _resolve_source_video(source, plan_path)
    clips = _clips_from_plan(plan)
    start = min(c["start"] for c in clips)
    end = max(c["end"] for c in clips)
    out_dir = Path(args.out) if args.out else ROOT / "data" / "output" / "face_diagnostics" / plan_path.stem
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    summary = diagnose_clip_faces(source, start, end, out_dir, frame_count=args.frames)
    print(f"Wrote {out_dir / 'report.json'}")
    return 0 if summary.get("two_person_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
