from __future__ import annotations

from pathlib import Path

from .transcribe import transcribe
from .candidates import mine_candidates, save_candidates


def preprocess_video(video: Path, cfg: dict, *, force: bool = False) -> dict:
    stem = video.stem
    transcript_dir = Path(cfg["paths"]["transcripts"])
    candidate_dir = Path(cfg["paths"]["candidates"])
    transcript_json = transcript_dir / f"{stem}.transcript.json"
    transcript_srt = transcript_dir / f"{stem}.srt"
    candidates_json = candidate_dir / f"{stem}.candidates.json"
    work_packet = candidate_dir / f"{stem}.work_packet.md"

    if transcript_json.exists() and not force:
        import json
        transcript = json.loads(transcript_json.read_text(encoding="utf-8"))
        print(f"[reuse] {transcript_json}")
    else:
        print(f"[transcribe] {video}")
        transcript = transcribe(video, transcript_json, transcript_srt, cfg)

    candidates = mine_candidates(transcript, cfg)
    save_candidates(video.name, candidates, candidates_json, work_packet)
    return {
        "video": str(video),
        "transcript_json": str(transcript_json),
        "transcript_srt": str(transcript_srt),
        "candidates_json": str(candidates_json),
        "work_packet": str(work_packet),
        "candidate_count": len(candidates),
    }
