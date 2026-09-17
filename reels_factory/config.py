from __future__ import annotations

from pathlib import Path
import yaml

DEFAULT_CONFIG = {
    "paths": {
        "inbox": "data/inbox",
        "transcripts": "data/transcripts",
        "candidates": "data/candidates",
        "edit_plans": "data/edit_plans",
        "output": "data/output",
        "approved": "data/approved",
        "rejected": "data/rejected",
        "refined_transcripts": "data/refined_transcripts",
        "refine_requests": "data/refine_requests",
    },
    "transcription": {
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "language": None,
        "vad_filter": True,
        "beam_size": 5,
    },
    "refine": {
        "model": "large-v3-turbo",
        "fallback_model": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "language": "fa",
        "vad_filter": True,
        "beam_size": 5,
    },
    "candidate_mining": {
        "min_seconds": 20,
        "target_seconds": 38,
        "max_seconds": 62,
        "max_candidates": 24,
        "max_overlap_ratio": 0.45,
    },
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "video_crf": 20,
        "audio_bitrate": "160k",
        "loudness_target_lufs": -16,
        "burn_captions": True,
        "subtitle_font": "Tahoma",
        "subtitle_font_size": 18,
        "subtitle_margin_v": 110,
        "subtitle_outline": 2,
        "layout": "stacked_faces",
        "face_sample_interval": 0.5,
        "face_score_threshold": 0.55,
        "face_min_size": 40,
        "face_track_min_hit_ratio": 0.4,
        "face_smooth_alpha": 0.35,
    },
}


def _merge(base: dict, extra: dict) -> dict:
    result = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(root: Path, config_path: Path | None = None) -> dict:
    path = config_path or (root / "config.yaml")
    user_cfg = {}
    if path.exists():
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULT_CONFIG, user_cfg)
    for key, rel in list(cfg["paths"].items()):
        p = Path(rel)
        cfg["paths"][key] = p if p.is_absolute() else (root / p)
    return cfg
