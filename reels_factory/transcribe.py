from __future__ import annotations

from pathlib import Path
from .utils import write_json, srt_ts


def transcribe(video: Path, out_json: Path, out_srt: Path, cfg: dict) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    tcfg = cfg["transcription"]
    model = WhisperModel(
        tcfg["model"],
        device=tcfg.get("device", "auto"),
        compute_type=tcfg.get("compute_type", "int8"),
    )
    segments_iter, info = model.transcribe(
        str(video),
        language=tcfg.get("language"),
        vad_filter=bool(tcfg.get("vad_filter", True)),
        beam_size=int(tcfg.get("beam_size", 5)),
        word_timestamps=True,
    )

    segments = []
    for i, seg in enumerate(segments_iter):
        words = []
        for w in (seg.words or []):
            words.append({
                "start": float(w.start),
                "end": float(w.end),
                "word": w.word,
                "probability": float(w.probability),
            })
        segments.append({
            "id": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })

    payload = {
        "source_video": str(video.resolve()),
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or (segments[-1]["end"] if segments else 0.0)),
        "segments": segments,
    }
    write_json(out_json, payload)

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    with out_srt.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            f.write(f"{idx}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n{seg['text']}\n\n")
    return payload
