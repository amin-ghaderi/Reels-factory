from __future__ import annotations

import math
import re
from pathlib import Path
from .utils import ts, write_json

HOOK_TERMS = {
    # Persian
    "اما", "ولی", "واقعیت", "حقیقت", "اشتباه", "بزرگترین", "بزرگ‌ترین", "هیچکس", "هیچ‌کس",
    "چرا", "چطور", "چگونه", "راز", "باور", "فکر", "مشکل", "مهم", "جالب", "تصور", "اگر", "اگه",
    "هیچ وقت", "هیچ‌وقت", "نمی‌دونی", "نمیدونی", "نمی‌دونید", "نمیدونید",
    # English
    "but", "truth", "mistake", "biggest", "nobody", "why", "how", "secret", "believe",
    "problem", "important", "imagine", "if", "never", "actually", "here's", "here’s",
    # Swedish / German common hook words
    "varför", "hur", "misstag", "sanningen", "aldrig", "warum", "wie", "fehler", "wahrheit", "nie",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06FF'-]+", text.lower(), flags=re.UNICODE)


def _score_window(text: str, duration: float, target: float) -> tuple[float, dict]:
    toks = _tokens(text)
    if not toks:
        return -999.0, {"reason": "empty"}

    opening = " ".join(toks[:18])
    all_text = " ".join(toks)
    hook_hits = sum(1 for term in HOOK_TERMS if term in opening)
    global_hits = sum(1 for term in HOOK_TERMS if term in all_text)
    questions = text.count("?") + text.count("؟")
    exclaims = text.count("!")
    numbers = len(re.findall(r"\d", text))
    unique_ratio = len(set(toks)) / max(1, len(toks))

    duration_score = max(0.0, 1.0 - abs(duration - target) / max(target, 1.0)) * 2.0
    score = (
        duration_score
        + min(3.0, hook_hits * 1.25)
        + min(1.5, global_hits * 0.25)
        + min(1.5, questions * 0.75)
        + min(0.8, exclaims * 0.4)
        + min(0.5, numbers * 0.08)
        + unique_ratio
    )

    if len(toks) < 35:
        score -= 1.2
    if len(toks) > 220:
        score -= 0.6
    if text.strip().endswith((".", "!", "?", "؟", "؛")):
        score += 0.35

    reasons = []
    if hook_hits:
        reasons.append(f"{hook_hits} hook term(s) near opening")
    if questions:
        reasons.append("question-driven")
    if duration_score > 1.5:
        reasons.append("good short-form duration")
    if unique_ratio > 0.7:
        reasons.append("high lexical variety")
    return score, {"reasons": reasons, "token_count": len(toks)}


def _overlap_ratio(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return inter / max(shorter, 1e-9)


def mine_candidates(transcript: dict, cfg: dict) -> list[dict]:
    segs = transcript.get("segments", [])
    ccfg = cfg["candidate_mining"]
    min_s = float(ccfg["min_seconds"])
    target_s = float(ccfg["target_seconds"])
    max_s = float(ccfg["max_seconds"])
    max_candidates = int(ccfg["max_candidates"])
    max_overlap = float(ccfg.get("max_overlap_ratio", 0.45))

    pool = []
    for i in range(len(segs)):
        text_parts = []
        for j in range(i, len(segs)):
            start = float(segs[i]["start"])
            end = float(segs[j]["end"])
            duration = end - start
            if duration > max_s:
                break
            text_parts.append(segs[j]["text"].strip())
            if duration < min_s:
                continue
            text = " ".join(x for x in text_parts if x)
            score, meta = _score_window(text, duration, target_s)
            pool.append({
                "start": start,
                "end": end,
                "duration": round(duration, 3),
                "start_ts": ts(start),
                "end_ts": ts(end),
                "score": round(score, 4),
                "text": text,
                "segment_start_id": int(segs[i]["id"]),
                "segment_end_id": int(segs[j]["id"]),
                "signals": meta,
            })

    pool.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for cand in pool:
        if all(_overlap_ratio(cand, existing) <= max_overlap for existing in selected):
            cand = dict(cand)
            cand["candidate_id"] = f"C{len(selected)+1:02d}"
            selected.append(cand)
            if len(selected) >= max_candidates:
                break
    return selected


def save_work_packet(source_name: str, candidates: list[dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reels Factory — Candidate Packet: {source_name}",
        "",
        "Use this packet as the cheap first-pass input. Select only candidates worth deeper review.",
        "For each selected candidate, you may choose a cold-open hook from anywhere inside the source transcript.",
        "",
    ]
    for c in candidates:
        lines += [
            f"## {c['candidate_id']} — {c['start_ts']} → {c['end_ts']} — {c['duration']:.1f}s — heuristic {c['score']:.2f}",
            "",
            c["text"],
            "",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def save_candidates(source_name: str, candidates: list[dict], out_json: Path, out_md: Path) -> None:
    write_json(out_json, {"source": source_name, "candidates": candidates})
    save_work_packet(source_name, candidates, out_md)
