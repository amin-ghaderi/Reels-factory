from pathlib import Path

from reels_factory.candidates import save_candidates


def test_save_candidates_writes_work_packet(tmp_path: Path):
    candidates = [{
        "candidate_id": "C01",
        "start_ts": "00:14:40.000",
        "end_ts": "00:15:34.000",
        "duration": 54.0,
        "score": 4.2,
        "text": "اما بزرگ‌ترین اشتباه این بود که فکر می‌کردیم مسئله فقط قیمت است.",
    }]
    json_path = tmp_path / "episode.candidates.json"
    md_path = tmp_path / "episode.work_packet.md"
    save_candidates("episode.mp4", candidates, json_path, md_path)
    assert json_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "# Reels Factory — Candidate Packet: episode.mp4" in text
    assert "C01" in text
    assert "00:14:40.000 → 00:15:34.000" in text
    assert "بزرگ‌ترین اشتباه" in text
