from pathlib import Path

from reels_factory.subtitles import build_rebased_srt


def test_rebased_srt_follows_cold_open_timeline(tmp_path: Path):
    transcript = {
        "segments": [
            {"start": 880.0, "end": 890.0, "text": "story start"},
            {"start": 907.2, "end": 912.7, "text": "hook line"},
            {"start": 920.0, "end": 930.0, "text": "payoff"},
        ]
    }
    clips = [
        {"label": "hook", "start": 907.2, "end": 912.7},
        {"label": "body_before_hook", "start": 880.0, "end": 907.2},
        {"label": "body_after_hook", "start": 912.7, "end": 934.0},
    ]
    out = tmp_path / "reel.srt"
    build_rebased_srt(transcript, clips, out)
    text = out.read_text(encoding="utf-8")
    assert "hook line" in text
    assert "story start" in text
    assert "payoff" in text
    # Hook is first on the reel timeline, starting at 00:00:00,000
    assert "00:00:00,000 --> 00:00:05,500" in text
    assert text.index("hook line") < text.index("story start")
    assert text.index("story start") < text.index("payoff")
