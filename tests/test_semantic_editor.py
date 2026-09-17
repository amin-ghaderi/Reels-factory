import json

import pytest

from pathlib import Path

from reels_factory.ai_normalizer import compact_segments_for_normalizer, merge_corrections, timestamps_unchanged
from reels_factory.cursor_ai import extract_json_payload
from reels_factory.plan_validate import PlanValidationError, estimated_duration, validate_semantic_plan
from reels_factory.render import _clips_from_plan
from reels_factory.semantic_editor import compact_normalized_for_editor, enrich_reel


def test_normalization_patch_merge_keeps_timestamps():
    original = [
        {"segment_id": 1, "start": 10.0, "end": 12.5, "raw_text": "حضینه موندن"},
        {"segment_id": 2, "start": 12.5, "end": 15.0, "raw_text": "درسته"},
    ]
    merged = merge_corrections(
        original,
        [
            {"segment_id": 1, "clean_text": "هزینه ماندن", "start": 0, "end": 999},
            {"segment_id": 99, "clean_text": "ignore"},
        ],
    )
    assert timestamps_unchanged(original, merged)
    assert merged[0]["clean_text"] == "هزینه ماندن"
    assert merged[1]["clean_text"] == "درسته"
    assert merged[0]["start"] == 10.0
    assert merged[0]["end"] == 12.5


def test_compact_segments_preserve_whisper_times():
    transcript = {
        "segments": [
            {"id": 7, "start": 623.12, "end": 628.44, "text": " hello "},
        ]
    }
    rows = compact_segments_for_normalizer(transcript)
    assert rows == [{"segment_id": 7, "start": 623.12, "end": 628.44, "raw_text": "hello"}]


def test_compact_normalized_for_editor_uses_clean_text():
    rows = compact_normalized_for_editor({
        "segments": [
            {"segment_id": 1, "start": 1.5, "end": 2.0, "raw_text": "x", "clean_text": "y"},
        ]
    })
    assert rows[0]["text"] == "y"
    assert rows[0]["start"] == "00:00:01.500"


def test_extract_json_from_fenced_agent_output():
    text = "Here you go:\n```json\n{\"corrections\": []}\n```\n"
    assert extract_json_payload(text) == {"corrections": []}


def test_semantic_plan_non_chronological_playback():
    plan = {
        "reel_id": "demo",
        "source_video": "data/inbox/nabz-18.mp4",
        "segments": [
            {"start": "00:10:55.000", "end": "00:11:00.000", "role": "hook"},
            {"start": "00:10:24.000", "end": "00:10:35.000", "role": "setup"},
            {"start": "00:11:00.000", "end": "00:11:05.000", "role": "payoff"},
        ],
        "avoid_hook_repeat": True,
    }
    validate_semantic_plan(plan)
    clips = _clips_from_plan(plan)
    assert [c["label"] for c in clips] == ["hook", "setup", "payoff"]
    assert clips[0]["start"] > clips[1]["start"]
    assert estimated_duration(plan) == pytest.approx(21.0)


def test_segment_validation_rejects_start_after_end():
    plan = {
        "segments": [
            {"start": 10.0, "end": 9.0, "role": "hook"},
        ]
    }
    with pytest.raises(PlanValidationError, match="start >= end"):
        validate_semantic_plan(plan)


def test_segments_outside_source_duration():
    plan = {
        "segments": [
            {"start": 10.0, "end": 200.0, "role": "hook"},
        ]
    }
    with pytest.raises(PlanValidationError, match="past source duration"):
        validate_semantic_plan(plan, source_duration=100.0)


def test_overlap_detection():
    plan = {
        "segments": [
            {"start": 10.0, "end": 20.0, "role": "setup"},
            {"start": 15.0, "end": 25.0, "role": "core"},
        ]
    }
    with pytest.raises(PlanValidationError, match="overlap"):
        validate_semantic_plan(plan)


def test_duplicate_and_hook_repeat_protection():
    dup = {
        "segments": [
            {"start": 10.0, "end": 12.0, "role": "hook"},
            {"start": 10.0, "end": 12.0, "role": "core"},
        ],
        "avoid_hook_repeat": True,
    }
    with pytest.raises(PlanValidationError, match="duplicate"):
        validate_semantic_plan(dup)

    replay = {
        "segments": [
            {"start": 50.0, "end": 55.0, "role": "hook"},
            {"start": 10.0, "end": 20.0, "role": "setup"},
            {"start": 52.0, "end": 60.0, "role": "payoff"},
        ],
        "avoid_hook_repeat": True,
    }
    with pytest.raises(PlanValidationError, match="hook repeated"):
        validate_semantic_plan(replay)


def test_old_edit_plan_backward_compatibility():
    plan = {
        "hook": {"enabled": True, "start": 30.0, "end": 35.0},
        "body": {"start": 10.0, "end": 50.0},
        "avoid_hook_repeat": False,
    }
    clips = _clips_from_plan(plan)
    assert [c["label"] for c in clips] == ["hook", "body"]
    assert clips[1]["start"] == 10.0


def test_multi_segment_concatenation_order():
    plan = {
        "segments": [
            {"start": 100.0, "end": 104.0, "role": "hook"},
            {"start": 20.0, "end": 28.0, "role": "core"},
            {"start": 80.0, "end": 86.0, "role": "payoff"},
        ],
        "avoid_hook_repeat": True,
    }
    validate_semantic_plan(plan)
    clips = _clips_from_plan(plan)
    assert [round(c["end"] - c["start"], 3) for c in clips] == [4.0, 8.0, 6.0]
    assert clips[0]["start"] == 100.0
    assert clips[1]["start"] == 20.0


def test_enrich_reel_adds_durations():
    reel = {
        "reel_id": "x",
        "segments": [
            {"start": "00:00:10.000", "end": "00:00:15.000", "role": "hook"},
            {"start": "00:00:01.000", "end": "00:00:04.000", "role": "setup"},
        ],
        "avoid_hook_repeat": True,
        "context_integrity": "high",
    }
    out = enrich_reel(reel, source_video="data/inbox/nabz-18.mp4", source_duration=100.0)
    assert out["estimated_duration_s"] == 8.0
    assert out["original_source_span_s"] == 14.0


def test_normalizer_mock_and_cache(tmp_path):
    from reels_factory.ai_normalizer import normalize_transcript

    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "clip.transcript.json").write_text(
        json.dumps({
            "source_video": "clip.mp4",
            "language": "fa",
            "duration": 20.0,
            "segments": [
                {"id": 1, "start": 1.0, "end": 3.0, "text": "حضینه موندن"},
                {"id": 2, "start": 3.0, "end": 5.0, "text": "درسته"},
            ],
        }),
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    cfg = {
        "paths": {
            "transcripts": transcripts,
            "normalized_transcripts": tmp_path / "norm",
        },
        "ai_editor": {"normalization": {"model": "composer-2.5"}},
    }

    def fake_invoke(prompt, **kwargs):
        assert "Transcript" in prompt
        return json.dumps({"corrections": [{"segment_id": 1, "clean_text": "هزینه ماندن"}]})

    result = normalize_transcript(video, cfg, root=Path(__file__).resolve().parents[1], invoke=fake_invoke)
    assert result["correction_count"] == 1
    assert result["segments"][0]["start"] == 1.0
    assert result["segments"][0]["clean_text"] == "هزینه ماندن"
    cached = normalize_transcript(
        video,
        cfg,
        root=Path(__file__).resolve().parents[1],
        invoke=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    assert cached["correction_count"] == 1


def test_semantic_editor_cache_and_mock_call(tmp_path):
    from reels_factory.semantic_editor import semantic_edit

    normalized = {
        "source_video": "data/inbox/nabz-18.mp4",
        "language": "fa",
        "duration": 100.0,
        "segments": [
            {"segment_id": 1, "start": 10.0, "end": 14.0, "raw_text": "a", "clean_text": "a"},
            {"segment_id": 2, "start": 40.0, "end": 45.0, "raw_text": "b", "clean_text": "b"},
        ],
    }
    cfg = {
        "paths": {
            "normalized_transcripts": tmp_path,
            "semantic_plans": tmp_path / "plans",
        },
        "ai_editor": {"semantic_editor": {"model": "grok-4.6"}},
    }
    (tmp_path / "clip.normalized.json").write_text(json.dumps(normalized), encoding="utf-8")

    def fake_invoke(prompt, **kwargs):
        assert "Normalized transcript" in prompt
        assert kwargs["model"] == "grok-4.6"
        return json.dumps({
            "clearest_example_reel_id": "clip_semantic_01",
            "reels": [{
                "reel_id": "clip_semantic_01",
                "title": "t",
                "editorial_summary": "s",
                "segments": [
                    {"start": "00:00:40.000", "end": "00:00:45.000", "role": "hook", "why": "later line"},
                    {"start": "00:00:10.000", "end": "00:00:14.000", "role": "setup", "why": "context"},
                ],
                "avoid_hook_repeat": True,
                "context_integrity": "high",
            }],
        })

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    result = semantic_edit(video, cfg, root=Path(__file__).resolve().parents[1], invoke=fake_invoke)
    assert result["reel_count"] == 1
    cached = semantic_edit(video, cfg, root=Path(__file__).resolve().parents[1], invoke=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    assert cached["cached"] is True
