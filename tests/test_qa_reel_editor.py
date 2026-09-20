import json

from pathlib import Path

from reels_factory.make_reels import make_reels, render_config_for_qa
from reels_factory.qa_reel_editor import (
    edit_program_qa_units,
    evaluate_control_gate,
    is_closing_message,
    is_qa_unit,
    is_transition_unit,
    parse_qa_editor_payload,
    qa_reel_id,
    slice_normalized_for_unit,
    slice_words_for_unit,
    unit_bounds,
)
from reels_factory.utils import read_json, ts

ROOT = Path(__file__).resolve().parents[1]


def _unit(uid="Q01", **kwargs):
    base = {
        "unit_id": uid,
        "topic": "nuclear file",
        "question_type": "main_question",
        "references_to_previous_unit": [],
        "question": {
            "start": "00:00:10.000",
            "end": "00:00:20.000",
            "core_start": "00:00:12.000",
            "core_end": "00:00:20.000",
            "text": "concede or escalate?",
        },
        "answer": {
            "start": "00:00:22.000",
            "end": "00:00:40.000",
            "core_start": "00:00:24.000",
            "status": "complete",
            "summary": "will not declare; compromise chance very low",
        },
    }
    base.update(kwargs)
    return base


def _transition(uid="T01"):
    return {
        "unit_id": uid,
        "topic": "open",
        "question": None,
        "answer": None,
        "question_type": "transition",
        "start": "00:00:01.000",
        "end": "00:00:05.000",
        "notes": "intro",
    }


def test_transition_units_are_not_qa():
    assert is_transition_unit(_transition()) is True
    assert is_qa_unit(_transition()) is False
    assert is_qa_unit(_unit()) is True


def test_reel_id_does_not_collide_with_experiments():
    assert qa_reel_id("nabz-18", "Q04") == "nabz18_qa_Q04"
    assert qa_reel_id("nabz-18", "Q04") != "nabz18_q04"


def test_slice_window_keeps_words_for_speaker_splits():
    normalized = {
        "segments": [
            {"segment_id": 1, "start": 9.0, "end": 20.0, "raw_text": "q", "clean_text": "question"},
            {"segment_id": 2, "start": 22.0, "end": 30.0, "raw_text": "a", "clean_text": "answer"},
            {"segment_id": 9, "start": 80.0, "end": 90.0, "raw_text": "other", "clean_text": "other"},
        ]
    }
    transcript = {
        "segments": [{
            "id": 1,
            "start": 9.0,
            "end": 20.0,
            "text": "question",
            "words": [
                {"start": 12.0, "end": 12.4, "word": "concede"},
                {"start": 19.5, "end": 20.0, "word": "or"},
            ],
        }]
    }
    start, end = unit_bounds(_unit())
    segs = slice_normalized_for_unit(normalized, start, end)
    words = slice_words_for_unit(transcript, start, end)
    assert [row["id"] for row in segs] == [1, 2]
    assert words[0]["word"] == "concede"


def test_parse_skip_and_plan_payload():
    skipped = parse_qa_editor_payload(
        {"skip": True, "reason": "cannot stand alone"},
        reel_id="show_qa_Q02",
        unit_id="Q02",
    )
    assert skipped["skip"] is True
    plan = parse_qa_editor_payload(
        {
            "segments": [
                {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core"},
                {"start": "00:00:24.000", "end": "00:00:30.000", "role": "central_answer"},
            ],
            "hook_used": False,
        },
        reel_id="show_qa_Q01",
        unit_id="Q01",
    )
    assert plan["skip"] is False
    assert plan["reel_id"] == "show_qa_Q01"
    assert plan["hook_used"] is False


def test_edit_program_skips_transitions_and_caches(tmp_path):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"0")
    cfg = {
        "paths": {
            "transcripts": tmp_path / "transcripts",
            "normalized_transcripts": tmp_path / "norm",
            "program_maps": tmp_path / "maps",
            "qa_plans": tmp_path / "qa_plans",
        },
        "ai_editor": {"qa_reel_editor": {"model": "grok-4.6"}},
    }
    normalized = {
        "source_video": str(video),
        "language": "fa",
        "duration": 80.0,
        "segments": [
            {"segment_id": 1, "start": 10.0, "end": 20.0, "raw_text": "q", "clean_text": "concede or escalate?"},
            {"segment_id": 2, "start": 22.0, "end": 40.0, "raw_text": "a", "clean_text": "will not declare"},
        ],
    }
    transcript = {
        "source_video": str(video),
        "duration": 80.0,
        "segments": [{"id": 1, "start": 10.0, "end": 20.0, "text": "q", "words": []}],
    }
    program_map = {
        "units": [
            _transition(),
            _unit("Q01"),
            _unit(
                "Q02",
                question={
                    "start": "00:00:50.000",
                    "end": "00:00:55.000",
                    "core_start": "00:00:50.000",
                    "core_end": "00:00:55.000",
                    "text": "and then?",
                },
                answer={
                    "start": "00:00:55.000",
                    "end": "00:01:05.000",
                    "core_start": "00:00:55.000",
                    "status": "complete",
                },
            ),
        ]
    }
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        assert "Do not force hooks" in prompt
        if "show_qa_Q02" in prompt:
            return json.dumps({"skip": True, "reason": "cannot stand alone"})
        return json.dumps({
            "title": "nuclear choice",
            "editorial_summary": "chronological Q04-style reel",
            "hook_used": False,
            "hook_rationale": "or-question must be heard first",
            "segments": [
                {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core", "why": "the choice"},
                {"start": "00:00:24.000", "end": "00:00:30.000", "role": "central_answer", "why": "stance"},
                {"start": "00:00:30.000", "end": "00:00:36.000", "role": "payoff", "why": "qualified close"},
            ],
            "avoid_hook_repeat": True,
            "context_integrity": "high",
        })

    result = edit_program_qa_units(
        video,
        cfg,
        root=ROOT,
        invoke=fake_invoke,
        program_map=program_map,
        normalized=normalized,
        transcript=transcript,
    )
    assert len(calls) == 2
    assert len(result["plans"]) == 1
    assert result["plans"][0]["reel_id"] == "show_qa_Q01"
    assert result["plans"][0]["hook_used"] is False
    reasons = {row["unit_id"]: row["reason"] for row in result["index"]["skipped"]}
    assert reasons["T01"] == "transition/intro/outro"
    assert "cannot stand alone" in reasons["Q02"]

    cached = edit_program_qa_units(
        video,
        cfg,
        root=ROOT,
        invoke=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
        program_map=program_map,
        normalized=normalized,
        transcript=transcript,
    )
    assert cached["plans"][0]["cached"] is True
    assert len(calls) == 2


def _pipeline_cfg(tmp_path: Path) -> dict:
    return {
        "paths": {
            "transcripts": tmp_path / "transcripts",
            "normalized_transcripts": tmp_path / "norm",
            "program_maps": tmp_path / "maps",
            "qa_plans": tmp_path / "qa_plans",
            "output": tmp_path / "output",
        },
        "ai_editor": {
            "normalization": {"model": "composer-2.5"},
            "program_mapper": {"model": "grok-4.6"},
            "qa_reel_editor": {"model": "grok-4.6"},
        },
        "render": {
            "burn_captions": True,
            "layout": "single_face",
            "face_crop": {"zoom": 1.32},
        },
    }


def test_make_reels_mocked_pipeline_and_render_settings(tmp_path):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"0")
    cfg = _pipeline_cfg(tmp_path)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "show.transcript.json").write_text(
        json.dumps({
            "source_video": str(video),
            "language": "fa",
            "duration": 80.0,
            "segments": [
                {"id": 1, "start": 10.0, "end": 20.0, "text": "concede or escalate", "words": []},
                {"id": 2, "start": 22.0, "end": 40.0, "text": "will not declare", "words": []},
            ],
        }),
        encoding="utf-8",
    )
    counts = {"norm": 0, "map": 0, "edit": 0, "render": 0, "transcribe": 0}

    def fake_transcribe(*_args, **_kwargs):
        counts["transcribe"] += 1
        raise AssertionError("Whisper should be cached")

    def fake_norm(prompt, **kwargs):
        counts["norm"] += 1
        return json.dumps({"corrections": []})

    def fake_map(prompt, **kwargs):
        counts["map"] += 1
        assert "units" in (ROOT / "prompts" / "program_mapper.md").read_text(encoding="utf-8")
        return json.dumps({
            "program_duration": "00:01:20.000",
            "units": [_transition(), _unit("Q01")],
        })

    def fake_edit(prompt, **kwargs):
        counts["edit"] += 1
        assert "Do not force hooks" in prompt
        return json.dumps({
            "title": "nuclear choice",
            "editorial_summary": "Q04-style chronological reel",
            "hook_used": False,
            "segments": [
                {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core", "why": "question"},
                {"start": "00:00:24.000", "end": "00:00:36.000", "role": "central_answer", "why": "answer"},
            ],
            "avoid_hook_repeat": True,
            "context_integrity": "high",
        })

    def fake_render(plan_path, transcript_path, render_cfg):
        counts["render"] += 1
        assert render_cfg["render"]["burn_captions"] is False
        assert render_cfg["render"]["layout"] == "stacked_faces"
        assert render_cfg["render"]["face_crop"]["zoom"] == 1.32
        plan = read_json(plan_path)
        out = Path(cfg["paths"]["output"]) / f"{plan['reel_id']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    result = make_reels(
        video,
        cfg,
        root=ROOT,
        transcribe_fn=fake_transcribe,
        invoke_normalize=fake_norm,
        invoke_map=fake_map,
        invoke_editor=fake_edit,
        render_fn=fake_render,
    )
    assert counts == {"norm": 1, "map": 1, "edit": 1, "render": 1, "transcribe": 0}
    assert result["plan_count"] == 1
    assert result["rendered"][0]["mp4"].endswith("show_qa_Q01.mp4")
    assert result["rendered"][0]["package"] is None
    assert any(row["unit_id"] == "T01" for row in result["skipped_units"])

    again = make_reels(
        video,
        cfg,
        root=ROOT,
        transcribe_fn=fake_transcribe,
        invoke_normalize=lambda *a, **k: (_ for _ in ()).throw(AssertionError("norm")),
        invoke_map=lambda *a, **k: (_ for _ in ()).throw(AssertionError("map")),
        invoke_editor=lambda *a, **k: (_ for _ in ()).throw(AssertionError("edit")),
        render_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("render")),
    )
    assert again["rendered"][0]["cached"] is True
    assert counts["render"] == 1


def test_make_reels_force_redoes_ai_not_whisper(tmp_path):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"0")
    cfg = _pipeline_cfg(tmp_path)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "show.transcript.json").write_text(
        json.dumps({
            "source_video": str(video),
            "language": "fa",
            "duration": 80.0,
            "segments": [{"id": 1, "start": 10.0, "end": 20.0, "text": "q", "words": []}],
        }),
        encoding="utf-8",
    )
    counts = {"norm": 0, "map": 0, "edit": 0, "transcribe": 0}

    def fake_norm(prompt, **kwargs):
        counts["norm"] += 1
        return json.dumps({"corrections": []})

    def fake_map(prompt, **kwargs):
        counts["map"] += 1
        return json.dumps({"program_duration": "00:01:20.000", "units": [_unit("Q01")]})

    def fake_edit(prompt, **kwargs):
        counts["edit"] += 1
        return json.dumps({
            "hook_used": False,
            "segments": [
                {"start": "00:00:10.000", "end": "00:00:18.000", "role": "question_core"},
                {"start": "00:00:24.000", "end": "00:00:32.000", "role": "payoff"},
            ],
            "avoid_hook_repeat": True,
            "context_integrity": "high",
        })

    def fake_render(plan_path, transcript_path, render_cfg):
        plan = read_json(plan_path)
        out = Path(cfg["paths"]["output"]) / f"{plan['reel_id']}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    make_reels(
        video, cfg, root=ROOT, force=True,
        transcribe_fn=lambda *a, **k: counts.__setitem__("transcribe", counts["transcribe"] + 1),
        invoke_normalize=fake_norm, invoke_map=fake_map, invoke_editor=fake_edit, render_fn=fake_render,
    )
    make_reels(
        video, cfg, root=ROOT, force=True,
        transcribe_fn=lambda *a, **k: counts.__setitem__("transcribe", counts["transcribe"] + 1),
        invoke_normalize=fake_norm, invoke_map=fake_map, invoke_editor=fake_edit, render_fn=fake_render,
    )
    assert counts["transcribe"] == 0
    assert counts["norm"] == 2
    assert counts["map"] == 2
    assert counts["edit"] == 2


def test_render_config_forces_stacked_faces_captions_off():
    cfg = render_config_for_qa({
        "render": {"burn_captions": True, "layout": "center_crop", "face_crop": {"zoom": 1.32}},
        "packaging": {"burn_subtitles": True, "write_srt": True},
        "paths": {},
    })
    assert cfg["render"]["burn_captions"] is False
    assert cfg["render"]["layout"] == "stacked_faces"
    assert cfg["render"]["face_crop"]["zoom"] == 1.32
    assert cfg["packaging"]["burn_subtitles"] is False
    assert cfg["packaging"]["write_srt"] is False


def test_make_reels_cli_exists():
    from run_pipeline import build_parser
    parser = build_parser()
    args = parser.parse_args(["make-reels", "--video", "data/inbox/nabz-18.mp4"])
    assert args.video == "data/inbox/nabz-18.mp4"
    assert args.force is False


def test_invalid_plan_skips_unit_without_crashing(tmp_path):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"0")
    cfg = {
        "paths": {
            "transcripts": tmp_path / "transcripts",
            "normalized_transcripts": tmp_path / "norm",
            "program_maps": tmp_path / "maps",
            "qa_plans": tmp_path / "qa_plans",
        },
        "ai_editor": {"qa_reel_editor": {"model": "grok-4.6"}},
    }
    result = edit_program_qa_units(
        video,
        cfg,
        root=ROOT,
        invoke=lambda prompt, **kwargs: json.dumps({
            "segments": [
                {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core"},
                {"start": "00:00:15.000", "end": "00:00:25.000", "role": "payoff"},
            ]
        }),
        program_map={"units": [_unit("Q01")]},
        normalized={"duration": 80.0, "segments": [
            {"segment_id": 1, "start": 10.0, "end": 20.0, "raw_text": "q", "clean_text": "q"}
        ]},
        transcript={"segments": []},
    )
    assert result["plans"] == []
    assert "overlap" in result["skipped"][0]["reason"]


def _gate_plan(segments, **extra):
    plan = {
        "reel_id": "show_qa_Q01",
        "segments": segments,
        "playback_check": {
            "subject_clear": True,
            "answers_the_question": True,
            "qualifications_kept": "uncertainty kept",
            "ending_complete": True,
        },
    }
    plan.update(extra)
    return plan


def test_control_gate_passes_compact_reel():
    gate = evaluate_control_gate(_gate_plan([
        {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core"},
        {"start": "00:00:24.000", "end": "00:00:40.000", "role": "central_answer"},
        {"start": "00:00:40.000", "end": "00:00:55.000", "role": "payoff"},
    ]))
    assert gate["verdict"] == "PASS"
    assert gate["duration_le_180"] is True
    assert gate["answer_block_count"] == 2


def test_control_gate_rejects_over_ceiling_unless_closing_message():
    long_plan = _gate_plan([
        {"start": "00:00:10.000", "end": "00:02:10.000", "role": "question_core"},
        {"start": "00:02:10.000", "end": "00:04:20.000", "role": "central_answer"},
    ])
    assert evaluate_control_gate(long_plan)["verdict"] == "FAIL"
    assert evaluate_control_gate(long_plan)["duration_le_180"] is False
    closing = evaluate_control_gate(long_plan, closing_message=True)
    assert closing["duration_le_180"] is True
    assert closing["verdict"] == "PASS"


def test_control_gate_rejects_fragment_montage_without_justification():
    segs = [{"start": "00:00:01.000", "end": "00:00:03.000", "role": "question_core"}]
    t = 10.0
    for _ in range(6):
        segs.append({"start": ts(t), "end": ts(t + 2), "role": "reasoning"})
        t += 4.0
    bare = evaluate_control_gate(_gate_plan(segs))
    assert bare["no_fragment_montage"] is False
    assert bare["verdict"] == "FAIL"
    justified = evaluate_control_gate(_gate_plan(segs, fragment_justification="source beats are disjoint"))
    assert justified["no_fragment_montage"] is True
    assert justified["verdict"] == "PASS"


def test_closing_message_requires_one_continuous_answer_block():
    unit = _unit(
        "Q06",
        topic="keepsake message to the people",
        notes="closing keepsake; keep the guest message continuous",
    )
    assert is_closing_message(unit) is True
    two_blocks = evaluate_control_gate(
        _gate_plan([
            {"start": "00:00:10.000", "end": "00:00:16.000", "role": "question_core"},
            {"start": "00:00:18.000", "end": "00:00:30.000", "role": "central_answer"},
            {"start": "00:00:40.000", "end": "00:00:50.000", "role": "payoff"},
        ]),
        closing_message=True,
    )
    assert two_blocks["no_fragment_montage"] is False
    assert two_blocks["verdict"] == "FAIL"


def test_over_ceiling_triggers_one_revision(tmp_path):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"0")
    cfg = {
        "paths": {
            "transcripts": tmp_path / "transcripts",
            "normalized_transcripts": tmp_path / "norm",
            "program_maps": tmp_path / "maps",
            "qa_plans": tmp_path / "qa_plans",
        },
        "ai_editor": {"qa_reel_editor": {"model": "grok-4.6"}},
    }
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        if "## CONTROL GATE FAILED — revision" in prompt:
            return json.dumps({
                "hook_used": False,
                "playback_check": {
                    "subject_clear": True,
                    "answers_the_question": True,
                    "qualifications_kept": "kept",
                    "ending_complete": True,
                },
                "segments": [
                    {"start": "00:00:10.000", "end": "00:00:20.000", "role": "question_core", "why": "q"},
                    {"start": "00:00:24.000", "end": "00:00:40.000", "role": "central_answer", "why": "a"},
                ],
            })
        return json.dumps({
            "hook_used": False,
            "playback_check": {
                "subject_clear": True,
                "answers_the_question": True,
                "qualifications_kept": "kept",
                "ending_complete": True,
            },
            "segments": [
                {"start": "00:00:10.000", "end": "00:02:10.000", "role": "question_core", "why": "q"},
                {"start": "00:02:10.000", "end": "00:04:20.000", "role": "central_answer", "why": "a"},
            ],
        })

    result = edit_program_qa_units(
        video,
        cfg,
        root=ROOT,
        invoke=fake_invoke,
        program_map={"units": [_unit("Q01")]},
        normalized={"duration": 400.0, "segments": [
            {"segment_id": 1, "start": 10.0, "end": 20.0, "raw_text": "q", "clean_text": "q"}
        ]},
        transcript={"segments": []},
    )
    assert len(calls) == 2
    assert "CONTROL GATE FAILED — revision" in calls[1]
    assert result["plans"][0]["control_gate"]["verdict"] == "PASS"
    assert result["plans"][0]["control_gate"]["revision_attempt"] == 2
    assert result["plans"][0]["editorial_pass"] == "compress_then_repair"
