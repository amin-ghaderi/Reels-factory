import pytest

from reels_factory.program_map_validate import ProgramMapError, validate_program_map


def _unit(**kwargs):
    base = {
        "unit_id": "Q01",
        "topic": "t",
        "question": {
            "start": "00:01:00.000",
            "end": "00:01:10.000",
            "core_start": "00:01:04.000",
            "core_end": "00:01:10.000",
            "text": "why?",
        },
        "answer": {
            "start": "00:01:10.000",
            "end": "00:02:00.000",
            "core_start": "00:01:12.000",
            "status": "complete",
        },
        "question_type": "main_question",
        "references_to_previous_unit": [],
    }
    base.update(kwargs)
    return base


def test_valid_program_map():
    validate_program_map(
        {"units": [_unit(), _unit(
            unit_id="Q02",
            question={
                "start": "00:02:00.000",
                "end": "00:02:08.000",
                "core_start": "00:02:03.000",
                "core_end": "00:02:08.000",
                "text": "and?",
            },
            answer={
                "start": "00:02:08.000",
                "end": "00:03:00.000",
                "core_start": "00:02:10.000",
                "status": "partial",
            },
            question_type="follow_up",
        )]},
        source_duration=200.0,
    )


def test_rejects_start_after_end():
    with pytest.raises(ProgramMapError, match="start >= end"):
        validate_program_map({"units": [_unit(answer={
            "start": "00:02:00.000",
            "end": "00:01:50.000",
            "status": "complete",
        })]})


def test_rejects_outside_duration():
    with pytest.raises(ProgramMapError, match="past source duration"):
        validate_program_map({"units": [_unit()]}, source_duration=50.0)


def test_rejects_duplicate_span_and_out_of_order():
    dup = _unit(unit_id="Q02")
    with pytest.raises(ProgramMapError, match="duplicates the same Q&A"):
        validate_program_map({"units": [_unit(), dup]})

    later_first = _unit(unit_id="Q09")
    earlier = _unit(
        unit_id="Q01b",
        question={
            "start": "00:00:10.000",
            "end": "00:00:12.000",
            "text": "x",
        },
        answer={"start": "00:00:12.000", "end": "00:00:20.000", "status": "complete"},
    )
    with pytest.raises(ProgramMapError, match="chronological"):
        validate_program_map({"units": [later_first, earlier]})


def test_transition_without_question_allowed():
    validate_program_map({"units": [{
        "unit_id": "T01",
        "topic": "open",
        "question": None,
        "answer": None,
        "question_type": "transition",
        "start": "00:00:17.000",
        "end": "00:00:40.000",
        "notes": "host open",
    }, _unit()]})
