from reels_factory.render import _clips_from_plan


def test_cold_open_omits_repeated_hook():
    # SOURCE: 14:40 story, 15:07 hook start, 15:12 hook end, 15:34 payoff end
    plan = {
        "hook": {"enabled": True, "start": 907.2, "end": 912.7},
        "body": {"start": 880.0, "end": 934.0},
        "avoid_hook_repeat": True,
    }
    clips = _clips_from_plan(plan)
    assert [c["label"] for c in clips] == [
        "hook",
        "body_before_hook",
        "body_after_hook",
    ]
    assert clips[0]["start"] == 907.2
    assert clips[0]["end"] == 912.7
    assert clips[1]["start"] == 880.0
    assert clips[1]["end"] == 907.2
    assert clips[2]["start"] == 912.7
    assert clips[2]["end"] == 934.0


def test_hook_at_body_start_does_not_repeat():
    plan = {
        "hook": {"enabled": True, "start": 10.0, "end": 15.0},
        "body": {"start": 10.0, "end": 40.0},
        "avoid_hook_repeat": True,
    }
    clips = _clips_from_plan(plan)
    assert [c["label"] for c in clips] == ["hook", "body_after_hook"]
    assert clips[1]["start"] == 15.0
    assert clips[1]["end"] == 40.0


def test_disabled_hook_uses_body_only():
    plan = {
        "hook": {"enabled": False, "start": 20.0, "end": 25.0},
        "body": {"start": 10.0, "end": 50.0},
        "avoid_hook_repeat": True,
    }
    clips = _clips_from_plan(plan)
    assert clips == [{"label": "body", "start": 10.0, "end": 50.0}]


def test_repeat_allowed_keeps_full_body():
    plan = {
        "hook": {"enabled": True, "start": 30.0, "end": 35.0},
        "body": {"start": 10.0, "end": 50.0},
        "avoid_hook_repeat": False,
    }
    clips = _clips_from_plan(plan)
    assert [c["label"] for c in clips] == ["hook", "body"]
    assert clips[1]["start"] == 10.0
    assert clips[1]["end"] == 50.0
