from pathlib import Path

from reels_factory.package_final import INTRO_S
from reels_factory.publication import (
    MAX_LINES,
    extract_plan_cues,
    polish_subtitle_text,
    trim_text_to_window,
    validate_publication_cues,
    write_srt,
)


def test_polish_fixes_asr_glitches_without_paraphrase():
    assert polish_subtitle_text("ولی معناش این نیست که مسائل آبی محل می‌شد.") == (
        "ولی معناش این نیست که مسائل آبی حل می‌شد."
    )
    assert polish_subtitle_text("اروپا بیشتر شکل از مسائل موقعاتی است.") == (
        "بیشتر شکل از مسائل موقتی است."
    )


def test_trim_uses_source_words_to_drop_cut_prefix():
    text = "و می‌خواهم بحث را از اینجا شروع بکنم با این سؤال که آیا بحران منابع آبی و بحران آب در ایران انقدر"
    words = [
        (75.76, 76.40, "و"),
        (76.40, 76.68, "میخم"),
        (76.68, 77.00, "بس"),
        (77.00, 77.16, "را"),
        (77.16, 77.24, "از"),
        (77.24, 77.44, "اینجا"),
        (77.44, 77.64, "شروع"),
        (77.64, 78.00, "بکنم"),
        (78.00, 78.20, "با"),
        (78.20, 78.28, "از"),
        (78.28, 78.80, "رتالی"),
        (78.80, 79.98, "که"),
        (79.98, 80.52, "آیا"),
        (80.52, 81.76, "بوهران"),
        (81.76, 82.58, "منابه"),
        (82.58, 82.90, "آبی"),
        (82.90, 83.04, "و"),
        (83.04, 83.50, "بوهران"),
        (83.50, 83.76, "آب"),
        (83.76, 83.92, "در"),
        (83.92, 84.16, "ایران"),
        (84.16, 84.76, "انغدر"),
    ]
    trimmed = trim_text_to_window(text, 75.76, 84.76, 79.98, 84.76, words)
    assert trimmed.startswith("آیا")
    assert "می‌خواهم بحث" not in trimmed


def test_cues_start_after_cover_and_follow_cuts(tmp_path: Path):
    plan = {
        "segments": [
            {"start": 10.0, "end": 14.0, "role": "question_core"},
            {"start": 20.0, "end": 24.0, "role": "central_answer"},
        ]
    }
    normalized = {
        "segments": [
            {"segment_id": 1, "start": 10.0, "end": 14.0, "clean_text": "آیا بحران آب حل می‌شود؟"},
            {"segment_id": 2, "start": 20.0, "end": 24.0, "clean_text": "جواب سؤال شما خیر است."},
        ]
    }
    cues = extract_plan_cues(plan, normalized, intro_s=INTRO_S, root=tmp_path)
    errors = validate_publication_cues(cues)
    assert errors == []
    assert cues[0]["start"] == INTRO_S
    assert abs(cues[0]["start"] - 2.0) < 1e-6
    answer = next(c for c in cues if "خیر" in c["text"])
    assert abs(answer["start"] - (INTRO_S + 4.0)) < 0.05
    assert all(len(c["lines"]) <= MAX_LINES for c in cues)
    srt = write_srt(cues, tmp_path / "t.srt").read_text(encoding="utf-8")
    assert "00:00:02,000" in srt
    assert "00:00:00,000" not in srt.split("-->")[0]
