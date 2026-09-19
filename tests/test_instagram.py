from reels_factory.config import DEFAULT_CONFIG
from reels_factory.instagram import build_instagram_caption
from reels_factory.package_final import burn_subtitles_enabled, instagram_captions_enabled, packaging_enabled


def test_v1_lock_disables_subtitles_and_keeps_captions():
    assert DEFAULT_CONFIG["render"]["burn_captions"] is False
    pack = DEFAULT_CONFIG["packaging"]
    assert pack["enabled"] is True
    assert pack["burn_subtitles"] is False
    assert pack["write_srt"] is False
    assert pack["instagram_captions"] is True
    assert pack["cover_hold_s"] == 1.75
    assert pack["transition_s"] == 0.25
    assert packaging_enabled(DEFAULT_CONFIG) is True
    assert instagram_captions_enabled(DEFAULT_CONFIG) is True
    assert burn_subtitles_enabled(DEFAULT_CONFIG) is False


def test_instagram_caption_is_source_faithful_not_headline():
    plan = {
        "title": "یک سال باران بحران آب ایران را حل نمی‌کند",
        "segments": [
            {
                "role": "payoff",
                "text": "ولی معناش این نیست که مسائل آبی حل می‌شد. آن هم نه در همه جای کشور.",
            }
        ],
    }
    meta = {
        "guest_name": "محسن سعیدی",
        "guest_role": "متخصص آب، محیط زیست و پایداری",
    }
    text = build_instagram_caption(plan, meta)
    assert "محسن سعیدی" in text
    assert "یک سال باران بحران آب ایران را حل نمی‌کند" in text
    assert "بارندگی به میانگین رسید؛ بحران حل نشد" not in text
    assert text.count("\n") >= 2
    assert "#ایران" in text
    assert "#بحران_آب" in text
    body, _, tags = text.strip().partition("\n\n")
    body_lines = [ln for ln in body.splitlines() if ln.strip()]
    assert 2 <= len(body_lines) <= 4
