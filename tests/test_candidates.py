from reels_factory.candidates import mine_candidates
from reels_factory.config import DEFAULT_CONFIG


def test_candidate_mining_returns_non_overlapping_shortlist():
    segments = []
    t = 0.0
    texts = [
        "این مقدمه‌ی کوتاه است و موضوع را باز می‌کند.",
        "اما بزرگ‌ترین اشتباه این بود که فکر می‌کردیم مسئله فقط قیمت است.",
        "بعد متوجه شدیم چرا مشتری اصلاً تصمیم نمی‌گیرد و مشکل جای دیگری است.",
        "وقتی داده‌ها را بررسی کردیم نتیجه کاملاً متفاوت بود.",
        "در نهایت راه‌حل ساده‌تر از چیزی بود که تصور می‌کردیم.",
        "این هم نتیجه‌ای است که از تجربه گرفتیم و می‌شود مستقل فهمید.",
    ]
    for i in range(18):
        text = texts[i % len(texts)]
        segments.append({"id": i, "start": t, "end": t + 5.0, "text": text, "words": []})
        t += 5.0
    transcript = {"segments": segments}
    result = mine_candidates(transcript, DEFAULT_CONFIG)
    assert result
    assert all(20 <= c["duration"] <= 62 for c in result)
    assert result[0]["candidate_id"] == "C01"
