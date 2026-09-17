from pathlib import Path
from types import SimpleNamespace

import pytest

from reels_factory.refine import (
    parse_refine_request,
    rebase_segment,
    refined_output_paths,
    refined_work_packet_path,
    validate_window,
    write_refined_work_packet,
)
from reels_factory.utils import parse_timestamp, to_source_time, ts


def test_parse_timestamp_hms_and_seconds():
    assert parse_timestamp("00:10:15") == pytest.approx(615.0)
    assert parse_timestamp("00:10:15.500") == pytest.approx(615.5)
    assert parse_timestamp("10:15") == pytest.approx(615.0)
    assert parse_timestamp(615) == pytest.approx(615.0)
    assert parse_timestamp("615.5") == pytest.approx(615.5)
    assert parse_timestamp("00:00:20,500") == pytest.approx(20.5)


def test_parse_timestamp_rejects_invalid():
    with pytest.raises(ValueError):
        parse_timestamp("")
    with pytest.raises(ValueError):
        parse_timestamp("not-a-time")
    with pytest.raises(ValueError):
        parse_timestamp(-1)
    with pytest.raises(ValueError):
        parse_timestamp("1:02:03:04")


def test_source_offset_matches_editorial_example():
    source_start = parse_timestamp("00:10:15")
    local = parse_timestamp("00:00:20.500")
    source = to_source_time(local, source_start)
    assert source == pytest.approx(parse_timestamp("00:10:35.500"))
    assert ts(source) == "00:10:35.500"


def test_rebase_segment_offsets_words_and_sentence():
    source_start = parse_timestamp("00:10:15")
    seg = SimpleNamespace(
        start=20.5,
        end=24.1,
        text=" جمله آزمایشی ",
        words=[
            SimpleNamespace(start=20.5, end=21.0, word="جمله", probability=0.9),
            SimpleNamespace(start=21.0, end=24.1, word="آزمایشی", probability=0.8),
        ],
    )
    out = rebase_segment(seg, source_start)
    assert out["local_start"] == pytest.approx(20.5)
    assert out["source_start"] == pytest.approx(635.5)
    assert out["source_end"] == pytest.approx(639.1)
    assert out["words"][0]["source_start"] == pytest.approx(635.5)
    assert out["words"][1]["source_end"] == pytest.approx(639.1)
    assert out["text"] == "جمله آزمایشی"


def test_invalid_ranges():
    with pytest.raises(ValueError, match="before end"):
        validate_window(670.0, 615.0)
    with pytest.raises(ValueError, match="before end"):
        validate_window(615.0, 615.0)
    with pytest.raises(ValueError, match="past source duration"):
        validate_window(1040.0, 1200.0, duration=1153.8)
    validate_window(1040.0, 1090.0, duration=1153.8)


def test_output_naming():
    out_dir = Path("data/refined_transcripts")
    paths = refined_output_paths(out_dir, "nabz-18", "C08")
    assert paths["json"].name == "nabz-18.C08.refined.json"
    assert paths["md"].name == "nabz-18.C08.refined.md"
    assert paths["srt"].name == "nabz-18.C08.refined.srt"
    combo = refined_output_paths(out_dir, "nabz-18", "C05_C01")
    assert combo["json"].name == "nabz-18.C05_C01.refined.json"
    assert refined_work_packet_path(out_dir, "nabz-18").name == "nabz-18.refined_work_packet.md"


def test_parse_refine_request(tmp_path: Path):
    video = tmp_path / "nabz-18.mp4"
    video.write_bytes(b"fake")
    parsed = parse_refine_request(
        {
            "video": str(video),
            "windows": [
                {"id": "C08", "start": "00:10:15", "end": "00:11:10"},
                {"id": "C05_C01", "start": "00:06:20", "end": "00:07:30"},
            ],
        },
        tmp_path,
    )
    assert parsed["video"] == video.resolve()
    assert parsed["windows"][0]["id"] == "C08"
    assert parsed["windows"][0]["start"] == pytest.approx(615.0)
    assert parsed["windows"][0]["end"] == pytest.approx(670.0)
    assert parsed["windows"][1]["id"] == "C05_C01"


def test_parse_refine_request_rejects_bad_payloads(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(ValueError, match="windows"):
        parse_refine_request({"video": str(video), "windows": []}, tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_refine_request(
            {
                "video": str(video),
                "windows": [
                    {"id": "C08", "start": "00:10:15", "end": "00:11:10"},
                    {"id": "C08", "start": "00:11:10", "end": "00:12:00"},
                ],
            },
            tmp_path,
        )
    with pytest.raises(ValueError, match="before end"):
        parse_refine_request(
            {
                "video": str(video),
                "windows": [{"id": "C08", "start": "00:11:10", "end": "00:10:15"}],
            },
            tmp_path,
        )


def test_missing_video_is_a_useful_error(tmp_path: Path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError, match="Source video not found"):
        parse_refine_request(
            {
                "video": str(missing),
                "windows": [{"id": "C08", "start": "00:10:15", "end": "00:11:10"}],
            },
            tmp_path,
        )


def test_work_packet_contains_only_shortlisted_windows(tmp_path: Path):
    payloads = [
        {
            "candidate_id": "C08",
            "source_start": 615.0,
            "source_end": 670.0,
            "text": "متن دوم",
            "segments": [
                {"source_start": 635.5, "source_end": 639.1, "text": "جمله اول"},
            ],
        }
    ]
    out = tmp_path / "nabz-18.refined_work_packet.md"
    write_refined_work_packet("nabz-18.mp4", payloads, out)
    text = out.read_text(encoding="utf-8")
    assert "C08" in text
    assert "00:10:15.000" in text
    assert "00:10:35.500" in text
    assert "متن دوم" in text
    assert "19-minute" not in text
