from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .cover import generate_cover, generate_cover_finaltest
from .instagram import write_instagram_caption
from .utils import read_json, require_binary, run, write_json

COVER_HOLD_S = 1.75
TRANSITION_S = 0.25
INTRO_S = COVER_HOLD_S + TRANSITION_S
WHOOSH_DELAY_S = 1.72
WHOOSH_VOLUME = 0.26
MASTER_PORTRAIT = "data/portraits/17-05_guest_master_v2.png"

# Every 17-05 package uses the source-profile cleanframe render.
REEL_VIDEO = {
    "Q01": "data/output/1705_qa_Q01_cleanframe.mp4",
    "Q02": "data/output/1705_qa_Q02_cleanframe.mp4",
    "Q04": "data/output/1705_qa_Q04_cleanframe.mp4",
    "Q05": "data/output/1705_qa_Q05_cleanframe.mp4",
    "Q06": "data/output/1705_qa_Q06_cleanframe.mp4",
    "Q07": "data/output/1705_qa_Q07_cleanframe.mp4",
    "Q08": "data/output/1705_qa_Q08_cleanframe.mp4",
    "Q09": "data/output/1705_qa_Q09_cleanframe.mp4",
}

# Editorial copy locked from each Q&A plan. Not auto-trimmed from title.
HEADLINES = {
    "Q01": {
        "headline": "بارندگی به میانگین رسید؛ بحران حل نشد",
        "lines": ["بارندگی به میانگین رسید", "بحران حل نشد"],
        "concept": "یک سال باران",
        "candidates": [
            "بارندگی به میانگین رسید؛ بحران حل نشد",
            "یک سال باران مسئله آب را حل نمی‌کند",
            "با یک سال پُرباران مسائل آبی حل نمی‌شود",
            "آیا یک سال بارندگی بحران آب را تمام می‌کند؟",
            "میانگین بارش مسئله آب را حل نکرد",
        ],
    },
    "Q02": {
        "headline": "بحران آب ریشه‌ای است نه زودگذر",
        "lines": ["بحران آب ریشه‌ای است", "نه زودگذر"],
        "concept": "حق‌آبه محیط زیست",
        "candidates": [
            "بحران آب ریشه‌ای است نه زودگذر",
            "بحران عمیق و ریشه‌ای است نه زودگذر",
            "آب تجدیدپذیر حساب بانکی سالانه نیست",
            "بحران آب ایران ساختاری است نه گذرا",
        ],
    },
    "Q04": {
        "headline": "باران انگار از پیش مصرف شده",
        "lines": ["باران انگار از پیش مصرف شده"],
        "concept": "بحران منطقه‌ای",
        "candidates": [
            "باران انگار از پیش مصرف شده",
            "بحران آب ایران یکپارچه نیست",
            "هر منطقه بحران آبی خودش را دارد",
            "پرسش اصلی فقط میزان باران نیست",
        ],
    },
    "Q05": {
        "headline": "تغییر اقلیم سهم دارد؛ علت اصلی برداشت بی‌رویه است",
        "lines": ["تغییر اقلیم سهم دارد", "علت اصلی برداشت بی‌رویه است"],
        "concept": "چاه‌های بی‌مجوز",
        "candidates": [
            "تغییر اقلیم سهم دارد؛ علت اصلی برداشت بی‌رویه است",
            "بحران آب به‌خاطر تغییر اقلیم نیست",
            "اقلیم اثر دارد اما علت اصلی برداشت است",
            "برداشت بی‌رویه علت اصلی بحران آب است",
        ],
    },
    "Q06": {
        "headline": "باران ظرفیت از دست‌رفته آبخوان را برنمی‌گرداند",
        "lines": ["باران ظرفیت از دست‌رفته", "آبخوان را برنمی‌گرداند"],
        "concept": "تراکم آبخوان",
        "candidates": [
            "باران ظرفیت از دست‌رفته آبخوان را برنمی‌گرداند",
            "باران ذخیره از دست‌رفته آبخوان را بازنمی‌گرداند",
            "بعد از تراکم، باران ظرفیت اولیه را پر نمی‌کند",
            "آبخوان فشرده دیگر گنجایش پیشین ندارد",
        ],
    },
    "Q07": {
        "headline": "توسعه بیش از آب حوزه بارگذاری می‌شود",
        "lines": ["توسعه بیش از آب حوزه", "بارگذاری می‌شود"],
        "concept": "مدیریت تقاضا",
        "candidates": [
            "توسعه بیش از آب حوزه بارگذاری می‌شود",
            "برنامه‌ریزی توسعه بیش از بارش حوزه بار می‌گذارد",
            "آب حوزه در برنامه‌های توسعه لحاظ نشده",
            "ریشه بحران در تقاضا، مصرف و تخصیص است",
        ],
    },
    "Q08": {
        "headline": "اول توقف تخریب؛ بعد احیای آب",
        "lines": ["اول توقف تخریب", "بعد احیای آب"],
        "concept": "منافع بلندمدت",
        "candidates": [
            "اول توقف تخریب؛ بعد احیای آب",
            "تغییر سیاسی بحران آب را خودبه‌خود حل نمی‌کند",
            "اول تخریب را متوقف کنید، بعد احیا کنید",
            "احیا بعد از توقف روند تخریب ممکن است",
        ],
    },
    "Q09": {
        "headline": "هنوز برای ایران امیدوارم",
        "lines": ["هنوز برای ایران امیدوارم"],
        "concept": "عمل جمعی",
        "candidates": [
            "هنوز برای ایران امیدوارم",
            "امید به ایران مبتنی بر دانش و مسئولیت است",
            "ایران محکوم به اضمحلال نیست",
            "اگر به مردم اعتماد کنیم ایران آباد می‌شود",
        ],
    },
}


def _ffprobe_json(path: Path) -> dict:
    ffprobe = require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout or "{}")


def _duration(path: Path) -> float:
    data = _ffprobe_json(path)
    return float((data.get("format") or {}).get("duration") or 0.0)


def ensure_whoosh(root: Path) -> Path:
    path = root / "assets" / "sounds" / "whoosh_soft.wav"
    if path.is_file() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_binary("ffmpeg")
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=d=0.32:c=pink:r=96000:a=0.55",
            "-af",
            "highpass=f=450,lowpass=f=5200,afade=t=in:st=0:d=0.045,afade=t=out:st=0.09:d=0.23,volume=0.55",
            str(path),
        ]
    )
    return path


def _make_cover_intro(ffmpeg: str, cover_jpg: Path, intro_mp4: Path, fps: int = 30) -> None:
    frames = int(round(INTRO_S * fps))
    hold = int(round(COVER_HOLD_S * fps))
    zoom_expr = (
        f"if(lt(in,{hold}),1,min(1.05,1+(in-{hold - 1})*0.00625))"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(cover_jpg),
            "-vf",
            (
                f"scale=1080:1920:force_original_aspect_ratio=disable,"
                f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s=1080x1920:fps={fps},format=yuv420p"
            ),
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(intro_mp4),
        ]
    )


def package_final_mp4(
    *,
    cover_jpg: Path,
    reel_mp4: Path,
    whoosh: Path,
    output_mp4: Path,
    fps: int = 30,
) -> dict:
    ffmpeg = require_binary("ffmpeg")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = int(round(INTRO_S * 1000))
    whoosh_ms = int(round(WHOOSH_DELAY_S * 1000))
    with tempfile.TemporaryDirectory(prefix="final_pack_") as td:
        intro = Path(td) / "intro.mp4"
        _make_cover_intro(ffmpeg, cover_jpg, intro, fps=fps)
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(intro),
                "-i",
                str(reel_mp4),
                "-i",
                str(whoosh),
                "-filter_complex",
                (
                    "[0:v][1:v]concat=n=2:v=1:a=0[v];"
                    f"[1:a]adelay={delay_ms}|{delay_ms},aformat=sample_fmts=fltp:sample_rates=96000:channel_layouts=stereo[ra];"
                    f"[2:a]adelay={whoosh_ms}|{whoosh_ms},aformat=sample_fmts=fltp:sample_rates=96000:channel_layouts=stereo,"
                    f"volume={WHOOSH_VOLUME}[wh];"
                    "[ra][wh]amix=inputs=2:duration=longest:dropout_transition=0,"
                    "alimiter=limit=0.95[a]"
                ),
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "96000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output_mp4),
            ]
        )
    probe = _ffprobe_json(output_mp4)
    streams = {s.get("codec_type"): s for s in probe.get("streams") or []}
    video = streams.get("video") or {}
    audio = streams.get("audio") or {}
    duration = float((probe.get("format") or {}).get("duration") or 0.0)
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "duration": round(duration, 3),
        "has_audio": audio.get("codec_name") == "aac",
        "sample_rate": int(audio.get("sample_rate") or 0),
    }


def package_1705_finals(
    root: Path,
    cfg: dict,
    *,
    regenerate_covers: bool = True,
) -> list[dict]:
    root = Path(root)
    whoosh = ensure_whoosh(root)
    meta = root / "data" / "metadata" / "17-05.json"
    portrait = root / MASTER_PORTRAIT
    final_dir = root / "data" / "final"
    cover_dir = final_dir / "covers"
    cover_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, copy in HEADLINES.items():
        plan = root / "data" / "qa_plans" / f"17-05.{uid}.qa.json"
        reel = root / REEL_VIDEO[uid]
        if not reel.is_file():
            raise FileNotFoundError(f"Approved reel missing: {reel}")
        cover_jpg = cover_dir / f"1705_qa_{uid}_cover.jpg"
        cover_json = cover_dir / f"1705_qa_{uid}_cover.json"
        if regenerate_covers:
            print(f"[final] cover {uid}", flush=True)
            cover = generate_cover_finaltest(
                plan,
                meta,
                cfg,
                root=root,
                master_portrait=portrait,
                output_jpg=cover_jpg,
                output_json=cover_json,
                headline=copy["headline"],
                headline_lines=copy["lines"],
                headline_candidates=copy["candidates"],
            )
        else:
            if not cover_jpg.is_file():
                raise FileNotFoundError(f"Approved cover missing: {cover_jpg}")
            cover = json.loads(cover_json.read_text(encoding="utf-8")) if cover_json.is_file() else {}
            print(f"[final] reuse cover {uid} <- {cover_jpg.name}", flush=True)
        out_mp4 = final_dir / f"1705_qa_{uid}_final.mp4"
        print(f"[final] package {uid} <- {reel.name}", flush=True)
        stats = package_final_mp4(
            cover_jpg=cover_jpg,
            reel_mp4=reel,
            whoosh=whoosh,
            output_mp4=out_mp4,
        )
        errors = []
        if stats["width"] != 1080 or stats["height"] != 1920:
            errors.append(f"size {stats['width']}x{stats['height']}")
        if not stats["has_audio"]:
            errors.append("missing audio")
        expected = INTRO_S + _duration(reel)
        if abs(stats["duration"] - expected) > 0.35:
            errors.append(f"duration {stats['duration']} vs expected {expected:.3f}")
        if cover.get("guest_portrait", "").replace("\\", "/").endswith("17-05_guest_master_v2.png") is False:
            errors.append("wrong master portrait")
        if cover.get("guest_name") != "محسن سعیدی":
            errors.append("guest name")
        row = {
            "reel": uid,
            "reel_id": f"1705_qa_{uid}",
            "headline": copy["headline"],
            "concept": copy["concept"],
            "cover_duration_s": COVER_HOLD_S,
            "transition_duration_s": TRANSITION_S,
            "whoosh": str(whoosh.as_posix()),
            "final_duration_s": stats["duration"],
            "final_mp4": str(out_mp4.as_posix()),
            "cover_jpg": str(cover_jpg.as_posix()),
            "source_reel": str(reel.as_posix()),
            "validation": {
                "size": f"{stats['width']}x{stats['height']}",
                "audio": stats["has_audio"],
                "errors": errors,
            },
        }
        rows.append(row)
        print(
            f"[final] {uid} {stats['width']}x{stats['height']} "
            f"{stats['duration']:.3f}s errors={errors or 'none'}",
            flush=True,
        )
    write_json(final_dir / "1705_final_index.json", {
        "master_portrait": str(portrait.as_posix()),
        "whoosh": str(whoosh.as_posix()),
        "intro_s": INTRO_S,
        "reels": rows,
    })
    return rows


def packaging_enabled(cfg: dict) -> bool:
    pack = cfg.get("packaging") or {}
    return bool(pack.get("enabled", True))


def instagram_captions_enabled(cfg: dict) -> bool:
    pack = cfg.get("packaging") or {}
    return bool(pack.get("instagram_captions", True))


def burn_subtitles_enabled(cfg: dict) -> bool:
    """Production default is off. Reel SRTs are never written unless this is true."""
    pack = cfg.get("packaging") or {}
    if pack.get("burn_subtitles") or pack.get("write_srt"):
        return True
    return bool((cfg.get("render") or {}).get("burn_captions", False))


def package_production_reel(
    *,
    plan_path: Path,
    reel_mp4: Path,
    cfg: dict,
    root: Path,
    metadata_path: Path,
) -> dict:
    """Cover + whoosh intro + final MP4 + Instagram caption. No subtitles."""
    root = Path(root)
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    meta = read_json(metadata_path)
    reel_id = plan.get("reel_id") or plan_path.stem
    final_dir = Path(cfg["paths"].get("final") or (root / "data" / "final"))
    cover_dir = final_dir / "covers"
    cover_dir.mkdir(parents=True, exist_ok=True)
    cover_jpg = cover_dir / f"{reel_id}_cover.jpg"
    cover_json = cover_dir / f"{reel_id}_cover.json"
    cover = generate_cover(
        plan_path,
        metadata_path,
        cfg,
        root=root,
        output_jpg=cover_jpg,
        output_json=cover_json,
    )
    whoosh = ensure_whoosh(root)
    out_mp4 = final_dir / f"{reel_id}_final.mp4"
    print(f"[final] package {reel_id} <- {reel_mp4.name}", flush=True)
    stats = package_final_mp4(
        cover_jpg=cover_jpg,
        reel_mp4=reel_mp4,
        whoosh=whoosh,
        output_mp4=out_mp4,
        fps=int((cfg.get("render") or {}).get("fps") or 30),
    )
    caption_path = None
    if instagram_captions_enabled(cfg):
        caption_path = write_instagram_caption(
            final_dir / f"{reel_id}_instagram.txt",
            plan,
            meta,
        )
    return {
        "reel_id": reel_id,
        "final_mp4": str(out_mp4.as_posix()),
        "cover_jpg": str(cover_jpg.as_posix()),
        "cover_json": str(cover_json.as_posix()),
        "instagram_txt": str(caption_path.as_posix()) if caption_path else None,
        "headline": cover.get("selected_headline"),
        "duration": stats["duration"],
    }
