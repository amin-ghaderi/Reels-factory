# Reels Factory V1 — local-first, low-cost

This is the first implementation of a **local-first Reels Factory** designed to minimize paid model/video processing.

The expensive intelligence layer should see **timestamped text first**, not the entire long video. Local tools do transcription, candidate mining, cutting, cropping, captions, audio normalization, and export.

## V1 flow

```text
Raw video
  -> faster-whisper (local transcript + timestamps)
  -> local heuristic candidate miner
  -> compact Work packet
  -> ChatGPT Work chooses finalists + cold-open hook
  -> JSON edit plans
  -> FFmpeg renders 9:16 reels + subtitles
  -> Work/human QA
  -> approved/
```

## Why this is cheap

- Full transcription is local with `faster-whisper`.
- The model receives a compact candidate packet before any full transcript/video inspection.
- FFmpeg handles all deterministic editing locally.
- The source video is inspected by Work only for finalists when necessary.
- No paid API is required for V1.

## Requirements

- Windows/macOS/Linux
- Python **3.12 x64 recommended** (`py -3.12`; 3.11 also works)
- FFmpeg and ffprobe in PATH
- Enough disk space for video renders

GPU is optional. CPU works; a GPU makes transcription faster.

## Windows setup

From PowerShell inside this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

If you prefer manual setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe run_pipeline.py setup
.\.venv\Scripts\python.exe scripts\healthcheck.py
```

## Health check

```powershell
.\.venv\Scripts\python.exe scripts\healthcheck.py
```

## First real run

1. Copy one source video to:

```text
data/inbox/
```

2. Run the one-shot inbox launcher:

```powershell
.\scripts\run_inbox.ps1
```

Or the equivalent Python command:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py inbox
```

3. The important low-token file will appear at:

```text
data/candidates/<video>.work_packet.md
```

4. Give ChatGPT Work the instruction in:

```text
prompts/work_reel_director.md
```

Tell Work to read the candidate packet first and create JSON edit plans in `data/edit_plans/`.

If first-pass Whisper `small` Persian is too noisy, refine only the shortlisted windows instead of re-transcribing the whole video:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py refine `
  --video data/inbox/nabz-18.mp4 `
  --start 00:10:15 `
  --end 00:11:10 `
  --id C08
```

Or batch several windows:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py refine-batch `
  --request data/refine_requests/nabz-18.json
```

The compact second-pass packet is:

```text
data/refined_transcripts/<video>.refined_work_packet.md
```

5. Render one approved plan:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py render `
  --plan data/edit_plans/reel_001.json `
  --transcript data/transcripts/<video>.transcript.json
```

Rendered video appears in:

```text
data/output/
```

## Configuration

Run `setup` once to create `config.yaml` from `config.example.yaml`.

The first setting to tune is the Whisper model:

```yaml
transcription:
  model: small
  device: cpu
  compute_type: int8
```

For a CPU-first cheap setup, `small` is a sensible starting point. If Persian transcription quality is insufficient, try `medium` or `large-v3` locally; this costs compute time, not API tokens.

## Current V1 limitations

- 9:16 crop is centered; no face/speaker tracking yet.
- Candidate mining is heuristic. ChatGPT Work performs the editorial judgment.
- Subtitle styling is intentionally simple.
- No B-roll generation, auto-publishing, analytics feedback loop, or Instagram integration yet.
- No API is used; therefore Work is the orchestration/editorial layer rather than a background API call inside Python.

## V2 roadmap

1. Speaker/face tracking for dynamic crop.
2. Word-level animated captions.
3. Silence/filler removal.
4. Candidate preview generation only for finalists.
5. Content analytics database and hook-performance feedback loop.
6. Optional publishing/integration layer, with final human approval.
