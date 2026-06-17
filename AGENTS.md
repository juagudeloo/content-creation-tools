# Agent Instructions

This repository contains CLI scripts for transcribing, translating, and creating short video reels from Spanish MP4s:
- [scripts/generate_english_subtitles.py](scripts/generate_english_subtitles.py) — Generate English subtitles from Spanish video.
- [scripts/burn_subtitles.py](scripts/burn_subtitles.py) — Burn an existing `.srt` into a video.
- [scripts/create_reel_from_video.py](scripts/create_reel_from_video.py) — Create keyword-based reels with generated SRT and transcript.

Keep changes focused on these scripts and use [CLAUDE.md](CLAUDE.md) for architecture details. Reference [README.md](README.md) for the recommended workflow.

## Working Rules

- Preserve the CLI and SRT output format unless the task explicitly requires a breaking change.
- Put dependency changes in [requirements.txt](requirements.txt); the script bootstraps its own `.venv` on first run.
- `ffmpeg` is required on `PATH` and is invoked as a subprocess, so keep subtitle-path escaping intact when touching burn-in logic.
- Subtitle timing and line wrapping belong in the cue-shaping helpers (`split_text_for_subtitles`, `chunk_by_words`, `wrap_for_srt`, `split_segment_into_cues`), not in the SRT writer.
- Reel creation: keyword matching and segment merging logic lives in `create_reel_from_video.py`; this is where LLM refinement can be added.
- Prefer small, targeted edits over broad refactors.

## Validation

- There is no test suite or linter config in the repo.
- When behavior changes, validate with a focused run of the affected script or a narrow Python syntax check before widening scope.

