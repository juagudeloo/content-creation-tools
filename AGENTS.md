# Agent Instructions

This repository is a single-script CLI for generating English subtitles from Spanish MP4 videos. Keep changes focused on [scripts/generate_english_subtitles.py](scripts/generate_english_subtitles.py) and use [CLAUDE.md](CLAUDE.md) as the canonical source for runtime and architecture details.

## Working Rules

- Preserve the CLI and SRT output format unless the task explicitly requires a breaking change.
- Put dependency changes in [requirements.txt](requirements.txt); the script bootstraps its own `.venv` on first run.
- `ffmpeg` is required on `PATH` and is invoked as a subprocess, so keep subtitle-path escaping intact when touching burn-in logic.
- Subtitle timing and line wrapping belong in the cue-shaping helpers (`split_text_for_subtitles`, `chunk_by_words`, `wrap_for_srt`, `split_segment_into_cues`), not in the SRT writer.
- Prefer small, targeted edits over broad refactors.

## Validation

- There is no test suite or linter config in the repo.
- When behavior changes, validate with a focused run of the script or a narrow Python syntax check before widening scope.
