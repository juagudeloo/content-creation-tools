# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-script CLI tool that generates English subtitles for Spanish-language MP4 videos. It transcribes-and-translates the audio with `faster-whisper`, writes a well-formed `.srt`, and can optionally burn the subtitles into a new video via `ffmpeg`.

There is only one source file: `scripts/generate_english_subtitles.py`. There is no test suite, linter config, or package manifest beyond `requirements.txt`.

## Running it

```bash
python3 scripts/generate_english_subtitles.py <input_video.mp4> \
    [--output-srt PATH] [--burned-video PATH] \
    [--model small] [--language es] [--device cpu] \
    [--compute-type auto] [--beam-size 5]
```

- No manual setup is required: the script bootstraps its own `.venv` at the project root and installs `requirements.txt` into it on first run (see "Self-bootstrapping venv" below). Just invoke it with the system `python3`.
- `--output-srt` defaults to `<input_stem>.en.srt` next to the input video.
- `--burned-video`, if passed, produces a second MP4 with subtitles hard-coded into the picture.
- `ffmpeg` must be installed and on `PATH` — it is invoked as a subprocess, not as a pip dependency.

## Architecture

The script runs as a single linear pipeline in `main()`:

1. **Self-bootstrapping venv** (`ensure_virtual_environment`): on every invocation, the script checks the `SUBTITLE_BOOTSTRAP_READY` env var. If unset, it creates `.venv/` (if missing), checks whether `faster_whisper` is importable inside it, installs `requirements.txt` if not, then re-executes itself (`os.execve`) using the venv's Python interpreter with the env var set. This means the body of `main()` always effectively runs under `.venv`'s interpreter, even though the user invokes the system `python3`. Keep this guard in mind when adding new dependencies — add them to `requirements.txt`, not to the system environment.
2. **Audio extraction** (`extract_audio`): shells out to `ffmpeg` to pull mono 16kHz WAV audio into a temp dir.
3. **Model load & device/compute resolution** (`load_whisper_model`, `resolve_device`): `--device auto` probes `ctranslate2.get_cuda_device_count()` to pick `cuda` vs `cpu`. Compute type tries a short list of candidates (`float16`/`float32` first, then `int8` variants) and falls back on `ValueError`, since not every backend supports every compute type.
4. **Transcription** (`collect_cues`): calls `WhisperModel.transcribe(..., task="translate")` so Whisper directly outputs English text from Spanish (or `--language`) audio, with VAD filtering enabled.
5. **Cue shaping** (`split_text_for_subtitles`, `chunk_by_words`, `wrap_for_srt`, `split_segment_into_cues`): Whisper's raw segments are often too long/run-on for subtitles. This stage splits long segment text on punctuation boundaries (falling back to word-chunking) to respect `MAX_CUE_CHARS`, then distributes timing across the resulting pieces proportionally to word count (bounded by `MIN_CUE_DURATION` per piece), and wraps each piece's text to `MAX_LINE_WIDTH` for two-line display. This is the most intricate part of the script — changes to subtitle pacing/formatting should go through here rather than in the SRT writer.
6. **SRT writing** (`write_srt`, `format_timestamp`): standard SRT index/timestamp/text/blank-line format.
7. **Optional burn-in** (`render_burned_video`): re-invokes `ffmpeg` with a `subtitles=filename=...` video filter; `escape_path_for_ffmpeg` escapes the path for ffmpeg's filtergraph syntax (colons, backslashes, quotes) since the subtitle path is passed inline in `-vf`.

## Notes for changes

- This is not a git repository, so there's no commit history or branch state to inspect.
- Tunable constants live at the top of the file (`MAX_CUE_CHARS`, `MAX_LINE_WIDTH`, `MIN_CUE_DURATION`, `DEFAULT_MODEL`, `DEFAULT_LANGUAGE`, `DEFAULT_OUTPUT_SUFFIX`) — prefer adjusting these over hardcoding new magic numbers elsewhere.
