# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A collection of CLI scripts for processing Spanish-language MP4 videos: transcribing and translating audio to English subtitles via `faster-whisper`, burning subtitle files into video, and cutting keyword-based short reels. All scripts invoke `ffmpeg` as a subprocess — it must be on `PATH`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_english_subtitles.py` | Transcribe + translate audio → `.srt` (self-bootstrapping venv) |
| `scripts/burn_subtitles.py` | Burn an existing `.srt` into a video |
| `scripts/create_reel_from_video.py` | Cut keyword-matched segments into a short reel |
| `scripts/_utils.py` | Shared helpers: `render_burned_video`, `escape_path_for_ffmpeg` |

## Running each script

**Generate subtitles** (self-bootstrapping — invoke with system `python3`):
```bash
python3 scripts/generate_english_subtitles.py <input.mp4> \
    [--output-srt PATH] [--model small] [--language es] \
    [--task translate|transcribe] \
    [--device auto|cpu|cuda] [--compute-type auto] [--beam-size 5]
```
`--task translate` (default) outputs English. `--task transcribe` keeps the original language and names the output `<stem>.<language>.srt`.

**Burn-only mode** (skip transcription, use existing SRT):
```bash
python3 scripts/generate_english_subtitles.py <input.mp4> \
    --srt <subtitles.en.srt> --burned-video <output.mp4>
```

**Standalone burn**:
```bash
python3 scripts/burn_subtitles.py --video <input.mp4> --srt <subtitles.en.srt> [--out <output.mp4>]
```

**Create keyword reel** (auto-transcribes if no SRT is provided):
```bash
python3 scripts/create_reel_from_video.py <input.mp4> <keyword> \
    [--sub-file <subtitles.en.srt>] [--out <reel.mp4>] [--max-seconds 60]
```
Outputs: `<keyword>-reel.mp4`, `<keyword>-reel-script.txt`, `<keyword>-reel-en.srt`.

## Architecture

### generate_english_subtitles.py pipeline

1. **Self-bootstrapping venv** (`ensure_virtual_environment`): checks `SUBTITLE_BOOTSTRAP_READY` env var; if unset, creates `.venv/`, installs `requirements.txt` if `faster_whisper` is not importable, then `os.execve`s itself under the venv's Python with the var set. `main()` body always runs under `.venv`. New dependencies go in `requirements.txt`, not the system env.
2. **Audio extraction** (`extract_audio`): `ffmpeg` → mono 16kHz WAV in a temp dir.
3. **Device/compute resolution** (`resolve_device`, `load_whisper_model`): `--device auto` probes `ctranslate2.get_cuda_device_count()`. Compute type tries `float16`/`float32` then `int8` variants, falling back on `ValueError`.
4. **Transcription** (`collect_cues`): `WhisperModel.transcribe(..., task="translate")` outputs English directly from Spanish audio, with VAD filtering.
5. **Cue shaping** (`split_text_for_subtitles`, `chunk_by_words`, `wrap_for_srt`, `split_segment_into_cues`): splits long Whisper segments on punctuation (fallback: word-chunking) to respect `MAX_CUE_CHARS`, distributes timing proportionally by word count (floor: `MIN_CUE_DURATION`), wraps to `MAX_LINE_WIDTH`. This is the most intricate stage — subtitle pacing/formatting changes belong here, not in the SRT writer.
6. **SRT writing** (`write_srt`, `format_timestamp`): standard index/timestamp/text/blank-line format.

### Deliberate two-step design

`generate_english_subtitles.py` will NOT transcribe and burn in one command. Passing `--burned-video` without `--srt` raises an error. Use `--srt` + `--burned-video` together (burn-only mode) or use `scripts/burn_subtitles.py`.

### Burn-in helpers (scripts/_utils.py)

`render_burned_video` shells out to `ffmpeg` with a `subtitles=filename=...` video filter (libx264, crf 18, aac 192k). `escape_path_for_ffmpeg` escapes colons, backslashes, and quotes for ffmpeg's filtergraph syntax. Touch this carefully — path escaping failures are silent and produce corrupt video.

### create_reel_from_video.py

Has its own self-bootstrapping venv (env var `REEL_BOOTSTRAP_READY`) that installs `sentence-transformers`.

If no `--sub-file` is given, calls `generate_english_subtitles.py` with `--task transcribe --language es` to produce a Spanish SRT (preserving original phrasing for better embedding accuracy).

Cues are grouped into topic **paragraphs** by two rules: silence gap > `PARAGRAPH_GAP` (1.5 s) or paragraph length > `MAX_PARAGRAPH_DURATION` (90 s). The duration cap is essential for videos with continuous speech — without it, the entire video can collapse into 2–3 paragraphs and the semantic signal gets diluted.

Each paragraph is embedded with `paraphrase-multilingual-MiniLM-L12-v2` (default) and ranked by cosine similarity to the query. Paragraphs above `SIMILARITY_THRESHOLD` (0.25) are selected. The budget (`--max-seconds`, default 60) is filled by taking the **highest-scoring** paragraphs first (not the earliest), then re-sorting them chronologically for the video cut. This prevents a lower-relevance early paragraph from consuming the budget before the truly relevant section is reached.

After the reel is created, a **verification report** is printed: the full reel transcript is embedded against the query and a relevance score (0–1) is shown alongside a transcript preview.

## Tunable constants

In `generate_english_subtitles.py`: `MAX_CUE_CHARS` (84), `MAX_LINE_WIDTH` (42), `MIN_CUE_DURATION` (0.7 s), `DEFAULT_MODEL`, `DEFAULT_LANGUAGE`, `DEFAULT_OUTPUT_SUFFIX`.

In `create_reel_from_video.py`: `MAX_REEL_SECONDS` (60 s), `PARAGRAPH_GAP` (1.5 s), `MAX_PARAGRAPH_DURATION` (90 s), `SIMILARITY_THRESHOLD` (0.25), `CLUSTER_GAP` (60 s).

Prefer adjusting these constants over hardcoding magic numbers elsewhere.
