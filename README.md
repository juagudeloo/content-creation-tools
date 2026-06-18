# subtitle-generator

Generate English subtitles from Spanish MP4 videos.

## What it does

This repository contains CLI scripts for transcribing, translating, and creating short "reels" from Spanish-language MP4 videos:

- **generate_english_subtitles.py** — Extracts audio, transcribes and translates with `faster-whisper`, and writes an `.srt` subtitle file.
- **burn_subtitles.py** — Burns an existing `.srt` into a video.
- **create_reel_from_video.py** — Assembles short clips from a video based on keyword matching, with accompanying SRT and transcript.

## Scripts

- **[generate_english_subtitles.py](scripts/generate_english_subtitles.py)** — Generate English subtitles from Spanish MP4 videos. Outputs a `.en.srt` file. See [docs/GENERATE_SUBTITLES_TOOL.md](docs/GENERATE_SUBTITLES_TOOL.md).
- **[burn_subtitles.py](scripts/burn_subtitles.py)** — Burn an existing `.srt` file into a video using ffmpeg. See [docs/burn_subtitles.md](docs/burn_subtitles.md).
- **[create_reel_from_video.py](scripts/create_reel_from_video.py)** — Create a keyword-based reel from a video using an existing `.srt`. Outputs a trimmed MP4, SRT, and script. See [docs/create_reel_from_video.md](docs/create_reel_from_video.md).

## Recommended Workflow

1. **Generate subtitles** from your Spanish video:
   ```bash
   python3 scripts/generate_english_subtitles.py input.mp4
   ```
   Output: `input.en.srt`

2. **Review and edit** the SRT file manually to fix any Whisper translation errors.

3. **Create a reel** based on a keyword or phrase (supports multi-word keywords):
   ```bash
   python3 scripts/create_reel_from_video.py input.mp4 "keyword or phrase" --sub-file input.en.srt
   ```
   Outputs:
   - `input-keyword or phrase-reel.mp4` (trimmed video)
   - `input-keyword or phrase-reel-en.srt` (SRT for the reel)
   - `input-keyword or phrase-reel-script.txt` (transcript of the reel)

4. **Burn subtitles** into the reel (or any video):
   ```bash
   python3 scripts/burn_subtitles.py --video "input-keyword or phrase-reel.mp4" --srt "input-keyword or phrase-reel-en.srt" --out "input-keyword or phrase-reel.burned.mp4"
   ```
   Output: `input-keyword or phrase-reel.burned.mp4`

## Gradio app

A single interface wrapping all of the above lives in [app.py](app.py). It
bootstraps its own `.venv` (like the scripts) and installs `gradio`:

```bash
python3 app.py
```

Videos and subtitles are chosen with native **file pickers** (`gr.File`), output
folders with an in-app **path browser** (`gradio-path-selector`), and every
result is shown in a player / download box.

Tabs: **Generate subtitles**, **Burn subtitles** (with a live SRT editor — save
with `Ctrl+S`), and **Create reel** with two modes — *Clip creation*
(semantic keyword cut) and *Reel compilation* (snapshot a region, pick time
intervals, and compile the selected clips into one vertical reel).

> The app and scripts share one `.venv`, pinned to the gradio-4.44 era because
> `gradio-path-selector` is a gradio 4.x component (see the note in
> [requirements.txt](requirements.txt)). It's all installed automatically on
> first run.

## How to use it (basic)

```bash
python3 scripts/generate_english_subtitles.py <input_video.mp4> \
    [--output-srt PATH] [--model small] [--language es] [--device cpu] \
    [--compute-type auto] [--beam-size 5]
```

See the individual script documentation in `docs/` and `[CLAUDE.md](CLAUDE.md)` for detailed architecture and usage notes.

## Requirements

- `python3`
- `ffmpeg` on `PATH`

The script bootstraps its own local `.venv` and installs [requirements.txt](requirements.txt) on first run.
