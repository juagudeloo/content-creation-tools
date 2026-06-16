# subtitle-generator

Generate English subtitles from Spanish MP4 videos.

## What it does

This repository contains a single CLI script, [scripts/generate_english_subtitles.py](scripts/generate_english_subtitles.py), which:

- extracts audio from an MP4 with `ffmpeg`
- transcribes and translates the speech with `faster-whisper`
- writes an `.srt` subtitle file
- optionally burns the subtitles into a new MP4

## How to use it

```bash
python3 scripts/generate_english_subtitles.py <input_video.mp4> \
    [--output-srt PATH] [--burned-video PATH] \
    [--model small] [--language es] [--device cpu] \
    [--compute-type auto] [--beam-size 5]
```

See [docs/GENERATE_SUBTITLES_TOOL.md](docs/GENERATE_SUBTITLES_TOOL.md) for a step-by-step explanation of how the script works.

## Requirements

- `python3`
- `ffmpeg` on `PATH`

The script bootstraps its own local `.venv` and installs [requirements.txt](requirements.txt) on first run.
