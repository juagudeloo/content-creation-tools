# burn_subtitles.py — Burn an existing SRT into a video

`scripts/burn_subtitles.py` takes an existing `.srt` file and hard-codes the subtitles into a new MP4 using `ffmpeg`. It is the dedicated burn tool; for an integrated transcribe-then-burn workflow, see [generate_english_subtitles.md](generate_english_subtitles.md).

## How it works

1. **Parse CLI args**: `--video` (required), `--srt` (required), `--out` (optional, defaults to `<video_stem>.burned.mp4`).
2. **Resolve paths** and delegate to `render_burned_video()` in `scripts/_utils.py`.
3. `render_burned_video()` builds an `ffmpeg` command with a `subtitles=filename='...'` video filter and runs it via `subprocess.run(..., check=True)`. Output is re-encoded as libx264 (CRF 18, preset medium) with AAC audio at 192 kbps.

`escape_path_for_ffmpeg()` in `_utils.py` escapes colons, backslashes, and quotes in the subtitle path before it is embedded in the `-vf` filtergraph string. This is required for paths that contain any of those characters.

## Command

```bash
python3 scripts/burn_subtitles.py \
    --video path/to/video.mp4 \
    --srt path/to/video.en.srt \
    --out path/to/video.burned.mp4
```

## Notes

- `ffmpeg` must be installed and on `PATH`.
- The script re-encodes the video (unlike the `ffmpeg -c copy` used in reel creation). Expect longer processing time proportional to video length.
- If the subtitle path escaping causes issues, run the `ffmpeg` command manually to see the full error.
