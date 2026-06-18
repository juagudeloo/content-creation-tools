# create_reel_from_video.py — Two-stage reel pipeline

`scripts/create_reel_from_video.py` is a two-stage tool. The stages are intentionally separate because crop coordinates and segment boundaries are video-specific and need a human review step before committing to a lossy re-encode.

```
Stage 1 — extract-clips   Find and cut the semantically relevant sections of a video.
Stage 2 — apply-format    Convert those clips to 9:16 vertical format using a YAML layout spec.
```

---

## Stage 1: extract-clips

```bash
python3 scripts/create_reel_from_video.py extract-clips <video.mp4> "<query>" [options]
```

Finds the transcript sections whose *meaning* is most similar to the query, cuts them from the source video, and writes one MP4 per detected occurrence of the topic.

### How it works

1. **Bootstrap the venv**: checks `REEL_BOOTSTRAP_READY`, creates `.venv` if needed, installs `requirements.txt` (which includes `sentence-transformers` and `pyyaml`), then re-execs under the venv Python.

2. **Auto-transcribe if needed**: if `--sub-file` is not provided, calls `generate_english_subtitles.py` with `--task transcribe --language es`, producing `<video_stem>.es.srt` in the original Spanish. This preserves the speaker's exact phrasing for the embedding step.

3. **Parse the SRT** into `(start_seconds, end_seconds, text)` entries.

4. **Group cues into topic paragraphs**: consecutive cues are merged as long as two conditions are both false:
   - the silence gap between cues exceeds `PARAGRAPH_GAP` (1.5 s) — topic shift signal
   - the paragraph would exceed `MAX_PARAGRAPH_DURATION` (90 s) — safety cap for continuous speech

5. **Embed and rank**: the query and all paragraph texts are encoded with the multilingual model on CPU. Cosine similarity is computed for each paragraph.

6. **Select relevant paragraphs**: paragraphs with similarity ≥ `SIMILARITY_THRESHOLD` (0.25) are kept, ranked highest-score first. If nothing clears the threshold, the top-3 by score are used as a fallback.

7. **Fill budget by relevance, then sort chronologically**: `trim_to_max_length` takes paragraphs from the highest-relevance end of the ranked list until the `--max-seconds` budget is filled. The kept segments are then re-sorted by start time so the reel plays in the original video order.

8. **Cluster into separate reels**: segments more than `CLUSTER_GAP` (60 s) apart start a new cluster — and a new reel. This separates distinct occurrences of a topic (e.g. the initial explanation and the closing summary).

9. **Clip and concatenate with ffmpeg** (`-c copy`, no re-encode) once per cluster.

10. **Verification report per reel**: the script embeds the reel's transcript against the query and prints a relevance score and short preview.

### Outputs per reel

- `<stem>-<query>-reel.mp4` (single reel) or `<stem>-<query>-reel-1.mp4`, `-2.mp4` … (multiple)
- Same pattern for `-script.txt` and `-en.srt`

### Options

| Flag | Default | Description |
|---|---|---|
| `--sub-file` | — | Existing SRT to skip transcription. |
| `--out` | next to input | Output path or directory. |
| `--max-seconds` | 60 | Per-reel length cap. |
| `--embed-model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model. |

### Tunable constants

| Constant | Default | Effect |
|---|---|---|
| `PARAGRAPH_GAP` | 1.5 s | Silence gap that signals a topic shift. |
| `MAX_PARAGRAPH_DURATION` | 90 s | Hard cap on paragraph length. Lower for finer topic separation. |
| `SIMILARITY_THRESHOLD` | 0.25 | Minimum cosine similarity to include a paragraph. |
| `CLUSTER_GAP` | 60 s | Gap that starts a new reel. |
| `MAX_REEL_SECONDS` | 60 s | Default `--max-seconds`. Applied per reel, not total. |

### Examples

```bash
# Auto-transcribe and search (Spanish video):
python3 scripts/create_reel_from_video.py extract-clips video.mp4 "cómo el modelo evita el olvido catastrófico"

# With an existing SRT (skips transcription):
python3 scripts/create_reel_from_video.py extract-clips video.mp4 "data augmentation" \
    --sub-file video.es.srt

# Custom output location and length:
python3 scripts/create_reel_from_video.py extract-clips video.mp4 "fine tuning" \
    --out /path/to/output/ --max-seconds 90
```

---

## Stage 2: apply-format

```bash
python3 scripts/create_reel_from_video.py apply-format <format.yaml> [--out-dir DIR]
```

Takes the reel files produced by `extract-clips` and converts each one to 1080×1920 (9:16) vertical format. You specify which time ranges should use a face crop vs. a screen crop.

### YAML format

```yaml
# Crop regions in the SOURCE video (w:h:x:y — ffmpeg crop filter syntax).
# For a 1920×1080 source: a ~607:1080 width gives a 9:16 aspect ratio.
crops:
  face:   "607:1080:1313:0"   # ADJUST: x offset where your webcam appears
  screen: "607:1080:0:0"      # ADJUST: x offset where the code/screen is

videos:
  video-data-augmentation-reel-1.mp4:
    segments:
      - {start: 0,  end: 18, type: face}
      - {start: 18, end: 50, type: screen}
      - {start: 50, end: 60, type: face}

  video-data-augmentation-reel-2.mp4:
    crops:                           # per-video override (merged with global crops)
      face: "607:1080:1000:0"
    output: summary-reel-formatted.mp4   # optional custom output name
    segments:
      - {start: 0, end: 60, type: screen}
```

**Key points:**

- `crops` at the top level are document-wide defaults.
- Each video entry can add its own `crops` block to override specific keys for that video.
- `output` is optional — defaults to `<stem>-formatted.mp4` next to the input, or inside `--out-dir`.
- Segment `start`/`end` are **seconds relative to the reel file** (i.e. 0-based after `extract-clips` has cut the clip).
- Each segment is re-encoded with `libx264 CRF 23 + AAC 192k`, then segments are concatenated without re-encoding.

### Finding your crop coordinates

The crop value `"w:h:x:y"` defines a rectangle in the source video:

- `w` and `h` — width and height of the region to keep
- `x` and `y` — top-left corner offset

For a 1920×1080 source and 9:16 output: keep `h=1080` and set `w=607` (1080×9/16 ≈ 607). Then adjust `x` to center on the area of interest. A 9:16 crop starting at the far right is `607:1080:1313:0`; starting at the far left is `607:1080:0:0`.

Use `ffprobe` or a video player with coordinate display to identify where your face and screen appear in the frame.

### Example

```bash
# After extract-clips has produced reel-1.mp4 and reel-2.mp4:
python3 scripts/create_reel_from_video.py apply-format format.yaml

# Place all formatted videos in a dedicated folder:
python3 scripts/create_reel_from_video.py apply-format format.yaml --out-dir /path/to/formatted/
```

### Verify output dimensions

```bash
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 \
    video-data-augmentation-reel-1-formatted.mp4
# expected: 1080,1920
```

---

## Requirements

- `ffmpeg` must be on `PATH`.
- `sentence-transformers` and `pyyaml` are installed automatically into `.venv` on first run.
- The default multilingual embedding model (~120 MB) is downloaded once from HuggingFace on first use.
- If your GPU is not compatible with the installed PyTorch build, the script forces CPU — no manual action needed.

## Troubleshooting

- **Wrong topic in the reel**: check the verification score printed after `extract-clips`. If below 0.25, rephrase the query. If moderate but wrong content, lower `MAX_PARAGRAPH_DURATION`.
- **No segments found**: the query may be too specific. Try a shorter phrase.
- **Reel cuts off mid-sentence**: increase `PARAGRAPH_GAP` to absorb longer pauses within the same topic.
- **`apply-format` crops the wrong area**: adjust the `x` offset in the crop string. Use a video player to find the pixel coordinates of your face or screen region.
- **ffmpeg concat warnings about non-monotonic DTS**: cosmetic, output plays correctly.
