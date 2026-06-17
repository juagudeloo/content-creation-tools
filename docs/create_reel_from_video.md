# create_reel_from_video.py — How to build a semantically-curated reel

`scripts/create_reel_from_video.py` assembles a short reel from an MP4 by finding the transcript sections whose *meaning* is most similar to a given topic. It uses `sentence-transformers` to embed both the query and each paragraph, ranks by cosine similarity, and cuts only the most relevant blocks. Everything runs locally — no API key, no internet connection beyond the one-time model download.

## Design rationale

Keyword matching requires the exact word to appear. Semantic embedding understands meaning: a query like "how the model avoids forgetting old knowledge" will surface sections that discuss catastrophic forgetting even if none of those exact words appear.

The default model (`paraphrase-multilingual-MiniLM-L12-v2`) is multilingual and understands Spanish, so it works correctly with Spanish-language source videos even when technical terms are in English. Auto-transcription keeps the original language (Spanish) rather than translating, so the embedding comparison operates on the same language the speaker used.

## Step-by-step

1. **Bootstrap the venv**: checks `REEL_BOOTSTRAP_READY`, creates `.venv` if needed, installs `requirements.txt` (which includes `sentence-transformers`), then re-execs under the venv Python.

2. **Auto-transcribe if needed**: if `--sub-file` is not provided, calls `generate_english_subtitles.py` with `--task transcribe --language es`, producing `<video_stem>.es.srt` in the original Spanish. This preserves the speaker's exact phrasing for the embedding step.

3. **Parse the SRT** into `(start_seconds, end_seconds, text)` entries.

4. **Group cues into topic paragraphs**: consecutive cues are merged as long as two conditions are both false:
   - the silence gap between cues exceeds `PARAGRAPH_GAP` (1.5 s) — topic shift signal
   - the paragraph would exceed `MAX_PARAGRAPH_DURATION` (90 s) — safety cap

   The duration cap is critical for videos with continuous speech and few pauses: without it, the entire video can collapse into 2–3 giant paragraphs, which dilutes the semantic signal and makes it impossible to distinguish topics from each other.

5. **Embed and rank**: the query and all paragraph texts are encoded into vectors with the multilingual model running on CPU. Cosine similarity is computed between the query vector and each paragraph vector.

6. **Select relevant paragraphs**: paragraphs with similarity ≥ `SIMILARITY_THRESHOLD` (0.25) are kept, ranked highest-score first. If nothing clears the threshold, the top-3 by score are used as a fallback.

7. **Fill budget by relevance, then sort chronologically**: `trim_to_max_length` takes paragraphs from the highest-relevance end of the ranked list until the `--max-seconds` budget is filled. The kept segments are then re-sorted by start time so the reel plays in the original video order.

   This order matters: picking by relevance first (not by earliest timestamp) ensures the most relevant section of the video ends up in the reel even if it appears after a less-relevant section that also cleared the threshold.

8. **Cluster into separate reels**: the selected segments are sorted chronologically and grouped by proximity. Any gap larger than `CLUSTER_GAP` (60 s) between two relevant segments starts a new cluster — and a new reel. This separates distinct occurrences of a topic (e.g. the initial concept explanation and the closing summary) instead of merging them into one video. Each cluster is trimmed independently to `--max-seconds`.

9. **Clip and concatenate with ffmpeg** (once per cluster): each range is cut with `-c copy` (no re-encode) and joined via a concat list file.

10. **Write outputs per reel**: if only one cluster is found, the original naming is used. Multiple clusters add a numeric suffix.
    - `<stem>-<query>-reel.mp4` (single) or `<stem>-<query>-reel-1.mp4`, `-2.mp4`, … (multiple)
    - Same pattern for `-script.txt` and `-en.srt`

11. **Verification report per reel**: after each reel is created, the script embeds its full transcript against the original query and prints a relevance score and short preview. This lets you confirm each reel covers the right topic and compare which occurrence scored highest.

## Commands

```bash
# Without a pre-existing SRT (auto-transcribes in Spanish):
python3 scripts/create_reel_from_video.py video.mp4 "cómo el modelo evita el olvido catastrófico"

# With an existing SRT (skips transcription):
python3 scripts/create_reel_from_video.py video.mp4 "data augmentation" \
    --sub-file video.en.srt

# Custom output location, length cap, and embedding model:
python3 scripts/create_reel_from_video.py video.mp4 "fine tuning on drone images" \
    --sub-file video.en.srt --out /path/to/output/ --max-seconds 90 \
    --embed-model paraphrase-multilingual-MiniLM-L12-v2
```

`--out` can be a file path or a directory. When it is a directory, the reel is placed inside it with the auto-generated name.

## Tunable constants

| Constant | Default | Effect |
|---|---|---|
| `PARAGRAPH_GAP` | 1.5 s | Silence gap that signals a topic shift between cues. |
| `MAX_PARAGRAPH_DURATION` | 90 s | Hard cap on paragraph length. Prevents continuous speech from collapsing into a single giant paragraph. Lower this if topics are short; raise it if you want broader context per segment. |
| `SIMILARITY_THRESHOLD` | 0.25 | Minimum cosine similarity to include a paragraph. Lower to include more sections; raise to be stricter. |
| `CLUSTER_GAP` | 60 s | Gap between relevant segments that starts a new reel. Raise if the same topic recurs quickly; lower if you want tighter separation. |
| `MAX_REEL_SECONDS` | 60 s | Default `--max-seconds` value. Applied per reel, not total. |

## Requirements

- `ffmpeg` must be on `PATH`.
- `sentence-transformers` is installed automatically into `.venv` on first run. The default multilingual model is ~120 MB and is downloaded once from HuggingFace on first use.
- PyTorch (pulled in by `sentence-transformers`) will attempt to use CUDA. If your GPU is not compatible with the installed PyTorch build, the script forces CPU via `device="cpu"` on the `SentenceTransformer` constructor — no manual action needed.

## Troubleshooting

- **Wrong topic in the reel**: check the verification score. If it is below 0.25, rephrase the query. If the score is moderate but the content is wrong, lower `MAX_PARAGRAPH_DURATION` so topics are more finely separated.
- **No segments found**: the query may be too specific or in a language the model isn't handling well. Try a shorter phrase or a multilingual model.
- **Reel cuts off mid-sentence**: the matched paragraph ended there. Increase `PARAGRAPH_GAP` to absorb longer pauses within the same topic.
- **ffmpeg concat warnings about non-monotonic DTS**: these are cosmetic. They appear when two clips from different positions in the source video are joined with `-c copy`. The output plays correctly.
