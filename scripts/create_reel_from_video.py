#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import List, Tuple

MAX_REEL_SECONDS = 60
PARAGRAPH_GAP = 1.5          # silence gap (seconds) that signals a new topic block
MAX_PARAGRAPH_DURATION = 90  # force-split paragraphs longer than this (seconds)
SIMILARITY_THRESHOLD = 0.25  # minimum cosine similarity to include a paragraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
BOOTSTRAP_ENV_VAR = "REEL_BOOTSTRAP_READY"


def ensure_virtual_environment() -> None:
    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return
    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)
    check = subprocess.run(
        [str(venv_python), "-c", "import sentence_transformers"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=PROJECT_ROOT)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True, cwd=PROJECT_ROOT)
    env = os.environ.copy()
    env[BOOTSTRAP_ENV_VAR] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a short reel by finding the transcript sections most semantically relevant to a topic."
    )
    parser.add_argument("input_video", type=Path, help="Input video file (MP4)")
    parser.add_argument("query", type=str, help="Topic or sentence describing what you want the reel to cover")
    parser.add_argument(
        "--sub-file", type=Path,
        help="Existing SRT to use as the transcription source. If absent, transcribes the video in its original language.",
    )
    parser.add_argument("--out", type=Path, help="Output path or directory for the reel. Defaults to next to the input video.")
    parser.add_argument("--max-seconds", type=int, default=MAX_REEL_SECONDS, help="Maximum reel length in seconds (default 60)")
    parser.add_argument(
        "--embed-model", default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model for semantic matching. Default is multilingual.",
    )
    return parser.parse_args()


def generate_srt(input_video: Path) -> Path:
    """Transcribe in the original language (no translation) to preserve meaning for embedding."""
    srt = input_video.with_name(f"{input_video.stem}.es.srt")
    subprocess.run(
        [
            "python3",
            str(PROJECT_ROOT / "scripts" / "generate_english_subtitles.py"),
            str(input_video),
            "--output-srt", str(srt),
            "--task", "transcribe",
            "--language", "es",
        ],
        check=True,
    )
    return srt


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    text = srt_path.read_text(encoding="utf-8")
    entries = []
    for part in re.split(r"\n\s*\n", text.strip()):
        lines = part.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
        if not m:
            continue
        def ts(s: str) -> float:
            h, mn, rest = s.split(":")
            sec, ms = rest.split(",")
            return int(h) * 3600 + int(mn) * 60 + int(sec) + int(ms) / 1000.0
        entries.append((ts(m.group(1)), ts(m.group(2)), "\n".join(lines[2:])))
    return entries


def group_into_paragraphs(entries: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Merge consecutive cues into topic paragraphs.

    A new paragraph starts when either:
    - there is a silence gap longer than PARAGRAPH_GAP, or
    - the current paragraph would exceed MAX_PARAGRAPH_DURATION.
    """
    if not entries:
        return []
    paragraphs = []
    p_start, p_end, texts = entries[0][0], entries[0][1], [entries[0][2]]
    for start, end, text in entries[1:]:
        silence_break = start - p_end > PARAGRAPH_GAP
        duration_break = (end - p_start) > MAX_PARAGRAPH_DURATION
        if silence_break or duration_break:
            paragraphs.append((p_start, p_end, " ".join(texts)))
            p_start, p_end, texts = start, end, [text]
        else:
            p_end = end
            texts.append(text)
    paragraphs.append((p_start, p_end, " ".join(texts)))
    return paragraphs


def load_embed_model(embed_model: str):
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(embed_model, device="cpu")


def find_relevant_segments(
    paragraphs: List[Tuple[float, float, str]],
    query: str,
    model,
) -> List[Tuple[float, float]]:
    """Return (start, end) pairs sorted by relevance score descending.

    trim_to_max_length will pick from the front of this list (highest score first)
    and then re-sort the selected segments chronologically for the video cut.
    """
    from sentence_transformers import util  # type: ignore

    texts = [text for _, _, text in paragraphs]
    query_vec = model.encode(query, convert_to_tensor=True)
    para_vecs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_vec, para_vecs)[0].tolist()

    scored = sorted(zip(scores, paragraphs), key=lambda x: x[0], reverse=True)

    # Keep paragraphs above threshold, ranked by score (highest first)
    selected = [(s, e) for score, (s, e, _) in scored if score >= SIMILARITY_THRESHOLD]
    if not selected:
        selected = [(s, e) for _, (s, e, _) in scored[:3]]

    return selected  # relevance order, NOT chronological


def verify_reel(
    reel_entries: List[Tuple[float, float, str]],
    query: str,
    model,
) -> None:
    from sentence_transformers import util  # type: ignore

    reel_text = " ".join(text for _, _, text in reel_entries)
    query_vec = model.encode(query, convert_to_tensor=True)
    reel_vec = model.encode(reel_text, convert_to_tensor=True)
    score = util.cos_sim(query_vec, reel_vec).item()

    print("\n--- Reel verification ---")
    print(f'Query:            "{query}"')
    print(f"Relevance score:  {score:.2f}", end="  ")
    if score >= 0.35:
        print("✓ Strong match")
    elif score >= 0.20:
        print("~ Moderate match")
    else:
        print("✗ Weak match — reel content may not correspond to the query")
    print("\nReel transcript:")
    preview = reel_text[:600] + (" …" if len(reel_text) > 600 else "")
    print(preview)


def trim_to_max_length(ranges: List[Tuple[float, float]], max_seconds: int) -> List[Tuple[float, float]]:
    """Fill the budget with the highest-relevance segments (ranges is relevance-ordered),
    then return the kept segments sorted chronologically for the video cut."""
    kept = []
    acc = 0.0
    for s, e in ranges:
        length = e - s
        if acc + length <= max_seconds:
            kept.append((s, e))
            acc += length
        else:
            remaining = max_seconds - acc
            if remaining > 0:
                kept.append((s, s + remaining))
            break
    return sorted(kept)  # chronological order for ffmpeg


def ffmpeg_trim_and_concat(input_video: Path, ranges: List[Tuple[float, float]], out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        parts = []
        for i, (s, e) in enumerate(ranges):
            out = td / f"part_{i}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(s), "-to", str(e), "-i", str(input_video), "-c", "copy", str(out)],
                check=True,
            )
            parts.append(out)
        listfile = td / "files.txt"
        listfile.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts))
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out_path)],
            check=True,
        )


def write_reel_srt(entries: List[Tuple[float, float, str]], out_srt: Path) -> None:
    def fmt(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3_600_000)
        mn, rem = divmod(rem, 60_000)
        sec, msn = divmod(rem, 1000)
        return f"{h:02d}:{mn:02d}:{sec:02d},{msn:03d}"
    lines = []
    for idx, (s, e, text) in enumerate(entries, start=1):
        lines += [str(idx), f"{fmt(s)} --> {fmt(e)}", text, ""]
    out_srt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_virtual_environment()
    args = parse_args()
    input_video = args.input_video.resolve()
    safe_query = re.sub(r"[^\w\s-]", "", args.query).strip().replace(" ", "-")[:40]
    default_name = f"{input_video.stem}-{safe_query}-reel.mp4"
    if args.out:
        out = args.out.resolve()
        if out.is_dir():
            out = out / default_name
    else:
        out = input_video.with_name(default_name)

    if args.sub_file:
        srt = args.sub_file.resolve()
    else:
        print(f"No --sub-file provided. Transcribing {input_video.name} in Spanish…")
        srt = generate_srt(input_video)

    entries = parse_srt(srt)
    paragraphs = group_into_paragraphs(entries)
    print(f"Loading embedding model ({args.embed_model})…")
    embed_model = load_embed_model(args.embed_model)

    print(f"Scoring {len(paragraphs)} topic segments against query…")
    hits = find_relevant_segments(paragraphs, args.query, embed_model)
    if not hits:
        raise SystemExit("No relevant segments found for that query.")

    hits = trim_to_max_length(hits, args.max_seconds)
    ffmpeg_trim_and_concat(input_video, hits, out)

    gathered = []
    reel_entries = []
    for s, e in hits:
        for start, end, text in entries:
            if not (end <= s or start >= e):
                gathered.append(text)
                reel_entries.append((max(start, s), min(end, e), text))

    script_txt = out.parent / f"{out.stem}-script.txt"
    script_txt.write_text("\n\n".join(gathered), encoding="utf-8")
    srt_out = out.parent / f"{out.stem}-en.srt"
    write_reel_srt(reel_entries, srt_out)

    print(f"\nReel created: {out}\nScript: {script_txt}\nSRT: {srt_out}")
    verify_reel(reel_entries, args.query, embed_model)


if __name__ == "__main__":
    main()
