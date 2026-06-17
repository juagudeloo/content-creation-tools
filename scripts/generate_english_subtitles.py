#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faster_whisper
from scripts._utils import escape_path_for_ffmpeg, render_burned_video


DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "es"
DEFAULT_OUTPUT_SUFFIX = ".en.srt"
MAX_CUE_CHARS = 84
MAX_LINE_WIDTH = 42
MIN_CUE_DURATION = 0.7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
BOOTSTRAP_ENV_VAR = "SUBTITLE_BOOTSTRAP_READY"


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create English subtitles for a Spanish video, bootstrapping a local virtual environment on demand."
    )
    parser.add_argument("input_video", type=Path, help="Input MP4 file.")
    parser.add_argument(
        "--output-srt",
        type=Path,
        help="Output SRT path. Defaults to the input video name with an .en.srt suffix.",
    )
    parser.add_argument(
        "--burned-video",
        type=Path,
        help="Optional output path for a video with burned-in subtitles. Defaults to <input>.en.mp4.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model size to use, for example tiny, base, small, medium, or large-v3.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Input speech language. Defaults to Spanish.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Inference device to use. Defaults to CPU for the most reliable automatic path.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type. Defaults to an automatic backend-safe choice.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for Whisper decoding.",
    )
    parser.add_argument(
        "--srt",
        type=Path,
        help="Optional path to an existing SRT. When provided together with --burned-video, the script will only burn that SRT into the video and exit.",
    )
    parser.add_argument(
        "--task",
        default="translate",
        choices=["translate", "transcribe"],
        help="Whisper task. 'translate' outputs English text from any language (default); 'transcribe' keeps the original language.",
    )
    return parser.parse_args()


def ensure_virtual_environment() -> None:
    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return

    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)

    dependency_check = subprocess.run(
        [str(venv_python), "-c", "import faster_whisper"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if dependency_check.returncode != 0:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            cwd=PROJECT_ROOT,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
            check=True,
            cwd=PROJECT_ROOT,
        )

    env = os.environ.copy()
    env[BOOTSTRAP_ENV_VAR] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def default_output_path(input_video: Path, task: str = "translate", language: str = DEFAULT_LANGUAGE) -> Path:
    suffix = DEFAULT_OUTPUT_SUFFIX if task == "translate" else f".{language}.srt"
    return input_video.with_name(f"{input_video.stem}{suffix}")


def extract_audio(input_video: Path, output_audio: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_audio),
    ]
    subprocess.run(command, check=True)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_whisper_model(model_name: str, device: str, compute_type: str):
    preferred_compute_types = ["float16" if device == "cuda" else "float32", "int8_float16", "int8"]
    if compute_type != "auto":
        preferred_compute_types.append(compute_type)

    last_error: Exception | None = None
    for candidate in preferred_compute_types:
        try:
            return faster_whisper.WhisperModel(model_name, device=device, compute_type=candidate)
        except ValueError as error:
            last_error = error

    if last_error is not None:
        raise last_error

    return faster_whisper.WhisperModel(model_name, device=device, compute_type="float32")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text_for_subtitles(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= MAX_CUE_CHARS:
        return [text]

    fragments = [fragment.strip() for fragment in re.split(r"(?<=[,;:.!?])\s+", text) if fragment.strip()]
    if len(fragments) == 1:
        return chunk_by_words(text)

    pieces: list[str] = []
    buffer = ""
    for fragment in fragments:
        candidate = f"{buffer} {fragment}".strip() if buffer else fragment
        if len(candidate) <= MAX_CUE_CHARS:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        if len(fragment) <= MAX_CUE_CHARS:
            buffer = fragment
        else:
            pieces.extend(chunk_by_words(fragment))
            buffer = ""
    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_by_words(text: str, max_chars: int = MAX_CUE_CHARS) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))
    return chunks


def wrap_for_srt(text: str) -> str:
    words = text.split()
    if len(words) <= 1:
        return text

    wrapped = textwrap.wrap(
        text,
        width=MAX_LINE_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) <= 2:
        return "\n".join(wrapped)

    midpoint = max(1, len(words) // 2)
    first_half = " ".join(words[:midpoint])
    second_half = " ".join(words[midpoint:])
    first_line = " ".join(textwrap.wrap(first_half, width=MAX_LINE_WIDTH, break_long_words=False, break_on_hyphens=False))
    second_line = " ".join(textwrap.wrap(second_half, width=MAX_LINE_WIDTH, break_long_words=False, break_on_hyphens=False))
    return f"{first_line}\n{second_line}".strip()


def split_segment_into_cues(start: float, end: float, text: str) -> list[Cue]:
    pieces = split_text_for_subtitles(text)
    if not pieces:
        return []

    if len(pieces) == 1 or (end - start) < MIN_CUE_DURATION * len(pieces):
        return [Cue(start=start, end=end, text=wrap_for_srt(pieces[0]))]

    duration = max(end - start, MIN_CUE_DURATION * len(pieces))
    total_weight = sum(max(len(piece.split()), 1) for piece in pieces)
    cursor = start
    cues: list[Cue] = []

    for index, piece in enumerate(pieces):
        if index == len(pieces) - 1:
            piece_end = end
        else:
            weight = max(len(piece.split()), 1)
            allocated = duration * (weight / total_weight)
            piece_end = max(cursor + MIN_CUE_DURATION, cursor + allocated)
            remaining_pieces = len(pieces) - index - 1
            max_end = end - remaining_pieces * MIN_CUE_DURATION
            piece_end = min(piece_end, max_end)
        cues.append(Cue(start=cursor, end=piece_end, text=wrap_for_srt(piece)))
        cursor = piece_end

    return cues


def collect_cues(model: Any, audio_path: Path, language: str, beam_size: int, task: str = "translate") -> list[Cue]:
    segments, _info = model.transcribe(
        str(audio_path),
        task=task,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )

    cues: list[Cue] = []
    for segment in segments:
        text = normalize_text(segment.text)
        if not text:
            continue
        cues.extend(split_segment_into_cues(segment.start, segment.end, text))
    return cues


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(cues: Iterable[Cue], output_srt: Path) -> None:
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    output_srt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# rendering helpers moved to scripts/_utils.py and imported above


def main() -> None:
    ensure_virtual_environment()
    args = parse_args()
    input_video = args.input_video.resolve()
    output_srt = (args.output_srt or default_output_path(input_video, args.task, args.language)).resolve()
    burned_video = args.burned_video.resolve() if args.burned_video else None
    provided_srt = args.srt.resolve() if args.srt else None
    device = resolve_device(args.device)

    # If the user provided an SRT and an explicit burned-video path, only burn and exit.
    if provided_srt is not None and burned_video is not None:
        render_burned_video(input_video, provided_srt, burned_video)
        return

    # Do not allow --burned-video without specifying --srt. To burn, either use --srt with this script
    # or use the dedicated `scripts/burn_subtitles.py` utility.
    if burned_video is not None and provided_srt is None:
        raise SystemExit("Refusing to run full pipeline and burn in one step. Provide --srt for burning or use scripts/burn_subtitles.py")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = Path(temp_dir) / "audio.wav"
        extract_audio(input_video, temp_audio)
        model = load_whisper_model(args.model, device=device, compute_type=args.compute_type)
        cues = collect_cues(model, temp_audio, args.language, args.beam_size, task=args.task)
        write_srt(cues, output_srt)
    # Generate only: burning must be done explicitly with --srt or with scripts/burn_subtitles.py


if __name__ == "__main__":
    main()