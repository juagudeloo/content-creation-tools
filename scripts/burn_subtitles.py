#!/usr/bin/env python3
from pathlib import Path
import argparse
from scripts._utils import render_burned_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burn an existing SRT file into a video using ffmpeg.")
    parser.add_argument("--video", type=Path, required=True, help="Input video file to burn subtitles into.")
    parser.add_argument("--srt", type=Path, required=True, help="SRT subtitle file to burn into the video.")
    parser.add_argument("--out", type=Path, required=False, help="Optional output path. Defaults to input-video.burned.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.video.resolve()
    srt_file = args.srt.resolve()
    output_video = args.out.resolve() if args.out else input_video.with_name(f"{input_video.stem}.burned.mp4")
    render_burned_video(input_video, srt_file, output_video)


if __name__ == "__main__":
    main()
