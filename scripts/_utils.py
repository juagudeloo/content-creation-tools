from pathlib import Path
import subprocess


def escape_path_for_ffmpeg(path: Path) -> str:
    resolved = path.resolve().as_posix()
    return resolved.replace("\\", r"\\\\").replace(":", r"\:").replace("'", r"\'")


def render_burned_video(input_video: Path, subtitle_file: Path, output_video: Path) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    subtitle_filter = f"subtitles=filename='{escape_path_for_ffmpeg(subtitle_file)}'"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video),
    ]
    subprocess.run(command, check=True)
