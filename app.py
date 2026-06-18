#!/usr/bin/env python3
"""Gradio front-end for the content-creator-tools pipeline.

Combines the existing CLI scripts into a single interface with three tabs:

1. Generate subtitles  -> scripts/generate_english_subtitles.py
2. Burn subtitles       -> scripts/_utils.render_burned_video (with an SRT editor)
3. Create reel
     (a) Clip creation     -> scripts/create_reel_from_video.py extract-clips
     (b) Reel compilation   -> scripts/create_reel_from_video.format_reel

This module is *only* the visualization / interaction layer. All the heavy
lifting (transcription, semantic clip extraction, ffmpeg cropping) lives in the
scripts and is reused here rather than re-implemented.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
# The app and the CLI scripts share a single .venv (see requirements.txt). The
# whole stack is pinned to the gradio-4.44 era so gradio-path-selector works.
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
BOOTSTRAP_ENV_VAR = "APP_BOOTSTRAP_READY"


# ── Self-bootstrapping venv (same pattern as the scripts) ─────────────────────

def ensure_virtual_environment() -> None:
    if os.environ.get(BOOTSTRAP_ENV_VAR) == "1":
        return
    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    check = subprocess.run(
        [str(venv_python), "-c", "import gradio, gradio_path_selector, PIL"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=PROJECT_ROOT)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True, cwd=PROJECT_ROOT)
    env = os.environ.copy()
    env[BOOTSTRAP_ENV_VAR] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


ensure_virtual_environment()

# Everything below runs under .venv -------------------------------------------
import gradio as gr  # noqa: E402

# Work around a known gradio 4.x API-schema bug: when a component's JSON schema
# has a boolean `additionalProperties` (as PathSelector's free-form JSON does),
# gradio_client.utils.get_type does `"const" in schema` on a bool and crashes,
# which in turn breaks launch()'s localhost health check. Tolerate bools.
import gradio_client.utils as _gc_utils  # noqa: E402

_gc_orig_get_type = _gc_utils.get_type
_gc_orig_js2pt = _gc_utils._json_schema_to_python_type


def _gc_get_type(schema):  # type: ignore[no-redef]
    if isinstance(schema, bool):
        return "Any"
    return _gc_orig_get_type(schema)


def _gc_js2pt(schema, defs=None):  # type: ignore[no-redef]
    if isinstance(schema, bool):
        return "Any"
    return _gc_orig_js2pt(schema, defs)


_gc_utils.get_type = _gc_get_type
_gc_utils._json_schema_to_python_type = _gc_js2pt

# gradio 4.44 refuses to launch unless a HEAD request to the local URL returns
# 200/401/302 within ~2.5s. That probe is a false-negative in some environments
# even though the server serves fine. This is a local single-user app, so skip it.
import gradio.networking as _gr_networking  # noqa: E402

_gr_networking.url_ok = lambda url: True

from gradio_path_selector import PathSelector  # noqa: E402
from gradio_image_annotation import image_annotator  # noqa: E402

# Reuse the existing scripts as a library (importing does NOT trigger their
# own bootstrap — that only happens inside their main()).
sys.path.insert(0, str(SCRIPTS_DIR))
from _utils import render_burned_video  # noqa: E402
import create_reel_from_video as reel  # noqa: E402
import generate_english_subtitles as subs  # noqa: E402

WORK_DIR = Path(tempfile.gettempdir()) / "content_creator_app"
WORK_DIR.mkdir(parents=True, exist_ok=True)

GEN_SUBS_SCRIPT = SCRIPTS_DIR / "generate_english_subtitles.py"
REEL_SCRIPT = SCRIPTS_DIR / "create_reel_from_video.py"


# ── Generic helpers ───────────────────────────────────────────────────────────

def _clean_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None
    return Path(raw).expanduser()


# Where the in-app folder browser starts (near the user's content).
FOLDER_BROWSER_START = PROJECT_ROOT.parent


def _selector_path(value: Any) -> str:
    """Extract the chosen folder from a PathSelector value (a dict) or a string."""
    if isinstance(value, dict):
        return value.get("current_path", "") or ""
    return value or ""


def video_dimensions(video: Path) -> Tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def video_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def grab_frame(video: Path, t: float, out_png: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(max(t, 0)), "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(out_png)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out_png


def stream_command(cmd: List[str]):
    """Run a command, yielding accumulated output as it is produced."""
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    buffer: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        buffer.append(line)
        yield "".join(buffer), False
    proc.wait()
    ok = proc.returncode == 0
    if not ok:
        buffer.append(f"\n[process exited with code {proc.returncode}]")
    yield "".join(buffer), ok


# ── Tab 1: Generate subtitles ─────────────────────────────────────────────────

def generate_subtitles(video_path: str, output_dir: str, model: str, task: str, language: str):
    video = _clean_path(video_path)
    if video is None or not video.exists():
        yield f"❌ Video not found: {video_path}", None
        return
    video = video.resolve()

    out_dir = _clean_path(_selector_path(output_dir))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = subs.DEFAULT_OUTPUT_SUFFIX if task == "translate" else f".{language}.srt"
        out_srt = (out_dir / f"{video.stem}{suffix}").resolve()
    else:
        out_srt = subs.default_output_path(video, task, language).resolve()

    cmd = ["python3", str(GEN_SUBS_SCRIPT), str(video),
           "--output-srt", str(out_srt), "--model", model,
           "--task", task, "--language", language]

    header = f"$ {' '.join(cmd)}\n\n"
    log = ""
    for log, ok in stream_command(cmd):
        yield header + log, None
    if out_srt.exists():
        yield f"{header}{log}\n\n✅ Subtitles written to: {out_srt}", str(out_srt)
    else:
        yield f"{header}{log}\n\n❌ Did not produce an SRT — see log above.", None


# ── Tab 2: Burn subtitles ─────────────────────────────────────────────────────

def load_burn_srt(srt_path: str):
    """Read an uploaded .srt into the editor."""
    srt = _clean_path(srt_path)
    if srt and srt.exists():
        return srt.read_text(encoding="utf-8"), "✅ Subtitles loaded — edit, then save with Ctrl+S."
    return "", "⚠️ Upload a subtitle (.srt) file."


def save_srt(srt_path: str, content: str):
    srt = _clean_path(srt_path)
    if srt is None:
        return "❌ No SRT path provided."
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text(content, encoding="utf-8")
    return f"💾 Saved {srt.name} ({len(content.splitlines())} lines)."


def apply_burning(video_path: str, srt_path: str, content: str, output_path: str):
    video = _clean_path(video_path)
    srt = _clean_path(srt_path)
    if video is None or not video.exists():
        return None, f"❌ Video not found: {video_path}"
    if srt is None:
        return None, "❌ No SRT path provided."
    # Persist the latest editor content before burning so what you see is burned.
    srt.write_text(content, encoding="utf-8")

    output_path = _selector_path(output_path)
    out = _clean_path(output_path)
    default_name = f"{video.stem}.burned.mp4"
    if out is None:
        out = video.with_name(default_name)
    # A folder (existing dir or trailing slash) means "put the file in here".
    elif out.is_dir() or output_path.strip().endswith(("/", os.sep)):
        out = out / default_name
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_burned_video(video.resolve(), srt.resolve(), out)
    except subprocess.CalledProcessError as exc:
        return None, f"❌ ffmpeg failed: {exc}"
    return str(out), f"✅ Burned video written to: {out}"


# ── Tab 3a: Clip creation ─────────────────────────────────────────────────────

def create_clips(video_path: str, sentence: str, output_dir: str, sub_file: str, max_seconds: int):
    """gr.Gallery only renders images, so created *videos* are exposed through a
    dropdown selector plus a video preview instead."""
    empty = gr.update(choices=[], value=None)
    video = _clean_path(video_path)
    if video is None or not video.exists():
        yield f"❌ Video not found: {video_path}", empty, None
        return
    video = video.resolve()
    if not sentence or not sentence.strip():
        yield "❌ Please provide a sentence / topic to search for.", empty, None
        return

    out_dir = _clean_path(_selector_path(output_dir))
    out_dir = out_dir.resolve() if out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python3", str(REEL_SCRIPT), "extract-clips", str(video), sentence.strip(),
           "--out", str(out_dir), "--max-seconds", str(int(max_seconds))]
    sub = _clean_path(sub_file)
    if sub and sub.exists():
        cmd += ["--sub-file", str(sub.resolve())]

    before = set(out_dir.glob("*.mp4"))
    header = f"$ {' '.join(cmd)}\n\n"
    log = ""
    for log, ok in stream_command(cmd):
        yield header + log, gr.update(), None

    new_clips = sorted(p for p in out_dir.glob("*-reel*.mp4") if p not in before)
    if not new_clips:
        new_clips = sorted(p for p in out_dir.glob("*-reel*.mp4"))
    choices = [(p.name, str(p)) for p in new_clips]
    first = str(new_clips[0]) if new_clips else None
    suffix = (f"\n\n✅ Created {len(new_clips)} clip(s) in {out_dir}"
              if new_clips else "\n\n⚠️ No clips detected — check the log above.")
    yield header + log + suffix, gr.update(choices=choices, value=first), first


# ── Tab 3b: Reel compilation ──────────────────────────────────────────────────

def default_crop(src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    """A centered 9:16 crop covering the full source height."""
    h = src_h
    w = min(src_w, int(round(h * 9 / 16)))
    x = max(0, (src_w - w) // 2)
    return x, 0, w, h


def load_reel_source(video_path: str):
    video = _clean_path(video_path)
    if video is None or not video.exists():
        msg = "Upload a video to begin." if not video_path else f"❌ Video not found: {video_path}"
        return (None, None, gr.update(), gr.update(), 0, 0, 0, 0, 0, msg, None)
    video = video.resolve()
    src_w, src_h = video_dimensions(video)
    duration = video_duration(video)
    x, y, w, h = default_crop(src_w, src_h)
    dims = {"w": src_w, "h": src_h, "duration": duration, "path": str(video)}
    status = f"✅ Loaded {video.name} — {src_w}×{src_h}, {duration:.1f}s"
    return (
        str(video), dims,
        gr.update(maximum=max(duration, 1), value=0),          # snapshot slider
        gr.update(maximum=max(duration, 1), value=round(duration, 1)),  # interval end slider
        x, y, w, h,                                            # crop x/y/w/h
        0,                                                     # interval start
        status,
        (x, y, w, h),                                          # crop_state guard
    )


# ── 9:16 reel-crop geometry (shared by the draw canvas and the number fields) ──

REEL_RATIO = 9 / 16  # width / height — a reel crop must always keep this shape.


def _clamp_crop(x, y, w, h, src_w, src_h):
    w = max(16, min(int(round(w)), src_w))
    h = max(16, min(int(round(h)), src_h))
    x = max(0, min(int(round(x)), src_w - w))
    y = max(0, min(int(round(y)), src_h - h))
    return x, y, w, h


def snap_to_reel(x, y, w, h, src_w, src_h, drive="h"):
    """Force a crop box to 9:16, deriving the missing side, clamped to the frame.

    drive='h' keeps the height and derives the width (used when drawing / editing
    height); drive='w' keeps the width and derives the height.
    """
    x, y, w, h = float(x), float(y), float(w), float(h)
    if drive == "w":
        h = w / REEL_RATIO
    else:
        w = h * REEL_RATIO
    # If the derived side overflows the frame, shrink from the other side.
    if h > src_h:
        h = src_h
        w = h * REEL_RATIO
    if w > src_w:
        w = src_w
        h = w / REEL_RATIO
    return _clamp_crop(x, y, w, h, src_w, src_h)


def crop_to_box(x, y, w, h):
    return {"xmin": int(x), "ymin": int(y), "xmax": int(x + w), "ymax": int(y + h),
            "label": "reel", "color": (0, 230, 0)}


def annot_value(image: Optional[str], x, y, w, h):
    if not image:
        return None
    return {"image": image, "boxes": [crop_to_box(x, y, w, h)]}


def take_snapshot(dims: Optional[Dict], t: float, x, y, w, h):
    if not dims:
        return None, None, "⚠️ Load a video first.", gr.update()
    video = Path(dims["path"])
    snap = WORK_DIR / f"snap_{uuid.uuid4().hex}.png"
    grab_frame(video, float(t), snap)
    x, y, w, h = snap_to_reel(x, y, w, h, dims["w"], dims["h"])
    return str(snap), annot_value(str(snap), x, y, w, h), f"📸 Snapshot at {float(t):.1f}s — drag the green box to choose the region.", (x, y, w, h)


def on_draw(annot, dims, snapshot_path, crop_state):
    """User drew/dragged the box on the canvas → snap to 9:16 and sync the numbers."""
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), crop_state)
    if not dims or not annot or not annot.get("boxes"):
        return noop
    box = annot["boxes"][0]
    x, y, w, h = box["xmin"], box["ymin"], box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    crop = snap_to_reel(x, y, w, h, dims["w"], dims["h"], drive="h")
    if crop == crop_state:                       # already canonical → stop the echo
        return noop
    cx, cy, cw, ch = crop
    return annot_value(snapshot_path, *crop), cx, cy, cw, ch, crop


def on_numbers(x, y, w, h, dims, snapshot_path, crop_state):
    """User edited a crop number → keep 9:16 and redraw the box."""
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), crop_state)
    if not dims:
        return noop
    px, py, pw, ph = crop_state or (x, y, w, h)
    drive = "w" if (int(w) != int(pw) and int(h) == int(ph)) else "h"
    crop = snap_to_reel(x, y, w, h, dims["w"], dims["h"], drive=drive)
    if crop == crop_state:
        return noop
    cx, cy, cw, ch = crop
    box = annot_value(snapshot_path, *crop) if snapshot_path else gr.update()
    return box, cx, cy, cw, ch, crop


def _render_clip(video: Path, start: float, end: float, crop: str) -> Path:
    out = WORK_DIR / f"clip_{uuid.uuid4().hex}.mp4"
    seg = {"start": float(start), "end": float(end), "type": "c"}
    reel.format_reel(video, [seg], {"c": crop}, out)
    return out


def _gallery_items(clips: List[Dict]) -> List[Tuple[str, str]]:
    return [(c["thumb"], f"#{i + 1}  {c['start']:.0f}-{c['end']:.0f}s")
            for i, c in enumerate(clips)]


def add_clip(dims, clips, start, end, x, y, w, h):
    clips = list(clips or [])
    if not dims:
        return clips, _gallery_items(clips), None, "⚠️ Load a video first.", -1
    if float(end) <= float(start):
        return clips, _gallery_items(clips), None, "❌ Interval end must be after start.", -1
    video = Path(dims["path"])
    crop = f"{int(w)}:{int(h)}:{int(x)}:{int(y)}"
    try:
        clip_path = _render_clip(video, start, end, crop)
    except subprocess.CalledProcessError as exc:
        return clips, _gallery_items(clips), None, f"❌ ffmpeg failed: {exc}", -1
    thumb = WORK_DIR / f"thumb_{uuid.uuid4().hex}.png"
    grab_frame(clip_path, 0.1, thumb)
    clips.append({
        "start": float(start), "end": float(end),
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "crop": crop, "video": str(clip_path), "thumb": str(thumb),
    })
    return clips, _gallery_items(clips), str(clip_path), f"✅ Added clip #{len(clips)}.", -1


def update_clip(dims, clips, idx, start, end, x, y, w, h):
    clips = list(clips or [])
    if idx is None or idx < 0 or idx >= len(clips):
        return clips, _gallery_items(clips), None, "⚠️ Select a clip from the carousel first.", idx
    if not dims:
        return clips, _gallery_items(clips), None, "⚠️ Load a video first.", idx
    if float(end) <= float(start):
        return clips, _gallery_items(clips), None, "❌ Interval end must be after start.", idx
    video = Path(dims["path"])
    crop = f"{int(w)}:{int(h)}:{int(x)}:{int(y)}"
    try:
        clip_path = _render_clip(video, start, end, crop)
    except subprocess.CalledProcessError as exc:
        return clips, _gallery_items(clips), None, f"❌ ffmpeg failed: {exc}", idx
    thumb = WORK_DIR / f"thumb_{uuid.uuid4().hex}.png"
    grab_frame(clip_path, 0.1, thumb)
    clips[idx] = {
        "start": float(start), "end": float(end),
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "crop": crop, "video": str(clip_path), "thumb": str(thumb),
    }
    return clips, _gallery_items(clips), str(clip_path), f"✅ Updated clip #{idx + 1}.", idx


def select_clip(dims, clips, evt: gr.SelectData):
    """Load the selected clip onto the player and back into the control center."""
    clips = list(clips or [])
    idx = evt.index
    if idx is None or idx < 0 or idx >= len(clips):
        return (None, idx, gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), None, "", gr.update())
    c = clips[idx]
    # Refresh the snapshot at the clip's start time with its crop box.
    annot = None
    snap_path = None
    if dims:
        snap = WORK_DIR / f"snap_{uuid.uuid4().hex}.png"
        grab_frame(Path(dims["path"]), c["start"], snap)
        snap_path = str(snap)
        annot = annot_value(snap_path, c["x"], c["y"], c["w"], c["h"])
    return (
        c["video"], idx,
        c["start"], c["end"], c["x"], c["y"], c["w"], c["h"],
        annot, snap_path,
        f"▶️ Loaded clip #{idx + 1} into the control center.",
        (c["x"], c["y"], c["w"], c["h"]),
    )


def delete_clip(clips, idx):
    clips = list(clips or [])
    if idx is None or idx < 0 or idx >= len(clips):
        return clips, _gallery_items(clips), None, "⚠️ Select a clip to delete.", -1
    removed = clips.pop(idx)
    return clips, _gallery_items(clips), None, f"🗑️ Deleted clip ({removed['start']:.0f}-{removed['end']:.0f}s).", -1


def compile_reel(dims, clips, output_path):
    clips = list(clips or [])
    if not dims:
        return None, "⚠️ Load a video first."
    if not clips:
        return None, "⚠️ Add at least one clip before compiling."
    video = Path(dims["path"])
    output_path = _selector_path(output_path)
    out = _clean_path(output_path)
    default_name = f"{video.stem}-reel-compiled.mp4"
    if out is None:
        out = WORK_DIR / default_name
    elif out.is_dir() or output_path.strip().endswith(("/", os.sep)):
        out = out / default_name
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = [{"start": c["start"], "end": c["end"], "type": f"c{i}"} for i, c in enumerate(clips)]
    crops = {f"c{i}": c["crop"] for i, c in enumerate(clips)}
    try:
        reel.format_reel(video, segments, crops, out)
    except subprocess.CalledProcessError as exc:
        return None, f"❌ ffmpeg failed: {exc}"
    return str(out), f"✅ Compiled {len(clips)} clip(s) into: {out}"


# ── UI ─────────────────────────────────────────────────────────────────────────

CSS = """
#apply-burn-btn, #create-reel-btn {
    background: linear-gradient(90deg,#16a34a,#15803d) !important;
    color: #fff !important; font-weight: 700 !important; border: none !important;
}
#apply-burn-btn:hover, #create-reel-btn:hover { filter: brightness(1.1); }
.srt-editor textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
    white-space: pre !important; line-height: 1.4 !important;
}
"""

# Ctrl+S in the Burn tab triggers the (hidden-id) save button.
HEAD = """
<script>
document.addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
    const btn = document.querySelector('#burn-save-btn');
    if (btn) { e.preventDefault(); btn.click(); }
  }
});
</script>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Content Creator Tools", css=CSS, head=HEAD) as demo:
        gr.Markdown("# 🎬 Content Creator Tools\nSubtitle generation, burning, and reel building in one place. "
                    "Click a file box to open your file manager and pick a video.")

        def with_folder_picker(label: str, placeholder: str = ""):
            """Optional output-folder browser (gradio-path-selector). The folder you
            navigate into is where a copy of the result is also saved."""
            with gr.Accordion(f"📁 {label} — browse to a folder to also save a copy there", open=False):
                ps = PathSelector(label=label, value=PathSelector.get_value(FOLDER_BROWSER_START))
            return ps

        with gr.Tabs():
            # ── Tab 1 ──────────────────────────────────────────────────────────
            with gr.Tab("Generate subtitles"):
                gr.Markdown("Transcribe/translate a video into an `.srt` file.")
                gs_video = gr.File(label="Video", file_types=["video", ".mp4", ".mov", ".mkv"])
                gs_output = with_folder_picker("Output folder (optional)",
                                               "Leave empty to just download the result below")
                with gr.Row():
                    gs_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"],
                                           value=subs.DEFAULT_MODEL, label="Whisper model")
                    gs_task = gr.Radio(["translate", "transcribe"], value="translate",
                                       label="Task (translate→English, transcribe→original)")
                    gs_lang = gr.Textbox(value=subs.DEFAULT_LANGUAGE, label="Language")
                gs_btn = gr.Button("Generate subtitles", variant="primary")
                gs_log = gr.Textbox(label="Log", lines=14, interactive=False)
                gs_result = gr.File(label="Generated subtitles (.srt) — click to download")
                gs_btn.click(generate_subtitles,
                             [gs_video, gs_output, gs_model, gs_task, gs_lang],
                             [gs_log, gs_result])

            # ── Tab 2 ──────────────────────────────────────────────────────────
            with gr.Tab("Burn subtitles"):
                gr.Markdown("Pick a video and its subtitles, edit them on the right, then burn them in. "
                            "Save with the button or **Ctrl+S**.")
                with gr.Row():
                    bs_video_file = gr.File(label="Video", file_types=["video", ".mp4", ".mov", ".mkv"])
                    bs_srt_file = gr.File(label="Subtitles (.srt)", file_types=[".srt"])
                bs_apply = gr.Button("🔥 Apply burning", elem_id="apply-burn-btn")
                bs_status = gr.Markdown()
                with gr.Row():
                    bs_video = gr.Video(label="Video", scale=1)
                    with gr.Column(scale=1):
                        bs_editor = gr.Textbox(label="Subtitles (.srt)", lines=20, max_lines=20,
                                               interactive=True, elem_classes=["srt-editor"])
                        bs_save = gr.Button("💾 Save (Ctrl+S)", elem_id="burn-save-btn")
                bs_out_path = with_folder_picker("Output folder (optional)",
                                                 "Leave empty to just download the result below")
                bs_burned = gr.Video(label="Burned result")

                bs_video_file.change(lambda p: p, bs_video_file, bs_video)
                bs_srt_file.change(load_burn_srt, bs_srt_file, [bs_editor, bs_status])
                bs_save.click(save_srt, [bs_srt_file, bs_editor], [bs_status])
                bs_apply.click(apply_burning,
                               [bs_video_file, bs_srt_file, bs_editor, bs_out_path],
                               [bs_burned, bs_status])

            # ── Tab 3 ──────────────────────────────────────────────────────────
            with gr.Tab("Create reel"):
                with gr.Tabs():
                    # ── 3a: Clip creation ──────────────────────────────────────
                    with gr.Tab("Clip creation"):
                        gr.Markdown("Find the segments most relevant to a sentence and cut them.")
                        cc_video = gr.File(label="Video", file_types=["video", ".mp4", ".mov", ".mkv"])
                        cc_sentence = gr.Textbox(label="Sentence / topic",
                                                 placeholder="What should the reel be about?")
                        cc_output = with_folder_picker("Output folder (optional)",
                                                       "Where to save the clips (download below otherwise)")
                        cc_sub = gr.File(label="Subtitle file (optional, reuse an existing .srt)",
                                         file_types=[".srt"])
                        cc_max = gr.Slider(10, 180, value=reel.MAX_REEL_SECONDS, step=5,
                                           label="Max reel length (seconds)")
                        cc_btn = gr.Button("Create clips", variant="primary")
                        cc_log = gr.Textbox(label="Log", lines=14, interactive=False)
                        cc_select = gr.Dropdown(label="Created clips (select to preview)",
                                                choices=[], interactive=True)
                        cc_preview = gr.Video(label="Clip preview")
                        cc_btn.click(create_clips,
                                     [cc_video, cc_sentence, cc_output, cc_sub, cc_max],
                                     [cc_log, cc_select, cc_preview])
                        cc_select.change(lambda p: p, cc_select, cc_preview)

                    # ── 3b: Reel compilation ──────────────────────────────────
                    with gr.Tab("Reel compilation"):
                        clips_state = gr.State([])
                        dims_state = gr.State(None)
                        snapshot_state = gr.State(None)
                        selected_state = gr.State(-1)
                        crop_state = gr.State(None)

                        rc_video_file = gr.File(label="Video", file_types=["video", ".mp4", ".mov", ".mkv"])
                        rc_compile = gr.Button("🎬 Create reel", elem_id="create-reel-btn")
                        rc_status = gr.Markdown()

                        with gr.Row():
                            # Left: source player
                            with gr.Column(scale=1):
                                gr.Markdown("### Source")
                                rc_video = gr.Video(label="Uploaded video")

                            # Middle: control center
                            with gr.Column(scale=1):
                                gr.Markdown("### Region & interval")
                                rc_snap_time = gr.Slider(0, 1, value=0, step=0.5,
                                                         label="Snapshot time (s)")
                                rc_snap_btn = gr.Button("📸 Take snapshot")
                                rc_annotator = image_annotator(
                                    label="Drag the green box (locked to 9:16 reel shape)",
                                    single_box=True, disable_edit_boxes=True, height=300,
                                    show_download_button=False, show_share_button=False,
                                )
                                with gr.Row():
                                    rc_x = gr.Number(label="Crop X", value=0, precision=0)
                                    rc_y = gr.Number(label="Crop Y", value=0, precision=0)
                                with gr.Row():
                                    rc_w = gr.Number(label="Crop W", value=0, precision=0)
                                    rc_h = gr.Number(label="Crop H", value=0, precision=0)
                                with gr.Row():
                                    rc_start = gr.Number(label="Interval start (s)", value=0, precision=1)
                                    rc_end = gr.Slider(0, 1, value=0, step=0.5, label="Interval end (s)")
                                with gr.Row():
                                    rc_add = gr.Button("➕ Add clip", variant="primary")
                                    rc_update = gr.Button("✏️ Update selected")
                                    rc_delete = gr.Button("🗑️ Delete selected")

                            # Right: clip carousel + preview
                            with gr.Column(scale=1):
                                gr.Markdown("### Session clips")
                                rc_preview = gr.Video(label="Selected clip preview")
                                # No fixed height: the gallery grows with the clips and the
                                # page scrolls, so every clip stays reachable/clickable.
                                rc_gallery = gr.Gallery(label="Carousel (click to load)",
                                                        columns=2, object_fit="cover",
                                                        allow_preview=False)

                        rc_reel_out = with_folder_picker("Reel output folder (optional)",
                                                         "Leave empty to just download the result below")
                        rc_final = gr.Video(label="Compiled reel")

                        # Wiring
                        rc_video_file.change(
                            load_reel_source, [rc_video_file],
                            [rc_video, dims_state, rc_snap_time, rc_end,
                             rc_x, rc_y, rc_w, rc_h, rc_start, rc_status, crop_state],
                        )
                        rc_snap_btn.click(
                            take_snapshot, [dims_state, rc_snap_time, rc_x, rc_y, rc_w, rc_h],
                            [snapshot_state, rc_annotator, rc_status, crop_state],
                        )
                        # Bidirectional crop editing: draw the box ↔ edit the numbers,
                        # always re-snapped to 9:16 (crop_state guards against echo loops).
                        crop_io = [rc_annotator, rc_x, rc_y, rc_w, rc_h, crop_state]
                        rc_annotator.change(
                            on_draw, [rc_annotator, dims_state, snapshot_state, crop_state], crop_io,
                        )
                        for ctrl in (rc_x, rc_y, rc_w, rc_h):
                            ctrl.change(
                                on_numbers,
                                [rc_x, rc_y, rc_w, rc_h, dims_state, snapshot_state, crop_state],
                                crop_io,
                            )
                        rc_add.click(
                            add_clip,
                            [dims_state, clips_state, rc_start, rc_end, rc_x, rc_y, rc_w, rc_h],
                            [clips_state, rc_gallery, rc_preview, rc_status, selected_state],
                        )
                        rc_update.click(
                            update_clip,
                            [dims_state, clips_state, selected_state, rc_start, rc_end,
                             rc_x, rc_y, rc_w, rc_h],
                            [clips_state, rc_gallery, rc_preview, rc_status, selected_state],
                        )
                        rc_delete.click(
                            delete_clip, [clips_state, selected_state],
                            [clips_state, rc_gallery, rc_preview, rc_status, selected_state],
                        )
                        rc_gallery.select(
                            select_clip, [dims_state, clips_state],
                            [rc_preview, selected_state, rc_start, rc_end,
                             rc_x, rc_y, rc_w, rc_h, rc_annotator, snapshot_state, rc_status, crop_state],
                        )
                        rc_compile.click(
                            compile_reel, [dims_state, clips_state, rc_reel_out],
                            [rc_final, rc_status],
                        )

    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        allowed_paths=["/home/juanessao", "/tmp", str(WORK_DIR)],
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
