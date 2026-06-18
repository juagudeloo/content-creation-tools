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

import functools
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


def video_fps(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    raw = out.stdout.strip()  # e.g. "30000/1001" or "25/1"
    try:
        num, den = raw.split("/")
        fps = float(num) / float(den)
        return fps if fps > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        try:
            return float(raw)
        except ValueError:
            return 30.0


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


# ── Razor/split timeline — sections model ─────────────────────────────────────
# A "section" is a contiguous, boundary-sharing slice of the source video:
#   {"start": float, "end": float, "crop": "w:h:x:y", "include": bool, "thumb": path}
# Sections always partition [0, duration] with shared boundaries, so two adjacent
# *included* sections concatenate as an exact continuation (no overlap/repeat).

def _frame_snap(dims, t: float) -> float:
    """Clamp t to [0, duration] and round to the nearest frame boundary."""
    fps = dims.get("fps", 30.0) or 30.0
    t = max(0.0, min(float(t), dims["duration"]))
    return round(t * fps) / fps


def _section_at(sections: List[Dict], t: float) -> int:
    for i, s in enumerate(sections):
        if s["start"] <= t < s["end"]:
            return i
    return len(sections) - 1 if sections else -1


def _render_section_thumb(video: Path, sec: Dict) -> str:
    t = (sec["start"] + sec["end"]) / 2.0
    thumb = WORK_DIR / f"sect_{uuid.uuid4().hex}.png"
    grab_frame(video, t, thumb)
    return str(thumb)


def _gallery_items(sections: List[Dict]) -> List[Tuple[str, str]]:
    items = []
    for i, s in enumerate(sections):
        mark = "✅" if s.get("include", True) else "🚫"
        items.append((s["thumb"], f"{mark} #{i + 1}  {s['start']:.1f}–{s['end']:.1f}s"))
    return items


def _render_section_preview(dims: Dict, sec: Dict) -> str:
    out = WORK_DIR / f"prev_{uuid.uuid4().hex}.mp4"
    seg = {"start": sec["start"], "end": sec["end"], "type": "c"}
    reel.format_reel(Path(dims["path"]), [seg], {"c": sec["crop"]}, out)
    return str(out)


def _crop_tuple(crop: str) -> Tuple[int, int, int, int]:
    """Parse a "w:h:x:y" crop string into (x, y, w, h)."""
    cw, ch, cx, cy = (int(v) for v in crop.split(":"))
    return cx, cy, cw, ch


def load_reel_source(video_path: str):
    if not video_path:
        return (None, None, gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), [], [], None, None, None,
                "Upload a video to begin.", None, -1, 0.0, gr.update())
    video = _clean_path(video_path)
    if video is None or not video.exists():
        return (None, None, gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), [], [], None, None, None,
                f"❌ Video not found: {video_path}", None, -1, 0.0, gr.update())
    video = video.resolve()
    src_w, src_h = video_dimensions(video)
    duration = video_duration(video)
    fps = video_fps(video)
    x, y, w, h = default_crop(src_w, src_h)
    crop = f"{w}:{h}:{x}:{y}"
    dims = {"w": src_w, "h": src_h, "duration": duration, "path": str(video),
            "fps": fps, "frame_step": 1.0 / fps}
    snap = WORK_DIR / f"snap_{uuid.uuid4().hex}.png"
    grab_frame(video, 0.0, snap)
    section = {"start": 0.0, "end": duration, "crop": crop, "include": True, "thumb": ""}
    section["thumb"] = _render_section_thumb(video, section)
    sections = [section]
    status = (f"✅ Loaded {video.name} — {src_w}×{src_h}, {duration:.1f}s, {fps:.2f}fps. "
              "Scrub the playhead and press ✂️ Split to chop sections.")
    return (
        str(video), dims,
        gr.update(maximum=max(duration, 1), step=round(1.0 / fps, 4), value=0),  # rc_playhead
        gr.update(value=0.0),                                  # rc_time
        x, y, w, h,                                            # rc_x/y/w/h
        sections, _gallery_items(sections),                    # sections_state, rc_gallery
        annot_value(str(snap), x, y, w, h),                    # rc_annotator
        str(snap),                                             # snapshot_state
        None,                                                  # rc_preview
        status,                                                # rc_status
        (x, y, w, h),                                          # crop_state
        0,                                                     # selected_state
        0.0,                                                   # playhead_state
        gr.update(value=True),                                 # rc_include
    )


# ── Playhead scrubbing (frame preview only on release / step buttons) ──────────

def _frame_at(dims, sections, sel_idx, t) -> Tuple[str, dict, Tuple[int, int, int, int]]:
    """Render the frame at t and return it as an annotator value carrying the
    selected section's crop box (or the default crop if no section is selected)."""
    snap = WORK_DIR / f"snap_{uuid.uuid4().hex}.png"
    grab_frame(Path(dims["path"]), float(t), snap)
    if sections and 0 <= sel_idx < len(sections):
        cx, cy, cw, ch = _crop_tuple(sections[sel_idx]["crop"])
    else:
        cx, cy, cw, ch = default_crop(dims["w"], dims["h"])
    return str(snap), annot_value(str(snap), cx, cy, cw, ch), (cx, cy, cw, ch)


def render_playhead(dims, sections, sel_idx, t):
    if not dims:
        return (gr.update(),) * 10
    t = _frame_snap(dims, t)
    snap, annot, (cx, cy, cw, ch) = _frame_at(dims, sections, sel_idx, t)
    return (snap, annot, cx, cy, cw, ch, (cx, cy, cw, ch), t,
            gr.update(value=round(t, 3)), gr.update(value=t))


def step_frame(dims, sections, sel_idx, t, n_frames):
    fs = dims["frame_step"] if dims else 1.0 / 30
    return render_playhead(dims, sections, sel_idx, float(t) + n_frames * fs)


def step_seconds(dims, sections, sel_idx, t, secs):
    return render_playhead(dims, sections, sel_idx, float(t) + secs)


# ── Crop editing — retargeted to write into the selected section ───────────────

def _write_crop_to_section(sections, sel_idx, x, y, w, h):
    if 0 <= sel_idx < len(sections):
        sections = list(sections)
        sections[sel_idx] = {**sections[sel_idx], "crop": f"{int(w)}:{int(h)}:{int(x)}:{int(y)}"}
        return sections, _gallery_items(sections)
    return sections, gr.update()


def on_draw(annot, dims, snapshot_path, crop_state, sections, sel_idx):
    """User drew/dragged the box → snap to 9:16, sync numbers, store on section."""
    sections = list(sections or [])
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            crop_state, sections, gr.update())
    if not dims or not annot or not annot.get("boxes"):
        return noop
    box = annot["boxes"][0]
    x, y, w, h = box["xmin"], box["ymin"], box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    crop = snap_to_reel(x, y, w, h, dims["w"], dims["h"], drive="h")
    if crop == crop_state:                       # already canonical → stop the echo
        return noop
    cx, cy, cw, ch = crop
    sections, gallery = _write_crop_to_section(sections, sel_idx, cx, cy, cw, ch)
    return annot_value(snapshot_path, *crop), cx, cy, cw, ch, crop, sections, gallery


def on_numbers(x, y, w, h, dims, snapshot_path, crop_state, sections, sel_idx):
    """User edited a crop number → keep 9:16, redraw box, store on section."""
    sections = list(sections or [])
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            crop_state, sections, gr.update())
    if not dims:
        return noop
    px, py, pw, ph = crop_state or (x, y, w, h)
    drive = "w" if (int(w) != int(pw) and int(h) == int(ph)) else "h"
    crop = snap_to_reel(x, y, w, h, dims["w"], dims["h"], drive=drive)
    if crop == crop_state:
        return noop
    cx, cy, cw, ch = crop
    box = annot_value(snapshot_path, *crop) if snapshot_path else gr.update()
    sections, gallery = _write_crop_to_section(sections, sel_idx, cx, cy, cw, ch)
    return box, cx, cy, cw, ch, crop, sections, gallery


# ── Razor split / merge / select / include ─────────────────────────────────────

def split_section(dims, sections, sel_idx, t):
    """Split the section containing the playhead into [start, t] and [t, end]."""
    sections = list(sections or [])
    if not dims or not sections:
        return sections, _gallery_items(sections), "⚠️ Load a video first.", sel_idx, None
    t = _frame_snap(dims, t)
    i = _section_at(sections, t)
    s = sections[i]
    eps = dims["frame_step"]
    if not (s["start"] + eps <= t <= s["end"] - eps):
        return (sections, _gallery_items(sections),
                "⚠️ Playhead is on a section boundary — move it inside a section to split.",
                sel_idx, None)
    video = Path(dims["path"])
    left = {"start": s["start"], "end": t, "crop": s["crop"], "include": s["include"], "thumb": ""}
    right = {"start": t, "end": s["end"], "crop": s["crop"], "include": s["include"], "thumb": ""}
    left["thumb"] = _render_section_thumb(video, left)
    right["thumb"] = _render_section_thumb(video, right)
    sections[i:i + 1] = [left, right]
    return (sections, _gallery_items(sections),
            f"✂️ Split at {t:.2f}s — {len(sections)} sections.", i + 1, None)


def merge_section(dims, sections, sel_idx):
    """Join the selected section with the next one (or the previous one if it is
    the last), keeping the left section's crop and include flag."""
    sections = list(sections or [])
    if not dims or not sections or sel_idx is None or sel_idx < 0:
        return sections, _gallery_items(sections), "⚠️ Select a section to merge.", sel_idx, None
    if len(sections) == 1:
        return sections, _gallery_items(sections), "⚠️ Only one section — nothing to merge.", 0, None
    left = sel_idx if sel_idx < len(sections) - 1 else sel_idx - 1
    a, b = sections[left], sections[left + 1]
    merged = {"start": a["start"], "end": b["end"], "crop": a["crop"],
              "include": a["include"], "thumb": ""}
    merged["thumb"] = _render_section_thumb(Path(dims["path"]), merged)
    sections[left:left + 2] = [merged]
    return (sections, _gallery_items(sections),
            f"🔗 Merged — {len(sections)} sections.", left, None)


def select_section(dims, sections, evt: gr.SelectData):
    """Load the clicked section into the editor: crop box, numbers, playhead,
    include checkbox, and a cropped preview."""
    sections = list(sections or [])
    idx = evt.index
    if not dims or idx is None or not (0 <= idx < len(sections)):
        return (gr.update(),) * 14
    s = sections[idx]
    cx, cy, cw, ch = _crop_tuple(s["crop"])
    t = s["start"]
    snap = WORK_DIR / f"snap_{uuid.uuid4().hex}.png"
    grab_frame(Path(dims["path"]), t, snap)
    annot = annot_value(str(snap), cx, cy, cw, ch)
    try:
        preview = _render_section_preview(dims, s)
    except subprocess.CalledProcessError:
        preview = None
    return (idx, cx, cy, cw, ch, annot, str(snap), (cx, cy, cw, ch),
            gr.update(value=t), gr.update(value=round(t, 3)), t,
            gr.update(value=s["include"]), preview,
            f"▶️ Section #{idx + 1} selected ({s['start']:.1f}–{s['end']:.1f}s).")


def toggle_include(sections, sel_idx, value):
    sections = list(sections or [])
    if sel_idx is None or not (0 <= sel_idx < len(sections)):
        return sections, _gallery_items(sections), "⚠️ Select a section first."
    sections[sel_idx] = {**sections[sel_idx], "include": bool(value)}
    n_inc = sum(1 for s in sections if s["include"])
    verb = "✅ Included" if value else "🚫 Excluded"
    return sections, _gallery_items(sections), f"{verb} section #{sel_idx + 1} ({n_inc} in reel)."


def delete_section(sections, sel_idx):
    """Remove the selected section entirely. Remaining sections keep their own
    ranges (the deleted span is simply dropped from the reel)."""
    sections = list(sections or [])
    if sel_idx is None or not (0 <= sel_idx < len(sections)):
        return (sections, _gallery_items(sections),
                "⚠️ Click a section below to select it, then delete.", -1, None)
    removed = sections.pop(sel_idx)
    if sections:
        msg = f"🗑️ Deleted section {removed['start']:.1f}–{removed['end']:.1f}s — {len(sections)} left."
    else:
        msg = "🗑️ Deleted the last section — upload a video to start over."
    return sections, _gallery_items(sections), msg, -1, None


def compile_reel(dims, sections, fade, output_path):
    sections = list(sections or [])
    if not dims:
        return None, "⚠️ Load a video first."
    included = [s for s in sections if s.get("include", True)]
    if not included:
        return None, "⚠️ Mark at least one section as included before compiling."
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
    segments = [{"start": s["start"], "end": s["end"], "type": f"c{i}"} for i, s in enumerate(included)]
    crops = {f"c{i}": s["crop"] for i, s in enumerate(included)}
    try:
        reel.format_reel(video, segments, crops, out, fade=float(fade or 0.0))
    except subprocess.CalledProcessError as exc:
        return None, f"❌ ffmpeg failed: {exc}"
    return str(out), f"✅ Compiled {len(included)} section(s) into: {out}"


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

                    # ── 3b: Reel compilation (razor/split timeline editor) ─────
                    with gr.Tab("Reel compilation"):
                        sections_state = gr.State([])    # ordered contiguous sections
                        dims_state = gr.State(None)
                        snapshot_state = gr.State(None)  # current playhead frame PNG
                        selected_state = gr.State(-1)    # selected section index
                        crop_state = gr.State(None)      # echo guard for crop sync
                        playhead_state = gr.State(0.0)   # frame-snapped playhead time

                        rc_video_file = gr.File(label="Video", file_types=["video", ".mp4", ".mov", ".mkv"])
                        rc_compile = gr.Button("🎬 Create reel", elem_id="create-reel-btn")
                        rc_status = gr.Markdown()

                        with gr.Row():
                            # Left: source player
                            with gr.Column(scale=1):
                                gr.Markdown("### Source")
                                rc_video = gr.Video(label="Uploaded video")

                            # Middle: timeline editor + crop
                            with gr.Column(scale=1):
                                gr.Markdown("### Playhead & crop")
                                rc_playhead = gr.Slider(0, 1, value=0, step=1 / 30,
                                                        label="Playhead (s)")
                                with gr.Row():
                                    rc_back_s = gr.Button("⏪ −1s")
                                    rc_back_f = gr.Button("◀ −1f")
                                    rc_time = gr.Number(label="Time (s)", value=0.0,
                                                        precision=3, interactive=False)
                                    rc_fwd_f = gr.Button("+1f ▶")
                                    rc_fwd_s = gr.Button("+1s ⏩")
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
                                    rc_split = gr.Button("✂️ Split at playhead", variant="primary")
                                    rc_merge = gr.Button("🔗 Merge selected")
                                rc_include = gr.Checkbox(label="Include selected section in reel",
                                                         value=True)

                            # Right: sections list + preview
                            with gr.Column(scale=1):
                                gr.Markdown("### Sections")
                                rc_preview = gr.Video(label="Selected section preview")
                                # No fixed height: the gallery grows with the sections and
                                # the page scrolls, so every section stays reachable.
                                rc_gallery = gr.Gallery(label="Sections (click to select)",
                                                        columns=2, object_fit="cover",
                                                        allow_preview=False)
                                rc_delete = gr.Button("🗑️ Delete selected section", variant="stop")

                        rc_fade = gr.Slider(0, 2.0, value=0.0, step=0.05,
                                            label="Fade in/out per section (s, 0 = off)")
                        rc_reel_out = with_folder_picker("Reel output folder (optional)",
                                                         "Leave empty to just download the result below")
                        rc_final = gr.Video(label="Compiled reel")

                        # ── Wiring ──
                        rc_video_file.change(
                            load_reel_source, [rc_video_file],
                            [rc_video, dims_state, rc_playhead, rc_time,
                             rc_x, rc_y, rc_w, rc_h, sections_state, rc_gallery,
                             rc_annotator, snapshot_state, rc_preview, rc_status,
                             crop_state, selected_state, playhead_state, rc_include],
                        )

                        # Playhead: render the frame only on release + step buttons.
                        playhead_out = [snapshot_state, rc_annotator, rc_x, rc_y, rc_w, rc_h,
                                        crop_state, playhead_state, rc_time, rc_playhead]
                        playhead_in = [dims_state, sections_state, selected_state, rc_playhead]
                        step_in = [dims_state, sections_state, selected_state, playhead_state]
                        rc_playhead.release(render_playhead, playhead_in, playhead_out)
                        rc_back_s.click(functools.partial(step_seconds, secs=-1.0), step_in, playhead_out)
                        rc_fwd_s.click(functools.partial(step_seconds, secs=1.0), step_in, playhead_out)
                        rc_back_f.click(functools.partial(step_frame, n_frames=-1), step_in, playhead_out)
                        rc_fwd_f.click(functools.partial(step_frame, n_frames=1), step_in, playhead_out)

                        # Crop editing (draw ↔ numbers), retargeted to the selected section.
                        crop_io = [rc_annotator, rc_x, rc_y, rc_w, rc_h, crop_state,
                                   sections_state, rc_gallery]
                        rc_annotator.change(
                            on_draw,
                            [rc_annotator, dims_state, snapshot_state, crop_state,
                             sections_state, selected_state],
                            crop_io,
                        )
                        for ctrl in (rc_x, rc_y, rc_w, rc_h):
                            ctrl.change(
                                on_numbers,
                                [rc_x, rc_y, rc_w, rc_h, dims_state, snapshot_state,
                                 crop_state, sections_state, selected_state],
                                crop_io,
                            )

                        # Razor split / merge.
                        section_edit_out = [sections_state, rc_gallery, rc_status,
                                            selected_state, rc_preview]
                        rc_split.click(split_section,
                                       [dims_state, sections_state, selected_state, playhead_state],
                                       section_edit_out)
                        rc_merge.click(merge_section,
                                       [dims_state, sections_state, selected_state],
                                       section_edit_out)
                        rc_delete.click(delete_section,
                                        [sections_state, selected_state],
                                        section_edit_out)

                        # Include toggle.
                        rc_include.change(toggle_include,
                                          [sections_state, selected_state, rc_include],
                                          [sections_state, rc_gallery, rc_status])

                        # Select a section.
                        rc_gallery.select(
                            select_section, [dims_state, sections_state],
                            [selected_state, rc_x, rc_y, rc_w, rc_h, rc_annotator,
                             snapshot_state, crop_state, rc_playhead, rc_time,
                             playhead_state, rc_include, rc_preview, rc_status],
                        )

                        # Compile.
                        rc_compile.click(
                            compile_reel, [dims_state, sections_state, rc_fade, rc_reel_out],
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
