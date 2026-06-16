# Subtitle Generator Docs

This project has one source file, [scripts/generate_english_subtitles.py](../scripts/generate_english_subtitles.py), and one dependency file, [requirements.txt](../requirements.txt). The script is a single linear pipeline that extracts audio from a Spanish MP4, translates it to English subtitles, writes an SRT file, and can optionally burn the subtitles into a new video.

## What the script does, step by step

### 1. Parse the command-line arguments

`main()` starts by calling `parse_args()`. The script accepts:

- the input MP4 path
- an optional output SRT path
- an optional burned-in output video path
- Whisper model selection
- input language
- device selection
- compute type selection
- beam size

Defaults are defined near the top of the script so they are easy to change without hunting through the implementation.

### 2. Bootstrap a local virtual environment

Before any work happens, `ensure_virtual_environment()` checks the `SUBTITLE_BOOTSTRAP_READY` environment variable.

If the flag is missing, the script:

- creates `.venv/` if needed
- checks whether `faster_whisper` can be imported inside that environment
- installs [requirements.txt](../requirements.txt) if the dependency is missing
- re-executes itself with the `.venv` Python interpreter

This is why you can run the tool with the system `python3` even though the real execution happens inside the project-local virtual environment.

### 3. Resolve the output paths and device

`main()` resolves the input video path, chooses the output SRT path, and resolves the optional burned video path.

If `--device auto` is used, `resolve_device()` probes CUDA availability through `ctranslate2`. Otherwise it respects the explicit `cpu` or `cuda` choice.

### 4. Extract mono 16 kHz audio

`extract_audio()` calls `ffmpeg` to convert the MP4 into a temporary WAV file.

The command forces:

- one audio channel
- 16 kHz sampling rate
- WAV output

This normalized audio format is what Whisper expects for stable transcription.

### 5. Load the Whisper model

`load_whisper_model()` imports `faster_whisper.WhisperModel` and tries a small set of compute types until one works.

The ordering matters:

- `float16` on CUDA or `float32` on CPU first
- then `int8_float16`
- then `int8`

If a backend does not support a candidate compute type, the code falls back to the next one instead of failing immediately.

### 6. Transcribe and translate the speech

`collect_cues()` calls `model.transcribe(..., task="translate")`.

That `task="translate"` setting is the key behavior: Whisper converts the Spanish speech directly into English text instead of just transcribing the original language.

The transcription also enables VAD filtering so silence and pauses do not become subtitle noise.

### 7. Split the transcription into subtitle cues

Whisper output is usually too long for readable subtitles, so the script reshapes each segment before writing it.

The cue-shaping flow is:

- `normalize_text()` trims and compresses whitespace
- `split_text_for_subtitles()` breaks long text on punctuation boundaries first
- `chunk_by_words()` is the fallback when punctuation is not enough
- `split_segment_into_cues()` assigns timing to the resulting pieces
- `wrap_for_srt()` formats the text for one or two display lines

This is the most important formatting stage in the project. It controls subtitle length, line wrapping, and how time is distributed across the smaller cues.

### 8. Write the `.srt` file

`write_srt()` serializes the cues using standard SRT formatting:

- numeric cue index
- start and end timestamp
- cue text
- blank line separator

`format_timestamp()` converts seconds into the `HH:MM:SS,mmm` format expected by SRT readers.

### 9. Optionally burn subtitles into a new video

If `--burned-video` is provided, `render_burned_video()` runs `ffmpeg` again with a `subtitles=` video filter.

`escape_path_for_ffmpeg()` is important here because subtitle paths are embedded inside the filtergraph string, so colons, backslashes, and quotes must be escaped correctly.

### 10. Clean up temporary files automatically

`main()` creates a temporary directory for the intermediate audio file. When the context manager exits, the temporary WAV is removed automatically.

The final outputs that remain are the SRT file and, if requested, the burned-in MP4.

## Important implementation notes

- Subtitle timing logic belongs in the cue-shaping helpers, not in the SRT writer.
- Dependency changes belong in [requirements.txt](../requirements.txt) because the script manages its own `.venv`.
- `ffmpeg` must be installed on `PATH`; it is not a Python dependency.
- The project currently has no test suite or linter configuration.

## If you want to change the behavior

- To adjust subtitle readability, change the constants near the top of [scripts/generate_english_subtitles.py](../scripts/generate_english_subtitles.py).
- To change how long text is split or wrapped, edit the cue-shaping helpers.
- To support a different model or dependency set, update [requirements.txt](../requirements.txt) and verify the bootstrap flow still works.
