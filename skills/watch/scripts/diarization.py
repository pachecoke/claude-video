#!/usr/bin/env python3
"""Subprocess bridge for optional WhisperX speaker diarization.

Runs the diarization worker (``diarize.py``) inside the managed WhisperX venv
(``~/.cache/watch/whisperx-venv``) that ``setup.py --install-whisperx`` builds,
so diarization reuses the same Torch/WhisperX/pyannote install as upstream's
local transcription. The Hugging Face token is injected through the child
environment, never placed on argv.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import get_config, load_hf_token  # noqa: E402
from frames import format_time  # noqa: E402
from whisper import extract_audio  # noqa: E402


def _shift_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Shift focus-range segments back into source time, preserving speaker labels.

    (whisper.shift_segments keeps only start/end/text, so it is not reused here.)
    """
    if not offset_seconds:
        return segments
    shifted = []
    for seg in segments:
        item = dict(seg)
        item["start"] = round(float(seg["start"]) + offset_seconds, 2)
        item["end"] = round(float(seg["end"]) + offset_seconds, 2)
        shifted.append(item)
    return shifted


def diarize_script() -> Path:
    configured = os.environ.get("WATCH_DIARIZE_SCRIPT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return SCRIPT_DIR / "diarize.py"


def _managed_venv() -> Path:
    return Path.home() / ".cache" / "watch" / "whisperx-venv"


def discover_diarize_python(cfg: dict | None = None) -> Path | None:
    """Locate the managed WhisperX interpreter that runs the diarization worker."""
    cfg = get_config(backend_override="whisperx") if cfg is None else cfg
    windows = platform.system() == "Windows"
    candidates: list[Path] = []
    override = (os.environ.get("WATCH_WHISPERX_PYTHON", "").strip()
                or cfg.get("whisperx_python", ""))
    if override:
        candidates.append(Path(override).expanduser())
    # Sibling of the configured whisperx executable (Scripts/ or bin/).
    whisperx_bin = cfg.get("whisperx_bin")
    if whisperx_bin:
        parent = Path(whisperx_bin).parent
        candidates += [parent / "python.exe", parent / "python"]
    venv = _managed_venv()
    candidates += ([venv / "Scripts" / "python.exe"] if windows
                   else [venv / "bin" / "python"])
    candidates += [venv / "Scripts" / "python.exe", venv / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def diarization_available() -> bool:
    return discover_diarize_python() is not None and diarize_script().is_file()


def build_command(
    python_path: Path,
    script_path: Path,
    audio_path: Path,
    model: str,
    *,
    device: str | None = None,
    compute_type: str | None = None,
    diarize_model: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    threshold: float | None = None,
) -> list[str]:
    command = [
        str(python_path),
        str(script_path),
        str(audio_path),
        "--model",
        model,
    ]
    if device:
        command.extend(["--device", device])
    if compute_type:
        command.extend(["--compute-type", compute_type])
    if diarize_model:
        command.extend(["--diarize-model", diarize_model])
    if num_speakers is not None:
        command.extend(["--num-speakers", str(num_speakers)])
    else:
        if min_speakers is not None:
            command.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            command.extend(["--max-speakers", str(max_speakers)])
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])
    return command


def label_speakers(segments: list[dict]) -> list[dict]:
    """Map raw speaker IDs to stable labels in first-appearance order."""
    labels: dict[str, str] = {}
    for segment in segments:
        raw = str(segment.get("speaker") or "UNKNOWN")
        if raw.upper() in ("UNKNOWN", "NONE"):
            continue
        if raw not in labels:
            labels[raw] = f"Speaker {len(labels) + 1}"

    output = []
    last_label = next(iter(labels.values()), "Speaker 1")
    for segment in segments:
        raw = str(segment.get("speaker") or "UNKNOWN")
        if raw.upper() in ("UNKNOWN", "NONE"):
            label = last_label
        else:
            label = labels[raw]
            last_label = label
        item = dict(segment)
        item["speaker"] = label
        output.append(item)
    return output


def format_diarized_transcript(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        lines.append(
            f"[{format_time(float(segment['start']))}] "
            f"{segment['speaker']}: {segment['text']}"
        )
    return "\n".join(lines)


def _child_env() -> dict[str, str]:
    """Inject the resolved HF token (gated pyannote models) without logging it."""
    env = dict(os.environ)
    token = load_hf_token()
    if token:
        env.setdefault("HF_TOKEN", token)
        env["HUGGING_FACE_HUB_TOKEN"] = token
    env.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    env.setdefault("PYTHONWARNINGS", "ignore")
    return env


def _extract_focus_audio(
    video_path: str,
    out_path: Path,
    start_seconds: float | None,
    end_seconds: float | None,
) -> Path:
    """Extract mono 16 kHz MP3 for diarization.

    0.3.2's whisper.extract_audio() pulls the full track and no longer trims,
    so when a --start/--end focus range is set we trim here with ffmpeg. The
    clip is 0-based; run_diarization shifts segment times back by start_seconds.
    """
    if start_seconds is None and end_seconds is None:
        return extract_audio(video_path, out_path)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += ["-i", str(Path(video_path).resolve())]
    if end_seconds is not None:
        duration = max(0.0, end_seconds - (start_seconds or 0.0))
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
            "-b:a", "64k", str(out_path.resolve())]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio in the requested range")
    return out_path


def run_diarization(
    video_path: str,
    work_dir: Path,
    model: str = "large-v3",
    *,
    device: str | None = None,
    compute_type: str | None = None,
    diarize_model: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    threshold: float | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[list[dict], str]:
    python_path = discover_diarize_python()
    script_path = diarize_script()
    if python_path is None:
        raise SystemExit(
            "Managed WhisperX interpreter not found. Run "
            "setup.py --install-whisperx (with WATCH_WHISPERX_DIARIZE=true), "
            "or set WATCH_WHISPERX_PYTHON to the venv interpreter."
        )
    if not script_path.is_file():
        raise SystemExit(f"Diarization worker not found: {script_path}")
    if not load_hf_token():
        raise SystemExit(
            "No Hugging Face token found. Add HF_TOKEN to the environment or "
            "~/.config/watch/.env and accept the gated pyannote model terms."
        )

    audio_path = _extract_focus_audio(
        video_path,
        work_dir / "diarize-audio.mp3",
        start_seconds,
        end_seconds,
    )
    command = build_command(
        python_path,
        script_path,
        audio_path,
        model,
        device=device,
        compute_type=compute_type,
        diarize_model=diarize_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        threshold=threshold,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        env=_child_env(),
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        message = detail[-1] if detail else "unknown error"
        raise SystemExit(f"WhisperX diarization failed: {message}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"WhisperX returned invalid JSON: {exc}")
    segments = payload.get("segments") or []
    if not segments:
        raise SystemExit("WhisperX returned no transcript segments")
    shifted = _shift_segments(segments, start_seconds or 0.0)
    return label_speakers(shifted), str(payload.get("backend") or "whisperx")
