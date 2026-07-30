#!/usr/bin/env python3
"""Subprocess bridge for optional WhisperX speaker diarization."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import config_value  # noqa: E402
from frames import format_time  # noqa: E402
from whisper import extract_audio, shift_segments  # noqa: E402


def diarize_script() -> Path:
    configured = config_value("WATCH_DIARIZE_SCRIPT")
    if configured:
        return Path(configured).expanduser().resolve()
    return SCRIPT_DIR / "diarize.py"


def discover_diarize_python() -> Path | None:
    configured = config_value("WATCH_DIARIZE_PYTHON")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        Path.home() / "whisperx-env" / "Scripts" / "python.exe",
        Path.home() / "whisperx-env" / "bin" / "python",
    ])
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


def run_diarization(
    video_path: str,
    work_dir: Path,
    model: str = "large-v3",
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
            "WhisperX interpreter not found. Set WATCH_DIARIZE_PYTHON or create "
            "~/whisperx-env."
        )
    if not script_path.is_file():
        raise SystemExit(f"WhisperX wrapper not found: {script_path}")

    audio_path = extract_audio(
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
        num_speakers,
        min_speakers,
        max_speakers,
        threshold,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        env=os.environ.copy(),
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
    shifted = shift_segments(segments, start_seconds or 0.0)
    return label_speakers(shifted), str(payload.get("backend") or "whisperx")
