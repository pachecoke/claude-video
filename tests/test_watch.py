"""End-to-end routing of --detail through watch.py on a local clip."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _run(clip: Path, *args: str, env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_efficient_uses_keyframe_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient")
    assert "(keyframe" in out
    assert "**Detail:** efficient" in out


def test_balanced_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced")
    assert "(scene" in out
    assert "**Detail:** balanced" in out


def test_token_burner_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "token-burner")
    assert "(scene" in out


def test_transcript_skips_frames(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript")
    assert "skipped" in out
    assert "frame_0000.jpg" not in out


def test_flag_overrides_env(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient", env_extra={"WATCH_DETAIL": "balanced"})
    assert "(keyframe" in out


def test_default_is_balanced(cut_clip: Path):
    out = _run(cut_clip)  # no flag, WATCH_DETAIL cleared
    assert "**Detail:** balanced" in out
    assert "(scene" in out


def test_timestamps_add_cue_frames_to_detail(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" in out  # detail frames still present (additive)


def test_timestamps_with_transcript_detail_is_cue_only(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" not in out
    assert "reason=keyframe" not in out


def _frame_lines(out: str) -> int:
    normalized = out.replace("\\", "/")
    return sum(1 for line in normalized.splitlines() if "/frames/frame_" in line and "(t=" in line)


def test_dedup_collapses_static_by_default(static_clip: Path):
    out = _run(static_clip)  # solid blue → identical frames collapse to one
    assert "near-duplicate" in out
    assert _frame_lines(out) == 1


def test_no_dedup_preserves_static_frames(static_clip: Path):
    out = _run(static_clip, "--no-dedup")
    assert "near-duplicate" not in out
    assert _frame_lines(out) > 1


def test_report_file_is_utf8_and_complete(static_clip: Path, tmp_path: Path):
    report = tmp_path / "report.md"
    stdout = _run(static_clip, "--report-file", str(report))
    assert stdout == ""
    raw = report.read_bytes()
    assert raw.startswith(b"\n# watch: video report")
    text = raw.decode("utf-8")
    assert "**Source:**" in text

    assert "_Work dir:" in text
def test_failed_diarization_falls_back_without_crashing(tmp_path: Path):
    clip = tmp_path / "audio-video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", "1", "-i", "color=c=blue:s=320x240:r=10",
            "-f", "lavfi", "-t", "1", "-i", "sine=frequency=440:sample_rate=16000",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(clip),
        ],
        check=True,
    )
    failing = tmp_path / "failing-diarize.py"
    failing.write_text(
        "import sys\nprint('synthetic failure', file=sys.stderr)\nraise SystemExit(2)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update({
        "WATCH_DIARIZE_PYTHON": sys.executable,
        "WATCH_DIARIZE_SCRIPT": str(failing),
        "WATCH_DISABLE_LOCAL_WHISPER": "1",
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
    })
    for name in ("HF_TOKEN", "GROQ_API_KEY", "OPENAI_API_KEY"):
        env.pop(name, None)
    process = subprocess.run(
        [
            sys.executable,
            str(WATCH),
            str(clip),
            "--detail",
            "transcript",
            "--diarize",
            "--num-speakers",
            "2",
            "--no-whisper",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert process.returncode == 0
    assert "falling back to normal transcription" in process.stderr
    assert "**Transcript:** none available" in process.stdout
