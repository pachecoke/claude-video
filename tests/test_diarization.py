"""WhisperX diarization bridge behavior without loading models or credentials."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import diarization
import diarize as diarize_wrapper


def test_exact_speaker_count_overrides_range():
    command = diarization.build_command(
        Path("python"),
        Path("diarize.py"),
        Path("audio.mp3"),
        "large-v3",
        num_speakers=2,
        min_speakers=1,
        max_speakers=5,
    )
    assert command[-2:] == ["--num-speakers", "2"]
    assert "--min-speakers" not in command
    assert "--max-speakers" not in command


def test_build_command_adds_range_and_threshold():
    command = diarization.build_command(
        Path("python"),
        Path("diarize.py"),
        Path("audio.mp3"),
        "small",
        min_speakers=2,
        max_speakers=4,
        threshold=0.55,
    )
    assert command[-6:] == [
        "--min-speakers", "2",
        "--max-speakers", "4",
        "--threshold", "0.55",
    ]


def test_build_command_passes_device_and_model():
    command = diarization.build_command(
        Path("python"),
        Path("diarize.py"),
        Path("audio.mp3"),
        "small",
        device="cuda",
        compute_type="float16",
        diarize_model="pyannote/speaker-diarization-community-1",
    )
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--compute-type") + 1] == "float16"
    assert command[command.index("--diarize-model") + 1] == "pyannote/speaker-diarization-community-1"


def test_labels_follow_first_appearance():
    output = diarization.label_speakers([
        {"start": 0, "end": 1, "text": "a", "speaker": "SPEAKER_07"},
        {"start": 1, "end": 2, "text": "b", "speaker": "SPEAKER_02"},
        {"start": 2, "end": 3, "text": "c", "speaker": "SPEAKER_07"},
    ])
    assert [item["speaker"] for item in output] == [
        "Speaker 1", "Speaker 2", "Speaker 1"
    ]


def test_unknown_segment_does_not_create_extra_speaker():
    output = diarization.label_speakers([
        {"start": 0, "end": 1, "text": "a", "speaker": "raw-a"},
        {"start": 1, "end": 2, "text": "unclear", "speaker": "UNKNOWN"},
        {"start": 2, "end": 3, "text": "b", "speaker": "raw-b"},
    ])
    assert [item["speaker"] for item in output] == [
        "Speaker 1", "Speaker 1", "Speaker 2"
    ]


def test_formats_timestamped_speaker_lines():
    text = diarization.format_diarized_transcript([
        {"start": 65.0, "end": 66.0, "text": "hello", "speaker": "Speaker 1"}
    ])
    assert text == "[01:05] Speaker 1: hello"


def test_shift_preserves_speaker_label():
    shifted = diarization._shift_segments(
        [{"start": 1, "end": 2, "text": "a", "speaker": "raw-a"}], 60.0
    )
    assert shifted[0]["start"] == 61
    assert shifted[0]["speaker"] == "raw-a"


def test_run_parses_json_extracts_focus_and_shifts(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    script_path = tmp_path / "diarize.py"
    python_path.write_text("", encoding="utf-8")
    script_path.write_text("", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(diarization, "discover_diarize_python", lambda: python_path)
    monkeypatch.setattr(diarization, "diarize_script", lambda: script_path)
    monkeypatch.setattr(diarization, "load_hf_token", lambda: "test-token")

    def fake_focus(video, out, start, end):
        captured["range"] = (start, end)
        out.write_bytes(b"audio")
        return out

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env", {})
        payload = {
            "backend": "whisperx",
            "segments": [
                {"start": 1, "end": 2, "text": "a", "speaker": "raw-a"},
                {"start": 3, "end": 4, "text": "b", "speaker": "raw-b"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(diarization, "_extract_focus_audio", fake_focus)
    monkeypatch.setattr(diarization.subprocess, "run", fake_run)
    segments, backend = diarization.run_diarization(
        "video.mp4",
        tmp_path,
        num_speakers=2,
        start_seconds=60.0,
        end_seconds=90.0,
    )
    assert captured["range"] == (60.0, 90.0)
    assert "--num-speakers" in captured["command"]
    # The HF token is injected through the environment, never on argv.
    assert "test-token" not in captured["command"]
    assert captured["env"].get("HUGGING_FACE_HUB_TOKEN") == "test-token"
    assert segments[0]["start"] == 61
    assert {item["speaker"] for item in segments} == {"Speaker 1", "Speaker 2"}
    assert backend == "whisperx"


def test_missing_interpreter_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(diarization, "discover_diarize_python", lambda: None)
    with pytest.raises(SystemExit, match="Managed WhisperX interpreter not found"):
        diarization.run_diarization("video.mp4", tmp_path)


def test_missing_token_is_actionable(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    script_path = tmp_path / "diarize.py"
    python_path.write_text("", encoding="utf-8")
    script_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(diarization, "discover_diarize_python", lambda: python_path)
    monkeypatch.setattr(diarization, "diarize_script", lambda: script_path)
    monkeypatch.setattr(diarization, "load_hf_token", lambda: None)
    with pytest.raises(SystemExit, match="Hugging Face token"):
        diarization.run_diarization("video.mp4", tmp_path)


def test_invalid_json_is_rejected(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    script_path = tmp_path / "diarize.py"
    python_path.write_text("", encoding="utf-8")
    script_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(diarization, "discover_diarize_python", lambda: python_path)
    monkeypatch.setattr(diarization, "diarize_script", lambda: script_path)
    monkeypatch.setattr(diarization, "load_hf_token", lambda: "test-token")
    monkeypatch.setattr(
        diarization,
        "extract_audio",
        lambda video, out: out,
    )
    monkeypatch.setattr(
        diarization.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "not-json", ""),
    )
    with pytest.raises(SystemExit, match="invalid JSON"):
        diarization.run_diarization("video.mp4", tmp_path)


def test_focus_audio_no_range_delegates_to_extract_audio(monkeypatch, tmp_path):
    # Guards the 0.3.2 extract_audio arity: with no focus range it must call
    # extract_audio with exactly (video, out), not the old 4-arg form.
    calls = {}
    out = tmp_path / "a.mp3"

    def fake_extract(video, out_path):
        calls["args"] = (video, out_path)
        return out_path

    monkeypatch.setattr(diarization, "extract_audio", fake_extract)
    result = diarization._extract_focus_audio("video.mp4", out, None, None)
    assert result == out
    assert calls["args"] == ("video.mp4", out)


def test_wrapper_main_emits_json(monkeypatch, capsys):
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "raw"}
    ]
    monkeypatch.setattr(diarize_wrapper, "diarize", lambda *args, **kwargs: segments)
    monkeypatch.setattr(
        diarize_wrapper.sys,
        "argv",
        ["diarize.py", "audio.mp3", "--num-speakers", "2"],
    )
    assert diarize_wrapper.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "whisperx"
    assert payload["segments"] == segments
