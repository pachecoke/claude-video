#!/usr/bin/env python3
"""Local diarized transcription through WhisperX and pyannote.

This worker is executed by the *managed* WhisperX interpreter
(``~/.cache/watch/whisperx-venv``), never by the base watch runtime. It imports
torch/whisperx lazily so the module stays importable (for unit tests) under a
plain stdlib Python. The calling bridge (``diarization.py``) injects the
Hugging Face token into the environment; this worker never receives it on argv.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def _log(message: str) -> None:
    print(f"[diarize] {message}", file=sys.stderr, flush=True)


def _load_hf_token() -> str | None:
    """Read the token only from process environment (the bridge injects it)."""
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _certificate_bundle() -> str | None:
    configured = os.environ.get("WATCH_CA_BUNDLE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise SystemExit(f"WATCH_CA_BUNDLE does not exist: {path}")
        return str(path.resolve())
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None


@contextmanager
def model_network_context():
    """Temporarily configure certificate trust while models initialize."""
    bundle = _certificate_bundle()
    if not bundle:
        yield
        return

    env_names = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    previous_env = {name: os.environ.get(name) for name in env_names}
    original_client = None
    original_async_client = None
    try:
        for name in env_names:
            os.environ[name] = bundle
        try:
            import httpx
            original_client = httpx.Client.__init__
            original_async_client = httpx.AsyncClient.__init__

            def patched_client(self, *args, **kwargs):
                kwargs.setdefault("verify", bundle)
                original_client(self, *args, **kwargs)

            def patched_async_client(self, *args, **kwargs):
                kwargs.setdefault("verify", bundle)
                original_async_client(self, *args, **kwargs)

            httpx.Client.__init__ = patched_client
            httpx.AsyncClient.__init__ = patched_async_client
        except ImportError:
            pass
        yield
    finally:
        if original_client is not None:
            httpx.Client.__init__ = original_client
        if original_async_client is not None:
            httpx.AsyncClient.__init__ = original_async_client
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def resolve_device(requested: str | None) -> tuple[str, str]:
    """Return (device, compute_type), honoring the request but never promising a
    GPU that torch cannot see."""
    import torch

    want = (requested or "").strip().lower()
    if want == "cuda" and not torch.cuda.is_available():
        _log("WATCH_WHISPERX_DEVICE=cuda requested but CUDA is unavailable; using cpu.")
        want = "cpu"
    device = want if want in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def diarize(
    source: str,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    diarize_model: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_speakers: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    try:
        import whisperx
    except ImportError as exc:
        raise SystemExit(
            "WhisperX dependencies are missing from the managed environment. "
            "Run setup.py --install-whisperx with WATCH_WHISPERX_DIARIZE=true."
        ) from exc

    token = _load_hf_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN in the environment or "
            "~/.config/watch/.env and accept the gated model terms at "
            f"https://huggingface.co/{diarize_model or DEFAULT_DIARIZATION_MODEL}"
        )

    resolved_device, resolved_compute = resolve_device(device)
    if compute_type:
        resolved_compute = compute_type
    _log(f"device={resolved_device} compute_type={resolved_compute} model={model_size}")
    _log("loading audio...")
    audio = whisperx.load_audio(source)

    _log("transcribing with faster-whisper...")
    with model_network_context():
        asr = whisperx.load_model(
            model_size,
            resolved_device,
            compute_type=resolved_compute,
            language=language,
        )
    result = asr.transcribe(audio, batch_size=16)
    language_code = result.get("language", language or "en")
    _log(
        f"transcribed {len(result.get('segments', []))} segments "
        f"(lang={language_code})"
    )

    _log("aligning word timestamps...")
    try:
        with model_network_context():
            align_model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=resolved_device,
            )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            resolved_device,
            return_char_alignments=False,
        )
    except Exception as exc:
        _log(
            f"alignment unavailable for {language_code} ({exc}); "
            "continuing without word-level alignment"
        )

    from whisperx.diarize import DiarizationPipeline

    model_name = diarize_model or DEFAULT_DIARIZATION_MODEL
    _log(f"running pyannote diarization with {model_name}...")
    with model_network_context():
        diarizer = DiarizationPipeline(
            model_name=model_name,
            token=token,
            device=resolved_device,
        )

    if threshold is not None:
        try:
            pipeline = diarizer.model
            parameters = pipeline.parameters(instantiated=True)
            parameters["clustering"]["threshold"] = threshold
            pipeline.instantiate(parameters)
            _log(f"clustering.threshold={threshold}")
        except Exception as exc:
            _log(f"could not set clustering threshold ({exc}); using default")

    diarize_args = {}
    if num_speakers is not None:
        diarize_args["num_speakers"] = num_speakers
        _log(f"forcing exact num_speakers={num_speakers}")
    else:
        if min_speakers is not None:
            diarize_args["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_args["max_speakers"] = max_speakers
    diarized = diarizer(audio, **diarize_args)

    _log("assigning words to speakers...")
    result = whisperx.assign_word_speakers(diarized, result)
    segments = []
    for segment in result.get("segments", []):
        text = (segment.get("text") or "").strip()
        if text:
            segments.append({
                "start": round(float(segment.get("start") or 0.0), 2),
                "end": round(float(segment.get("end") or 0.0), 2),
                "text": text,
                "speaker": segment.get("speaker") or "UNKNOWN",
            })
    if not segments:
        raise SystemExit("WhisperX returned no transcript segments")
    _log(
        f"done: {len(segments)} segments across "
        f"{len({segment['speaker'] for segment in segments})} speaker(s)"
    )
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="diarize",
        description="Diarized transcription through WhisperX (managed-venv worker)",
    )
    parser.add_argument("source")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--diarize-model", default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    # Keep stdout clean for the JSON payload; model libraries print to stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        segments = diarize(
            args.source,
            model_size=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            diarize_model=args.diarize_model,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            num_speakers=args.num_speakers,
            threshold=args.threshold,
        )
    finally:
        sys.stdout = real_stdout

    print(json.dumps({"backend": "whisperx", "segments": segments}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
