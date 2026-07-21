"""Stage 3 — transcribe the dialogue with word-level timing + speaker diarization.

Produces one manifest segment per spoken line:
    {id, start, end, speaker, text_src}
The speaker labels (SPEAKER_00, ...) are what you map to voices in config.yaml.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError


def _diarization_pipeline(hf_token: str, device: str):
    # Import path moved between WhisperX versions; try both.
    try:
        from whisperx.diarize import DiarizationPipeline  # newer
    except Exception:
        from whisperx import DiarizationPipeline  # older
    return DiarizationPipeline(use_auth_token=hf_token, device=device)


def transcribe(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    try:
        import whisperx
    except ImportError as e:
        raise PipelineError(
            "[transcribe] whisperx is not installed. The ML extras are optional: "
            "pip install -r requirements-ml.txt (several GB incl. torch), or run "
            "the Analyze stage in Colab and finish locally — see README."
        ) from e

    if not cfg.hf_token:
        raise PipelineError("[transcribe] HF_TOKEN required for speaker diarization "
                            "(see .env.example).")

    vocals = manifest.data["vocals"]
    audio = whisperx.load_audio(vocals)

    model = whisperx.load_model(
        cfg.whisper_model, cfg.device, compute_type=cfg.compute_type,
        language=cfg.source_lang,
    )
    result = model.transcribe(audio, batch_size=16)

    # Word-level alignment for tight per-line timestamps.
    align_model, meta = whisperx.load_align_model(
        language_code=cfg.source_lang, device=cfg.device)
    result = whisperx.align(result["segments"], align_model, meta, audio, cfg.device)

    # Who spoke each line.
    diarize = _diarization_pipeline(cfg.hf_token, cfg.device)
    result = whisperx.assign_word_speakers(diarize(audio), result)

    segments = []
    for i, seg in enumerate(result["segments"]):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "id": i,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": seg.get("speaker"),  # may be None if diarization missed it
            "text_src": text,
        })
    manifest.segments = segments
    manifest.save()

    # Release GPU memory: in a long-lived GUI process, a second run would OOM
    # a T4 if the whisper/align/diarization models stayed resident.
    del model, align_model, diarize
    if cfg.device == "cuda":
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    speakers = sorted({s["speaker"] for s in segments if s["speaker"]})
    print(f"[transcribe] {len(segments)} lines, speakers: {speakers or 'none detected'}")
    print("[transcribe] map these speaker labels to voices in config.yaml, then run tts.")
