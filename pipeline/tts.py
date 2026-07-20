"""Stage 5 — synthesize each translated line in Georgian via ElevenLabs.

Billed per character, so this is the cheap path vs. the managed Dubbing product.
Clips are cached by (text, voice, model) hash: re-runs never re-bill unchanged
lines, and editing one translation only re-synthesizes that line.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .util import Manifest, PipelineError

# Credits per character by model, and $ per 1000 chars (API rates).
_CREDITS = {"eleven_flash_v2_5": 0.5, "eleven_turbo_v2_5": 0.5}  # default 1.0 otherwise
_USD_PER_1K = {0.5: 0.05, 1.0: 0.10}


def _rate(model: str) -> float:
    return _CREDITS.get(model, 1.0)


def clip_hash(text: str, voice_id: str, model_id: str, fmt: str) -> str:
    """Cache key for a synthesized clip. Also used by GUI voice previews."""
    return hashlib.sha256(
        f"{model_id}|{voice_id}|{fmt}|{text}".encode()
    ).hexdigest()[:16]


_hash = clip_hash  # backwards-compat alias


def estimate(manifest: Manifest, cfg: Config) -> dict:
    """Cost preview — no API calls."""
    chars = credits = usd = 0
    for s in manifest.segments:
        text = s.get("text_tgt")
        if not text:
            continue
        _, model = cfg.voice_for(s.get("speaker"))
        rate = _rate(model)
        chars += len(text)
        credits += len(text) * rate
        usd += len(text) / 1000 * _USD_PER_1K[rate]
    return {"chars": chars, "credits": round(credits), "usd": round(usd, 2)}


def synth(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    from elevenlabs.client import ElevenLabs

    if not cfg.elevenlabs_key:
        raise PipelineError("[tts] ELEVENLABS_API_KEY required (see .env.example).")

    client = ElevenLabs(api_key=cfg.elevenlabs_key)
    clips = workdir / "clips"
    ext = cfg.output_format.split("_")[0]  # mp3 / pcm / ...

    todo = [s for s in manifest.segments if s.get("text_tgt")]
    for s in tqdm(todo, desc="[tts]"):
        voice_id, model_id = cfg.voice_for(s.get("speaker"))
        if not voice_id or "REPLACE" in voice_id.upper():
            raise PipelineError(
                f"[tts] speaker {s.get('speaker')}: voice_id is unset or still a "
                f"placeholder ({voice_id!r}) — set real ElevenLabs voice IDs in config.yaml."
            )
        h = clip_hash(s["text_tgt"], voice_id, model_id, cfg.output_format)
        clip = clips / f'{s["id"]:04d}_{h}.{ext}'
        if not clip.exists():
            audio = client.text_to_speech.convert(
                voice_id=voice_id, model_id=model_id,
                text=s["text_tgt"], output_format=cfg.output_format,
            )
            clip.write_bytes(b"".join(audio))
        s["clip"] = str(clip)

    manifest.save()
    est = estimate(manifest, cfg)
    print(f"[tts] {len(todo)} clips ready  (~{est['credits']} credits, ~${est['usd']})")
