"""ElevenLabs voice catalog, previews, and voice-mapping persistence."""
from __future__ import annotations

from pathlib import Path

from pipeline.config import Config
from pipeline.tts import clip_hash

# Fixed Georgian sample sentence (~30 chars => ~15 credits flash / ~30 multilingual).
PREVIEW_TEXT = "გამარჯობა! ეს ჩემი ხმის სინჯია."

TTS_MODELS = [
    ("Flash v2.5 — 0.5 credits/char (cheap)", "eleven_flash_v2_5"),
    ("Multilingual v2 — 1 credit/char (better)", "eleven_multilingual_v2"),
]


def list_voices(api_key: str) -> list[dict]:
    """[{label, voice_id, preview_url}] from the user's ElevenLabs library.

    Raises RuntimeError with a readable message on failure — callers surface
    it in the UI instead of crashing the app.
    """
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (see .env).")
    try:
        from elevenlabs.client import ElevenLabs

        resp = ElevenLabs(api_key=api_key).voices.get_all()
        return [
            {
                "label": f"{v.name} ({v.category})" if v.category else v.name,
                "voice_id": v.voice_id,
                "preview_url": getattr(v, "preview_url", None) or "",
            }
            for v in resp.voices
        ]
    except Exception as e:  # noqa: BLE001 - surfaced in the UI
        raise RuntimeError(f"Could not fetch voices: {e}") from e


def georgian_preview(api_key: str, workdir: Path, voice_id: str,
                     model_id: str) -> str:
    """Synthesize (or reuse cached) Georgian sample for a voice; returns path.

    Cached by the same content hash as pipeline clips, so repeat listens are
    free — only the first click per (voice, model) bills (~15-30 credits).
    """
    if not voice_id or "REPLACE" in voice_id.upper():
        raise RuntimeError("Pick a voice first.")
    fmt = "mp3_44100_128"
    out_dir = workdir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip_hash(PREVIEW_TEXT, voice_id, model_id, fmt)}.mp3"
    if not path.exists():
        from elevenlabs.client import ElevenLabs

        audio = ElevenLabs(api_key=api_key).text_to_speech.convert(
            voice_id=voice_id, model_id=model_id,
            text=PREVIEW_TEXT, output_format=fmt,
        )
        path.write_bytes(b"".join(audio))
    return str(path)


def save_mapping(config_path: Path, mapping: dict[str, tuple[str, str]],
                 default_voice_id: str) -> str:
    """Persist speaker->voice choices to config.yaml; returns a status line."""
    voices = {
        speaker: {"voice_id": vid, "model_id": model}
        for speaker, (vid, model) in mapping.items()
        if vid and "REPLACE" not in vid.upper()
    }
    updates: dict = {"voices": voices}
    if default_voice_id and "REPLACE" not in default_voice_id.upper():
        updates["default_voice_id"] = default_voice_id
    Config.update_yaml(config_path, updates)
    return (f"Saved {len(voices)} speaker mapping(s)"
            + (" + default voice" if "default_voice_id" in updates else "")
            + f" → {config_path.name}")
