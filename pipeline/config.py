"""Load YAML config + .env, with auto device detection."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _auto_device() -> tuple[str, str]:
    """(device, compute_type) — CUDA float16 if a GPU is present, else CPU int8."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@dataclass
class Config:
    source_lang: str = "nl"
    target_lang: str = "ka"
    whisper_model: str = "large-v3"
    device: str = ""
    compute_type: str = ""
    translate_model: str = "claude-sonnet-5"
    chars_per_second: float = 15.0
    default_tts_model: str = "eleven_flash_v2_5"
    default_voice_id: str = ""
    output_format: str = "mp3_44100_128"
    voices: dict[str, dict[str, str]] = field(default_factory=dict)
    background_gain_db: float = -3.0
    max_speedup: float = 1.30
    # 1.0 = never slow speech down to fill a slot; short lines get natural silence.
    min_slowdown: float = 1.0

    # secrets (from .env)
    elevenlabs_key: str = ""
    anthropic_key: str = ""
    hf_token: str = ""

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        load_dotenv()
        raw: dict[str, Any] = {}
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        unknown = [k for k in raw if k not in cls.__dataclass_fields__]
        if unknown:
            print(f"[config] ignoring unknown keys in {config_path.name}: {unknown} "
                  "(typo?)")
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        if not cfg.device or not cfg.compute_type:
            dev, ct = _auto_device()
            cfg.device = cfg.device or dev
            cfg.compute_type = cfg.compute_type or ct

        cfg.elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        cfg.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        cfg.hf_token = os.environ.get("HF_TOKEN", "")
        return cfg

    @staticmethod
    def update_yaml(config_path: Path, updates: dict[str, Any]) -> None:
        """Merge updates into the YAML file (used by the GUI to persist choices).

        Round-trips the raw dict — never the dataclass — so auto-detected
        fields (device/compute_type) and secrets are never written to disk.
        Note: hand-written comments in config.yaml are lost on rewrite.
        """
        raw: dict[str, Any] = {}
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw.update(updates)
        config_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def voice_for(self, speaker: str | None) -> tuple[str, str]:
        """(voice_id, model_id) for a diarization speaker label, with fallback."""
        entry = self.voices.get(speaker or "", {})
        return (
            entry.get("voice_id") or self.default_voice_id,
            entry.get("model_id") or self.default_tts_model,
        )
