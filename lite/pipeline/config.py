"""Load YAML config + .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    source_lang: str = "nl"
    target_lang: str = "ka"

    # secrets (from .env)
    elevenlabs_key: str = ""

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
        cfg.elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        return cfg
