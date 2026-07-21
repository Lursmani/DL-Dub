#!/usr/bin/env python3
"""Auto-dub a video via the managed ElevenLabs Dubbing API.

Usage:
    python autodub.py episode.mp4 [--config config.yaml]
    python autodub.py episode.mp4 --target-lang ka --no-watermark
    python autodub.py episode.mp4 --yes            # skip the pre-spend confirmation

One call does everything: the API transcribes, translates, clones the
original voices and renders the dub (~$0.33/min watermarked, ~$0.50/min
clean). The result lands in work/<episode>/<episode>.<lang>.auto.mp4 and is
cached — delete it to re-dub.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import autodub as autodub_mod
from pipeline.config import Config
from pipeline.util import PipelineError, workdir_for


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-call dubbing via the managed ElevenLabs Dubbing API "
                    "— no local ML, clones original voices, ~$0.33-0.50/min")
    p.add_argument("video")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source-lang", help="default: source_lang from config")
    p.add_argument("--target-lang", help="default: target_lang from config")
    p.add_argument("--no-watermark", action="store_true",
                   help="clean output ($0.50/min instead of $0.33/min)")
    p.add_argument("--yes", action="store_true",
                   help="skip pre-spend confirmation")
    args = p.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    cfg = Config.load(Path(args.config))
    watermark = not args.no_watermark
    target_lang = args.target_lang or cfg.target_lang
    workdir = workdir_for(video)
    cached = autodub_mod.output_path(video, workdir, target_lang).exists()
    if not cached and not args.yes:
        est = autodub_mod.estimate(video, watermark)
        ok = input(f"\nAbout to auto-dub {est['minutes']} min via the ElevenLabs "
                   f"Dubbing API (~${est['usd']}). Proceed? [y/N] ")
        if ok.strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted before spending.")
    try:
        autodub_mod.autodub(video, workdir, cfg,
                            source_lang=args.source_lang or cfg.source_lang,
                            target_lang=target_lang, watermark=watermark)
    except PipelineError as e:
        # autodub raises PipelineError (thread-safe for the GUI); the CLI
        # converts it to a clean exit instead of a traceback.
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
