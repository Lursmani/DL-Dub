#!/usr/bin/env python3
"""Georgian dubbing pipeline — extract, dub cost-efficiently, re-splice.

Usage:
    python dub.py run episode.mp4 [--config config.yaml]
    python dub.py run episode.mp4 --start-at tts        # resume after editing config
    python dub.py run episode.mp4 --stop-after transcribe   # then map speakers -> voices
    python dub.py estimate episode.mp4                  # cost preview, no spend
    python dub.py run episode.mp4 --yes                 # skip the pre-spend confirmation

Stages run in order and are resumable — each writes work/<episode>/manifest.json,
so re-running skips work already done. `tts` prompts with a cost estimate before
calling ElevenLabs unless --yes is passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.assemble import assemble
from pipeline.config import Config
from pipeline.extract import extract
from pipeline.mux import mux
from pipeline.separate import separate
from pipeline.transcribe import transcribe
from pipeline.translate import translate
from pipeline.tts import estimate as tts_estimate
from pipeline.tts import synth
from pipeline.util import Manifest, PipelineError, workdir_for

STAGES = [
    ("extract", extract),
    ("separate", separate),
    ("transcribe", transcribe),
    ("translate", translate),
    ("tts", synth),
    ("assemble", assemble),
    ("mux", mux),
]
STAGE_NAMES = [name for name, _ in STAGES]


def _run(args: argparse.Namespace) -> None:
    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    cfg = Config.load(Path(args.config))
    if cfg.device == "cpu":
        print("[device] CPU — no CUDA/ROCm GPU detected. ML stages (separate, "
              "transcribe, diarize) run on CPU (~20-40 min per 6-min episode). "
              "Tip: set whisper_model: medium to speed up, or use a cloud GPU.")
    else:
        print(f"[device] {cfg.device} ({cfg.compute_type}) — GPU acceleration active.")
    workdir = workdir_for(video)
    manifest = Manifest.load_or_init(workdir, video)

    start = STAGE_NAMES.index(args.start_at) if args.start_at else 0
    stop = STAGE_NAMES.index(args.stop_after) if args.stop_after else len(STAGES) - 1

    for name, fn in STAGES[start: stop + 1]:
        # Re-running transcribe rebuilds segments and wipes translations; guard it.
        if name == "transcribe" and manifest.segments and not args.force:
            print("[transcribe] segments already exist — skipping. "
                  "Use --force to redo (this discards existing translations).")
            continue
        if name == "tts" and not args.yes:
            est = tts_estimate(manifest, cfg)
            if est["chars"] == 0:
                raise SystemExit("[tts] no translated lines — run the translate stage first.")
            ok = input(
                f"\nAbout to synthesize {est['chars']} characters "
                f"(~{est['credits']} credits, ~${est['usd']}). Proceed? [y/N] "
            )
            if ok.strip().lower() not in ("y", "yes"):
                raise SystemExit("Aborted before spending.")
        try:
            fn(video, workdir, manifest, cfg)
        except PipelineError as e:
            # Stages raise PipelineError (thread-safe for the GUI); the CLI
            # converts it to a clean exit instead of a traceback.
            raise SystemExit(str(e)) from None


def _estimate(args: argparse.Namespace) -> None:
    video = Path(args.video).resolve()
    cfg = Config.load(Path(args.config))
    manifest = Manifest.load_or_init(workdir_for(video), video)
    est = tts_estimate(manifest, cfg)
    if est["chars"] == 0:
        print("No translated lines yet — run `run --stop-after translate` first.")
    else:
        print(f"{est['chars']} chars  ~{est['credits']} credits  ~${est['usd']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Georgian cartoon dubbing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (resumable)")
    r.add_argument("video")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--start-at", choices=STAGE_NAMES)
    r.add_argument("--stop-after", choices=STAGE_NAMES)
    r.add_argument("--yes", action="store_true", help="skip pre-spend confirmation")
    r.add_argument("--force", action="store_true",
                   help="redo stages that already have results (e.g. transcribe)")
    r.set_defaults(func=_run)

    e = sub.add_parser("estimate", help="preview TTS cost, no spend")
    e.add_argument("video")
    e.add_argument("--config", default="config.yaml")
    e.set_defaults(func=_estimate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
