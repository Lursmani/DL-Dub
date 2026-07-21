"""Stage 7 — replace the video's audio with the mixed Georgian dub."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError, run


def mux(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    # Derived from the workdir, not manifest paths — those are absolute and go
    # stale when a work dir moves between machines (e.g. Colab -> local).
    final_audio = workdir / "final_audio.wav"
    if not final_audio.exists():
        raise PipelineError("[mux] final_audio.wav not found in the work dir — "
                            "run the assemble stage first.")
    out = workdir / f"{video.stem}.{cfg.target_lang}.mp4"
    run([
        "ffmpeg", "-y", "-i", video, "-i", final_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out,
    ], capture_output=True)
    manifest.data["output"] = str(out)
    manifest.save()
    print(f"[mux] dubbed video -> {out}")
