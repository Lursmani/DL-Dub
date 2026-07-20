"""Stage 7 — replace the video's audio with the mixed Georgian dub."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, run


def mux(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    final_audio = manifest.data["final_audio"]
    out = workdir / f"{video.stem}.ka.mp4"
    run([
        "ffmpeg", "-y", "-i", video, "-i", final_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out,
    ], capture_output=True)
    manifest.data["output"] = str(out)
    manifest.save()
    print(f"[mux] dubbed video -> {out}")
