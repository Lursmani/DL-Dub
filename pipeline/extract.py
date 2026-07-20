"""Stage 1 — pull the audio track out of the video as WAV."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, check_tool, ffprobe_duration, run


def extract(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    check_tool("ffmpeg")
    check_tool("ffprobe")
    audio = workdir / "audio.wav"
    if not audio.exists():
        # 44.1 kHz stereo PCM — a clean, lossless input for separation.
        run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100", audio],
            capture_output=True)
    manifest.data["duration"] = ffprobe_duration(video)
    manifest.data["audio"] = str(audio)
    manifest.save()
    print(f"[extract] audio -> {audio}  ({manifest.data['duration']:.1f}s)")
