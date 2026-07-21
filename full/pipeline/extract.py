"""Stage 1 — pull the audio track out of the video as WAV."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError, check_tool, ffprobe_duration, run


def extract(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    check_tool("ffmpeg")
    check_tool("ffprobe")
    audio = workdir / "audio.wav"
    duration = ffprobe_duration(video)
    # Workdirs are keyed by filename stem — a different video with the same
    # name would silently reuse this episode's cached analysis.
    prev = manifest.data.get("duration")
    if audio.exists() and prev is not None and abs(prev - duration) > 0.5:
        raise PipelineError(
            f"[extract] work/{video.stem} holds results for a different video "
            f"({prev:.1f}s vs {duration:.1f}s) — rename the file or delete "
            "that work directory.")
    if not audio.exists():
        # 44.1 kHz stereo PCM — a clean, lossless input for separation.
        run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100", audio],
            capture_output=True)
    manifest.data["duration"] = duration
    manifest.data["audio"] = str(audio)
    manifest.save()
    print(f"[extract] audio -> {audio}  ({manifest.data['duration']:.1f}s)")
