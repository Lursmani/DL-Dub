"""Stage 6 — fit each dubbed clip to its slot and mix over the music/SFX bed.

For each line: time-stretch the synthesized clip toward its original duration
(clamped so voices don't turn into chipmunks), lay it on a silent timeline at
the original start time, then mix that dialogue track over the background bed.
"""
from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from .config import Config
from .util import Manifest, run


def _fit_duration(clip: Path, target_s: float, cfg: Config, tmp: Path) -> AudioSegment:
    """Return the clip stretched toward target_s, within configured bounds."""
    seg = AudioSegment.from_file(clip)
    src_s = len(seg) / 1000.0
    if src_s <= 0 or target_s <= 0:
        return seg
    ratio = src_s / target_s  # >1 means clip is too long -> must speed up
    ratio = max(cfg.min_slowdown, min(cfg.max_speedup, ratio))
    if abs(ratio - 1.0) < 0.02:
        return seg
    # ffmpeg atempo does clean speech time-stretch (no pitch change); chain for >2x.
    out = tmp / f"fit_{clip.stem}.wav"
    filters, r = [], ratio
    while r > 2.0:
        filters.append("atempo=2.0"); r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5"); r /= 0.5
    filters.append(f"atempo={r:.4f}")
    run(["ffmpeg", "-y", "-i", clip, "-filter:a", ",".join(filters), out],
        capture_output=True)
    return AudioSegment.from_file(out)


def assemble(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    tmp = workdir / "clips"
    background = AudioSegment.from_file(manifest.data["background"])
    total_ms = len(background)

    dialogue = AudioSegment.silent(duration=total_ms)
    for s in manifest.segments:
        clip_path = s.get("clip")
        if not clip_path:
            continue
        target = s["end"] - s["start"]
        fitted = _fit_duration(Path(clip_path), target, cfg, tmp)
        dialogue = dialogue.overlay(fitted, position=int(s["start"] * 1000))

    mixed = background.apply_gain(cfg.background_gain_db).overlay(dialogue)
    final = workdir / "final_audio.wav"
    mixed.export(final, format="wav")
    manifest.data["final_audio"] = str(final)
    manifest.save()
    print(f"[assemble] mixed dub -> {final}")
