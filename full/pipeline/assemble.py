"""Stage 6 — fit each dubbed clip to its slot and mix over the music/SFX bed.

For each line: time-stretch the synthesized clip toward its original duration
(clamped so voices don't turn into chipmunks), lay it on a silent timeline at
the original start time, then mix that dialogue track over the background bed.
"""
from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from .config import Config
from .util import Manifest, PipelineError, run


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
    # Derived from the workdir, not manifest paths — those are absolute and go
    # stale when a work dir moves between machines (e.g. Colab -> local).
    background_path = workdir / "background.wav"
    if not background_path.exists():
        raise PipelineError("[assemble] background.wav not found in the work "
                            "dir — run the separate stage first.")
    background = AudioSegment.from_file(background_path)
    total_ms = len(background)

    # Fit all clips first so the timeline can grow to hold a final line that
    # overflows its slot — overlay silently drops audio past the timeline end.
    fitted_clips: list[tuple[int, AudioSegment]] = []
    end_ms = total_ms
    for s in manifest.segments:
        clip_path = s.get("clip")
        if not clip_path:
            continue
        clip = Path(clip_path)
        if not clip.exists():
            # Stale absolute path from another machine; same filename locally?
            clip = tmp / clip.name
            if not clip.exists():
                raise PipelineError(f"[assemble] missing clip for line "
                                    f"{s['id']} — re-run the tts stage.")
        pos = int(s["start"] * 1000)
        fitted = _fit_duration(clip, s["end"] - s["start"], cfg, tmp)
        fitted_clips.append((pos, fitted))
        end_ms = max(end_ms, pos + len(fitted))

    dialogue = AudioSegment.silent(duration=end_ms)
    for pos, fitted in fitted_clips:
        dialogue = dialogue.overlay(fitted, position=pos)

    bed = background.apply_gain(cfg.background_gain_db)
    if end_ms > total_ms:
        bed = bed + AudioSegment.silent(duration=end_ms - total_ms)
        print(f"[assemble] dialogue runs {(end_ms - total_ms) / 1000:.1f}s past "
              "the source audio; mux trims the result to the video length.")
    mixed = bed.overlay(dialogue)
    final = workdir / "final_audio.wav"
    mixed.export(final, format="wav")
    manifest.data["final_audio"] = str(final)
    manifest.save()
    print(f"[assemble] mixed dub -> {final}")
