"""Per-speaker stats and playable sample clips sliced from the vocals stem."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.util import Manifest


def speaker_stats(manifest: Manifest) -> pd.DataFrame:
    """Summary table shown after Analyze: lines / speech seconds / sample text."""
    per: dict[str, dict] = {}
    for s in manifest.segments:
        sp = s.get("speaker") or "(undetected)"
        d = per.setdefault(sp, {"lines": 0, "seconds": 0.0, "sample": ""})
        d["lines"] += 1
        d["seconds"] += s["end"] - s["start"]
        if len(s["text_src"]) > len(d["sample"]):
            d["sample"] = s["text_src"]
    rows = [
        {"speaker": sp, "lines": d["lines"], "speech (s)": round(d["seconds"], 1),
         "sample line": d["sample"][:80]}
        for sp, d in sorted(per.items())
    ]
    return pd.DataFrame(rows, columns=["speaker", "lines", "speech (s)", "sample line"])


def speaker_samples(
    workdir: Path, manifest: Manifest, k: int = 3,
    min_s: float = 1.5, max_s: float = 8.0,
) -> dict[str, list[dict]]:
    """{speaker: [{"path", "text", "dur"}, ...]} — the k longest lines each.

    Clips are cached by segment id in workdir/samples/; the Analyze handler
    deletes that directory when transcription actually re-runs (ids change).
    """
    from pydub import AudioSegment  # lazy: needs ffmpeg via pydub

    # Derived from the workdir, not manifest paths — those are absolute and go
    # stale when a work dir moves between machines (e.g. Colab -> local).
    vocals_path = workdir / "vocals.wav"
    if not vocals_path.exists():
        return {}
    out_dir = workdir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_speaker: dict[str, list[dict]] = {}
    for s in manifest.segments:
        sp = s.get("speaker")
        if sp:
            by_speaker.setdefault(sp, []).append(s)

    vocals = None  # load lazily, once, only if something needs exporting
    result: dict[str, list[dict]] = {}
    for sp, segs in sorted(by_speaker.items()):
        best = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        picked = [s for s in best if s["end"] - s["start"] >= min_s][:k] or best[:1]
        clips = []
        for seg in picked:
            clip_path = out_dir / f'{sp}_{seg["id"]:04d}.mp3'
            if not clip_path.exists():
                if vocals is None:
                    vocals = AudioSegment.from_file(vocals_path)
                start_ms = int(seg["start"] * 1000)
                end_ms = int(min(seg["end"], seg["start"] + max_s) * 1000)
                vocals[start_ms:end_ms].export(clip_path, format="mp3")
            clips.append({
                "path": str(clip_path),
                "text": seg["text_src"][:60],
                "dur": round(seg["end"] - seg["start"], 1),
            })
        result[sp] = clips
    return result
