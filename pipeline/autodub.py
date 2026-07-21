"""One-call dubbing via the managed ElevenLabs Dubbing API.

The opposite trade-off to the manual pipeline: billed per minute of source
audio (~$0.33/min watermarked, ~$0.50/min clean at API rates) instead of per
character — several times pricier — but it needs no local ML at all, and it
clones the original voices and preserves their intonation automatically.
This is the primary path for the lite (API-only) install.
"""
from __future__ import annotations

import time
from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError, ffprobe_duration

# API billing per minute of source audio, by watermark choice.
_USD_PER_MIN = {True: 0.33, False: 0.50}
_POLL_SECONDS = 5
_TIMEOUT_SECONDS = 45 * 60


def estimate(video: Path, watermark: bool) -> dict:
    """Cost preview from source duration — no API calls."""
    minutes = ffprobe_duration(video) / 60
    return {
        "minutes": round(minutes, 1),
        "usd": round(minutes * _USD_PER_MIN[watermark], 2),
    }


_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def output_path(video: Path, workdir: Path, target_lang: str) -> Path:
    # .auto. distinguishes it from the manual pipeline's <stem>.<lang>.mp4,
    # so both results can coexist for the same episode. The API returns mp4
    # for video input and mp3 for audio input regardless of the source format.
    ext = ".mp3" if video.suffix.lower() in _AUDIO_SUFFIXES else ".mp4"
    return workdir / f"{video.stem}.{target_lang}.auto{ext}"


def _api(client, new: str, old: str):
    """The SDK renamed the dubbing methods between 1.x and 2.x; take either."""
    return getattr(client.dubbing, new, None) or getattr(client.dubbing, old)


def autodub(video: Path, workdir: Path, manifest: Manifest, cfg: Config,
            *, source_lang: str = "", target_lang: str = "",
            watermark: bool = True) -> Path:
    """Upload, poll until dubbed, download. Progress goes to stdout so the
    GUI's runner machinery can stream it. Stage-compatible signature
    (manifest is unused) so it can run through gui.runner.run_stages."""
    from elevenlabs.client import ElevenLabs

    if not cfg.elevenlabs_key:
        raise PipelineError("[autodub] ELEVENLABS_API_KEY required (see .env.example).")
    source_lang = (source_lang or "").strip() or "auto"
    target_lang = (target_lang or "").strip() or cfg.target_lang

    out = output_path(video, workdir, target_lang)
    if out.exists():
        print(f"[autodub] cached result exists — delete it to re-dub: {out}")
        return out

    est = estimate(video, watermark)
    print(f"[autodub] {est['minutes']} min of source -> ~${est['usd']} "
          f"({'watermarked' if watermark else 'clean'})")

    client = ElevenLabs(api_key=cfg.elevenlabs_key)
    print(f"[autodub] uploading {video.name} ({source_lang} -> {target_lang})…")
    with open(video, "rb") as f:
        job = _api(client, "create", "dub_a_video_or_an_audio_file")(
            file=f, source_lang=source_lang, target_lang=target_lang,
            watermark=watermark,
        )

    print(f"[autodub] dubbing_id={job.dubbing_id} — ElevenLabs is transcribing, "
          "translating, cloning voices, and rendering (typically ~1-2x realtime).")
    start = time.monotonic()
    while True:
        meta = _api(client, "get", "get_dubbing_project_metadata")(job.dubbing_id)
        if meta.status == "dubbed":
            print()  # finish the \r status line
            break
        if meta.status != "dubbing":
            raise PipelineError(f"[autodub] job ended as {meta.status!r}: "
                                f"{getattr(meta, 'error', None) or 'no detail'}")
        elapsed = time.monotonic() - start
        if elapsed > _TIMEOUT_SECONDS:
            raise PipelineError(f"[autodub] still not done after "
                                f"{_TIMEOUT_SECONDS // 60} min — check the "
                                f"ElevenLabs dashboard for job {job.dubbing_id}.")
        # \r so the status becomes one self-updating line (terminal and GUI).
        print(f"[autodub] status: {meta.status}… ({elapsed:.0f}s)", end="\r")
        time.sleep(_POLL_SECONDS)

    print("[autodub] downloading result…")
    if hasattr(client.dubbing, "audio"):  # 2.x
        stream = client.dubbing.audio.get(job.dubbing_id, target_lang)
    else:  # 1.x
        stream = client.dubbing.get_dubbed_file(job.dubbing_id, target_lang)
    with open(out, "wb") as fo:
        for chunk in stream:
            fo.write(chunk)
    print(f"[autodub] done -> {out}")
    return out
