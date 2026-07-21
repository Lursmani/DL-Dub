"""Stage 2 — split audio into dialogue (vocals) and a music+SFX bed (no_vocals).

Demucs is trained on music, but for dialogue-forward cartoons the vocals stem
captures speech well and the residual is a usable M&E bed. Sung segments are the
weak spot — expect artifacts there.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError, check_tool, run


def separate(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    check_tool(
        "demucs",
        hint="The ML extras are optional: pip install -r requirements-ml.txt "
             "(several GB incl. torch), or run the Analyze stage in Colab and "
             "finish locally — see README.",
    )
    audio = Path(manifest.data["audio"])
    out_root = workdir / "demucs"
    vocals = workdir / "vocals.wav"
    background = workdir / "background.wav"

    if not (vocals.exists() and background.exists()):
        # --two-stems=vocals -> {vocals.wav, no_vocals.wav}
        run(["demucs", "--two-stems", "vocals", "-o", out_root, audio],
            capture_output=True)
        # Demucs writes to <out_root>/<model>/<audio_stem>/{vocals,no_vocals}.wav
        produced = list(out_root.rglob("vocals.wav"))
        if not produced:
            raise PipelineError("[separate] demucs produced no output — check its logs.")
        stem_dir = produced[0].parent
        (stem_dir / "vocals.wav").replace(vocals)
        (stem_dir / "no_vocals.wav").replace(background)

    manifest.data["vocals"] = str(vocals)
    manifest.data["background"] = str(background)
    manifest.save()
    print(f"[separate] dialogue -> {vocals}\n[separate] music/SFX bed -> {background}")
