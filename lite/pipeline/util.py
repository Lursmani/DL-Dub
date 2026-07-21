"""Shared helpers: workdir layout and ffprobe wrapper."""
from __future__ import annotations

import subprocess
from pathlib import Path


class PipelineError(RuntimeError):
    """The dub failed in an expected way (bad input, missing key, API error).

    Raised instead of SystemExit: the GUI runs the dub in a worker thread, and
    Python threads silently swallow SystemExit — errors would look like
    successful no-op runs. The CLI (autodub.py) converts it back to SystemExit.
    """


# Anchor work/ to the project root so results don't depend on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def workdir_for(video: Path, root: Path | None = None) -> Path:
    """One stable working directory per source video; results land here."""
    wd = (root if root is not None else _REPO_ROOT / "work") / video.stem
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def ffprobe_duration(path: Path) -> float:
    """Media duration in seconds via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())
