"""Shared helpers: workdir layout, manifest state, ffmpeg wrappers."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    """A stage failed in an expected way (bad input, missing key/tool, API error).

    Stages raise this instead of SystemExit: the GUI runs stages in a worker
    thread, and Python threads silently swallow SystemExit — errors would look
    like successful no-op runs. The CLI (dub.py) converts it back to SystemExit.
    """


def check_tool(name: str) -> None:
    """Fail fast with a clear message if a required CLI isn't on PATH."""
    if shutil.which(name) is None:
        raise PipelineError(
            f"Required tool '{name}' not found on PATH. See README setup section."
        )


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a subprocess; on failure, surface its captured stderr before raising."""
    try:
        return subprocess.run([str(c) for c in cmd], check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        err = e.stderr
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        print(f"[run] command failed: {' '.join(str(c) for c in cmd)}")
        if err:
            print(err.strip())
        raise


# Anchor work/ to the repo root so results don't depend on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def workdir_for(video: Path, root: Path | None = None) -> Path:
    """One stable working directory per source video; stages read/write here."""
    wd = (root if root is not None else _REPO_ROOT / "work") / video.stem
    (wd / "clips").mkdir(parents=True, exist_ok=True)
    return wd


class Manifest:
    """Resumable pipeline state, persisted as work/<episode>/manifest.json.

    Each stage fills in more fields on each segment, so any stage can be re-run
    without redoing the expensive ones (separation, transcription) before it.
    """

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def load_or_init(cls, workdir: Path, video: Path) -> "Manifest":
        path = workdir / "manifest.json"
        if path.exists():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        return cls(path, {"video": str(video), "segments": []})

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def segments(self) -> list[dict[str, Any]]:
        return self.data["segments"]

    @segments.setter
    def segments(self, value: list[dict[str, Any]]) -> None:
        self.data["segments"] = value


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
