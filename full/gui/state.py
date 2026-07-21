"""Pipeline status derived from disk, tab gating, and the review-table round-trip.

Disk (manifest.json + config.yaml) is the single source of truth: every
function here reloads state on entry so the GUI survives tunnel drops and
stays interchangeable with the CLI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import pandas as pd

from pipeline.config import Config
from pipeline.translate import char_budget
from pipeline.util import Manifest, workdir_for

from . import runner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

REVIEW_COLUMNS = ["id", "speaker", "time", "Dutch", "Georgian (editable)",
                  "budget", "chars", "fit"]


@dataclass
class Status:
    has_video: bool = False
    has_segments: bool = False
    n_segments: int = 0
    n_translated: int = 0
    speakers: list[str] = field(default_factory=list)
    voices_ok: bool = False
    has_output: bool = False
    output_path: str = ""


def _load(video: str) -> tuple[Config, Manifest]:
    cfg = Config.load(CONFIG_PATH)
    manifest = Manifest.load_or_init(workdir_for(Path(video)), Path(video))
    return cfg, manifest


def pipeline_status(video: str) -> Status:
    if not video or not Path(video).exists():
        return Status()
    cfg, manifest = _load(video)
    segs = manifest.segments
    speakers = sorted({s["speaker"] for s in segs if s.get("speaker")})

    def _mapped(sp: str | None) -> bool:
        vid, _ = cfg.voice_for(sp, manifest.voices)
        return bool(vid) and "REPLACE" not in vid.upper()

    output = manifest.data.get("output", "")
    return Status(
        has_video=True,
        has_segments=bool(segs),
        n_segments=len(segs),
        n_translated=sum(1 for s in segs if s.get("text_tgt")),
        speakers=speakers,
        # every detected speaker (and the None-fallback) must resolve to a real voice
        voices_ok=bool(segs) and all(_mapped(s.get("speaker")) for s in segs),
        has_output=bool(output) and Path(output).exists(),
        output_path=output,
    )


def gate(status: Status) -> tuple:
    """(analyze, voices, translate, dub) tab interactivity updates."""
    return (
        gr.update(interactive=status.has_video),
        gr.update(interactive=status.has_segments),
        gr.update(interactive=status.has_segments),
        gr.update(interactive=status.n_translated > 0 and status.voices_ok),
    )


def status_markdown(status: Status) -> str:
    if not status.has_video:
        return "No episode loaded."
    parts = [
        f"**{status.n_segments}** lines transcribed" if status.has_segments
        else "not yet analyzed",
        f"**{status.n_translated}/{status.n_segments}** translated",
        ("voices mapped ✓" if status.voices_ok else "voices not fully mapped"),
        ("dubbed output ready ✓" if status.has_output else "not yet dubbed"),
    ]
    return "Episode status: " + " · ".join(parts)


def discover_episodes() -> list[str]:
    """Video paths of episodes that already have a work dir (for resuming)."""
    found = []
    for mf in sorted((REPO_ROOT / "work").glob("*/manifest.json")):
        try:
            video = json.loads(mf.read_text(encoding="utf-8")).get("video", "")
            if video and Path(video).exists():
                found.append(video)
        except Exception:
            continue
    return found


# --- review table ---------------------------------------------------------


def _fit(text_tgt: str, budget: int) -> str:
    over = len(text_tgt) - budget
    return f"OVER +{over}" if over > 0 else "OK"


def manifest_to_df(video: str) -> tuple[pd.DataFrame, str]:
    """Review table + totals line, straight from disk."""
    if not video:
        return pd.DataFrame(columns=REVIEW_COLUMNS), "No episode loaded."
    cfg, manifest = _load(video)
    rows = []
    for s in manifest.segments:
        tgt = s.get("text_tgt") or ""
        budget = char_budget(s, cfg)
        rows.append({
            "id": s["id"],
            "speaker": s.get("speaker") or "?",
            "time": f'{s["start"]:.1f}–{s["end"]:.1f}',
            "Dutch": s["text_src"],
            "Georgian (editable)": tgt,
            "budget": budget,
            "chars": len(tgt),
            "fit": _fit(tgt, budget) if tgt else "",
        })
    df = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    over = sum(1 for r in rows if r["fit"].startswith("OVER"))
    translated = sum(1 for r in rows if r["Georgian (editable)"])
    totals = (f"{len(rows)} lines · {translated} translated · "
              f"{over} over budget" if rows else "No lines yet — run Analyze first.")
    return df, totals


def df_to_manifest(df: pd.DataFrame, video: str) -> tuple[pd.DataFrame, str]:
    """Write edited Georgian text back to the manifest; return refreshed table.

    Rows are matched by the id column — never by row order. Pandas hands back
    numerics as floats and empty cells as NaN, so coerce defensively.
    """
    if not video:
        return df, "No episode loaded."
    if runner.RUN_LOCK.locked():
        refreshed, totals = manifest_to_df(video)
        return refreshed, "⚠️ A run is in progress — edit not saved, try again after."

    cfg, manifest = _load(video)
    by_id = {s["id"]: s for s in manifest.segments}
    changed = 0
    for _, row in df.iterrows():
        try:
            seg = by_id.get(int(float(row["id"])))
        except (TypeError, ValueError):
            continue
        if seg is None:
            continue
        raw = row["Georgian (editable)"]
        new = "" if pd.isna(raw) else str(raw).strip()
        old = seg.get("text_tgt") or ""
        if new != old:
            if new:
                seg["text_tgt"] = new
            else:
                # Cell deliberately cleared — the line becomes re-translatable.
                seg.pop("text_tgt", None)
            changed += 1
    if changed:
        manifest.save()
    refreshed, totals = manifest_to_df(video)
    note = f" · saved {changed} edit(s) — only edited lines re-bill" if changed else ""
    return refreshed, totals + note
