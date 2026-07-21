"""Gradio app: one page around the managed ElevenLabs Dubbing API.

Layout + event wiring only — the dubbing logic lives in pipeline.autodub and
the log streaming in .runner.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr

from pipeline import autodub as autodub_mod
from pipeline.config import Config
from pipeline.util import workdir_for

from .runner import run_stages

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _setup_report() -> str:
    cfg = Config.load(CONFIG_PATH)
    return "\n".join([
        "### Environment",
        "- ELEVENLABS_API_KEY: "
        + ("✅ set" if cfg.elevenlabs_key else "❌ MISSING (edit .env)"),
        "- ffprobe (for the cost estimate): "
        + ("✅ found" if shutil.which("ffprobe") else "❌ not on PATH"),
    ])


def build_app() -> gr.Blocks:
    if not CONFIG_PATH.exists():  # first-run bootstrap
        example = REPO_ROOT / "config.example.yaml"
        if example.exists():
            shutil.copy(example, CONFIG_PATH)
    cfg0 = Config.load(CONFIG_PATH)

    with gr.Blocks(title="Auto-dub (ElevenLabs API)") as demo:
        gr.Markdown(
            "# ⚡ Auto-dub\n"
            "One call to ElevenLabs' managed **Dubbing API**: it transcribes, "
            "translates, **clones the original voices** and keeps their "
            "intonation — no speaker mapping, no local ML. Billed **per "
            "minute of source** (~$0.33/min watermarked, ~$0.50/min clean).")
        setup_md = gr.Markdown(_setup_report())

        video_tb = gr.Textbox(label="Video path",
                              placeholder="input/episode.mp4")
        with gr.Row():
            src = gr.Textbox(label="Source language", value=cfg0.source_lang)
            tgt = gr.Textbox(label="Target language", value=cfg0.target_lang)
            wm = gr.Checkbox(
                label="Watermark output ($0.33/min instead of $0.50/min)",
                value=True)
        est = gr.Markdown("")
        btn = gr.Button("🎬 Auto-dub (spends the amount above)",
                        variant="stop")
        log = gr.Textbox(label="Log", lines=10, max_lines=10,
                         autoscroll=True, interactive=False)
        out = gr.Video(label="Auto-dubbed episode", visible=False)
        dl = gr.DownloadButton("⬇️ Download", visible=False)

        def show_estimate(path: str, watermark: bool) -> str:
            if not path or not Path(path).exists():
                return ""
            try:
                e = autodub_mod.estimate(Path(path), watermark)
            except Exception as exc:  # noqa: BLE001 - surfaced in UI
                return f"⚠️ Could not read duration: {exc}"
            return (f"**Estimate: {e['minutes']} min of source ≈ "
                    f"${e['usd']}** (billed per minute of source audio)")

        for trigger in (video_tb.blur, wm.change, demo.load):
            trigger(show_estimate, inputs=[video_tb, wm], outputs=[est])

        # Re-check the environment (.env edits) on page load too.
        demo.load(_setup_report, outputs=[setup_md])

        def do_autodub(path: str, src_lang: str, tgt_lang: str,
                       watermark: bool):
            if not path or not Path(path).exists():
                yield ("Enter a valid video path first.",
                       gr.update(), gr.update())
                return
            video = Path(path)
            stage = [("autodub", lambda v, w, c: autodub_mod.autodub(
                v, w, c, source_lang=src_lang, target_lang=tgt_lang,
                watermark=watermark))]
            log_text = ""
            for log_text in run_stages(stage, video, CONFIG_PATH):
                yield log_text, gr.update(), gr.update()
            out_path = autodub_mod.output_path(
                video, workdir_for(video),
                (tgt_lang or "").strip()
                or Config.load(CONFIG_PATH).target_lang)
            if out_path.exists():
                yield (log_text,
                       gr.update(value=str(out_path), visible=True),
                       gr.update(value=str(out_path), visible=True))

        btn.click(do_autodub, inputs=[video_tb, src, tgt, wm],
                  outputs=[log, out, dl], concurrency_limit=1)

    demo.queue(max_size=16)
    return demo
