"""Gradio Blocks app: a tabbed wizard over the dubbing pipeline.

The layout adapts to the install (ML_AVAILABLE): full installs run the whole
5-step wizard locally; hybrid installs (no ML extras) get Colab guidance in
place of local Analyze and run the remaining stages here.

Layout + event wiring only — logic lives in runner/state/samples/voices.
Heavy buttons share concurrency_id="heavy" so GPU/manifest work is serialized;
light interactions stay responsive.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import gradio as gr

from pipeline.config import Config
from pipeline.translate import PROVIDERS, available_providers, effective_model
from pipeline.tts import estimate as tts_estimate
from pipeline.util import Manifest, workdir_for

from . import samples, voices
from .runner import RUN_LOCK, run_stages
from .state import (
    CONFIG_PATH,
    REPO_ROOT,
    REVIEW_COLUMNS,
    df_to_manifest,
    discover_episodes,
    gate,
    manifest_to_df,
    pipeline_status,
    status_markdown,
)

MAX_SPEAKERS = 8
HEAVY = {"concurrency_id": "heavy", "concurrency_limit": 1}

# Hybrid installs lack the ML extras (requirements-ml.txt); the GUI reshapes
# itself around that: the Analyze tab turns into run-it-in-Colab guidance.
ML_AVAILABLE = (shutil.which("demucs") is not None
                and importlib.util.find_spec("whisperx") is not None)

COLAB_URL = ("https://colab.research.google.com/github/Lursmani/DL-Dub/"
             "blob/main/full/colab_dub.ipynb")


def _noop(n: int) -> tuple:
    return tuple(gr.update() for _ in range(n))


def _dataframe(**kwargs) -> gr.Dataframe:
    """gr.Dataframe with static_columns when supported (gradio >= 5.13)."""
    try:
        return gr.Dataframe(**kwargs)
    except TypeError:
        kwargs.pop("static_columns", None)
        return gr.Dataframe(**kwargs)


def _setup_report() -> str:
    cfg = Config.load(CONFIG_PATH)
    lines = ["### Environment"]
    for label, ok in [
        ("ELEVENLABS_API_KEY", bool(cfg.elevenlabs_key)),
        ("HF_TOKEN (diarization)", bool(cfg.hf_token)),
    ]:
        lines.append(f"- {label}: {'✅ set' if ok else '❌ MISSING (edit .env)'}")
    avail = available_providers(cfg)
    marks = " · ".join(
        f"{PROVIDERS[p]['label']} {'✅' if ok else '❌'}"
        for p, ok in avail.items())
    lines.append(
        f"- Translation APIs (need ≥1, pick in the Translate tab): {marks}"
        + ("" if any(avail.values()) else " — **all missing, edit .env**"))
    lines.append(f"- ffmpeg: "
                 f"{'✅ found' if shutil.which('ffmpeg') else '❌ not on PATH'}")
    lines.append(
        "- local ML stages (demucs + whisperx): "
        + ("✅ installed" if ML_AVAILABLE else
           "⚠️ not installed — Analyze needs `pip install -r "
           "requirements-ml.txt` (several GB), or run Analyze in Colab; "
           "all other tabs work without it"))
    if cfg.device == "cuda":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown GPU"
        lines.append(f"- device: ✅ **cuda** ({cfg.compute_type}) — {name}")
    else:
        lines.append("- device: ⚠️ **CPU** — Analyze will take ~20-40 min per "
                     "6-min episode (fine in Colab with a GPU runtime)")
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    if not CONFIG_PATH.exists():  # first-run bootstrap
        example = REPO_ROOT / "config.example.yaml"
        if example.exists():
            shutil.copy(example, CONFIG_PATH)

    with gr.Blocks(title="Georgian Dub Studio") as demo:
        episode = gr.State("")  # video path; all other state re-read from disk

        gr.Markdown(
            "# 🎬 Georgian Dub Studio\n"
            + ("🟢 **Full mode** — local ML installed; the whole pipeline "
               "runs here." if ML_AVAILABLE else
               "🔀 **Hybrid mode** — no local ML. Run Analyze in Colab, "
               "then continue here. (For one-call API dubbing, use the "
               "separate lite project.)"))
        with gr.Row():
            status_banner = gr.Markdown("No episode loaded.")
            refresh_btn = gr.Button("🔄 Refresh from disk", size="sm", scale=0)

        # The tab set is fixed; only the Analyze tab's label reflects whether
        # the ML stages run here or in Colab.
        analyze_label = ("2 · Analyze" if ML_AVAILABLE
                         else "2 · Analyze (in Colab)")

        with gr.Tabs() as tabs:
            # ---------------- 1 · Setup ----------------
            with gr.Tab("1 · Setup", id="setup"):
                setup_md = gr.Markdown(_setup_report())
                video_tb = gr.Textbox(
                    label="Video path",
                    placeholder="/content/drive/MyDrive/dubbing/episode.mp4",
                )
                resume_dd = gr.Dropdown(
                    label="…or resume an existing episode",
                    choices=discover_episodes(), interactive=True,
                )
                load_btn = gr.Button("Load episode", variant="primary")

            # ---------------- 2 · Analyze ----------------
            with gr.Tab(analyze_label, id="analyze",
                        interactive=False) as tab_analyze:
                if ML_AVAILABLE:
                    gr.Markdown("Extract audio → separate dialogue from "
                                "music/SFX → transcribe + detect speakers. "
                                "**Free**, but the slow GPU part (~2-4 min on "
                                "a T4).")
                else:
                    gr.Markdown(
                        "**This install can't run the ML analysis locally.** "
                        "To analyze anyway:\n"
                        f"1. [Run Analyze in Colab]({COLAB_URL}) (free GPU) — "
                        "it writes `work/<episode>/` with the manifest and "
                        "speaker samples.\n"
                        "2. Copy that `work/<episode>/` folder into this "
                        "app's `work/` directory.\n"
                        "3. Click **🔄 Refresh from disk** above and continue "
                        "in the Voices tab.\n\n"
                        "To run Analyze locally instead, `pip install -r "
                        "requirements-ml.txt` (several GB) or use the Docker "
                        "image. For one-call API dubbing with cloned voices, "
                        "use the separate **lite** project.")
                force_cb = gr.Checkbox(
                    label="Force re-transcribe (discards existing translations)",
                    value=False, visible=ML_AVAILABLE)
                analyze_btn = gr.Button("Run analysis", variant="primary",
                                        visible=ML_AVAILABLE)
                analyze_log = gr.Textbox(label="Log", lines=14, max_lines=14,
                                         autoscroll=True, interactive=False,
                                         visible=ML_AVAILABLE)
                speakers_table = gr.Dataframe(
                    label="Detected speakers", interactive=False,
                    headers=["speaker", "lines", "speech (s)", "sample line"])

            # ---------------- 3 · Voices ----------------
            with gr.Tab("3 · Voices", id="voices",
                        interactive=False) as tab_voices:
                gr.Markdown(
                    "Listen to each detected speaker, then assign an ElevenLabs "
                    "voice. **Fetch voices** pulls your voice library (incl. any "
                    "cloned Georgian voices). Choices are saved per episode "
                    "(into its manifest); the default voice goes to "
                    "`config.yaml`.")
                with gr.Row():
                    fetch_btn = gr.Button("Fetch my ElevenLabs voices")
                    voices_status = gr.Markdown("")
                voices_state = gr.State([])       # [{label, voice_id, preview_url}]
                slot_speakers = gr.State([])      # slot index -> speaker label
                slots = []
                for i in range(MAX_SPEAKERS):
                    with gr.Group(visible=False) as grp:
                        title = gr.Markdown()
                        with gr.Row():
                            auds = [gr.Audio(label=f"sample {j+1}", type="filepath",
                                             visible=False) for j in range(3)]
                        with gr.Row():
                            dd = gr.Dropdown(label="ElevenLabs voice", choices=[],
                                             allow_custom_value=True, scale=2)
                            rd = gr.Radio(label="TTS model",
                                          choices=voices.TTS_MODELS,
                                          value="eleven_flash_v2_5", scale=2)
                        with gr.Row():
                            demo_aud = gr.Audio(label="voice demo (free, not Georgian)",
                                                visible=False, scale=2)
                            prev_btn = gr.Button(
                                "🔊 Georgian preview (bills ~15-30 credits, cached)",
                                size="sm", scale=1)
                            prev_aud = gr.Audio(label="Georgian preview",
                                                visible=False, scale=2)
                    slots.append({"grp": grp, "title": title, "auds": auds,
                                  "dd": dd, "rd": rd, "demo": demo_aud,
                                  "prev_btn": prev_btn, "prev": prev_aud})
                default_dd = gr.Dropdown(
                    label="Default voice (fallback for unmapped speakers)",
                    choices=[], allow_custom_value=True)
                save_btn = gr.Button("💾 Save voice mapping for this episode",
                                     variant="primary")

            # ---------------- 4 · Translate & review ----------------
            with gr.Tab("4 · Translate & review", id="translate",
                        interactive=False) as tab_translate:
                gr.Markdown(
                    "Translate Dutch → Georgian with your chosen API (costs "
                    "pennies), then edit any line below. Only the **Georgian** "
                    "column is editable; edits save automatically. Lines you edit "
                    "re-bill only themselves at dub time. Clear a cell to mark "
                    "the line for re-translation.")
                with gr.Row():
                    provider_dd = gr.Dropdown(
                        label="Translation API", choices=[], scale=2,
                        info="❌ = key missing from .env")
                    model_tb = gr.Textbox(
                        label="Model (blank = provider default)", scale=2)
                    with gr.Column(scale=2):
                        provider_md = gr.Markdown()
                translate_btn = gr.Button("Translate untranslated lines",
                                          variant="primary")
                translate_log = gr.Textbox(label="Log", lines=6, max_lines=6,
                                           autoscroll=True, interactive=False)
                totals_md = gr.Markdown("")
                review_df = _dataframe(
                    headers=REVIEW_COLUMNS,
                    datatype=["number", "str", "str", "str", "str",
                              "number", "number", "str"],
                    col_count=(len(REVIEW_COLUMNS), "fixed"),
                    interactive=True, wrap=True,
                    static_columns=[0, 1, 2, 3, 5, 6, 7],
                )

            # ---------------- 5 · Dub ----------------
            with gr.Tab("5 · Dub", id="dub",
                        interactive=False) as tab_dub:
                estimate_md = gr.Markdown("")
                dub_btn = gr.Button(
                    "🎙️ Synthesize + assemble + mux (spends the credits above)",
                    variant="stop")
                dub_log = gr.Textbox(label="Log", lines=10, max_lines=10,
                                     autoscroll=True, interactive=False)
                out_video = gr.Video(label="Dubbed episode", visible=False)
                with gr.Row():
                    download_btn = gr.DownloadButton("⬇️ Download", visible=False)
                    drive_btn = gr.Button("📁 Copy to Google Drive",
                                          visible="google.colab" in sys.modules)
                    dub_status = gr.Markdown("")

        gated_tabs = [tab_analyze, tab_voices, tab_translate, tab_dub]

        # ================= handlers =================

        def _slot_updates(video: str) -> list:
            """Visibility/titles/samples/current-choices for all 8 voice slots."""
            if not video:
                hidden = []
                for _ in range(MAX_SPEAKERS):
                    hidden.append(gr.update(visible=False))
                    hidden.append(gr.update(value=""))
                    hidden.extend(gr.update(visible=False, value=None)
                                  for _ in range(3))
                    hidden.append(gr.update(value=None))
                    hidden.append(gr.update())
                return hidden + [[]]
            cfg = Config.load(CONFIG_PATH)
            manifest = Manifest.load_or_init(workdir_for(Path(video)), Path(video))
            per_speaker = samples.speaker_samples(workdir_for(Path(video)), manifest)
            speakers = sorted(per_speaker)[:MAX_SPEAKERS]
            updates, mapped = [], []
            for i in range(MAX_SPEAKERS):
                if i < len(speakers):
                    sp = speakers[i]
                    clips = per_speaker[sp]
                    n_lines = sum(1 for s in manifest.segments
                                  if s.get("speaker") == sp)
                    vid, model = cfg.voice_for(sp, manifest.voices)
                    cur = "" if "REPLACE" in (vid or "").upper() else vid
                    updates.append(gr.update(visible=True))          # group
                    updates.append(gr.update(                        # title
                        value=f"### {sp} — {n_lines} lines"))
                    for j in range(3):                               # 3 samples
                        if j < len(clips):
                            c = clips[j]
                            updates.append(gr.update(
                                value=c["path"], visible=True,
                                label=f'“{c["text"]}” ({c["dur"]}s)'))
                        else:
                            updates.append(gr.update(visible=False, value=None))
                    updates.append(gr.update(value=cur or None))     # dropdown
                    updates.append(gr.update(value=model))           # radio
                    mapped.append(sp)
                else:
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(value=""))
                    updates.extend(gr.update(visible=False, value=None)
                                   for _ in range(3))
                    updates.append(gr.update(value=None))
                    updates.append(gr.update())
            return updates + [mapped]

        slot_outputs = []
        for s in slots:
            slot_outputs += [s["grp"], s["title"], *s["auds"], s["dd"], s["rd"]]
        slot_outputs.append(slot_speakers)

        def do_load(path: str, resume: str | None):
            video = (path or "").strip() or (resume or "")
            if not video or not Path(video).exists():
                return ("", "⚠️ Video not found — check the path.",
                        *_noop(len(gated_tabs)), gr.update(),
                        gr.update(choices=discover_episodes()))
            st = pipeline_status(video)
            return (video, status_markdown(st), *gate(st),
                    gr.update(selected="analyze" if not st.has_segments
                              else "voices"),
                    gr.update(choices=discover_episodes()))

        load_btn.click(
            do_load, inputs=[video_tb, resume_dd],
            outputs=[episode, status_banner, *gated_tabs, tabs, resume_dd],
        )

        def do_refresh(video: str):
            st = pipeline_status(video)
            df, totals = (manifest_to_df(video) if st.has_segments
                          else (gr.update(), ""))
            return (status_markdown(st), *gate(st), df, totals,
                    gr.update(value=samples.speaker_stats(
                        Manifest.load_or_init(workdir_for(Path(video)),
                                              Path(video)))
                        if st.has_segments else None),
                    _setup_report())

        refresh_btn.click(
            do_refresh, inputs=[episode],
            outputs=[status_banner, *gated_tabs, review_df, totals_md,
                     speakers_table, setup_md],
        )
        # Re-check the environment (.env edits, new installs) on page load too.
        demo.load(_setup_report, outputs=[setup_md])

        # --- Analyze ---
        def do_analyze(video: str, force: bool):
            n_out = 8  # log, table, banner, 4 tabs, button
            if not video:
                yield ("Load an episode first (Setup tab).", *_noop(n_out - 1))
                return
            st = pipeline_status(video)
            if st.has_segments and not force:
                yield ("Segments already exist — tick 'Force re-transcribe' "
                       "to redo (this discards translations).", *_noop(n_out - 1))
                return
            if force:
                shutil.rmtree(workdir_for(Path(video)) / "samples",
                              ignore_errors=True)
            log = ""
            for log in run_stages(["extract", "separate", "transcribe"],
                                  Path(video), CONFIG_PATH):
                yield (log, *_noop(n_out - 2), gr.update(interactive=False))
            st = pipeline_status(video)
            manifest = Manifest.load_or_init(workdir_for(Path(video)), Path(video))
            yield (log, samples.speaker_stats(manifest), status_markdown(st),
                   *gate(st), gr.update(interactive=True))

        analyze_btn.click(
            do_analyze, inputs=[episode, force_cb],
            outputs=[analyze_log, speakers_table, status_banner, *gated_tabs,
                     analyze_btn],
            **HEAVY,
        )

        # --- Voices ---
        tab_voices.select(_slot_updates, inputs=[episode], outputs=slot_outputs)

        def do_fetch():
            cfg = Config.load(CONFIG_PATH)
            try:
                found = voices.list_voices(cfg.elevenlabs_key)
            except RuntimeError as e:
                return (*_noop(MAX_SPEAKERS + 1), [], f"⚠️ {e}")
            choices = [(v["label"], v["voice_id"]) for v in found]
            dd_updates = tuple(gr.update(choices=choices)
                               for _ in range(MAX_SPEAKERS + 1))
            return (*dd_updates, found, f"Fetched {len(found)} voices.")

        # (error branch below returns [] for voices_state — gr.State outputs
        # need real values, not gr.update())

        fetch_btn.click(
            do_fetch,
            outputs=[*(s["dd"] for s in slots), default_dd, voices_state,
                     voices_status],
        )

        for s in slots:
            def _demo(voice_id, catalog, _s=s):
                url = next((v["preview_url"] for v in (catalog or [])
                            if v["voice_id"] == voice_id and v["preview_url"]),
                           None)
                return gr.update(value=url, visible=bool(url))

            s["dd"].change(_demo, inputs=[s["dd"], voices_state],
                           outputs=[s["demo"]])

            def _preview(voice_id, model_id, video, _s=s):
                cfg = Config.load(CONFIG_PATH)
                try:
                    path = voices.georgian_preview(
                        cfg.elevenlabs_key, workdir_for(Path(video)),
                        voice_id, model_id)
                    return gr.update(value=path, visible=True), ""
                except Exception as e:  # noqa: BLE001 - surfaced in UI
                    return gr.update(), f"⚠️ Preview failed: {e}"

            s["prev_btn"].click(_preview,
                                inputs=[s["dd"], s["rd"], episode],
                                outputs=[s["prev"], voices_status])

        def do_save(video, speakers, default_voice, *dd_rd_values):
            if not video:
                st = pipeline_status(video)
                return ("⚠️ Load an episode first (Setup tab).",
                        status_markdown(st), *gate(st))
            if RUN_LOCK.locked():
                st = pipeline_status(video)
                return ("⚠️ A run is in progress — voices not saved, "
                        "try again after.", status_markdown(st), *gate(st))
            dds = dd_rd_values[:MAX_SPEAKERS]
            rds = dd_rd_values[MAX_SPEAKERS:]
            mapping = {sp: (dds[i] or "", rds[i])
                       for i, sp in enumerate(speakers or [])}
            manifest = Manifest.load_or_init(workdir_for(Path(video)),
                                             Path(video))
            msg = voices.save_mapping(CONFIG_PATH, manifest, mapping,
                                      default_voice or "")
            st = pipeline_status(video)
            return (msg, status_markdown(st), *gate(st))

        save_btn.click(
            do_save,
            inputs=[episode, slot_speakers, default_dd,
                    *(s["dd"] for s in slots), *(s["rd"] for s in slots)],
            outputs=[voices_status, status_banner, *gated_tabs],
        )

        # --- Translate & review ---
        def _provider_line(cfg: Config) -> str:
            provider = (cfg.translate_provider or "anthropic").lower()
            if provider not in PROVIDERS:
                return f"⚠️ Unknown provider `{provider}` in config.yaml."
            model = effective_model(provider, cfg.translate_model)
            ok = available_providers(cfg).get(provider, False)
            return (f"Using **{PROVIDERS[provider]['label']}** · `{model}`"
                    + ("" if ok else
                       f"<br>⚠️ `{PROVIDERS[provider]['env']}` is not set — "
                       "translation will fail until you add it to .env "
                       "(or Colab Secrets) and relaunch."))

        def refresh_provider_ui():
            cfg = Config.load(CONFIG_PATH)
            avail = available_providers(cfg)
            choices = [
                (f"{PROVIDERS[p]['label']} {'✅' if avail[p] else '❌'}", p)
                for p in PROVIDERS]
            cur = ((cfg.translate_provider or "anthropic").lower()
                   if (cfg.translate_provider or "").lower() in PROVIDERS
                   else "anthropic")
            return (gr.update(choices=choices, value=cur),
                    gr.update(value=cfg.translate_model or ""),
                    _provider_line(cfg))

        def set_provider(provider: str, model: str):
            model = (model or "").strip()
            if model and effective_model(provider, model) != model:
                model = ""  # stale model from another provider — clear it
            Config.update_yaml(CONFIG_PATH, {
                "translate_provider": provider,
                "translate_model": model,
            })
            return (gr.update(value=model),
                    _provider_line(Config.load(CONFIG_PATH)))

        tab_translate.select(refresh_provider_ui,
                             outputs=[provider_dd, model_tb, provider_md])
        provider_dd.input(set_provider, inputs=[provider_dd, model_tb],
                          outputs=[model_tb, provider_md])
        model_tb.blur(set_provider, inputs=[provider_dd, model_tb],
                      outputs=[model_tb, provider_md])

        def do_translate(video: str):
            n_out = 8  # log, df, totals, 4 tabs, button
            if not video:
                yield ("Load an episode first.", *_noop(n_out - 1))
                return
            log = ""
            for log in run_stages(["translate"], Path(video), CONFIG_PATH):
                yield (log, *_noop(n_out - 2), gr.update(interactive=False))
            # repopulate from disk even after partial failure — partials saved
            df, totals = manifest_to_df(video)
            st = pipeline_status(video)
            yield (log, df, totals, *gate(st), gr.update(interactive=True))

        translate_btn.click(
            do_translate, inputs=[episode],
            outputs=[translate_log, review_df, totals_md, *gated_tabs,
                     translate_btn],
            **HEAVY,
        )

        review_df.input(df_to_manifest, inputs=[review_df, episode],
                        outputs=[review_df, totals_md])

        # --- Dub ---
        def show_estimate(video: str):
            if not video:
                return "Load an episode first."
            cfg = Config.load(CONFIG_PATH)
            manifest = Manifest.load_or_init(workdir_for(Path(video)), Path(video))
            est = tts_estimate(manifest, cfg)
            if est["chars"] == 0:
                return "Nothing to synthesize yet — run Translate first."
            st = pipeline_status(video)
            recap_lines = []
            for sp in st.speakers:
                vid, model = cfg.voice_for(sp, manifest.voices)
                recap_lines.append(f"- `{sp}` → `{vid}` ({model})")
            recap = "\n".join(recap_lines)
            return (f"### Cost estimate\n**{est['chars']} characters ≈ "
                    f"{est['credits']} credits ≈ ${est['usd']}**\n\n"
                    f"Voices about to be used:\n{recap}\n\n"
                    "Already-synthesized identical lines are cached and won't "
                    "re-bill.")

        tab_dub.select(show_estimate, inputs=[episode], outputs=[estimate_md])

        def do_dub(video: str):
            n_out = 9  # log, video, download, status, 4 tabs, button
            if not video:
                yield ("Load an episode first.", *_noop(n_out - 1))
                return
            st = pipeline_status(video)
            cfg = Config.load(CONFIG_PATH)
            manifest = Manifest.load_or_init(workdir_for(Path(video)), Path(video))
            if tts_estimate(manifest, cfg)["chars"] == 0:
                yield ("Nothing to synthesize — run Translate first.",
                       *_noop(n_out - 1))
                return
            if not st.voices_ok:
                yield ("Voices are not fully mapped — finish the Voices tab.",
                       *_noop(n_out - 1))
                return
            log = ""
            for log in run_stages(["tts", "assemble", "mux"], Path(video),
                                  CONFIG_PATH):
                yield (log, *_noop(n_out - 2), gr.update(interactive=False))
            st = pipeline_status(video)
            if st.has_output:
                yield (log,
                       gr.update(value=st.output_path, visible=True),
                       gr.update(value=st.output_path, visible=True),
                       f"✅ Done: `{st.output_path}`",
                       *gate(st), gr.update(interactive=True))
            else:
                yield (log, *_noop(2), "⚠️ Run did not produce an output — "
                       "check the log above.", *gate(st),
                       gr.update(interactive=True))

        dub_btn.click(
            do_dub, inputs=[episode],
            outputs=[dub_log, out_video, download_btn, dub_status, *gated_tabs,
                     dub_btn],
            **HEAVY,
        )

        def do_drive_copy(video: str):
            st = pipeline_status(video)
            if not st.has_output:
                return "⚠️ No dubbed output yet."
            dest = Path("/content/drive/MyDrive/dubbing")
            if not dest.exists():
                return "⚠️ Drive not mounted at /content/drive."
            shutil.copy(st.output_path, dest)
            return f"✅ Copied to Drive: dubbing/{Path(st.output_path).name}"

        drive_btn.click(do_drive_copy, inputs=[episode], outputs=[dub_status])

    demo.queue(max_size=16)
    return demo
