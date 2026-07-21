# dub-pipeline — full

Automated dubbing for a Dutch cartoon → Georgian, built to be **cheap**: it uses
the managed tools only where they're irreplaceable (ElevenLabs for Georgian voice)
and free/local tools for everything else. Per-episode cost is roughly the
characters of dialogue you synthesize — on the order of **$0.30–0.50 for ~6 minutes**,
versus ~€22 for ElevenLabs' automatic Dubbing Studio.

> Want the opposite trade-off — one API call, cloned original voices, zero
> setup, ~$0.33–0.50 **per minute**? That's the separate [lite project](../lite).

## What it does

```
video ──▶ extract ──▶ separate ──▶ transcribe ──▶ translate ──▶ tts ──▶ assemble ──▶ mux ──▶ dubbed video
 (mp4)    (ffmpeg)    (Demucs)     (WhisperX +     (LLM API      (Eleven   (fit + mix    (ffmpeg)
                     dialogue vs   diarization)    nl→ka,        Labs,     over music
                     music/SFX bed  per-speaker    length-       per char) bed)
                                    labels         budgeted)
```

Only the **tts** stage spends money (per character). Everything else is free/local.

## Project layout

```
input/     drop source videos here (e.g. input/episode.mp4)
output/    a place to collect finished dubs
work/      per-episode intermediates + the canonical result:
             work/<episode>/<episode>.ka.mp4
pipeline/  the pipeline stages (extract … mux)
gui/       Gradio web app  (python -m gui)
```

`input/` and `output/` are tracked empty for structure; their contents are
git-ignored so large media is never committed. The pipeline itself still reads
the video path you pass it and writes results under `work/` — these folders are
a convention for keeping sources and finished files tidy.

## The GUI (recommended)

`gui/` is a Gradio web app that wraps the whole workflow as a 5-step wizard:
**Setup → Analyze → Voices → Translate & review → Dub**. It runs the pipeline
in-process and keeps `work/<episode>/manifest.json` as the single source of
truth, so the GUI and CLI are fully interchangeable — you can start in one and
finish in the other.

- **In Colab (primary)**: the notebook's "Launch the GUI" cell prints a public
  `*.gradio.live` link — open it in a new browser tab. If the link dies mid-run,
  re-run the cell and click *Refresh from disk*; all progress is on disk.
- **Locally**: `python -m gui` (port 7861; add `--share` for a public link —
  note anyone with that link can run stages and spend your API credits). Fine
  for the Voices/Translate/Dub steps (they're API calls); Analyze runs on CPU
  locally.

Highlights: per-speaker audio samples so you can hear who's who before picking
voices; your ElevenLabs voice library in a dropdown (with free demos and a
cached paid Georgian preview per voice); an editable translation table with
per-line character budgets; and a cost estimate that must be on screen before
the spend button.

The GUI adapts to the install. With the ML extras present (**full mode**) the
whole wizard runs locally. Without them (**hybrid mode**, no
`requirements-ml.txt`) the Analyze tab becomes run-it-in-Colab guidance, and
the other tabs work as usual for episodes analyzed there — copy the episode's
`work/` folder over and hit *Refresh from disk*. For one-call API dubbing
with cloned voices, use the separate [lite project](../lite) instead.

## Cloud GPU (recommended) — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lursmani/DL-Dub/blob/main/full/colab_dub.ipynb)

The GPU stages (Demucs, WhisperX) assume CUDA, which AMD cards can't provide on
Windows, so the simplest path is a free Colab T4. Click the badge to open
**`colab_dub.ipynb`** and run it top to bottom:

1. **Runtime → Change runtime type → T4 GPU**.
2. Add `ELEVENLABS_API_KEY`, `HF_TOKEN`, and one translation key (`ANTHROPIC_API_KEY`
   / `OPENAI_API_KEY` / `GOOGLE_API_KEY`) in Colab's 🔑 Secrets panel.
3. Run the clone + install cells. After install finishes, **restart the runtime once**
   (pip upgrades numpy/torch), then continue.
4. Drop your episode in `MyDrive/dubbing/`; the dubbed `*.ka.mp4` is written back there.

A 6-minute episode processes in ~2–4 min of GPU time. The rest of this README covers
running the same pipeline **locally** (CUDA or CPU).

## Prerequisites (local runs)

1. **Python 3.10+**
2. **ffmpeg + ffprobe** on PATH — `winget install Gyan.FFmpeg` (Windows) / `brew install ffmpeg`.
3. **A GPU is strongly recommended** for WhisperX + Demucs. CPU works but is slow;
   the pipeline auto-detects CUDA and falls back to CPU (`int8`) otherwise.
4. Python deps — split so the heavy ML stack is **optional**:
   - `pip install -r requirements.txt` — the core install (a few hundred MB).
     Runs the GUI and every stage except `separate`/`transcribe`: perfect if you run
     Analyze in Colab and only do Voices/Translate/Dub locally.
   - `pip install -r requirements-ml.txt` — adds Demucs + WhisperX for local Analyze
     (pulls in `torch`, several GB). For CUDA, install the matching torch build first
     (see https://pytorch.org). CPU-only torch installs by default.

## Setup

```bash
cp .env.example .env            # fill in the 3 keys
cp config.example.yaml config.yaml
```

- **ELEVENLABS_API_KEY** — from the ElevenLabs dashboard.
- One translation key: **ANTHROPIC_API_KEY**, **OPENAI_API_KEY**, or
  **GOOGLE_API_KEY** — pick the provider in the GUI's Translate tab
  (or `translate_provider` in config.yaml).
- **HF_TOKEN** — a HuggingFace token, AND you must accept the license once (logged in) at
  [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and
  [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0), or diarization fails.

## Docker

```bash
cp .env.example .env                  # fill in the API keys
cp config.example.yaml config.yaml

docker compose up gui                 # full image (~5 GB): the whole pipeline
```

The GUI is at http://localhost:7861 (the full project uses port 7861; lite
uses 7860, so both GUIs can run alongside each other). `work/`, `input/`, `output/`, and `config.yaml`
are mounted from the host, so container and native runs are fully
interchangeable — start an episode in Colab, finish it in the container. The
image also mounts a `model-cache` volume so the ~4 GB of Whisper/pyannote/
Demucs weights download only once. For the hybrid workflow there's a no-ML
image too (~1.5 GB): `docker build --target core -t dub-full:core .`

CLI instead of the GUI:

```bash
docker compose run --rm gui python dub.py estimate input/episode.mp4
```

GPU note: the `full` image installs **CPU torch** by default and runs anywhere
Docker runs. On a host with an NVIDIA GPU, build with
`--build-arg TORCH_CHANNEL=cu126` and run with `--gpus all` (requires the
NVIDIA Container Toolkit; on Windows that means the WSL2 backend). AMD GPUs
can't accelerate these stages in Docker — use Colab for the Analyze step
instead.

## Usage — the two-pass workflow

Because voices are assigned per detected speaker, you run in two passes:

**Pass 1 — get the speaker labels:**
```bash
python dub.py run episode.mp4 --stop-after transcribe
```
Open `work/episode/manifest.json`, read the lines, and note which `SPEAKER_xx`
is which character. Fill those into `config.yaml` under `voices:` with a
`voice_id` per character (grab IDs from your ElevenLabs voice library or clone
your Georgian voice actors — cloned voices cost nothing to reuse). The GUI's
Voices tab does the same thing per episode — it saves into the episode's
manifest, since speaker labels mean a different character in every episode;
`config.yaml`'s `voices:` acts as a global fallback.

**Pass 2 — translate, preview cost, dub, splice:**
```bash
python dub.py run episode.mp4 --start-at translate
```
Before synthesizing, it prints a character/credit/$ estimate and asks to confirm.
The finished file lands at `work/episode/episode.ka.mp4`.

Preview cost anytime without spending:
```bash
python dub.py estimate episode.mp4
```

## Cost levers (already wired in)

- **Per-character TTS**, not per-video-minute — the core saving.
- **Flash model default** (`eleven_flash_v2_5`, 0.5 credits/char); give only lead
  characters the pricier `eleven_multilingual_v2` in `config.yaml`.
- **Length-budgeted translation** — Claude is told each line's character budget so
  Georgian stays short enough to fit the slot (fewer characters = lower cost + better timing).
- **Clip caching** — re-runs never re-bill; editing one translation re-synthesizes only that line.

## Tuning notes

- **Songs** are Demucs's weak spot — musical numbers will have artifacts in the bed and
  odd separation. Consider leaving sung segments in the original language.
- **Timing fit**: `max_speedup` / `min_slowdown` in config bound how much a clip is
  stretched to fit its slot before it's allowed to overflow (avoids chipmunk voices).
- **Lip-sync**: not attempted — cartoon mouth flaps only need duration matching, which
  the assemble stage does.
- Library versions (WhisperX, ElevenLabs SDK) move fast; if an import/API call breaks,
  check the installed version against the calls in `pipeline/`.
