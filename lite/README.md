# dub-pipeline — lite

One-call dubbing via ElevenLabs' managed **Dubbing API**: upload the video,
the API transcribes, translates, **clones the original voices** and preserves
their intonation, then you download the finished dub. No local ML, no speaker
mapping, nothing to tune. The trade-off is price: billed **per minute of
source audio** (~$0.33/min watermarked, ~$0.50/min clean) instead of the
pennies-per-episode of the [full pipeline](../full).

## Prerequisites

1. **Python 3.10+**
2. **ffmpeg + ffprobe** on PATH (only used for the pre-spend duration
   estimate) — `winget install Gyan.FFmpeg` (Windows) / `brew install ffmpeg`.
3. `pip install -r requirements.txt` — four small packages.

## Setup

```bash
cp .env.example .env            # fill in ELEVENLABS_API_KEY
cp config.example.yaml config.yaml
```

## Usage

CLI:

```bash
python autodub.py input/episode.mp4                # confirm the estimate, dub
python autodub.py input/episode.mp4 --no-watermark # clean output, $0.50/min
python autodub.py input/episode.mp4 --yes          # skip the confirmation
```

GUI (same thing in the browser, at http://localhost:7860):

```bash
python -m gui
```

The result lands at `work/<episode>/<episode>.<lang>.auto.mp4` and is
**cached** — re-running the same episode returns it without re-billing;
delete the file to re-dub. Audio-only inputs (mp3/wav/…) come back as mp3.

## Docker

```bash
cp .env.example .env
cp config.example.yaml config.yaml
docker compose up gui           # GUI at http://localhost:7860
```

## Project layout

```
autodub.py   CLI entry point
pipeline/    the API client (autodub.py) + config/util helpers
gui/         Gradio web app  (python -m gui)
input/       drop source videos here
output/      a place to collect finished dubs
work/        per-episode results (created on first run)
```

Want control over voices, translations, and cost — and a much cheaper
per-episode price? Use the [full pipeline](../full).
