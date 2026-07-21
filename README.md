# DL-Dub

Automated dubbing (built for a Dutch cartoon → Georgian), as **two independent
projects** — pick the trade-off you want:

| | [`lite/`](lite) ⚡ | [`full/`](full) 🎛️ |
|---|---|---|
| How | One call to ElevenLabs' managed Dubbing API | Local 7-stage pipeline: extract → separate → transcribe → translate → tts → assemble → mux |
| Voices | Clones the original voices automatically | You pick an ElevenLabs voice per character |
| Control | None — upload, wait, download | Full: edit translations, budget line lengths, tune the mix |
| Cost | ~$0.33–0.50 **per minute** of source | Per character synthesized — roughly **$0.30–0.50 per 6-min episode** |
| Hardware | None (pure API) | GPU recommended for analysis (or free Colab — notebook included) |
| Install | 4 small packages | Core packages + optional several-GB ML extras |

Each project is fully self-contained: its own `requirements.txt`, config,
`.env`, GUI, CLI, Dockerfile and compose file. Run everything from inside the
project's directory. The GUIs use different ports (lite 7860, full 7861), so
both can run at once.

- **[lite/README.md](lite/README.md)** — setup and usage for the API-only dubber.
- **[full/README.md](full/README.md)** — setup, the two-pass workflow, Colab, cost levers.
