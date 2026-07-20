"""Stage 4 — translate nl -> ka in one batched, context-aware Claude call.

Batching the whole episode in a single request gives the model full context
(better pronoun/register choices) and we push it to keep each line within a
character budget so the dubbed audio fits the original timing slot.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .util import Manifest

_SYSTEM = (
    "You are a professional dubbing translator. Translate each numbered line "
    "from {src} to {tgt} for a children's cartoon dub. Keep it natural, warm, "
    "and age-appropriate. CRITICAL: each line is spoken aloud in a fixed time "
    "slot, so stay at or under the given character budget — prefer shorter, "
    "punchier phrasing over literal completeness. Preserve names. "
    "Return ONLY a JSON object mapping the line id (as a string) to its "
    "translation, e.g. {{\"0\": \"...\", \"1\": \"...\"}}."
)


def translate(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    import anthropic

    if not cfg.anthropic_key:
        raise SystemExit("[translate] ANTHROPIC_API_KEY required (see .env.example).")

    todo = [s for s in manifest.segments if not s.get("text_tgt")]
    if not todo:
        print("[translate] nothing to do (all lines already translated).")
        return

    lines = []
    for s in todo:
        budget = max(12, int((s["end"] - s["start"]) * cfg.chars_per_second))
        lines.append(f'{s["id"]} (<= {budget} chars): {s["text_src"]}')
    user = "\n".join(lines)

    client = anthropic.Anthropic(api_key=cfg.anthropic_key)
    msg = client.messages.create(
        model=cfg.translate_model,
        max_tokens=8000,
        system=_SYSTEM.format(src=cfg.source_lang, tgt=cfg.target_lang),
        messages=[{"role": "user", "content": user}],
    )
    if msg.stop_reason == "max_tokens":
        raise SystemExit(
            "[translate] response truncated at max_tokens — too many lines for one "
            "batch. Raise max_tokens in pipeline/translate.py or split the episode."
        )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    # Be tolerant of stray prose around the JSON.
    text = text[text.find("{"): text.rfind("}") + 1]
    mapping = json.loads(text)

    by_id = {s["id"]: s for s in manifest.segments}
    for k, v in mapping.items():
        seg = by_id.get(int(k))
        if seg is not None:
            seg["text_tgt"] = v.strip()
    manifest.save()  # save first: a re-run only re-requests the still-missing lines

    missing = [s["id"] for s in todo if not s.get("text_tgt")]
    if missing:
        raise SystemExit(
            f"[translate] model response was missing line ids {missing} — "
            "translated lines were saved; re-run the translate stage to fill the rest."
        )
    print(f"[translate] translated {len(mapping)} lines -> {cfg.target_lang}")
