"""Stage 4 — translate nl -> ka in one batched, context-aware LLM call.

Batching the whole episode in a single request gives the model full context
(better pronoun/register choices) and we push it to keep each line within a
character budget so the dubbed audio fits the original timing slot.

Supports multiple providers (anthropic / openai / google) — pick one via
`translate_provider` in config.yaml or the GUI's Translate tab; the matching
API key comes from .env. Only the chosen provider's SDK is imported.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .util import Manifest, PipelineError

_SYSTEM = (
    "You are a professional dubbing translator. Translate each numbered line "
    "from {src} to {tgt} for a children's cartoon dub. Keep it natural, warm, "
    "and age-appropriate. CRITICAL: each line is spoken aloud in a fixed time "
    "slot, so stay at or under the given character budget — prefer shorter, "
    "punchier phrasing over literal completeness. Preserve names. "
    "Return ONLY a JSON object mapping the line id (as a string) to its "
    "translation, e.g. {{\"0\": \"...\", \"1\": \"...\"}}."
)

_MAX_TOKENS = 8000

PROVIDERS: dict[str, dict[str, str]] = {
    "anthropic": {"label": "Anthropic (Claude)", "env": "ANTHROPIC_API_KEY",
                  "default_model": "claude-sonnet-5"},
    "openai": {"label": "OpenAI (GPT)", "env": "OPENAI_API_KEY",
               "default_model": "gpt-5-mini"},
    "google": {"label": "Google (Gemini)", "env": "GOOGLE_API_KEY",
               "default_model": "gemini-2.5-flash"},
}

# Model-name prefixes that imply a provider — used to catch a stale
# translate_model left over from a provider switch.
_PREFIX_PROVIDER = {"claude": "anthropic", "gpt": "openai", "o1": "openai",
                    "o3": "openai", "o4": "openai", "gemini": "google"}


def available_providers(cfg: Config) -> dict[str, bool]:
    """{provider: key-is-set} — what the GUI shows as available."""
    return {p: bool(cfg.translate_key(p)) for p in PROVIDERS}


def effective_model(provider: str, model: str) -> str:
    """The model that would actually run for this provider.

    A model name whose prefix implies a DIFFERENT provider (e.g. a stale
    'claude-…' left in config after switching to openai) falls back to the
    provider's default. Used by both the pipeline and the GUI display.
    """
    model = (model or "").strip()
    if model:
        implied = next((p for pref, p in _PREFIX_PROVIDER.items()
                        if model.lower().startswith(pref)), None)
        if implied and implied != provider:
            model = ""
    return model or PROVIDERS[provider]["default_model"]


def resolve_translation(cfg: Config) -> tuple[str, str, str]:
    """(provider, model, api_key) for the configured provider, validated."""
    provider = (cfg.translate_provider or "anthropic").strip().lower()
    if provider not in PROVIDERS:
        raise PipelineError(
            f"[translate] unknown translate_provider '{provider}' — "
            f"choose one of {sorted(PROVIDERS)}.")
    key = cfg.translate_key(provider)
    if not key:
        raise PipelineError(
            f"[translate] {PROVIDERS[provider]['env']} required for provider "
            f"'{provider}' (see .env.example).")
    model = effective_model(provider, cfg.translate_model)
    if (cfg.translate_model or "").strip() not in ("", model):
        print(f"[translate] model '{cfg.translate_model}' doesn't match provider "
              f"'{provider}' — using its default '{model}' instead")
    return provider, model, key


def _call_anthropic(system: str, user: str, model: str, key: str) -> tuple[str, bool]:
    import anthropic

    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=_MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.stop_reason == "max_tokens"


def _call_openai(system: str, user: str, model: str, key: str) -> tuple[str, bool]:
    import openai

    resp = openai.OpenAI(api_key=key).chat.completions.create(
        model=model, max_completion_tokens=_MAX_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    choice = resp.choices[0]
    return choice.message.content or "", choice.finish_reason == "length"


def _call_google(system: str, user: str, model: str, key: str) -> tuple[str, bool]:
    from google import genai
    from google.genai import types

    resp = genai.Client(api_key=key).models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=_MAX_TOKENS),
    )
    finish = str(getattr(resp.candidates[0], "finish_reason", "")) \
        if getattr(resp, "candidates", None) else ""
    return resp.text or "", "MAX_TOKENS" in finish.upper()


_CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai,
            "google": _call_google}


def char_budget(seg: dict, cfg: Config) -> int:
    """Character budget for a line so its dubbed audio fits the time slot.

    Single source of truth: used both in the translation prompt and by the
    GUI's review table.
    """
    return max(12, int((seg["end"] - seg["start"]) * cfg.chars_per_second))


def translate(video: Path, workdir: Path, manifest: Manifest, cfg: Config) -> None:
    provider, model, key = resolve_translation(cfg)

    todo = [s for s in manifest.segments if not s.get("text_tgt")]
    if not todo:
        print("[translate] nothing to do (all lines already translated).")
        return

    lines = []
    for s in todo:
        lines.append(f'{s["id"]} (<= {char_budget(s, cfg)} chars): {s["text_src"]}')
    user = "\n".join(lines)

    print(f"[translate] provider={provider} model={model} ({len(todo)} lines)")
    system = _SYSTEM.format(src=cfg.source_lang, tgt=cfg.target_lang)
    text, truncated = _CALLERS[provider](system, user, model, key)
    if truncated:
        raise PipelineError(
            "[translate] response truncated at the token limit — too many lines "
            "for one batch. Raise _MAX_TOKENS in pipeline/translate.py or split "
            "the episode."
        )
    text = text.strip()
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
        raise PipelineError(
            f"[translate] model response was missing line ids {missing} — "
            "translated lines were saved; re-run the translate stage to fill the rest."
        )
    print(f"[translate] translated {len(mapping)} lines -> {cfg.target_lang}")
