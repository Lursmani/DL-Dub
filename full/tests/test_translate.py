import pytest

from pipeline import translate as tr
from pipeline.config import Config
from pipeline.util import Manifest, PipelineError


def test_effective_model_keeps_matching_model():
    assert tr.effective_model("anthropic", "claude-opus-4-8") == "claude-opus-4-8"


def test_effective_model_drops_stale_model_from_other_provider():
    # a 'claude-…' model left in config after switching to openai
    assert tr.effective_model("openai", "claude-sonnet-5") == \
        tr.PROVIDERS["openai"]["default_model"]


def test_effective_model_blank_gives_provider_default():
    for provider, info in tr.PROVIDERS.items():
        assert tr.effective_model(provider, "") == info["default_model"]
        assert tr.effective_model(provider, "  ") == info["default_model"]


def test_char_budget_scales_with_slot_and_has_floor():
    cfg = Config(chars_per_second=15.0)
    assert tr.char_budget({"start": 0.0, "end": 2.0}, cfg) == 30
    assert tr.char_budget({"start": 0.0, "end": 0.1}, cfg) == 12  # floor


def test_resolve_translation_rejects_unknown_provider():
    cfg = Config(translate_provider="mistral")
    with pytest.raises(PipelineError, match="unknown translate_provider"):
        tr.resolve_translation(cfg)


def test_resolve_translation_requires_key():
    cfg = Config(translate_provider="anthropic", anthropic_key="")
    with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY"):
        tr.resolve_translation(cfg)


# --- response parsing (the model is untrusted input) -----------------------


def _manifest(tmp_path, n=2):
    m = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    m.segments = [{"id": i, "start": float(i), "end": float(i) + 2.0,
                   "speaker": "SPEAKER_00", "text_src": f"regel {i}"}
                  for i in range(n)]
    m.save()
    return m


def _translate_with_response(tmp_path, monkeypatch, response, truncated=False):
    cfg = Config(translate_provider="anthropic", anthropic_key="k")
    manifest = _manifest(tmp_path)
    monkeypatch.setitem(tr._CALLERS, "anthropic",
                        lambda system, user, model, key: (response, truncated))
    tr.translate(tmp_path / "ep.mp4", tmp_path, manifest, cfg)
    return manifest


def test_translate_applies_mapping_and_tolerates_prose(tmp_path, monkeypatch):
    m = _translate_with_response(
        tmp_path, monkeypatch, 'Here you go: {"0": "კარგი", "1": "დიახ"}')
    assert [s["text_tgt"] for s in m.segments] == ["კარგი", "დიახ"]


def test_translate_skips_junk_keys_and_values(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="missing line ids"):
        _translate_with_response(
            tmp_path, monkeypatch,
            '{"0": "კარგი", "notanid": "x", "1": ["not", "a", "string"]}')
    # the good line was saved before the error (partial progress persists)
    m = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    assert m.segments[0]["text_tgt"] == "კარგი"
    assert "text_tgt" not in m.segments[1]


def test_translate_malformed_json_is_pipeline_error(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="malformed JSON"):
        _translate_with_response(tmp_path, monkeypatch, '{"0": "კარგი", }')


def test_translate_no_json_is_pipeline_error(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="no JSON object"):
        _translate_with_response(tmp_path, monkeypatch, "Sorry, I cannot help.")


def test_translate_truncated_response_is_pipeline_error(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="truncated"):
        _translate_with_response(tmp_path, monkeypatch, '{"0": "კარგ',
                                 truncated=True)
