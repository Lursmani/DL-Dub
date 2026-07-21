import yaml

from pipeline.config import Config


def test_voice_for_precedence_episode_then_config_then_default():
    cfg = Config(
        default_voice_id="default-v", default_tts_model="default-m",
        voices={"SPEAKER_00": {"voice_id": "cfg-v", "model_id": "cfg-m"}},
    )
    episode = {"SPEAKER_00": {"voice_id": "ep-v", "model_id": "ep-m"}}

    assert cfg.voice_for("SPEAKER_00", episode) == ("ep-v", "ep-m")
    assert cfg.voice_for("SPEAKER_00") == ("cfg-v", "cfg-m")
    # unknown speaker and the None-fallback both hit the defaults
    assert cfg.voice_for("SPEAKER_01", episode) == ("default-v", "default-m")
    assert cfg.voice_for(None, episode) == ("default-v", "default-m")


def test_voice_for_partial_entry_falls_back_per_field():
    cfg = Config(default_voice_id="default-v", default_tts_model="default-m")
    episode = {"SPEAKER_00": {"voice_id": "ep-v"}}  # no model_id
    assert cfg.voice_for("SPEAKER_00", episode) == ("ep-v", "default-m")


def test_load_ignores_unknown_keys(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("target_lang: ka\nwhisper_modle: large-v3\n",
                    encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.target_lang == "ka"
    assert cfg.whisper_model == "large-v3"  # dataclass default kept
    assert "whisper_modle" in capsys.readouterr().out  # typo warned about


def test_update_yaml_merges_without_touching_other_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("source_lang: nl\ntarget_lang: ka\n", encoding="utf-8")
    Config.update_yaml(path, {"default_voice_id": "v-123"})
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw == {"source_lang": "nl", "target_lang": "ka",
                   "default_voice_id": "v-123"}
