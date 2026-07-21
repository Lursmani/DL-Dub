import pytest

from pipeline import extract as ex
from pipeline.config import Config
from pipeline.util import Manifest, PipelineError


def _no_tools(monkeypatch, duration):
    monkeypatch.setattr(ex, "check_tool", lambda *a, **k: None)
    monkeypatch.setattr(ex, "ffprobe_duration", lambda _: duration)


def test_extract_rejects_workdir_of_a_different_video(tmp_path, monkeypatch):
    # Workdirs are keyed by filename stem — a different video with the same
    # name must not silently reuse the cached analysis.
    _no_tools(monkeypatch, duration=412.0)
    (tmp_path / "audio.wav").write_bytes(b"riff")
    manifest = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    manifest.data["duration"] = 360.0

    with pytest.raises(PipelineError, match="different video"):
        ex.extract(tmp_path / "ep.mp4", tmp_path, manifest, Config())


def test_extract_accepts_matching_cached_audio(tmp_path, monkeypatch):
    # Same duration (e.g. the same file moved Colab -> local) is fine.
    _no_tools(monkeypatch, duration=360.2)
    (tmp_path / "audio.wav").write_bytes(b"riff")
    manifest = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    manifest.data["duration"] = 360.0

    ex.extract(tmp_path / "ep.mp4", tmp_path, manifest, Config())
    assert manifest.data["duration"] == 360.2
