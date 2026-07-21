import sys

import pytest

from pipeline.util import Manifest, PipelineError, run, workdir_for


def test_workdir_for_uses_stem_and_creates_clips(tmp_path):
    wd = workdir_for(tmp_path / "episode.mp4", root=tmp_path / "work")
    assert wd == tmp_path / "work" / "episode"
    assert (wd / "clips").is_dir()


def test_manifest_init_records_video(tmp_path):
    m = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    assert m.data["video"] == str(tmp_path / "ep.mp4")
    assert m.segments == []
    assert m.voices == {}


def test_manifest_load_self_heals_stale_video_path(tmp_path):
    # A manifest written on another machine (e.g. Colab) holds a stale
    # absolute video path; loading with the local path must win.
    m = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    m.data["video"] = "/content/drive/MyDrive/dubbing/ep.mp4"
    m.save()

    local = tmp_path / "local" / "ep.mp4"
    m2 = Manifest.load_or_init(tmp_path, local)
    assert m2.data["video"] == str(local)


def test_manifest_voices_roundtrip(tmp_path):
    m = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    m.data["voices"] = {"SPEAKER_00": {"voice_id": "v1", "model_id": "m1"}}
    m.save()
    m2 = Manifest.load_or_init(tmp_path, tmp_path / "ep.mp4")
    assert m2.voices["SPEAKER_00"]["voice_id"] == "v1"


def test_run_wraps_failure_in_pipeline_error():
    with pytest.raises(PipelineError, match="exit 3"):
        run([sys.executable, "-c", "import sys; sys.exit(3)"],
            capture_output=True)
