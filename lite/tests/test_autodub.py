from pathlib import Path

from pipeline import autodub as ad


def test_output_path_video_input_gets_mp4():
    out = ad.output_path(Path("input/episode.mp4"), Path("work/episode"), "ka")
    assert out == Path("work/episode/episode.ka.auto.mp4")


def test_output_path_audio_input_gets_mp3():
    # The API returns mp3 for audio input regardless of the source format.
    for suffix in (".mp3", ".WAV", ".flac"):
        out = ad.output_path(Path(f"song{suffix}"), Path("work/song"), "ka")
        assert out.suffix == ".mp3", suffix


def test_estimate_prices_by_watermark(monkeypatch):
    monkeypatch.setattr(ad, "ffprobe_duration", lambda _: 360.0)  # 6 minutes
    assert ad.estimate(Path("ep.mp4"), watermark=True) == {
        "minutes": 6.0, "usd": 1.98}
    assert ad.estimate(Path("ep.mp4"), watermark=False) == {
        "minutes": 6.0, "usd": 3.0}
