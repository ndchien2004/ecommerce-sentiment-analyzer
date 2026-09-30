import zipfile

import pytest

from src.inference import ensure_distilbert_weights


def _make_archive(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("distilbert/config.json", "{}")
        zf.writestr("distilbert/model.safetensors", "weights")
    return path


def test_downloads_and_extracts_missing_weights(tmp_path):
    archive = _make_archive(tmp_path / "distilbert.zip")
    model_dir = tmp_path / "models" / "distilbert"

    ensure_distilbert_weights(model_dir, url=archive.as_uri())

    assert (model_dir / "model.safetensors").read_text() == "weights"
    assert (model_dir / "config.json").exists()


def test_skips_download_when_weights_exist(tmp_path):
    model_dir = tmp_path / "distilbert"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("local")

    ensure_distilbert_weights(model_dir, url="https://invalid.example/never-called.zip")

    assert (model_dir / "model.safetensors").read_text() == "local"


def test_download_failure_is_reported_as_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not download"):
        ensure_distilbert_weights(tmp_path / "distilbert", url=(tmp_path / "nope.zip").as_uri())
