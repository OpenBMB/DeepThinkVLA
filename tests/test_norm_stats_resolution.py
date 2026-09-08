import json
import importlib.util
from pathlib import Path

import pytest


_NORMALIZE_PATH = Path(__file__).parents[1] / "src" / "dt_datasets" / "normalize.py"
_NORMALIZE_SPEC = importlib.util.spec_from_file_location("deepthinkvla_normalize", _NORMALIZE_PATH)
normalize = importlib.util.module_from_spec(_NORMALIZE_SPEC)
assert _NORMALIZE_SPEC.loader is not None
_NORMALIZE_SPEC.loader.exec_module(normalize)


def test_resolve_norm_stats_path_prefers_local_checkpoint(tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps({"action": {}}), encoding="utf-8")

    assert normalize.resolve_norm_stats_path(tmp_path) == stats_path


def test_resolve_norm_stats_path_downloads_hub_repository(monkeypatch, tmp_path):
    downloaded_path = tmp_path / "cached" / "norm_stats.json"
    downloaded_path.parent.mkdir()
    downloaded_path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_hf_hub_download(*, repo_id, filename):
        calls.append((repo_id, filename))
        return str(downloaded_path)

    monkeypatch.setattr(normalize, "hf_hub_download", fake_hf_hub_download)

    resolved = normalize.resolve_norm_stats_path("org/model")

    assert resolved == downloaded_path
    assert calls == [("org/model", "norm_stats.json")]


def test_resolve_norm_stats_path_reports_missing_local_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        normalize.resolve_norm_stats_path(tmp_path / "missing-checkpoint")


def test_missing_absolute_path_does_not_trigger_hub_lookup(monkeypatch, tmp_path):
    def fail_hf_hub_download(**kwargs):
        raise AssertionError(f"unexpected Hub lookup: {kwargs}")

    monkeypatch.setattr(normalize, "hf_hub_download", fail_hf_hub_download)

    with pytest.raises(FileNotFoundError, match="local checkpoint"):
        normalize.resolve_norm_stats_path(tmp_path / "missing" / "checkpoint")
