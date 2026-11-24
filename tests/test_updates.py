import argparse
import io
import json
import sys
import tomllib
from pathlib import Path

from midas import __version__, cli
from midas_mcp import updates


def test_package_and_runtime_versions_match():
    metadata = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert tomllib.loads(metadata.read_text())["project"]["version"] == __version__


def test_latest_release_ignores_prereleases_and_yanked_releases(monkeypatch):
    def open_pypi(url, timeout):
        assert url == "https://pypi.org/pypi/midas-memory/json"
        assert timeout == 5
        return io.BytesIO(json.dumps({"releases": {
            "1.9.0": [{"yanked": False}],
            "1.10.0": [{"yanked": False}],
            "2.0.0rc1": [{"yanked": False}],
            "3.0.0": [{"yanked": True}],
            "4.0.0": [],
        }}).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_pypi)
    assert updates._latest_release() == "1.10.0"


def test_automatic_check_uses_one_cache_for_all_clients(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("MIDAS_UPDATE_CHECK", "1")
    calls = []

    def latest():
        calls.append("PyPI")
        return "9.0.0"

    monkeypatch.setattr(updates, "_latest_release", latest)
    assert updates.check_updates(cached_only=True)["status"] == "unknown"
    assert not calls
    assert updates.check_updates()["status"] == "available"
    assert updates.check_updates(cached_only=True)["latest"] == "9.0.0"
    assert updates.check_updates()["latest"] == "9.0.0"
    assert calls == ["PyPI"]


def test_offline_check_is_throttled_and_last_known_notice_survives(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("MIDAS_UPDATE_CHECK", "1")
    cache = tmp_path / ".midas" / "update-check.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"checked": 1, "latest": "9.0.0"}))
    calls = []

    def offline():
        calls.append("PyPI")
        raise OSError("offline")

    monkeypatch.setattr(updates, "_latest_release", offline)
    assert updates.check_updates()["status"] == "available"
    assert updates.check_updates()["status"] == "available"
    assert calls == ["PyPI"]
    assert updates.check_updates(force=True)["status"] == "unknown"
    assert updates.check_updates(cached_only=True)["latest"] == "9.0.0"


def test_update_check_is_read_only_and_can_be_requested_when_disabled(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("MIDAS_UPDATE_CHECK", "0")
    monkeypatch.setattr(updates, "_latest_release", lambda: "9.0.0")

    def unexpected_install(*args, **kwargs):
        raise AssertionError("a version check must not run an installer")

    monkeypatch.setattr(cli.subprocess, "run", unexpected_install)
    assert updates.check_updates()["status"] == "disabled"
    assert cli.cmd_update(argparse.Namespace(check=True)) == 0
    assert "Midas 9.0.0 is available" in capsys.readouterr().out


def test_update_uses_the_current_environment_installer(monkeypatch, tmp_path):
    prefix = tmp_path / "midas-memory"
    prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert cli._update_command() == [
        sys.executable, "-m", "pip", "install", "--upgrade", "midas-memory",
    ]

    (prefix / "pipx_metadata.json").touch()
    assert cli._update_command() == ["pipx", "upgrade", "midas-memory"]
    (prefix / "uv-receipt.toml").touch()
    assert cli._update_command() == ["uv", "tool", "upgrade", "midas-memory"]
    assert cli._update_command(pip=True)[:3] == [sys.executable, "-m", "pip"]
