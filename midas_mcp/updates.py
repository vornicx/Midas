"""Cached, read-only checks for public Midas releases."""

import json
import os
import re
import time
from pathlib import Path

from ._version import VERSION


_PYPI_URL = "https://pypi.org/pypi/midas-memory/json"
_CHECK_INTERVAL = 24 * 60 * 60
_MAX_RESPONSE_BYTES = 512 * 1024


def _version(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([0-9]{1,8})\.([0-9]{1,8})\.([0-9]{1,8})", value)
    if match:
        major, minor, patch = match.groups()
        return int(major), int(minor), int(patch)
    return None


def _latest_release() -> str | None:
    from urllib.request import urlopen

    with urlopen(_PYPI_URL, timeout=5) as response:
        body = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(body) > _MAX_RESPONSE_BYTES:
        return None
    data = json.loads(body)
    releases = data.get("releases")
    if not isinstance(releases, dict):
        releases = {data["info"]["version"]: data.get("urls", [])}
    stable = [(parsed, version) for version, files in releases.items()
              if (parsed := _version(version)) is not None and isinstance(files, list)
              and any(isinstance(file, dict) and not file.get("yanked") for file in files)]
    return max(stable)[1] if stable else None


def _read_cache(path: Path) -> dict:
    try:
        with path.open("rb") as cache:
            value = json.loads(cache.read(4096))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _fresh(entry: dict) -> bool:
    checked = entry.get("checked")
    now = time.time()
    return isinstance(checked, (int, float)) and now - _CHECK_INTERVAL < checked <= now


def _result(entry: dict) -> dict:
    latest = entry.get("latest")
    parsed = _version(latest)
    if parsed is None:
        return {"status": "unknown", "installed": VERSION, "latest": None}
    status = "available" if parsed > _version(VERSION) else "current"
    return {"status": status, "installed": VERSION, "latest": latest}


def check_updates(*, force: bool = False, cached_only: bool = False) -> dict:
    """Look up stable releases at most daily, across all clients using this account."""
    if not force and os.getenv("MIDAS_UPDATE_CHECK", "1") == "0":
        return {"status": "disabled", "installed": VERSION, "latest": None}

    from . import try_lock

    path = Path.home() / ".midas" / "update-check.json"
    entry = _read_cache(path)
    result = _result(entry)
    if cached_only or (not force and _fresh(entry)):
        return result

    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with try_lock(path.with_suffix(".lock")) as acquired:
            if not acquired:
                return _result(_read_cache(path))
            entry = _read_cache(path)
            if not force and _fresh(entry):
                return _result(entry)
            try:
                latest = _latest_release()
            except (OSError, ValueError, AttributeError, KeyError, TypeError):
                latest = None
            entry = {"checked": time.time(), "latest": latest or entry.get("latest")}
            result = _result({"latest": latest} if force else entry)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            temporary.replace(path)
    except OSError:
        # Read-only homes and offline environments still run Midas normally.
        pass
    return result


def update_notice(info: dict) -> str | None:
    if info["status"] == "available":
        return f"Midas {info['latest']} is available. Run `midas update` to install it."
    return None
