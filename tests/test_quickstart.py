"""The repo's quickstart.py must actually run — guards against a broken first impression (it's the first
thing a new user executes). Runs it as a real subprocess and checks the end-to-end output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_quickstart_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(root / "quickstart.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "assembled context" in out
    assert "PostgreSQL" in out                 # recall surfaced the real match
    assert "October 1" in out                  # supersession: the live state is the revised value
    assert "guard" in out                      # the governance demo ran
