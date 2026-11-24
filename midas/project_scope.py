"""Local repository identity shared across linked Git worktrees."""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectScope:
    namespace: str
    name: str
    legacy_namespace: str


def git_output(directory: str | Path, *args: str) -> str:
    # An MCP worker inherits the first launcher's environment. Its Git overrides must not select
    # the repository for all later clients.
    env = {key: value for key, value in os.environ.items()
           if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    try:
        result = subprocess.run(
            ["git", *args], cwd=directory, env=env, capture_output=True,
            encoding="utf-8", errors="surrogateescape", timeout=2,
        )
        return result.stdout.rstrip("\r\n") if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def project_scope(directory: str | Path) -> ProjectScope:
    cwd = Path(directory).resolve()
    common = git_output(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        common = git_output(cwd, "rev-parse", "--git-common-dir")
    if common:
        identity = (cwd / common).resolve()
        root = git_output(cwd, "rev-parse", "--show-toplevel")
        worktree = Path(root) if root else cwd
        # Normal clones keep .git in the main worktree. External Git dirs can use another name.
        name = identity.parent.name if identity.name == ".git" else identity.name
        prefix = "repo"
    else:
        identity = worktree = cwd
        name = cwd.name or str(cwd)
        prefix = "folder"
    digest = hashlib.sha256(os.fsencode(os.path.normcase(str(identity)))).hexdigest()[:24]
    return ProjectScope(f"{prefix}:{digest}", name, worktree.name or name)
