"""Small stdio relay for clients that launch one MCP command per session."""

import errno
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ._version import VERSION

PROTOCOL = 2
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
ENGINE_SETTINGS = {
    "MIDAS_MCP_EMBEDDER": "local",
    "MIDAS_MCP_MODEL_THREADS": "2",
    "MIDAS_MCP_ANN": "0",
    "MIDAS_MCP_KEY": "",
    "MIDAS_MCP_MAX_RECORDS": "0",
    "MIDAS_MCP_MIN_IMPORTANCE": "2",
    "MIDAS_MCP_SUPERSEDE": "1",
    "MIDAS_MCP_SUPERSEDE_CONVO": "0",
    "MIDAS_MCP_NLI": "0",
    "MIDAS_MCP_PINNED": "2",
    "MIDAS_MCP_AUTO_MAINTAIN": "0",
    "MIDAS_MCP_TTL": "",
}


def engine_fingerprint() -> str:
    settings = [os.getenv(name, default) for name, default in ENGINE_SETTINGS.items()]
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()


def memory_db(db: str | None = None) -> str:
    db = db or os.getenv("MIDAS_MCP_DB", str(Path.home() / ".midas" / "memory.sqlite3"))
    return os.path.normcase(str(Path(db).expanduser().resolve()))


@contextmanager
def try_lock(path: Path):
    """Hold an OS lock until the handle closes, including on a process crash."""
    with path.open("ab") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            yield False
        else:
            yield True


def _endpoint(db: str) -> Path:
    run_dir = Path.home() / ".midas" / "run"
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt" and (run_dir.stat().st_uid != os.getuid() or run_dir.stat().st_mode & 0o077):
        raise OSError(f"Midas needs a private runtime directory: {run_dir}")
    key = hashlib.sha256(os.fsencode(db)).hexdigest()[:24]
    return run_dir / f"{key}.json"


def _start_worker(db: str, endpoint: Path) -> subprocess.Popen:
    env = {**os.environ, "MIDAS_MCP_DB": db}
    # Detach from the MCP client and its pipes so other sessions survive that client's exit.
    detach = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} \
        if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(
        [sys.executable, "-P", "-m", "midas_mcp", str(endpoint)], cwd=endpoint.parent, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach,
    )


def connect(db: str | None = None, *, capture: bool = False) -> socket.socket:
    """Find or start the worker and authenticate before forwarding protocol traffic."""
    db = memory_db(db)
    endpoint = _endpoint(db)
    deadline = time.monotonic() + (2 if capture else 15)
    worker = None
    while time.monotonic() < deadline:
        sock = None
        ready = False
        try:
            address = json.loads(endpoint.read_text(encoding="utf-8"))
            sock = socket.create_connection(("127.0.0.1", address["port"]), timeout=2)
            sock.sendall(json.dumps({
                "token": address["token"], "protocol": PROTOCOL, "engine": engine_fingerprint(),
                "version": VERSION,
                "cwd": os.getcwd(), "client": os.getenv("MIDAS_MCP_CLIENT", "agent"),
                "actor": os.getenv("MIDAS_MCP_ACTOR", ""), "namespace": os.getenv("MIDAS_MCP_NAMESPACE", ""),
                "session": uuid.uuid4().hex, "capture": capture,
            }).encode() + b"\n")
            status = sock.recv(1)
            if status == b"!":
                raise RuntimeError("Midas is already active with a different version or engine settings. "
                                   "Close the Midas clients and allow the previous worker to exit.")
            if status == b"+":
                sock.settimeout(None)
                ready = True
                return sock
        except (OSError, ValueError, KeyError):
            pass  # The endpoint may be absent or left by a worker that exited.
        finally:
            if sock is not None and not ready:
                sock.close()

        if not capture and (worker is None or worker.poll() is not None):
            # Only the current worker publishes an endpoint. The startup lock prevents a wave of
            # simultaneous clients from spawning Python interpreters while that endpoint appears.
            with try_lock(endpoint.with_suffix(".start")) as acquired:
                if acquired:
                    with try_lock(endpoint.with_suffix(".lock")) as stopped:
                        if stopped:
                            endpoint.unlink(missing_ok=True)
                    if stopped:
                        worker = _start_worker(db, endpoint)
                        while (time.monotonic() < deadline and worker.poll() is None
                               and not endpoint.exists()):
                            time.sleep(0.05)
        time.sleep(0.05)
    raise RuntimeError("Midas could not start its local worker.")


def _relay(sock: socket.socket) -> None:
    stdin = sys.stdin.fileno()
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(stdin, os.O_BINARY)

    def send_requests() -> None:
        try:
            while chunk := os.read(stdin, 65536):
                sock.sendall(chunk)
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    threading.Thread(target=send_requests, daemon=True).start()
    while chunk := sock.recv(65536):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def _capture_prompt() -> None:
    import argparse
    import base64

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-prompt", choices=("claude-code", "codex"), required=True)
    parser.add_argument("--db-encoded")
    parser.add_argument("--project-scoped", action="store_true")
    args = parser.parse_args()
    if importlib.util.find_spec("mcp") is None:
        return
    try:
        db = base64.urlsafe_b64decode(args.db_encoded).decode() if args.db_encoded else None
        payload = json.loads(sys.stdin.buffer.read(MAX_MESSAGE_BYTES))
        event = {field: payload[field] for field in ("prompt", "session_id", "cwd")}
        event.update(client=args.capture_prompt, namespace="auto" if args.project_scoped else "")
        message = json.dumps(event, ensure_ascii=False).encode() + b"\n"
        prompt = event["prompt"]
        if len(message) <= MAX_MESSAGE_BYTES and isinstance(prompt, str) and prompt.strip():
            with connect(db, capture=True) as sock:
                sock.settimeout(2)
                sock.sendall(message)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"[midas-mcp] prompt capture unavailable: {type(exc).__name__}", file=sys.stderr)


def main() -> None:
    if "--capture-prompt" in sys.argv[1:]:
        _capture_prompt()
        return
    if os.getenv("MIDAS_MCP_DB") == ":memory:" or os.getenv("MIDAS_MCP_MODE") == "standalone":
        from midas.mcp_server import run_server

        run_server()
        return
    if importlib.util.find_spec("mcp") is None:
        raise SystemExit('Midas MCP server needs the MCP SDK: pip install "midas-memory[mcp]"')
    try:
        with connect() as sock:
            from .updates import check_updates, update_notice

            if notice := update_notice(check_updates(cached_only=True)):
                print(f"[midas-mcp] {notice}", file=sys.stderr)
            _relay(sock)
    except (BrokenPipeError, ConnectionResetError):
        return
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"[midas-mcp] {exc}") from None
