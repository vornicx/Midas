"""Internal entry point for the detached local worker."""

import sys
from pathlib import Path

from . import try_lock


def main() -> None:
    endpoint = Path(sys.argv[1])
    with try_lock(endpoint.with_suffix(".lock")) as acquired:
        if acquired:
            from midas.mcp_worker import run_worker

            run_worker(endpoint)


if __name__ == "__main__":
    main()
