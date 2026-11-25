"""Run a repo script with the repository-root .env loaded in-process.

Mirrors the pattern in conference_pipeline.config/cli (load_dotenv before
provider work) so live stage runners can be launched without exporting
credentials into shell state.

Usage:
    .venv/bin/python scripts/run_with_env.py <script.py> [args...]

Never prints secret values; load_dotenv only populates os.environ for this
process and its children.
"""

from __future__ import annotations

import os
import runpy
import sys

from dotenv import load_dotenv


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_env.py <script.py> [args...]")
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
